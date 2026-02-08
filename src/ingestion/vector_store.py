"""
ChromaDB Vector Store Module

Manages persistent storage of embedded chunks in ChromaDB.
Provides retrieval with clinical metadata filtering for citations.
Updated to support rich clinical metadata from NG12 guidelines.

Interview Discussion Points:
---------------------------
1. Why ChromaDB?
   - Native metadata filtering (essential for clinical filtering)
   - Persistent storage with simple file-based backend
   - No external service required (embedded mode)
   - Easy Docker volume mounting

2. Collection design:
   - Single collection for NG12 (could extend to multiple guidelines)
   - Rich metadata schema for clinical filtering
   - Supports urgency, cancer type, symptom filtering

3. Query capabilities:
   - Semantic similarity search
   - Clinical metadata filtering (urgency, cancer type, symptoms)
   - Hybrid queries combining semantic + metadata
   - Page/section-based retrieval for citations

4. Clinical filtering strategies:
   - Filter by urgency level (urgent_2_week, routine, etc.)
   - Filter by cancer type for focused retrieval
   - Filter by investigation type (CT, MRI, etc.)
   - Combine filters for precise clinical queries

5. Production considerations:
   - Would migrate to managed service (Pinecone, Vertex AI Matching Engine)
   - Add caching layer for frequent queries
   - Implement query logging for analytics
   - Consider hybrid search (BM25 + semantic)
"""

from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
import sys

#sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from our embedder module
from src.ingestion.embedder import (
    RichChunk,
    EmbeddedChunk,
    ClinicalMetadata,
    cosine_similarity,
)


# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    ChromaSettings = None




# =============================================================================
# Configuration (standalone defaults)
# =============================================================================

class DefaultSettings:
    """Default settings when config module not available."""
    VECTORSTORE_DIR = Path("./vectorstore")
    CHROMA_COLLECTION_NAME = "ng12_chunks"

# Try to import settings, use defaults if not available
try:
    from src.config.settings import settings
except ImportError:
    settings = DefaultSettings()


# =============================================================================
# Retrieved Chunk with Clinical Metadata
# =============================================================================

@dataclass
class RetrievedChunk:
    """
    A chunk retrieved from the vector store with similarity score.
    
    Enhanced with full clinical metadata for filtering and display.
    
    Attributes:
        chunk_id: Unique identifier
        text: Chunk content
        score: Similarity score (higher = more similar, 0-1 for cosine)
        page_start: First page number
        page_end: Last page number
        section: Section header
        section_hierarchy: Hierarchical section path (e.g., "1|1.5|1.5.1")
        content_type: Type of content (text, table, list)
        clinical: Clinical metadata (cancer types, symptoms, urgency, etc.)
        metadata: Additional metadata dict
    """
    chunk_id: str
    text: str
    score: float
    page_start: int
    page_end: int
    section: str
    section_hierarchy: str = ""
    content_type: str = "text"
    clinical: ClinicalMetadata = field(default_factory=ClinicalMetadata)
    metadata: dict = field(default_factory=dict)
    
    def get_citation(self) -> str:
        """Generate a citation string for this chunk."""
        if self.page_start == self.page_end:
            page_ref = f"p.{self.page_start}"
        else:
            page_ref = f"pp.{self.page_start}-{self.page_end}"
        
        # Use section number from hierarchy if available
        # Hierarchy format: "1|1.4|1.4.1" - take the last (most specific) part
        if self.section_hierarchy:
            section_num = self.section_hierarchy.split("|")[-1]
            return f"[NG12 Section {section_num}, {page_ref}]"
        elif self.section:
            # Extract section number from section text (e.g., "1.4 Lung cancers" -> "1.4")
            section_short = self.section.split()[0] if self.section else ""
            return f"[NG12 {section_short}, {page_ref}]"
        return f"[NG12 {page_ref}]"
    
    def get_urgency_display(self) -> str:
        """Get human-readable urgency level."""
        urgency_map = {
            "urgent_2_week": "🔴 Urgent (2-week pathway)",
            "urgent": "🔴 Urgent",
            "soon": "🟡 Soon",
            "routine": "🟢 Routine",
            "": "⚪ Not specified"
        }
        return urgency_map.get(self.clinical.urgency, self.clinical.urgency)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "section_hierarchy": self.section_hierarchy,
            "content_type": self.content_type,
            "citation": self.get_citation(),
            "urgency_display": self.get_urgency_display(),
            # Clinical metadata
            "cancer_types": self.clinical.cancer_types,
            "symptoms": self.clinical.symptoms,
            "urgency": self.clinical.urgency,
            "actions": self.clinical.actions,
            "investigations": self.clinical.investigations,
            "is_urgent": self.clinical.is_urgent,
            "has_recommendation": self.clinical.has_recommendation,
            "has_table": self.clinical.has_table,
            # Extra metadata
            "metadata": self.metadata
        }
    
    @classmethod
    def from_chroma_result(
        cls,
        chunk_id: str,
        text: str,
        score: float,
        metadata: dict
    ) -> "RetrievedChunk":
        """Create from ChromaDB query result."""
        # Extract clinical metadata fields
        clinical = ClinicalMetadata(
            has_recommendation=metadata.get("has_recommendation", False),
            has_table=metadata.get("has_table", False),
            is_urgent=metadata.get("is_urgent", False),
            cancer_types=metadata.get("cancer_types", ""),
            symptoms=metadata.get("symptoms", ""),
            age_thresholds=metadata.get("age_thresholds", ""),
            timeframes=metadata.get("timeframes", ""),
            urgency=metadata.get("urgency", ""),
            actions=metadata.get("actions", ""),
            risk_factors=metadata.get("risk_factors", ""),
            investigations=metadata.get("investigations", ""),
        )
        
        return cls(
            chunk_id=chunk_id,
            text=text,
            score=score,
            page_start=metadata.get("page_start", 0),
            page_end=metadata.get("page_end", 0),
            section=metadata.get("section", ""),
            section_hierarchy=metadata.get("section_hierarchy", ""),
            content_type=metadata.get("content_type", "text"),
            clinical=clinical,
            metadata={
                "token_count": metadata.get("token_count", 0),
                "chunk_index": metadata.get("chunk_index", 0),
                "semantic_density": metadata.get("semantic_density", 0.0),
                "source": metadata.get("source", "NG12"),
            }
        )


# =============================================================================
# Vector Store
# =============================================================================

class VectorStore:
    """
    ChromaDB-based vector store for NG12 guideline chunks.
    
    Provides:
    - Persistent storage of embedded chunks with rich clinical metadata
    - Semantic similarity search
    - Clinical metadata filtering (urgency, cancer type, symptoms)
    - Citation-friendly retrieval
    
    Usage:
        # Initialize store
        store = VectorStore()
        
        # Add embedded chunks
        store.add_chunks(embedded_chunks)
        
        # Search with clinical filters
        results = store.search_by_urgency(query_embedding, "urgent_2_week")
        results = store.search_by_cancer_type(query_embedding, "lung")
        
        # Get citations
        for result in results:
            print(f"{result.get_citation()}: {result.text[:100]}...")
    """
    
    # Metadata fields stored in ChromaDB
    METADATA_FIELDS = [
        # Document location
        "page_start", "page_end", "section", "section_hierarchy",
        "content_type", "chunk_index","next_chunk_index", "prev_chunk_index",
        # Size metrics
        "token_count", "char_count", "semantic_density",
        # Source
        "source",
        # Clinical metadata (stored as strings for ChromaDB compatibility)
        "has_recommendation", "has_table", "is_urgent",
        "cancer_types", "symptoms", "age_thresholds", "timeframes",
        "urgency", "actions", "risk_factors", "investigations",
    ]
    
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        if not CHROMADB_AVAILABLE:
            raise RuntimeError(
                "ChromaDB not installed. Run: pip install chromadb"
            )
        
        self.persist_dir = Path(persist_dir or settings.VECTORSTORE_DIR)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        
        # Ensure directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self._collection = None
    
    def _get_collection(self):
        """Get or create the collection lazily."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "NICE NG12 Cancer Guidelines chunks with clinical metadata",
                    "source": "NG12",
                    "hnsw:space": "cosine"  # Use cosine similarity
                }
            )
        return self._collection
    
    def _chunk_to_metadata(self, chunk: RichChunk) -> dict:
        """
        Convert RichChunk to ChromaDB metadata dict.
        
        ChromaDB metadata values must be str, int, float, or bool.
        """
        metadata = {
            # Document location
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section": chunk.section or "",
            "section_hierarchy": chunk.section_hierarchy or "",
            "content_type": chunk.content_type,
            "chunk_index": chunk.chunk_index,
            "next_chunk_index": chunk.next_chunk_id or "",
            "prev_chunk_index": chunk.prev_chunk_id or "",
            # Size metrics
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "semantic_density": chunk.semantic_density,
            # Source
            "source": chunk.source or "",
            # Clinical metadata
            "has_recommendation": chunk.clinical.has_recommendation,
            "has_table": chunk.clinical.has_table,
            "is_urgent": chunk.clinical.is_urgent,
            "cancer_types": chunk.clinical.cancer_types or "",
            "symptoms": chunk.clinical.symptoms or "",
            "age_thresholds": chunk.clinical.age_thresholds or "",
            "timeframes": chunk.clinical.timeframes or "",
            "urgency": chunk.clinical.urgency or "",
            "actions": chunk.clinical.actions or "",
            "risk_factors": chunk.clinical.risk_factors or "",
            "investigations": chunk.clinical.investigations or "",
            }
        return  {k: v for k, v in metadata.items() if v is not None}
    
    def add_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
        show_progress: bool = True
    ) -> int:
        """
        Add embedded chunks to the vector store.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            show_progress: Whether to print progress
            
        Returns:
            Number of chunks added
        """
        if not embedded_chunks:
            return 0
        
        if show_progress:
            print(f"💾 Adding {len(embedded_chunks)} chunks to vector store...")
        
        collection = self._get_collection()
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for ec in embedded_chunks:
            ids.append(ec.chunk.chunk_id)
            embeddings.append(ec.embedding)
            documents.append(ec.chunk.text)
            metadatas.append(self._chunk_to_metadata(ec.chunk))
        
        # Add to collection (ChromaDB handles batching internally)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        if show_progress:
            print(f"   ✅ Added {len(embedded_chunks)} chunks")
            # Show clinical metadata summary
            urgent_count = sum(1 for ec in embedded_chunks if ec.chunk.clinical.is_urgent)
            rec_count = sum(1 for ec in embedded_chunks if ec.chunk.clinical.has_recommendation)
            print(f"   📊 {urgent_count} urgent, {rec_count} with recommendations")
        
        return len(embedded_chunks)
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
        min_score: Optional[float] = None
    ) -> list[RetrievedChunk]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            where: Metadata filter (ChromaDB where clause)
            min_score: Minimum similarity score (0-1 for cosine)
            
        Returns:
            List of RetrievedChunk objects, sorted by similarity
        """
        collection = self._get_collection()
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to RetrievedChunk objects
        retrieved = []
        
        if not results["ids"][0]:
            return retrieved
        
        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances, convert to similarity scores
            # For cosine distance: similarity = 1 - distance
            distance = results["distances"][0][i]
            score = 1 - distance
            
            # Apply minimum score filter
            if min_score is not None and score < min_score:
                continue
            
            metadata = results["metadatas"][0][i]
            text = results["documents"][0][i]
            
            retrieved.append(RetrievedChunk.from_chroma_result(
                chunk_id=chunk_id,
                text=text,
                score=score,
                metadata=metadata
            ))
        
        return retrieved
    
    # =========================================================================
    # Clinical Metadata Filters
    # =========================================================================
    
    def search_by_urgency(
        self,
        query_embedding: list[float],
        urgency: str,
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Search filtered by urgency level.
        
        Args:
            query_embedding: Query vector
            urgency: Urgency level (e.g., "urgent_2_week", "routine")
            top_k: Number of results
            
        Returns:
            List of RetrievedChunk objects
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"urgency": urgency}
        )
    
    def search_urgent_only(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Search only urgent chunks (is_urgent=True).
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of urgent chunks
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"is_urgent": True}
        )
    
    def search_by_cancer_type(
        self,
        query_embedding: list[float],
        cancer_type: str,
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Search filtered by cancer type.
        
        Uses $contains operator to match cancer type in comma-separated list.
        
        Args:
            query_embedding: Query vector
            cancer_type: Cancer type to filter (e.g., "lung", "breast")
            top_k: Number of results
            
        Returns:
            List of RetrievedChunk objects
        """
        # Get more results than needed, then filter post-hoc
        results = self.search(
            query_embedding=query_embedding,
            top_k=top_k * 3  # Get extra results to account for filtering
        )
        
        # Filter to only chunks containing this cancer type
        filtered = [
            r for r in results 
            if cancer_type.lower() in r.clinical.cancer_types.lower()
        ]
        
        return filtered[:top_k]
    
    def search_by_symptom(
        self,
        query_embedding: list[float],
        symptom: str,
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Search filtered by symptom.
        
        Args:
            query_embedding: Query vector
            symptom: Symptom to filter (e.g., "weight_loss", "cough")
            top_k: Number of results
            
        Returns:
            List of RetrievedChunk objects
        """
        # Get more results than needed, then filter post-hoc
        results = self.search(
            query_embedding=query_embedding,
            top_k=top_k * 3
        )
        
        # Filter to only chunks containing this symptom
        filtered = [
            r for r in results 
            if symptom.lower() in r.clinical.symptoms.lower()
        ]
        
        return filtered[:top_k]
    
    def search_by_investigation(
        self,
        query_embedding: list[float],
        investigation: str,
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Search filtered by investigation type.
        
        Args:
            query_embedding: Query vector
            investigation: Investigation to filter (e.g., "ct", "mri", "fit")
            top_k: Number of results
            
        Returns:
            List of RetrievedChunk objects
        """
        # Get more results than needed, then filter post-hoc
        results = self.search(
            query_embedding=query_embedding,
            top_k=top_k * 3
        )
        
        # Filter to only chunks containing this investigation
        filtered = [
            r for r in results 
            if investigation.lower() in r.clinical.investigations.lower()
        ]
        
        return filtered[:top_k]
    
    def search_recommendations(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Search specifically for recommendation chunks.
        
        Useful for risk assessment where we need actual guidelines.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of recommendation chunks
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"has_recommendation": True}
        )
    
    def search_tables(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Search specifically for table content.
        
        Tables often contain structured clinical criteria.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of table chunks
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"has_table": True}
        )
    
    def search_by_content_type(
        self,
        query_embedding: list[float],
        content_type: str,
        top_k: int = 5
    ) -> list[RetrievedChunk]:
        """
        Search filtered by content type.
        
        Args:
            query_embedding: Query vector
            content_type: One of: text, table, list, etc.
            top_k: Number of results
            
        Returns:
            List of RetrievedChunk objects
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"content_type": content_type}
        )
    
    def search_with_clinical_filter(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        cancer_type: Optional[str] = None,
        urgency: Optional[str] = None,
        is_urgent: Optional[bool] = None,
        has_recommendation: Optional[bool] = None,
        symptom: Optional[str] = None,
        investigation: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Search with multiple clinical filters combined.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            cancer_type: Filter by cancer type
            urgency: Filter by urgency level
            is_urgent: Filter by urgent flag
            has_recommendation: Filter for recommendations
            symptom: Filter by symptom
            investigation: Filter by investigation
            
        Returns:
            List of RetrievedChunk objects matching all filters
        """
        # Build filter conditions
        conditions = []
        
        if cancer_type:
            conditions.append({"cancer_types": {"$contains": cancer_type}})
        if urgency:
            conditions.append({"urgency": urgency})
        if is_urgent is not None:
            conditions.append({"is_urgent": is_urgent})
        if has_recommendation is not None:
            conditions.append({"has_recommendation": has_recommendation})
        if symptom:
            conditions.append({"symptoms": {"$contains": symptom}})
        if investigation:
            conditions.append({"investigations": {"$contains": investigation}})
        
        # Combine with $and if multiple conditions
        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}
        
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where
        )
    
    # =========================================================================
    # Page/Section Retrieval
    # =========================================================================
    
    def get_by_page(self, page_number: int) -> list[RetrievedChunk]:
        """
        Get all chunks from a specific page.
        
        Useful for exploring what's on a page cited elsewhere.
        
        Args:
            page_number: Page number to retrieve
            
        Returns:
            List of chunks from that page
        """
        collection = self._get_collection()
        
        results = collection.get(
            where={
                "$and": [
                    {"page_start": {"$lte": page_number}},
                    {"page_end": {"$gte": page_number}}
                ]
            },
            include=["documents", "metadatas"]
        )
        
        retrieved = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            retrieved.append(RetrievedChunk.from_chroma_result(
                chunk_id=chunk_id,
                text=results["documents"][i],
                score=1.0,  # Not a similarity search
                metadata=metadata
            ))
        
        return retrieved
    
    def get_by_section(self, section_prefix: str) -> list[RetrievedChunk]:
        """
        Get all chunks from a section (by hierarchy prefix).
        
        Args:
            section_prefix: Section hierarchy prefix (e.g., "1|1.5" for section 1.5.x)
            
        Returns:
            List of chunks in that section
        """
        collection = self._get_collection()
        
        # Get all and filter (ChromaDB doesn't support startswith)
        results = collection.get(include=["documents", "metadatas"])
        
        retrieved = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            hierarchy = metadata.get("section_hierarchy", "")
            
            if hierarchy.startswith(section_prefix):
                retrieved.append(RetrievedChunk.from_chroma_result(
                    chunk_id=chunk_id,
                    text=results["documents"][i],
                    score=1.0,
                    metadata=metadata
                ))
        
        return retrieved
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[RetrievedChunk]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            RetrievedChunk or None if not found
        """
        collection = self._get_collection()
        
        results = collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return None
        
        return RetrievedChunk.from_chroma_result(
            chunk_id=results["ids"][0],
            text=results["documents"][0],
            score=1.0,
            metadata=results["metadatas"][0]
        )
    
    # =========================================================================
    # Statistics and Management
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        collection = self._get_collection()
        count = collection.count()
        
        # Get all metadata for detailed stats
        if count > 0:
            all_data = collection.get(include=["metadatas"])
            metadatas = all_data["metadatas"]
            
            # Count by various fields
            urgent_count = sum(1 for m in metadatas if m.get("is_urgent"))
            rec_count = sum(1 for m in metadatas if m.get("has_recommendation"))
            table_count = sum(1 for m in metadatas if m.get("has_table"))
            
            # Unique cancer types
            cancer_types = set()
            for m in metadatas:
                if m.get("cancer_types"):
                    cancer_types.update(m["cancer_types"].split(","))
            
            # Unique urgency levels
            urgency_levels = set(m.get("urgency", "") for m in metadatas if m.get("urgency"))
        else:
            urgent_count = rec_count = table_count = 0
            cancer_types = set()
            urgency_levels = set()
        
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": str(self.persist_dir),
            "urgent_chunks": urgent_count,
            "recommendation_chunks": rec_count,
            "table_chunks": table_count,
            "cancer_types": sorted(cancer_types),
            "urgency_levels": sorted(urgency_levels),
        }
    
    def clear(self):
        """Clear all data from the collection."""
        collection = self._get_collection()
        # Get all IDs and delete
        all_data = collection.get()
        if all_data["ids"]:
            collection.delete(ids=all_data["ids"])
            print(f"   🗑️ Cleared {len(all_data['ids'])} chunks")
        else:
            print("   🗑️ Collection already empty")
    
    def reset(self):
        """Delete and recreate the collection."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection may not exist
        self._collection = None
        print(f"   🔄 Reset collection: {self.collection_name}")


# =============================================================================
# Mock Vector Store (for testing without ChromaDB)
# =============================================================================

class MockVectorStore:
    """
    In-memory mock vector store for testing.
    
    Stores chunks in memory and performs brute-force similarity search.
    Useful for unit tests without ChromaDB dependency.
    """
    
    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = "mock"):
        self.collection_name = collection_name
        self._chunks: dict[str, EmbeddedChunk] = {}
    
    def add_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
        show_progress: bool = True
    ) -> int:
        if show_progress:
            print(f"💾 Mock adding {len(embedded_chunks)} chunks...")
        
        for ec in embedded_chunks:
            self._chunks[ec.chunk.chunk_id] = ec
        
        if show_progress:
            print(f"   ✅ Added {len(embedded_chunks)} chunks")
        
        return len(embedded_chunks)
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
        min_score: Optional[float] = None
    ) -> list[RetrievedChunk]:
        # Compute similarities
        scored = []
        for chunk_id, ec in self._chunks.items():
            # Apply basic where filter
            if where and not self._matches_filter(ec.chunk, where):
                continue
            
            score = cosine_similarity(query_embedding, ec.embedding)
            
            if min_score is not None and score < min_score:
                continue
            
            scored.append((ec, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        results = []
        for ec, score in scored[:top_k]:
            results.append(RetrievedChunk(
                chunk_id=ec.chunk.chunk_id,
                text=ec.chunk.text,
                score=score,
                page_start=ec.chunk.page_start,
                page_end=ec.chunk.page_end,
                section=ec.chunk.section,
                section_hierarchy=ec.chunk.section_hierarchy,
                content_type=ec.chunk.content_type,
                clinical=ec.chunk.clinical,
                metadata={
                    "token_count": ec.chunk.token_count,
                    "source": ec.chunk.source,
                }
            ))
        
        return results
    
    def _matches_filter(self, chunk: RichChunk, where: dict) -> bool:
        """Check if chunk matches the filter."""
        for key, value in where.items():
            if key == "$and":
                return all(self._matches_filter(chunk, cond) for cond in value)
            elif key == "$or":
                return any(self._matches_filter(chunk, cond) for cond in value)
            
            # Get attribute value
            if hasattr(chunk, key):
                attr_val = getattr(chunk, key)
            elif hasattr(chunk.clinical, key):
                attr_val = getattr(chunk.clinical, key)
            else:
                return False
            
            # Handle operators
            if isinstance(value, dict):
                if "$contains" in value:
                    if value["$contains"] not in str(attr_val):
                        return False
                elif "$eq" in value:
                    if attr_val != value["$eq"]:
                        return False
            else:
                if attr_val != value:
                    return False
        
        return True
    
    def search_urgent_only(self, query_embedding: list[float], top_k: int = 5):
        return self.search(query_embedding, top_k, where={"is_urgent": True})
    
    def search_by_cancer_type(self, query_embedding: list[float], cancer_type: str, top_k: int = 5):
        return self.search(query_embedding, top_k, where={"cancer_types": {"$contains": cancer_type}})
    
    def search_recommendations(self, query_embedding: list[float], top_k: int = 5):
        return self.search(query_embedding, top_k, where={"has_recommendation": True})
    
    def get_stats(self) -> dict:
        return {
            "collection_name": self.collection_name,
            "total_chunks": len(self._chunks),
            "persist_dir": "memory",
        }
    
    def clear(self):
        self._chunks.clear()
        print("   🗑️ Cleared mock store")
    
    def reset(self):
        self.clear()


# =============================================================================
# Factory Function
# =============================================================================

def get_vector_store(
    persist_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    mock: bool = False
) -> VectorStore | MockVectorStore:
    """
    Factory function to get a vector store instance.
    
    Args:
        persist_dir: Override default persistence directory
        collection_name: Override default collection name
        mock: If True, return mock store for testing
        
    Returns:
        VectorStore or MockVectorStore instance
    """
    if mock or not CHROMADB_AVAILABLE:
        return MockVectorStore(persist_dir, collection_name or "mock")
    
    return VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name
    )


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    from embedder import RichChunk, MockEmbedder, get_embedder
    
    print("=" * 60)
    print("Vector Store Demo")
    print("=" * 60)
    
    # Create sample chunks
    sample_chunks = [
        {
            "chunk_id": "ng12_001",
            "text": "Refer patients with unexplained splenomegaly using urgent 2-week pathway for suspected lymphoma.",
            "page_start": 22,
            "page_end": 22,
            "section": "1.10 Haematological cancers",
            "section_hierarchy": "1|1.10",
            "content_type": "text",
            "cancer_types": "lymphoma",
            "symptoms": "splenomegaly",
            "urgency": "urgent_2_week",
            "is_urgent": True,
            "has_recommendation": True,
            "actions": "refer",
        },
        {
            "chunk_id": "ng12_002",
            "text": "Consider chest X-ray for patients over 40 with persistent cough lasting more than 3 weeks.",
            "page_start": 15,
            "page_end": 15,
            "section": "1.4 Lung and pleural cancers",
            "section_hierarchy": "1|1.4",
            "content_type": "text",
            "cancer_types": "lung",
            "symptoms": "cough",
            "urgency": "routine",
            "is_urgent": False,
            "has_recommendation": True,
            "actions": "consider",
            "investigations": "chest_xray",
            "age_thresholds": "40",
        },
        {
            "chunk_id": "ng12_003",
            "text": "Breast cancer symptoms include breast lump, skin changes, and nipple discharge.",
            "page_start": 8,
            "page_end": 8,
            "section": "1.2 Breast cancer",
            "section_hierarchy": "1|1.2",
            "content_type": "text",
            "cancer_types": "breast",
            "symptoms": "lump,skin_changes,nipple_discharge",
            "urgency": "urgent_2_week",
            "is_urgent": True,
            "has_recommendation": False,
        },
    ]
    
    # Create RichChunks
    chunks = [RichChunk.from_dict(c) for c in sample_chunks]
    print(f"\n📄 Created {len(chunks)} sample chunks")
    
    # Embed chunks
    embedder = get_embedder(mock=False)
    embedded = embedder.embed_chunks(chunks, include_metadata=True, show_progress=False)
    print(f"📊 Embedded {len(embedded)} chunks")
    
    # Create mock vector store
    store = get_vector_store(mock=False)
    store.add_chunks(embedded, show_progress=False)
    
    # Test searches
    print("\n" + "-" * 40)
    print("Search Tests")
    print("-" * 40)
    
    # Search for lymphoma
    query = "lymphoma splenomegaly referral"
    query_emb = embedder.embed_query(query)
    
    print(f"\n🔍 Query: '{query}'")
    results = store.search(query_emb, top_k=3)
    for r in results:
        print(f"   [{r.score:.4f}] {r.get_citation()} - {r.text[:50]}...")
    
    # Search urgent only
    print(f"\n🔍 Urgent chunks only:")
    results = store.search_urgent_only(query_emb, top_k=3)
    for r in results:
        print(f"   [{r.score:.4f}] {r.get_urgency_display()} - {r.chunk_id}")
    
    # Search by cancer type
    print(f"\n🔍 Lung cancer chunks:")
    results = store.search_by_cancer_type(query_emb, "lung", top_k=3)
    for r in results:
        print(f"   [{r.score:.4f}] {r.get_citation()} - {r.clinical.symptoms}")
    
    # Stats
    print(f"\n📊 Store stats: {store.get_stats()}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")