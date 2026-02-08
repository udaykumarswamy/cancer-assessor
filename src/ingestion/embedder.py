"""
Vertex AI Embeddings Module

Creates embeddings using Google's Vertex AI text-embedding-004 model.
Handles batching, rate limiting, and error recovery.
Updated to support rich clinical metadata chunks from NG12 guidelines.

Interview Discussion Points:
---------------------------
1. Why text-embedding-004?
   - Latest Vertex AI embedding model (as of 2024)
   - 768 dimensions (good balance of quality vs. storage)
   - Optimized for retrieval tasks
   - Supports task_type hints for better embeddings

2. Batching strategy:
   - Vertex AI has batch limits (250 texts per request)
   - Batching reduces API calls and latency
   - We process in parallel where possible

3. Error handling:
   - Exponential backoff for rate limits
   - Retry logic for transient failures
   - Graceful degradation (return partial results)

4. Task types for embeddings:
   - RETRIEVAL_DOCUMENT: For chunks being indexed
   - RETRIEVAL_QUERY: For search queries
   - Different task types optimize embedding space

5. Metadata-enhanced embeddings:
   - Option to prepend structured metadata to text
   - Improves retrieval for clinical queries
   - Configurable metadata fields for embedding
"""

from typing import Optional, Protocol, runtime_checkable
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Try to import tenacity for retry logic, fallback to simple implementation
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Try to import Google Cloud libraries
try:
    from google.api_core import exceptions as google_exceptions
    import vertexai
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    google_exceptions = None

# Try to import settings, provide defaults if not available
try:
    from src.config.settings import settings
except ImportError:
    class DefaultSettings:
        GCP_PROJECT_ID = "cancer-assessor-poc"
        GCP_LOCATION = "us-central1"
        EMBEDDING_MODEL = "text-embedding-004"
        EMBEDDING_DIMENSION = 768
    settings = DefaultSettings()


# =============================================================================
# Rich Clinical Chunk Data Classes
# =============================================================================

@dataclass
class ClinicalMetadata:
    """
    Clinical metadata extracted from NG12 guideline chunks.
    
    Interview Discussion Points:
    - Structured metadata enables filtered retrieval
    - Urgency levels map to clinical pathways
    - Cancer types and symptoms enable semantic matching
    """
    has_recommendation: bool = False
    has_table: bool = False
    is_urgent: bool = False
    cancer_types: str = ""  # Comma-separated
    symptoms: str = ""  # Comma-separated
    age_thresholds: str = ""
    timeframes: str = ""
    urgency: str = ""  # e.g., "urgent_2_week"
    actions: str = ""  # e.g., "offer,refer,consider"
    risk_factors: str = ""
    investigations: str = ""
    
    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalMetadata":
        """Create from dictionary, handling missing fields gracefully."""
        return cls(
            has_recommendation=data.get("has_recommendation", False),
            has_table=data.get("has_table", False),
            is_urgent=data.get("is_urgent", False),
            cancer_types=data.get("cancer_types", ""),
            symptoms=data.get("symptoms", ""),
            age_thresholds=data.get("age_thresholds", ""),
            timeframes=data.get("timeframes", ""),
            urgency=data.get("urgency", ""),
            actions=data.get("actions", ""),
            risk_factors=data.get("risk_factors", ""),
            investigations=data.get("investigations", ""),
        )
    
    def to_embedding_prefix(self) -> str:
        """
        Convert metadata to a text prefix for embedding enhancement.
        
        This creates a structured text representation that helps the
        embedding model understand the clinical context.
        """
        parts = []
        
        if self.cancer_types:
            cancer_list = self.cancer_types.replace(",", ", ").replace("_", " ")
            parts.append(f"Cancer types: {cancer_list}")
        
        if self.symptoms:
            symptom_list = self.symptoms.replace(",", ", ").replace("_", " ")
            parts.append(f"Symptoms: {symptom_list}")
        
        if self.urgency:
            urgency_text = self.urgency.replace("_", " ")
            parts.append(f"Urgency: {urgency_text}")
        
        if self.actions:
            action_list = self.actions.replace(",", ", ")
            parts.append(f"Actions: {action_list}")
        
        if self.investigations:
            inv_list = self.investigations.replace(",", ", ").upper()
            parts.append(f"Investigations: {inv_list}")
        
        if self.age_thresholds:
            parts.append(f"Age criteria: {self.age_thresholds}")
        
        return " | ".join(parts) if parts else ""


@dataclass
class RichChunk:
    """
    A chunk with rich clinical metadata from NG12 guidelines.
    
    This is the updated chunk format that includes:
    - Full document context (section hierarchy, page range)
    - Clinical metadata (cancer types, symptoms, urgency)
    - Linking information (prev/next chunk IDs)
    - Quality metrics (semantic density, token count)
    """
    # Core content
    chunk_id: str
    text: str
    
    # Document location
    page_start: int
    page_end: int
    section: str = ""
    section_hierarchy: str = ""
    
    # Content classification
    content_type: str = "text"  # text, table, list, etc.
    
    # Size metrics
    token_count: int = 0
    char_count: int = 0
    chunk_index: int = 0
    
    # Linking
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    
    # Quality metrics
    semantic_density: float = 0.0
    
    # Metadata
    created_at: str = ""
    source: str = "NG12"
    
    # Clinical metadata (flattened for easy access)
    clinical: ClinicalMetadata = field(default_factory=ClinicalMetadata)
    
    @classmethod
    def from_dict(cls, data: dict) -> "RichChunk":
        """Create RichChunk from dictionary (e.g., loaded from JSON)."""
        # Extract clinical metadata fields
        clinical_fields = {
            "has_recommendation", "has_table", "is_urgent", "cancer_types",
            "symptoms", "age_thresholds", "timeframes", "urgency", "actions",
            "risk_factors", "investigations"
        }
        clinical_data = {k: v for k, v in data.items() if k in clinical_fields}
        clinical = ClinicalMetadata.from_dict(clinical_data)
        
        return cls(
            chunk_id=data.get("chunk_id", ""),
            text=data.get("text", ""),
            page_start=data.get("page_start", 0),
            page_end=data.get("page_end", 0),
            section=data.get("section", ""),
            section_hierarchy=data.get("section_hierarchy", ""),
            content_type=data.get("content_type", "text"),
            token_count=data.get("token_count", 0),
            char_count=data.get("char_count", 0),
            chunk_index=data.get("chunk_index", 0),
            prev_chunk_id=data.get("prev_chunk_id"),
            next_chunk_id=data.get("next_chunk_id"),
            semantic_density=data.get("semantic_density", 0.0),
            created_at=data.get("created_at", ""),
            source=data.get("source", "NG12"),
            clinical=clinical,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "section_hierarchy": self.section_hierarchy,
            "content_type": self.content_type,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "chunk_index": self.chunk_index,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "semantic_density": self.semantic_density,
            "created_at": self.created_at,
            "source": self.source,
            # Flatten clinical metadata
            **asdict(self.clinical),
        }
        return result
    
    def get_embedding_text(self, include_metadata: bool = True) -> str:
        """
        Get the text to embed, optionally with metadata prefix.
        
        Args:
            include_metadata: If True, prepend clinical metadata
            
        Returns:
            Text optimized for embedding
        """
        if not include_metadata:
            return self.text
        
        prefix = self.clinical.to_embedding_prefix()
        if prefix:
            return f"{prefix}\n\n{self.text}"
        return self.text


@dataclass
class EmbeddedChunk:
    """
    A chunk with its embedding vector.
    
    Attributes:
        chunk: Original RichChunk object
        embedding: Embedding vector (768 dimensions for text-embedding-004)
        metadata_enhanced: Whether metadata was included in embedding
    """
    chunk: RichChunk
    embedding: list[float]
    metadata_enhanced: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "chunk": self.chunk.to_dict(),
            "embedding": self.embedding,
            "metadata_enhanced": self.metadata_enhanced,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddedChunk":
        """Load from dictionary."""
        return cls(
            chunk=RichChunk.from_dict(data["chunk"]),
            embedding=data["embedding"],
            metadata_enhanced=data.get("metadata_enhanced", False),
        )


# =============================================================================
# Embedder Protocol (for type hints)
# =============================================================================

@runtime_checkable
class Embedder(Protocol):
    """Protocol defining the embedder interface."""
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...
    
    def embed_chunks(
        self,
        chunks: list[RichChunk],
        show_progress: bool = True,
        include_metadata: bool = True
    ) -> list[EmbeddedChunk]:
        """Embed multiple chunks."""
        ...
    
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        ...
    
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed multiple queries."""
        ...


# =============================================================================
# Vertex AI Embedder
# =============================================================================

class VertexAIEmbedder:
    """
    Embedder using Vertex AI text-embedding-004.
    
    Handles:
    - Initialization of Vertex AI client
    - Batched embedding generation
    - Rate limit handling
    - Different task types for documents vs. queries
    - Metadata-enhanced embeddings for clinical content
    
    Usage:
        embedder = VertexAIEmbedder()
        
        # Embed chunks for indexing (with metadata enhancement)
        embedded = embedder.embed_chunks(chunks, include_metadata=True)
        
        # Embed a query for retrieval
        query_embedding = embedder.embed_query("symptoms for lung cancer")
    """
    
    # Vertex AI batch limits
    MAX_BATCH_SIZE = 250
    
    # Task types for different use cases
    TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
    TASK_QUERY = "RETRIEVAL_QUERY"
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the embedder.
        
        Args:
            project_id: GCP project ID (defaults to settings)
            location: GCP region (defaults to settings)
            model_name: Embedding model name (defaults to settings)
        """
        self.project_id = project_id or settings.GCP_PROJECT_ID
        self.location = location or settings.GCP_LOCATION
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        self._model = None
        self._initialized = False
        
        if not VERTEX_AI_AVAILABLE:
            print("⚠️ Warning: Vertex AI libraries not installed. Use MockEmbedder for testing.")
    
    def _ensure_initialized(self):
        """Initialize Vertex AI client lazily."""
        if not VERTEX_AI_AVAILABLE:
            raise RuntimeError(
                "Vertex AI libraries not installed. "
                "Run: pip install google-cloud-aiplatform vertexai"
            )
        
        if not self._initialized:
            print(f"   Initializing Vertex AI ({self.project_id}, {self.location})...")
            vertexai.init(
                project=self.project_id,
                location=self.location
            )
            self._model = TextEmbeddingModel.from_pretrained(self.model_name)
            self._initialized = True
    
    def _embed_batch_with_retry(
        self,
        texts: list[str],
        task_type: str,
        max_retries: int = 3
    ) -> list[list[float]]:
        """
        Embed a batch of texts with manual retry logic.
        
        Falls back to this when tenacity is not available.
        """
        self._ensure_initialized()
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Create embedding inputs with task type
                inputs = [
                    TextEmbeddingInput(text=text, task_type=task_type)
                    for text in texts
                ]
                
                # Get embeddings
                embeddings = self._model.get_embeddings(inputs)
                return [emb.values for emb in embeddings]
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"   Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
        
        raise last_exception
    
    def _embed_batch_with_token_limit(
    self,
    chunks: list[RichChunk],
    include_metadata: bool = True,
    max_tokens_per_batch: int = 15000  # Conservative limit (model max is 20000)
    ) -> tuple[list[list[float]], list[RichChunk]]:
        """
        Embed a batch of chunks, respecting token limit.
        
        Splits chunks into smaller batches if total tokens exceed limit.
        """
        # Calculate total tokens for all chunks
        texts = [chunk.get_embedding_text(include_metadata) for chunk in chunks]
        
        # Simple token approximation: ~4 chars per token
        text_tokens = [len(t) // 4 for t in texts]
        
        # Group chunks into batches respecting token limit
        batches = []
        current_batch = []
        current_tokens = 0
        
        for chunk, text, tokens in zip(chunks, texts, text_tokens):
            if current_tokens + tokens > max_tokens_per_batch and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = tokens
            else:
                current_batch.append(chunk)
                current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
        
        # Embed all batches
        all_embeddings = []
        all_chunks = []
        
        for batch_idx, batch in enumerate(batches, 1):
            batch_texts = [chunk.get_embedding_text(include_metadata) for chunk in batch]
            batch_tokens = sum(len(t) // 4 for t in batch_texts)
            
            print(f"   Token-aware batch {batch_idx}/{len(batches)} ({len(batch)} chunks, ~{batch_tokens} tokens)...")
            
            try:
                embeddings = self._embed_batch_with_retry(batch_texts, self.TASK_DOCUMENT)
                all_embeddings.extend(embeddings)
                all_chunks.extend(batch)
            except Exception as e:
                print(f"   ⚠️ Error in batch {batch_idx}: {e}")
                continue
            
            # Small delay between batches
            if batch_idx < len(batches):
                time.sleep(0.5)
        
        return all_embeddings, all_chunks


    def embed_chunks(
        self,
        chunks: list[RichChunk],
        show_progress: bool = True,
        include_metadata: bool = True
    ) -> list[EmbeddedChunk]:
        """Generate embeddings for chunks with token-aware batching."""
        if not chunks:
            return []
        
        if show_progress:
            meta_str = " (with metadata)" if include_metadata else ""
            print(f"📊 Embedding {len(chunks)} chunks{meta_str}...")
        
        # Use token-aware batching instead of fixed batch size
        embeddings, embedded_chunks = self._embed_batch_with_token_limit(
            chunks,
            include_metadata=include_metadata,
            max_tokens_per_batch=15000  # Conservative: model supports 20000
        )
        
        # Pair chunks with embeddings
        embedded_list = []
        for chunk, embedding in zip(embedded_chunks, embeddings):
            embedded_list.append(EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                metadata_enhanced=include_metadata
            ))
        
        if show_progress:
            print(f"   ✅ Embedded {len(embedded_list)}/{len(chunks)} chunks")
        
        return embedded_list
    # def embed_chunks(
    #     self,
    #     chunks: list[RichChunk],
    #     show_progress: bool = True,
    #     include_metadata: bool = True
    # ) -> list[EmbeddedChunk]:
    #     """
    #     Embed multiple chunks for document indexing.
        
    #     Uses RETRIEVAL_DOCUMENT task type for optimal indexing embeddings.
    #     Processes in batches to handle rate limits.
        
    #     Args:
    #         chunks: List of RichChunk objects to embed
    #         show_progress: Whether to print progress
    #         include_metadata: Whether to include clinical metadata in embedding
            
    #     Returns:
    #         List of EmbeddedChunk objects
    #     """
    #     if not chunks:
    #         return []
        
    #     if show_progress:
    #         meta_str = " (with metadata)" if include_metadata else ""
    #         print(f"📊 Embedding {len(chunks)} chunks{meta_str}...")
        
    #     embedded_chunks = []
    #     total_batches = (len(chunks) + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE
        
    #     for batch_idx in range(0, len(chunks), self.MAX_BATCH_SIZE):
    #         batch = chunks[batch_idx:batch_idx + self.MAX_BATCH_SIZE]
    #         batch_num = batch_idx // self.MAX_BATCH_SIZE + 1
            
    #         if show_progress:
    #             print(f"   Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
    #         # Extract texts (with or without metadata)
    #         texts = [chunk.get_embedding_text(include_metadata) for chunk in batch]
            
    #         try:
    #             # Get embeddings with retry
    #             embeddings = self._embed_batch_with_retry(texts, self.TASK_DOCUMENT)
                
    #             # Pair chunks with embeddings
    #             for chunk, embedding in zip(batch, embeddings):
    #                 embedded_chunks.append(EmbeddedChunk(
    #                     chunk=chunk,
    #                     embedding=embedding,
    #                     metadata_enhanced=include_metadata
    #                 ))
                
    #         except Exception as e:
    #             print(f"   ⚠️ Error in batch {batch_num}: {e}")
    #             continue
            
    #         # Small delay between batches to avoid rate limits
    #         if batch_idx + self.MAX_BATCH_SIZE < len(chunks):
    #             time.sleep(0.5)
        
    #     if show_progress:
    #         print(f"   ✅ Embedded {len(embedded_chunks)}/{len(chunks)} chunks")
        
    #     return embedded_chunks
    
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query.
        
        Uses RETRIEVAL_QUERY task type for optimal query embeddings.
        Query embeddings are optimized to match document embeddings.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector (768 dimensions)
        """
        embeddings = self._embed_batch_with_retry([query], self.TASK_QUERY)
        return embeddings[0]
    
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """
        Embed multiple search queries.
        
        Args:
            queries: List of search query texts
            
        Returns:
            List of embedding vectors
        """
        return self._embed_batch_with_retry(queries, self.TASK_QUERY)
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return settings.EMBEDDING_DIMENSION


# =============================================================================
# Mock Embedder (for testing)
# =============================================================================

class MockEmbedder:
    """
    Mock embedder for testing without Vertex AI credentials.
    
    Generates deterministic embeddings based on text content.
    Useful for development and testing the pipeline.
    
    Interview Discussion Points:
    - Deterministic hashing ensures reproducible tests
    - Same dimension as real embedder for pipeline compatibility
    - Simulates metadata enhancement behavior
    """
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._seed = 42
    
    def _text_to_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        import random
        import hashlib
        
        # Use SHA256 for more uniform distribution
        text_hash = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        random.seed(text_hash)
        
        # Generate normalized embedding
        embedding = [random.gauss(0, 1) for _ in range(self.dimension)]
        
        # L2 normalize
        norm = sum(x**2 for x in embedding) ** 0.5
        return [x / norm for x in embedding]
    
    def embed_chunks(
        self,
        chunks: list[RichChunk],
        show_progress: bool = True,
        include_metadata: bool = True
    ) -> list[EmbeddedChunk]:
        """Generate mock embeddings for chunks."""
        if show_progress:
            meta_str = " (with metadata)" if include_metadata else ""
            print(f"📊 Mock embedding {len(chunks)} chunks{meta_str}...")
        
        embedded = []
        for chunk in chunks:
            # Get text (with or without metadata prefix)
            text = chunk.get_embedding_text(include_metadata)
            embedding = self._text_to_embedding(text)
            
            embedded.append(EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                metadata_enhanced=include_metadata
            ))
        
        if show_progress:
            print(f"   ✅ Mock embedded {len(embedded)} chunks")
        
        return embedded
    
    def embed_query(self, query: str) -> list[float]:
        """Generate mock embedding for query."""
        return self._text_to_embedding(query)
    
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Generate mock embeddings for queries."""
        return [self.embed_query(q) for q in queries]


# =============================================================================
# Utility Functions
# =============================================================================

def get_embedder(mock: bool = False) -> VertexAIEmbedder | MockEmbedder:
    """
    Factory function to get an embedder instance.
    
    Args:
        mock: If True, return mock embedder for testing
        
    Returns:
        Embedder instance
    """
    if mock or not VERTEX_AI_AVAILABLE:
        return MockEmbedder()
    return VertexAIEmbedder()


def load_chunks_from_json(filepath: str | Path) -> list[RichChunk]:
    """
    Load chunks from a JSON file.
    
    Args:
        filepath: Path to JSON file containing chunk list
        
    Returns:
        List of RichChunk objects
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle both list and single chunk
    if isinstance(data, list):
        return [RichChunk.from_dict(item) for item in data]
    return [RichChunk.from_dict(data)]


def save_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
    filepath: str | Path
) -> None:
    """
    Save embedded chunks to JSON file.
    
    Args:
        embedded_chunks: List of EmbeddedChunk objects
        filepath: Output file path
    """
    filepath = Path(filepath)
    data = [ec.to_dict() for ec in embedded_chunks]
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved {len(embedded_chunks)} embedded chunks to {filepath}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Vertex AI Embedder Demo (with Mock)")
    print("=" * 60)
    
    # Create sample chunk matching the JSON format
    sample_chunk_data = {
        "chunk_id": "ng12_p22_0035_9aa76837",
        "text": "Splenomegaly (unexplained) in adults may indicate Non-Hodgkin's lymphoma. Consider a suspected cancer pathway referral. When considering referral, take into account any associated symptoms, particularly fever, night sweats, shortness of breath, pruritus or weight loss.",
        "page_start": 22,
        "page_end": 22,
        "section": "1.13.1 Take into account the insight and knowledge of parents and carers",
        "section_hierarchy": "1|1.13|1.13.1",
        "content_type": "table",
        "token_count": 678,
        "char_count": 2712,
        "chunk_index": 35,
        "prev_chunk_id": "ng12_p21_0034_79c7cd4e",
        "next_chunk_id": "ng12_p22_0036_da01a2c9",
        "semantic_density": 0.258,
        "created_at": "2026-02-06T13:23:14.521858",
        "source": "NG12",
        "has_recommendation": False,
        "has_table": True,
        "is_urgent": True,
        "cancer_types": "lymphoma",
        "symptoms": "weight_loss,shortness_of_breath,night_sweats,fever,unexplained",
        "age_thresholds": "",
        "timeframes": "",
        "urgency": "urgent_2_week",
        "actions": "offer,refer,consider",
        "risk_factors": "",
        "investigations": "ct"
    }
    
    # Create chunk from dict
    chunk = RichChunk.from_dict(sample_chunk_data)
    print(f"\n📄 Sample chunk: {chunk.chunk_id}")
    print(f"   Cancer types: {chunk.clinical.cancer_types}")
    print(f"   Symptoms: {chunk.clinical.symptoms}")
    print(f"   Urgency: {chunk.clinical.urgency}")
    
    # Show metadata prefix
    print(f"\n📋 Metadata prefix for embedding:")
    print(f"   {chunk.clinical.to_embedding_prefix()}")
    
    # Create embedder
    embedder = get_embedder(mock=False)
    print(f"\n🔧 Using: {type(embedder).__name__}")
    
    # Embed with metadata
    print("\n--- With metadata enhancement ---")
    embedded_with = embedder.embed_chunks([chunk], include_metadata=True)
    
    # Embed without metadata
    print("\n--- Without metadata enhancement ---")
    embedded_without = embedder.embed_chunks([chunk], include_metadata=False)
    
    # Compare embeddings
    sim = cosine_similarity(
        embedded_with[0].embedding,
        embedded_without[0].embedding
    )
    print(f"\n📊 Similarity between with/without metadata: {sim:.4f}")
    print("   (Lower similarity = metadata has more impact)")
    
    # Test query embedding
    print("\n--- Query embedding ---")
    query = "unexplained splenomegaly lymphoma referral"
    query_embedding = embedder.embed_query(query)
    
    # Compare to chunk embeddings
    sim_with = cosine_similarity(query_embedding, embedded_with[0].embedding)
    sim_without = cosine_similarity(query_embedding, embedded_without[0].embedding)
    
    print(f"   Query: '{query}'")
    print(f"   Similarity (with metadata):    {sim_with:.4f}")
    print(f"   Similarity (without metadata): {sim_without:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")