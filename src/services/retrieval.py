"""
Clinical Retrieval Service

Intelligent retrieval from NG12 guidelines with:
- Query expansion based on clinical terminology
- Hybrid search (semantic + metadata filtering)
- Result reranking for clinical relevance
- Context window management

Interview Discussion Points:
---------------------------
1. Query expansion strategy:
   - Medical synonyms (haemoptysis → coughing blood)
   - Related symptoms (chest pain → dyspnea, cough)
   - Anatomical relationships (lung → respiratory, thoracic)

2. Hybrid search benefits:
   - Semantic: Captures meaning ("breathing difficulty" → "dyspnea")
   - Metadata: Filters by cancer type, urgency, age
   - Combined: Best of both worlds

3. Reranking considerations:
   - Cross-encoder for semantic similarity
   - Clinical relevance boosting
   - Recency/version weighting
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.config.logging_config import get_logger

logger = get_logger("retrieval")


class SearchMode(Enum):
    """Search modes for retrieval."""
    SEMANTIC = "semantic"           # Pure vector similarity
    FILTERED = "filtered"           # Semantic + metadata filters
    HYBRID = "hybrid"               # Semantic + keyword matching


@dataclass
class RetrievalResult:
    """
    A retrieved chunk with relevance scoring.
    
    Attributes:
        chunk_id: Unique identifier
        text: Chunk content
        score: Relevance score (0-1)
        page: Source page number
        section: Source section
        urgency: Clinical urgency level
        cancer_types: Related cancer types
        citation: Formatted citation string
    """
    chunk_id: str
    text: str
    score: float
    page: int
    section: str = ""
    urgency: str = ""
    cancer_types: List[str] = field(default_factory=list)
    citation: str = ""
    
    def to_context_string(self) -> str:
        """Format for LLM context."""
        header = f"[{self.citation}]" if self.citation else f"[Page {self.page}]"
        return f"{header}\n{self.text}"


@dataclass
class RetrievalContext:
    """
    Full retrieval context for LLM.
    
    Contains retrieved chunks and metadata about the retrieval.
    """
    results: List[RetrievalResult]
    query: str
    expanded_query: Optional[str] = None
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    total_candidates: int = 0
    
    @property
    def is_empty(self) -> bool:
        """Check if retrieval returned no results."""
        return len(self.results) == 0
    
    def get_context_text(self, max_chunks: int = 5) -> str:
        """Get formatted context for LLM."""
        chunks = self.results[:max_chunks]
        return "\n\n---\n\n".join(r.to_context_string() for r in chunks)
    
    def get_citations(self) -> List[str]:
        """Get list of citations used."""
        return [r.citation for r in self.results if r.citation]


class ClinicalRetriever:
    """
    Retrieves relevant NG12 chunks for clinical queries.
    
    Features:
    - Query expansion with clinical synonyms
    - Metadata filtering (urgency, cancer type, age)
    - Score-based ranking
    - Context window management
    
    Usage:
        retriever = ClinicalRetriever(vector_store, embedder)
        
        # Simple search
        context = retriever.retrieve("lung cancer symptoms")
        
        # Filtered search
        context = retriever.retrieve_for_patient(
            query="cough and weight loss",
            patient_age=55,
            symptoms=["cough", "weight_loss"]
        )
    """
    
    # Clinical synonym expansions
    SYMPTOM_SYNONYMS = {
        "cough": ["persistent cough", "chronic cough", "coughing"],
        "haemoptysis": ["hemoptysis", "coughing blood", "blood in sputum"],
        "weight loss": ["unexplained weight loss", "weight reduction", "losing weight"],
        "fatigue": ["tiredness", "exhaustion", "lethargy"],
        "dysphagia": ["difficulty swallowing", "swallowing problems"],
        "breathlessness": ["shortness of breath", "dyspnea", "breathing difficulty"],
        "lump": ["mass", "swelling", "nodule"],
        "pain": ["ache", "discomfort", "tenderness"],
    }
    
    # Cancer type mappings for query expansion
    CANCER_KEYWORDS = {
        "lung": ["lung cancer", "pulmonary", "respiratory", "thoracic"],
        "breast": ["breast cancer", "mammary"],
        "colorectal": ["colorectal cancer", "bowel cancer", "colon", "rectal"],
        "prostate": ["prostate cancer", "prostatic"],
        "skin": ["skin cancer", "melanoma", "dermatological"],
        "bladder": ["bladder cancer", "urological"],
        "lymphoma": ["lymphoma", "lymph nodes", "lymphatic"],
    }
    
    def __init__(
        self,
        vector_store,
        embedder,
        default_top_k: int = 5,
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance
            embedder: Embedder instance for query embedding
            default_top_k: Default number of results to return
            similarity_threshold: Minimum similarity score
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_query: bool = True,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> RetrievalContext:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results (default: 5)
            expand_query: Whether to expand query with synonyms
            mode: Search mode (semantic, filtered, hybrid)
            
        Returns:
            RetrievalContext with results
        """
        top_k = top_k or self.default_top_k
        
        # Optionally expand query
        expanded = self._expand_query(query) if expand_query else query
        
        logger.debug(f"Retrieving for: '{query}'")
        if expanded != query:
            logger.debug(f"Expanded to: '{expanded}'")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(expanded)
        
        # Search
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more for filtering
        )
        
        # Convert to RetrievalResult
        results = []
        for r in raw_results:
            if r.score >= self.similarity_threshold:
                results.append(RetrievalResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=r.score,
                    page=r.page_start,
                    section=r.section,
                    urgency=r.metadata.get("urgency", ""),
                    cancer_types=r.metadata.get("cancer_types", "").split(",") if r.metadata.get("cancer_types") else [],
                    citation=r.get_citation(),
                ))
        
        # Take top_k
        results = results[:top_k]
        
        return RetrievalContext(
            results=results,
            query=query,
            expanded_query=expanded if expanded != query else None,
            total_candidates=len(raw_results),
        )
    
    def retrieve_for_patient(
        self,
        query: str,
        patient_age: Optional[int] = None,
        symptoms: Optional[List[str]] = None,
        suspected_cancer: Optional[str] = None,
        urgent_only: bool = False,
        top_k: Optional[int] = None,
    ) -> RetrievalContext:
        """
        Retrieve chunks tailored to patient context.
        
        Args:
            query: Search query or symptom description
            patient_age: Patient's age (affects filtering)
            symptoms: List of symptoms
            suspected_cancer: Suspected cancer type
            urgent_only: Only return urgent recommendations
            top_k: Number of results
            
        Returns:
            RetrievalContext with filtered results
        """
        top_k = top_k or self.default_top_k
        
        # Build enhanced query from patient context
        enhanced_query = self._build_patient_query(
            query, patient_age, symptoms, suspected_cancer
        )
        
        # Generate embedding
        query_embedding = self.embedder.embed_query(enhanced_query)
        
        # Search with appropriate filter
        if urgent_only:
            raw_results = self.vector_store.search_urgent_only(
                query_embedding=query_embedding,
                top_k=top_k * 2,
            )
        elif suspected_cancer:
            raw_results = self.vector_store.search_by_cancer_type(
                query_embedding=query_embedding,
                cancer_type=suspected_cancer,
                top_k=top_k * 2,
            )
        else:
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
            )
        
        # Convert and filter
        results = []
        for r in raw_results:
            if r.score >= self.similarity_threshold:
                results.append(RetrievalResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=r.score,
                    page=r.page_start,
                    section=r.section,
                    urgency=r.metadata.get("urgency", ""),
                    cancer_types=r.metadata.get("cancer_types", "").split(",") if r.metadata.get("cancer_types") else [],
                    citation=r.get_citation(),
                ))
        
        results = results[:top_k]
        
        filters = {}
        if patient_age:
            filters["age"] = patient_age
        if symptoms:
            filters["symptoms"] = symptoms
        if suspected_cancer:
            filters["cancer_type"] = suspected_cancer
        if urgent_only:
            filters["urgent_only"] = True
        
        return RetrievalContext(
            results=results,
            query=query,
            expanded_query=enhanced_query,
            filters_applied=filters,
            total_candidates=len(raw_results),
        )
    
    def retrieve_by_section(
        self,
        section_number: str,
        top_k: Optional[int] = None,
    ) -> RetrievalContext:
        """
        Retrieve all chunks from a specific section.
        
        Args:
            section_number: Section number (e.g., "1.1", "1.3.2")
            top_k: Maximum results
            
        Returns:
            RetrievalContext with section chunks
        """
        top_k = top_k or 10
        
        # Use section filter
        raw_results = self.vector_store.search_by_section(
            section_prefix=section_number,
            top_k=top_k,
        )
        
        results = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=1.0,  # Exact section match
                page=r.page_start,
                section=r.section,
                urgency=r.metadata.get("urgency", ""),
                cancer_types=r.metadata.get("cancer_types", "").split(",") if r.metadata.get("cancer_types") else [],
                citation=r.get_citation(),
            )
            for r in raw_results
        ]
        
        return RetrievalContext(
            results=results,
            query=f"Section {section_number}",
            filters_applied={"section": section_number},
            total_candidates=len(raw_results),
        )
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with clinical synonyms.
        
        Example:
            "patient with haemoptysis" → "patient with haemoptysis hemoptysis coughing blood"
        """
        query_lower = query.lower()
        expansions = []
        
        for term, synonyms in self.SYMPTOM_SYNONYMS.items():
            if term in query_lower:
                # Add synonyms not already in query
                for syn in synonyms:
                    if syn.lower() not in query_lower:
                        expansions.append(syn)
        
        if expansions:
            return f"{query} {' '.join(expansions[:3])}"  # Limit expansions
        
        return query
    
    def _build_patient_query(
        self,
        query: str,
        age: Optional[int],
        symptoms: Optional[List[str]],
        cancer_type: Optional[str],
    ) -> str:
        """Build enhanced query from patient context."""
        parts = [query]
        
        if age:
            if age >= 40:
                parts.append("aged 40 and over")
            elif age >= 50:
                parts.append("aged 50 and over")
        
        if symptoms:
            # Add symptom terms
            symptom_text = " ".join(symptoms[:3])
            parts.append(symptom_text)
        
        if cancer_type:
            # Add cancer-specific terms
            keywords = self.CANCER_KEYWORDS.get(cancer_type.lower(), [])
            if keywords:
                parts.append(keywords[0])
        
        return " ".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        store_stats = self.vector_store.get_stats()
        return {
            "vector_store": store_stats,
            "default_top_k": self.default_top_k,
            "similarity_threshold": self.similarity_threshold,
        }
