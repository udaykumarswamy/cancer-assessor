"""
API Dependencies

Dependency injection for FastAPI routes.
Manages singleton instances of services.

"""

from typing import Optional
from functools import lru_cache

from src.config.logging_config import get_logger

logger = get_logger("api.dependencies")

# Global instances (singletons)
_vector_store = None
_embedder = None
_llm = None
_retriever = None
_assessment_service = None
_clinical_agent = None


def _use_mock() -> bool:
    """Check if we should use mock services."""
    import os
    return os.environ.get("USE_MOCK", "false").lower() == "true"


@lru_cache()
def get_vector_store():
    """Get or create vector store instance."""
    global _vector_store
    
    if _vector_store is None:
        logger.info("Initializing vector store...")
        from src.ingestion.vector_store import VectorStore
        _vector_store = VectorStore()
        
        stats = _vector_store.get_stats()
        logger.info(f"Vector store loaded with {stats.get('total_chunks', 0)} chunks")
    
    return _vector_store


@lru_cache()
def get_embedder():
    """Get or create embedder instance."""
    global _embedder
    
    if _embedder is None:
        logger.info("Initializing embedder...")
        from src.ingestion.embedder import get_embedder as _get_embedder
        _embedder = _get_embedder(mock=_use_mock())
    
    return _embedder


@lru_cache()
def get_llm():
    """Get or create LLM instance."""
    global _llm
    
    if _llm is None:
        logger.info("Initializing LLM...")
        from src.llm.gemini import get_llm as _get_llm
        _llm = _get_llm(mock=_use_mock())
    
    return _llm


@lru_cache()
def get_retriever():
    """Get or create retriever instance."""
    global _retriever
    
    if _retriever is None:
        logger.info("Initializing retriever...")
        from src.services.retrieval import ClinicalRetriever
        
        vector_store = get_vector_store()
        embedder = get_embedder()
        
        _retriever = ClinicalRetriever(
            vector_store=vector_store,
            embedder=embedder,
        )
    
    return _retriever


@lru_cache()
def get_assessment_service():
    """Get or create assessment service instance."""
    global _assessment_service
    
    if _assessment_service is None:
        logger.info("Initializing assessment service...")
        from src.services.assessment import ClinicalAssessmentService
        
        retriever = get_retriever()
        llm = get_llm()
        
        _assessment_service = ClinicalAssessmentService(
            retriever=retriever,
            llm=llm,
        )
    
    return _assessment_service


@lru_cache()
def get_clinical_agent():
    """Get or create clinical agent instance."""
    global _clinical_agent
    
    if _clinical_agent is None:
        logger.info("Initializing clinical agent (ReAct-based)...")
        from src.agents.clinical_agent import ClinicalAgent
        
        retriever = get_retriever()
        llm = get_llm()
        
        _clinical_agent = ClinicalAgent(
            retriever=retriever,
            llm=llm,
            max_steps=8,
            verbose=False
        )
    
    return _clinical_agent


@lru_cache()
def get_conversational_agent():
    """Get or create conversational clinical agent."""
    from src.agents.clinical_agent import ConversationalClinicalAgent
    
    clinical_agent = get_clinical_agent()
    return ConversationalClinicalAgent(clinical_agent)


def reset_dependencies():
    """
    Reset all cached dependencies.
    
    Useful for testing or configuration changes.
    """
    global _vector_store, _embedder, _llm, _retriever, _assessment_service, _clinical_agent
    
    _vector_store = None
    _embedder = None
    _llm = None
    _retriever = None
    _assessment_service = None
    _clinical_agent = None
    
    # Clear lru_cache
    get_vector_store.cache_clear()
    get_embedder.cache_clear()
    get_llm.cache_clear()
    get_retriever.cache_clear()
    get_assessment_service.cache_clear()
    get_clinical_agent.cache_clear()
    
    logger.info("All dependencies reset")


def initialize_services(mock: bool = False):
    """
    Configure services for lazy initialization on first request.
    
    Services are now initialized on-demand rather than at startup,
    reducing startup time from potentially minutes to seconds.
    
    Args:
        mock: Use mock services
    """
    import os
    if mock:
        os.environ["USE_MOCK"] = "true"
    
    logger.info("Services configured for lazy loading (on-demand initialization)")
    logger.info("API is ready to accept requests. Services will initialize on first use.")
