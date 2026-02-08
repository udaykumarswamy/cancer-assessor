"""
API Module

FastAPI application for clinical risk assessment.
"""

from src.api.routes import (
    assessment_router,
    chat_router,
    search_router,
    system_router,
)

from src.api.dependencies import (
    get_vector_store,
    get_embedder,
    get_llm,
    get_retriever,
    get_assessment_service,
    get_clinical_agent,
    initialize_services,
    reset_dependencies,
)

__all__ = [
    # Routers
    "assessment_router",
    "chat_router",
    "search_router",
    "system_router",
    # Dependencies
    "get_vector_store",
    "get_embedder",
    "get_llm",
    "get_retriever",
    "get_assessment_service",
    "get_clinical_agent",
    "initialize_services",
    "reset_dependencies",
]
