"""
API Routes

All route modules for the API.
"""

from src.api.routes.assessment import router as assessment_router
from src.api.routes.chat import router as chat_router
from src.api.routes.search import router as search_router
from src.api.routes.system import router as system_router

__all__ = [
    "assessment_router",
    "chat_router",
    "search_router",
    "system_router",
]
