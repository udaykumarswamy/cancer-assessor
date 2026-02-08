"""
Configuration Module

Application settings and logging configuration.
"""

try:
    from src.config.settings import settings
except ImportError:
    settings = None

try:
    from src.config.logging_config import (
        get_logger,
        get_ingestion_logger,
        get_retrieval_logger,
        get_agent_logger,
        get_api_logger,
    )
except ImportError:
    def get_logger(*args, **kwargs):
        import logging
        return logging.getLogger(__name__)
    get_ingestion_logger = get_logger
    get_retrieval_logger = get_logger
    get_agent_logger = get_logger
    get_api_logger = get_logger

__all__ = [
    "settings",
    "get_logger",
    "get_ingestion_logger",
    "get_retrieval_logger",
    "get_agent_logger",
    "get_api_logger",
]
