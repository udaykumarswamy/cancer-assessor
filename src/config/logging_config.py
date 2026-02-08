"""
Logging Configuration for NG12 Cancer Risk Assessor

Provides structured logging with configurable levels and formats.

Interview Discussion Points:
---------------------------
1. Why structured logging?
   - Easier to parse in production (JSON format option)
   - Consistent format across all modules
   - Filterable by level, module, timestamp

2. Log levels strategy:
   - DEBUG: Detailed flow (chunk processing, embedding batches)
   - INFO: High-level progress (pipeline stages)
   - WARNING: Recoverable issues (retry, fallback)
   - ERROR: Failures requiring attention
"""

import logging
import sys
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        use_colors: Use colored output for console
        
    Returns:
        Configured logger instance
        
    Usage:
        from src.config.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Format
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    if use_colors:
        formatter = ColoredFormatter(log_format, datefmt=date_format)
    else:
        formatter = logging.Formatter(log_format, datefmt=date_format)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Pre-configured loggers for main components
def get_ingestion_logger():
    """Logger for ingestion pipeline."""
    return get_logger("ingestion", level="INFO")


def get_retrieval_logger():
    """Logger for retrieval operations."""
    return get_logger("retrieval", level="INFO")


def get_agent_logger():
    """Logger for agent operations."""
    return get_logger("agent", level="INFO")


def get_api_logger():
    """Logger for API operations."""
    return get_logger("api", level="INFO")
