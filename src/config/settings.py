"""
Configuration settings for NG12 Cancer Risk Assessor
This module uses Pydantic Settings for type-safe configuration management.
Settings can be overridden via environment variables or .env file.
  
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    """
        Application settings with environment variable support.
        
        All settings can be overridden by setting environment variables
        with the same name (case-insensitive).
        
        Example:
            export GCP_PROJECT_ID="my-project"
            export CHROMA_PERSIST_DIR="/custom/path"
    """
        
    model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore"  # Ignore extra env vars
        )
        
    # ===================
    # Project Paths
    # ===================
        
    BASE_DIR: Path = Field(
            default=Path(__file__).parent.parent.parent,
            description="Project root directory",
        
    )
        
    DATA_DIR: Path = Field(
            default_factory=lambda: Path(__file__).parent.parent.parent / "data",
            description="Data directory for PDFs and JSON",
           
    )
        
    VECTORSTORE_DIR: Path = Field(
            default_factory=lambda: Path(__file__).parent.parent.parent / "vectorstore",
            description="ChromaDB persistent storage directory",

    )
        
    def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
    # ===================
    # Google Cloud / Vertex AI
    # ===================
        
    GCP_PROJECT_ID: str = Field(
            default="cancer-assessor-poc",
            description="Google Cloud Project ID"
        )
        
    GCP_LOCATION: str = Field(
            default="us-central1",
            description="Google Cloud region for Vertex AI"
        )
        
    # Embedding model - text-embedding-004 is the latest as of 2024
    EMBEDDING_MODEL: str = Field(
            default="text-embedding-004",
            description="Vertex AI embedding model name"
        )
        
    EMBEDDING_DIMENSION: int = Field(
            default=768,
            description="Embedding vector dimension"
        )
        
    # LLM model for agents
    LLM_MODEL: str = Field(
            default="gemini-1.5-pro",
            description="Gemini model for reasoning"
        )
        
    LLM_TEMPERATURE: float = Field(
            default=0.1,
            description="LLM temperature (low for deterministic clinical decisions)"
        )
        
    # ===================
    # ChromaDB Settings
    # ===================
        
    CHROMA_COLLECTION_NAME: str = Field(
            default="ng12_guidelines",
            description="ChromaDB collection name for NG12 chunks"
        )
        
    # ===================
    # PDF Ingestion Settings
    # ===================
        
    NG12_PDF_URL: str = Field(
            default="https://www.nice.org.uk/guidance/ng12/resources/suspected-cancer-recognition-and-referral-pdf-1837268071621",
            description="URL to download the NG12 PDF"
        )
        
    NG12_PDF_FILENAME: str = Field(
            default="ng12_suspected_cancer.pdf",
            description="Local filename for the downloaded PDF"
        )
        
    # ===================
    # Chunking Parameters
    # ===================
    # These are CRITICAL for RAG quality - interview discussion point
        
    CHUNK_SIZE_TOKENS: int = Field(
            default=512,
            description="Target chunk size in tokens (not characters)"
    )
        
    CHUNK_OVERLAP_TOKENS: int = Field(
            default=100,
            description="Overlap between chunks to preserve context"
    )
        
    MIN_CHUNK_SIZE_TOKENS: int = Field(
            default=50,
            description="Minimum chunk size (avoid tiny fragments)"
        )
        
    # ===================
    # Retrieval Settings
    # ===================
        
    DEFAULT_TOP_K: int = Field(
            default=5,
            description="Default number of chunks to retrieve"
        )
        
    SIMILARITY_THRESHOLD: float = Field(
            default=0.7,
            description="Minimum similarity score for retrieval"
        )
        
    # ===================
    # API Settings
    # ===================
        
    API_HOST: str = Field(
            default="0.0.0.0",
            description="API host binding"
        )
        
    API_PORT: int = Field(
            default=8000,
            description="API port"
        )
        
    # ===================
    # Session Settings (Chat)
    # ===================
        
    MAX_SESSION_HISTORY: int = Field(
            default=10,
            description="Maximum conversation turns to keep in memory"
        )
        
    SESSION_TIMEOUT_MINUTES: int = Field(
            default=60,
            description="Session expiry time"
        )
        
    @property
    def pdf_path(self) -> Path:
            """Full path to the NG12 PDF file."""
            return self.DATA_DIR / "ng12" / self.NG12_PDF_FILENAME
        
    @property
    def patients_json_path(self) -> Path:
            """Full path to the patients.json file."""
            return self.DATA_DIR / "patients.json"


# Singleton instance - import this in other modules
settings = Settings()

# ===================
# Design Decision Notes
# ===================
"""

CHUNK_SIZE_TOKENS = 512:
- Large enough to contain complete clinical recommendations
- Small enough for precise retrieval
- NG12 recommendations are typically 100-300 tokens
- 512 allows context around the recommendation

CHUNK_OVERLAP_TOKENS = 100:
- ~20% overlap ensures criteria aren't split
- Example: "Refer urgently if patient has X" might be at chunk boundary
- Overlap ensures both chunks contain the full recommendation

LLM_TEMPERATURE = 0.1:
- Clinical decisions require consistency
- Higher temperature could lead to different assessments for same patient
- Not 0.0 to avoid getting stuck in local optima

SIMILARITY_THRESHOLD = 0.7:
- Balances recall vs precision
- Lower would return irrelevant chunks
- Higher might miss relevant chunks with different wording
- Tuned based on embedding model characteristics

"""