"""
Ingestion Pipeline Module

Components for parsing, chunking, embedding, and storing NG12 guidelines.

NOTE: Imports are lazy to avoid loading heavy ML dependencies at module init time.
Import specific modules directly when needed:
  from src.ingestion.pdf_parser import MarkerPDFParser
  from src.ingestion.chunker import SemanticChunker
  from src.ingestion.embedder import get_embedder
"""

__all__ = []

# Lazy imports - only loaded when explicitly imported
# This avoids the heavy ML dependency chain at module initialization


# Vector Store
try:
    from src.ingestion.vector_store import VectorStore, get_vector_store, RetrievedChunk
    __all__.extend(["VectorStore", "get_vector_store", "RetrievedChunk"])
except ImportError as e:
    # Vector store might not be implemented yet
    pass
