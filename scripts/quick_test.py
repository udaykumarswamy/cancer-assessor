#!/usr/bin/env python3
"""Quick diagnostic test to check cache loading performance."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from scripts.parse_and_cache import load_from_cache

logger = get_logger('quick_test', level='INFO')

def main():
    logger.info("=" * 60)
    logger.info("QUICK CACHE LOAD TEST")
    logger.info("=" * 60)
    
    # Step 1: Load cache
    logger.info("\n1️⃣ Loading cached document...")
    start = time.time()
    doc = load_from_cache()
    elapsed = time.time() - start
    
    if not doc:
        logger.error("❌ Cache load failed!")
        return 1
    
    logger.info(f"✅ Cache loaded in {elapsed:.1f}s")
    logger.info(f"   Title: {doc.title}")
    logger.info(f"   Pages: {doc.total_pages}")
    
    # Step 2: Test chunking
    logger.info("\n2️⃣ Testing chunking...")
    from src.ingestion.chunker import SemanticChunker
    
    chunker = SemanticChunker()
    start = time.time()
    chunks = chunker.chunk_document(doc)
    elapsed = time.time() - start
    
    logger.info(f"✅ Chunked into {len(chunks)} chunks in {elapsed:.1f}s")
    
    # Step 3: Test embeddings (mock)
    logger.info("\n3️⃣ Testing embeddings (MOCK mode)...")
    from src.ingestion.embedder import get_embedder
    
    start = time.time()
    embedder = get_embedder(mock=True)
    logger.info(f"✅ Embedder initialized in {time.time() - start:.1f}s (mock)")
    
    start = time.time()
    sample_chunks = chunks[:5]  # Just first 5
    embedded = embedder.embed_chunks(sample_chunks, include_metadata=True)
    elapsed = time.time() - start
    
    logger.info(f"✅ Embedded {len(embedded)} chunks in {elapsed:.1f}s")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("=" * 60)
    logger.info("\nTo run full pipeline:")
    logger.info("  python scripts/test_e2e_pipeline.py --mock")
    logger.info("\nTo use Vertex AI (requires auth):")
    logger.info("  python scripts/test_e2e_pipeline.py")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
