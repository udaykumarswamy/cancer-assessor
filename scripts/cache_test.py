#!/usr/bin/env python3
"""Super simple cache test - just load the cache."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from scripts.parse_and_cache import load_from_cache

logger = get_logger('cache_test', level='INFO')

def main():
    logger.info("=" * 60)
    logger.info("SIMPLE CACHE LOAD TEST")
    logger.info("=" * 60)
    
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
    logger.info(f"   Content pages: {len(doc.get_content_pages())}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ CACHE LOAD TEST PASSED")
    logger.info("=" * 60)
    logger.info("\nThe cache loads successfully!")
    logger.info("To run tests with embeddings, use:")
    logger.info("  python scripts/test_e2e_pipeline.py --mock")
    
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
