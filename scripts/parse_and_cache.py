#!/usr/bin/env python3
"""
Parse PDF with Marker and CACHE the result.

Run Marker once, save output to disk, load instantly next time.

Usage:
    # First run: Uses Marker (slow)
    python scripts/parse_and_cache.py
    
    # Subsequent runs: Loads from cache (instant!)
    python scripts/parse_and_cache.py
    
    # Force re-parse
    python scripts/parse_and_cache.py --force
"""

import sys
import json
import time
import pickle
import traceback

from pathlib import Path
import logging
sys.path.insert(0, str(Path(__file__).parent.parent))

# LAZY: Don't import settings at module level
# from src.config.settings import settings
# from src.config.logging_config import get_logger

# Use a simple logger instead
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('parse_and_cache')

# Cache file location - determined lazily
_CACHE_DIR = None
_CACHE_FILE = None

def _get_cache_paths():
    """Get cache paths (lazy-loaded settings)."""
    global _CACHE_DIR, _CACHE_FILE
    if _CACHE_DIR is None:
        from src.config.settings import settings
        _CACHE_DIR = settings.DATA_DIR / "cache"
        _CACHE_FILE = _CACHE_DIR / "ng12_parsed.pkl"
    return _CACHE_DIR, _CACHE_FILE

def _get_cache_file():
    """Get cache file path."""
    _, cache_file = _get_cache_paths()
    return cache_file

def _get_cache_dir():
    """Get cache directory."""
    cache_dir, _ = _get_cache_paths()
    return cache_dir


def save_to_cache(doc):
    """Save parsed document to cache."""
    cache_dir = _get_cache_dir()
    cache_file = _get_cache_file()
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(doc, f)
    
    logger.debug(f"Cached to: {cache_file} ")
    logger.debug(f"Cache size: {cache_file.stat().st_size:,} bytes")


def load_from_cache():
    """Load parsed document from cache."""
    cache_file = _get_cache_file()
    
    if not cache_file.exists():
        return None
    
    logger.info(f"📂 Loading cache file ({cache_file.stat().st_size / 1024:.1f}KB)...")
    logger.info("   (This may take 10-30 seconds on first load, then caches...)")
    print(cache_file) 
    start = time.time()
    
    try:
        with open(cache_file, 'rb') as f:
            doc = pickle.load(f)
        
        elapsed = time.time() - start
        logger.info(f"   ✅ Cache loaded in {elapsed:.1f}s")
        return doc
        
    except Exception as e:
        logger.error(f"   ❌ Cache load failed: {e}")
        logger.warning("   Corrupted cache file - will need to re-parse")
        return None


def parse_with_cache(force: bool = False, lightweight: bool = False):
    """
    Parse PDF with caching.
    
    First run: Parses with Marker (slow)
    Subsequent runs: Loads from cache (instant!)
    """
    
    cache_file = _get_cache_file()
    
    # Check cache first (no heavy imports needed)
    if not force and cache_file.exists():
        logger.info("Loading from cache (instant!)...")
        start = time.time()
        doc = load_from_cache()
        elapsed = time.time() - start
        logger.info(f"   ✅ Loaded in {elapsed:.1f}s")
        if doc:
            logger.info(f"   Title: {doc.title}")
            logger.info(f"   Pages: {doc.total_pages}")
        return doc
    
    # Parse with Marker - LAZY IMPORT (only load when parsing is needed)
    logger.info("No cache found. Parsing with Marker...")
    logger.info("   (Loading heavy dependencies...this may take 30-60 seconds)")
    logger.info("   (This will be cached for instant loading next time)\n")
    
    # LAZY: Only import when actually parsing
    from src.ingestion.pdf_parser import MarkerPDFParser
    from src.config.settings import settings
    
    parser = MarkerPDFParser()
    start = time.time()
    
    if lightweight:
        doc = parser.parse_lightweight(settings.pdf_path)
    else:
        doc = parser.parse(settings.pdf_path)
    
    parse_time = time.time() - start
    logger.info(f"   ✅ Parse complete in {parse_time:.1f}s")
    
    # Save to cache
    save_to_cache(doc)
    logger.info(f"   ✅ Cached for instant future loading!")
    
    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse PDF with caching")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-parse (ignore cache)")
    parser.add_argument("--lightweight", "-l", action="store_true", help="Use lightweight mode")
    parser.add_argument("--info", "-i", action="store_true", help="Just show cache info")
    args = parser.parse_args()
    
    cache_file = _get_cache_file()
    
    # Just show info
    if args.info:
        if cache_file.exists():
            size = cache_file.stat().st_size
            logger.info(f"Cache exists: {cache_file}")
            doc = load_from_cache()
            if doc:
                logger.info(f" Title: {doc.title}")
        else:
            logger.info("No cache found.")
            logger.info(f" Expected at: {cache_file}")
        return 0
    
    # Parse (with cache)
    try:
        doc = parse_with_cache(force=args.force, lightweight=args.lightweight)
        
        if doc:
            logger.info(f"\nDocument Info:")
            logger.info(f"   Title: {doc.title}")
            logger.info(f"   Total pages: {doc.total_pages}")
            logger.info(f"   Content pages: {len(doc.get_content_pages())}")
            logger.info(f"   Sections: {len(doc.get_sections())}")
            logger.info(f"   Markdown: {len(doc.full_markdown):,} chars")
            
            logger.info(f"\n Next time, run without --force for instant loading!")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.print_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
