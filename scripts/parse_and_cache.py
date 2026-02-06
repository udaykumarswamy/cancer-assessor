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

from src.config.settings import settings
from src.ingestion.pdf_parser import MarkerPDFParser, ParsedDocument, ParsedPage
from src.config.logging_config import get_logger

logger = get_logger('parse_and_cache', level=logging.DEBUG)

# Cache file location
CACHE_DIR = settings.DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "ng12_parsed.pkl"


def save_to_cache(doc: ParsedDocument):
    """Save parsed document to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(doc, f)
    
    logger.debug(f"Cached to: {CACHE_FILE} ")
    logger.debug(f"Cache size: {CACHE_FILE.stat().st_size:,} bytes")


def load_from_cache() -> ParsedDocument:
    """Load parsed document from cache."""
    if not CACHE_FILE.exists():
        return None
    
    with open(CACHE_FILE, 'rb') as f:
        doc = pickle.load(f)
    
    return doc


def parse_with_cache(force: bool = False, lightweight: bool = False) -> ParsedDocument:
    """
    Parse PDF with caching.
    
    First run: Parses with Marker (slow)
    Subsequent runs: Loads from cache (instant!)
    """
    
    # Check cache first
    if not force and CACHE_FILE.exists():
        logger.info("Loading from cache (instant!)...")
        start = time.time()
        doc = load_from_cache()
        logger.debug(f" Loaded in {time.time() - start:.2f}s")
        logger.info(f" Title: {doc.title}")
        logger.info(f"  pages: {len(doc.total_pages())}")
        return doc
    
    # Parse with Marker
    logger.info("No cache found. Parsing with Marker...This may take several minutes...")
    logger.info("   (This will be cached for instant loading next time)\n")
    
    parser = MarkerPDFParser()
    start = time.time()
    
    if lightweight:
        doc = parser.parse_lightweight(settings.pdf_path)
    else:
        doc = parser.parse(settings.pdf_path)
    
    parse_time = time.time() - start
    logger.debug(f"\n  Parse time: {parse_time:.1f}s")
    
    # Save to cache
    save_to_cache(doc)
    logger.info(f"Parsing complete and cached! Next time will load in seconds.")
    
    
    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse PDF with caching")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-parse (ignore cache)")
    parser.add_argument("--lightweight", "-l", action="store_true", help="Use lightweight mode")
    parser.add_argument("--info", "-i", action="store_true", help="Just show cache info")
    args = parser.parse_args()
    
    # Just show info
    if args.info:
        if CACHE_FILE.exists():
            size = CACHE_FILE.stat().st_size
            logger.info(f"Cache exists: {CACHE_FILE}")
            doc = load_from_cache()
            logger.info(f" Title: {doc.title}")
        else:
            logger.info("No cache found.")
            logger.info(f" Expected at: {CACHE_FILE}")
        return 0
    
    # Parse (with cache)
    try:
        doc = parse_with_cache(force=args.force, lightweight=args.lightweight)
        
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
        logger.error(f"Error: {traceback.print_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
