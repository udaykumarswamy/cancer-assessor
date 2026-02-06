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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.ingestion.pdf_parser import MarkerPDFParser, ParsedDocument, ParsedPage


# Cache file location
CACHE_DIR = settings.DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "ng12_parsed.pkl"


def save_to_cache(doc: ParsedDocument):
    """Save parsed document to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(doc, f)
    
    print(f"💾 Cached to: {CACHE_FILE}")


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
        print("📦 Loading from cache (instant!)...")
        start = time.time()
        doc = load_from_cache()
        print(f"   ✅ Loaded in {time.time() - start:.2f}s")
        print(f"   Title: {doc.title}")
        print(f"   Pages: {doc.total_pages}")
        return doc
    
    # Parse with Marker
    print("🔄 No cache found. Parsing with Marker...")
    print("   (This will be cached for instant loading next time)\n")
    
    parser = MarkerPDFParser()
    start = time.time()
    
    if lightweight:
        doc = parser.parse_lightweight(settings.pdf_path)
    else:
        doc = parser.parse(settings.pdf_path)
    
    parse_time = time.time() - start
    print(f"\n⏱️  Parse time: {parse_time:.1f}s")
    
    # Save to cache
    save_to_cache(doc)
    
    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse PDF with caching")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-parse (ignore cache)")
    parser.add_argument("--lightweight", "-l", action="store_true", help="Use lightweight mode")
    parser.add_argument("--info", "-i", action="store_true", help="Just show cache info")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MARKER PARSER WITH CACHING")
    print("="*60)
    
    # Just show info
    if args.info:
        if CACHE_FILE.exists():
            size = CACHE_FILE.stat().st_size
            print(f"✅ Cache exists: {CACHE_FILE}")
            print(f"   Size: {size:,} bytes")
            doc = load_from_cache()
            print(f"   Title: {doc.title}")
            print(f"   Pages: {doc.total_pages}")
        else:
            print("❌ No cache found")
            print(f"   Expected at: {CACHE_FILE}")
        return 0
    
    # Parse (with cache)
    try:
        doc = parse_with_cache(force=args.force, lightweight=args.lightweight)
        
        print(f"\n📊 Document Info:")
        print(f"   Title: {doc.title}")
        print(f"   Total pages: {doc.total_pages}")
        print(f"   Content pages: {len(doc.get_content_pages())}")
        print(f"   Sections: {len(doc.get_sections())}")
        print(f"   Markdown: {len(doc.full_markdown):,} chars")
        
        print(f"\n✅ Next time, run without --force for instant loading!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
