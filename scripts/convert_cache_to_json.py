#!/usr/bin/env python3
"""Convert pickle cache to JSON for instant loading."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.parse_and_cache import load_from_cache

def convert_to_json():
    """Convert pickle cache to JSON."""
    print("Loading pickle cache (this takes 1-2 minutes on first load)...")
    doc = load_from_cache()
    
    if not doc:
        print("❌ Cache not found!")
        return 1
    
    print(f"✅ Cache loaded: {doc.title}")
    
    # Convert to JSON-serializable format
    print("Converting to JSON (faster format)...")
    
    cache_json = {
        "title": doc.title,
        "total_pages": doc.total_pages,
        "sections": [s.to_dict() for s in doc.get_sections()],
        "content_pages": [p.to_dict() for p in doc.get_content_pages()],
    }
    
    json_path = Path(__file__).parent.parent / "data" / "cache" / "ng12_cache.json"
    with open(json_path, 'w') as f:
        json.dump(cache_json, f)
    
    size_kb = json_path.stat().st_size / 1024
    print(f"✅ JSON cache created: {json_path} ({size_kb:.1f}KB)")
    print("   Next time, loading will be instant!")
    
    return 0

if __name__ == "__main__":
    sys.exit(convert_to_json())
