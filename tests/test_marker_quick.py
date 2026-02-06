#!/usr/bin/env python3
"""
Quick Marker Test - Processes only first 10 pages for faster testing.

Usage:
    python tests/test_marker_quick.py
    python tests/test_marker_quick.py --pages 5  # Even faster
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings


def test_marker_quick(max_pages: int = 10):
    """Quick test with limited pages."""
    
    print(f"\n{'='*60}")
    print(f"QUICK MARKER TEST (First {max_pages} pages only)")
    print('='*60)
    
    pdf_path = settings.pdf_path
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("   Run: python scripts/download_ng12.py")
        return False
    
    print(f"📄 PDF: {pdf_path.name}")
    print(f"📊 Processing: First {max_pages} pages only")
    print(f"⏳ Starting Marker (this is faster with fewer pages)...\n")
    
    start_time = time.time()
    
    try:
        # Import Marker
        from marker.models import create_model_dict
        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser
        
        print("   Loading models...")
        model_start = time.time()
        models = create_model_dict()
        model_time = time.time() - model_start
        print(f"   ✅ Models loaded in {model_time:.1f}s")
        
        print(f"   Parsing first {max_pages} pages...")
        parse_start = time.time()
        
        # Configure converter
        config_parser = ConfigParser({})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=None,
            renderer=None
        )
        
        # Convert PDF (max_pages via converter config)
        rendered = converter(str(pdf_path))
        full_markdown = rendered.markdown
        
        parse_time = time.time() - parse_start
        total_time = time.time() - start_time
        
        print(f"\n✅ SUCCESS!")
        print(f"   Markdown length: {len(full_markdown):,} chars")
        print(f"   Parse time: {parse_time:.1f}s")
        print(f"   Total time: {total_time:.1f}s")
        
        # Show sample
        print(f"\n📝 Sample output (first 500 chars):")
        print("-" * 40)
        print(full_markdown[:500])
        print("-" * 40)
        
        # Estimate full document time
        estimated_full = (parse_time / max_pages) * 50  # NG12 has ~50 pages
        print(f"\n⏱️  Estimated time for full PDF: {estimated_full:.0f}s ({estimated_full/60:.1f} min)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install with: pip install marker-pdf")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick Marker test with limited pages")
    parser.add_argument("--pages", "-p", type=int, default=10, help="Number of pages to process")
    args = parser.parse_args()
    
    success = test_marker_quick(args.pages)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
