#!/usr/bin/env python3
"""
Step-by-Step Marker PDF Parser Test

This script walks you through testing the Marker PDF parser interactively.
Run this after installing dependencies.

Usage:
    python tests/test_marker_step_by_step.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def step(number: int, title: str):
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"STEP {number}: {title}")
    print('='*60)


def check_dependencies():
    """Check if all dependencies are installed."""
    step(1, "Checking Dependencies")
    
    missing = []
    
    # Check marker
    try:
        import marker
        print(f"✅ marker-pdf installed (version: {getattr(marker, '__version__', 'unknown')})")
    except ImportError:
        print("❌ marker-pdf NOT installed")
        missing.append("marker-pdf")
    
    # Check PyMuPDF
    try:
        import fitz
        print(f"✅ PyMuPDF installed (version: {fitz.version[0]})")
    except ImportError:
        print("❌ PyMuPDF NOT installed")
        missing.append("pymupdf")
    
    # Check torch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"✅ PyTorch installed (version: {torch.__version__})")
        print(f"   CUDA available: {cuda_available}")
        if cuda_available:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch NOT installed")
        missing.append("torch")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("    pip install marker-pdf")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_pdf():
    """Check if PDF exists."""
    step(2, "Checking PDF File")
    
    from src.config.settings import settings
    
    pdf_path = settings.pdf_path
    print(f"Expected PDF location: {pdf_path}")
    
    if pdf_path.exists():
        size = pdf_path.stat().st_size
        print(f"✅ PDF found! Size: {size:,} bytes")
        return True
    else:
        print("❌ PDF not found!")
        print("\nDownload with:")
        print("    python scripts/download_ng12.py")
        return False


def test_lightweight_first():
    """Quick test with lightweight mode."""
    step(3, "Quick Test with Lightweight Mode (PyMuPDF)")
    
    print("Testing lightweight mode first to verify basic setup...")
    
    from src.config.settings import settings
    from src.ingestion.pdf_parser import MarkerPDFParser
    
    parser = MarkerPDFParser()
    
    try:
        doc = parser.parse_lightweight(settings.pdf_path)
        print(f"\n✅ Lightweight parsing successful!")
        print(f"   Title: {doc.title}")
        print(f"   Pages: {doc.total_pages}")
        print(f"   Content pages: {len(doc.get_content_pages())}")
        return True
    except Exception as e:
        print(f"\n❌ Lightweight parsing failed: {e}")
        return False


def test_marker_full():
    """Full test with Marker."""
    step(4, "Full Test with MARKER (Deep Learning)")
    
    print("⏳ This will load ML models (~2GB on first run)...")
    print("   Models are cached after first download.\n")
    
    from src.config.settings import settings
    from src.ingestion.pdf_parser import MarkerPDFParser
    
    parser = MarkerPDFParser()
    
    try:
        doc = parser.parse(settings.pdf_path)
        print(f"\n✅ MARKER parsing successful!")
        print(f"   Title: {doc.title}")
        print(f"   Pages: {doc.total_pages}")
        print(f"   Content pages: {len(doc.get_content_pages())}")
        print(f"   Markdown length: {len(doc.full_markdown):,} chars")
        return doc
    except ImportError as e:
        print(f"\n❌ Marker import error: {e}")
        print("\nTry reinstalling:")
        print("    pip uninstall marker-pdf")
        print("    pip install marker-pdf")
        return None
    except Exception as e:
        print(f"\n❌ Marker parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(marker_doc):
    """Compare Marker vs Lightweight results."""
    step(5, "Comparing Marker vs Lightweight")
    
    from src.config.settings import settings
    from src.ingestion.pdf_parser import MarkerPDFParser
    
    parser = MarkerPDFParser()
    lightweight_doc = parser.parse_lightweight(settings.pdf_path)
    
    print("\n📊 Comparison:")
    print(f"{'Metric':<30} {'Marker':<20} {'Lightweight':<20}")
    print("-" * 70)
    print(f"{'Total pages':<30} {marker_doc.total_pages:<20} {lightweight_doc.total_pages:<20}")
    print(f"{'Content pages':<30} {len(marker_doc.get_content_pages()):<20} {len(lightweight_doc.get_content_pages()):<20}")
    print(f"{'Markdown length':<30} {len(marker_doc.full_markdown):<20,} {len(lightweight_doc.full_markdown):<20,}")
    print(f"{'Sections detected':<30} {len(marker_doc.get_sections()):<20} {len(lightweight_doc.get_sections()):<20}")
    
    # Count tables
    marker_tables = sum(1 for p in marker_doc.pages if p.has_tables)
    lightweight_tables = sum(1 for p in lightweight_doc.pages if p.has_tables)
    print(f"{'Pages with tables':<30} {marker_tables:<20} {lightweight_tables:<20}")


def show_sample_content(doc):
    """Show sample content from Marker parsing."""
    step(6, "Sample Content from Marker")
    
    content_pages = doc.get_content_pages()
    
    if len(content_pages) < 5:
        sample_page = content_pages[0] if content_pages else None
    else:
        sample_page = content_pages[4]  # Page 5 usually has good content
    
    if sample_page:
        print(f"\n📄 Content from Page {sample_page.page_number}:")
        print("-" * 40)
        print(sample_page.markdown[:1500])
        if len(sample_page.markdown) > 1500:
            print(f"\n... [truncated, full page is {len(sample_page.markdown)} chars]")
    
    # Show a table if found
    print("\n\n📊 Looking for tables in the document...")
    for page in content_pages:
        if page.has_tables and '|' in page.markdown:
            lines = page.markdown.split('\n')
            table_lines = [l for l in lines if l.strip().startswith('|')]
            if len(table_lines) >= 3:
                print(f"\nTable found on page {page.page_number}:")
                print("-" * 40)
                for line in table_lines[:10]:
                    print(line)
                if len(table_lines) > 10:
                    print(f"... [{len(table_lines) - 10} more rows]")
                break
    else:
        print("No markdown tables found (tables may be formatted differently)")


def save_output(doc):
    """Save the parsed output for inspection."""
    step(7, "Saving Output for Inspection")
    
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save full markdown
    md_path = output_dir / "ng12_parsed.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {doc.title}\n\n")
        f.write(f"Source: {doc.source}\n")
        f.write(f"Pages: {doc.total_pages}\n\n")
        f.write("---\n\n")
        f.write(doc.full_markdown)
    
    print(f"✅ Full markdown saved to: {md_path}")
    
    # Save page-by-page
    pages_path = output_dir / "ng12_pages.md"
    with open(pages_path, "w", encoding="utf-8") as f:
        for page in doc.pages:
            f.write(f"\n\n{'='*60}\n")
            f.write(f"PAGE {page.page_number} ({page.content_type})\n")
            f.write(f"Sections: {page.sections}\n")
            f.write(f"Has tables: {page.has_tables}\n")
            f.write('='*60 + "\n\n")
            f.write(page.markdown)
    
    print(f"✅ Page-by-page saved to: {pages_path}")
    
    print(f"\n📂 Open these files to inspect the Marker output!")


def main():
    print("\n" + "🧪 " * 20)
    print("   MARKER PDF PARSER - STEP BY STEP TEST")
    print("🧪 " * 20)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return 1
    
    # Step 2: Check PDF
    if not check_pdf():
        print("\n❌ Please download the PDF first.")
        return 1
    
    # Step 3: Quick lightweight test
    if not test_lightweight_first():
        print("\n❌ Basic setup has issues. Fix before trying Marker.")
        return 1
    
    # Step 4: Full Marker test
    print("\n" + "⚡ " * 20)
    print("   NOW TESTING WITH MARKER (DEEP LEARNING)")
    print("⚡ " * 20)
    
    doc = test_marker_full()
    if doc is None:
        print("\n❌ Marker test failed.")
        return 1
    
    # Step 5: Compare results
    compare_results(doc)
    
    # Step 6: Show sample content
    show_sample_content(doc)
    
    # Step 7: Save output
    save_output(doc)
    
    print("\n" + "✅ " * 20)
    print("   ALL TESTS PASSED!")
    print("✅ " * 20)
    
    print("\n📋 Summary:")
    print("   - Marker parsed the NG12 PDF successfully")
    print("   - Check the output/ directory for the parsed markdown")
    print("   - Marker output should have better tables and structure")
    print("   - Ready to proceed to chunking!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
