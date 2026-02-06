#!/usr/bin/env python3
"""
Test Cached Marker Output

Use this to verify and explore the cached PDF parsing results.

Usage:
    python tests/test_cached_output.py
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config.logging_config import get_logger

logger = get_logger('test_cached_output', level=logging.DEBUG)

def load_cached_document():
    """Load the cached document."""
    from src.config.settings import settings
    import pickle
    
    cache_file = settings.DATA_DIR / "cache" / "ng12_parsed.pkl"
    
    if not cache_file.exists():
        logger.debug(f"Cache not found at: {cache_file}")
        logger.info("   Run first: python scripts/parse_and_cache.py")
        return None
    
    logger.info(f"Loading from: {cache_file}")
    with open(cache_file, 'rb') as f:
        doc = pickle.load(f)
    
    logger.info(f" Loaded instantly!\n")
    return doc


def test_basic_info(doc):
    """Test basic document info."""
    logger.debug("=" * 60)
    logger.debug("TEST 1: Basic Document Info")
    logger.debug("=" * 60)
    
    logger.debug(f"   Title: {doc.title}")
    logger.debug(f"   Source: {doc.source}")
    logger.debug(f"   Total pages: {doc.total_pages}")
    logger.debug(f"   Markdown length: {len(doc.full_markdown):,} characters")
    
    return True


def test_page_structure(doc):
    """Test page structure."""
    logger.debug("\n" + "=" * 60)
    logger.debug("TEST 2: Page Structure")
    logger.debug("=" * 60)
    
    # Count by type
    types = {}
    for page in doc.pages:
        types[page.content_type] = types.get(page.content_type, 0) + 1
    
    logger.info("   Page types:")
    for t, count in sorted(types.items()):
        logger.info(f"      {t}: {count}")
    
    content_pages = doc.get_content_pages()
    logger.debug(f"\n   Content pages (will be indexed): {len(content_pages)}")
    
    return True


def test_sections(doc):
    """Test section detection."""
    logger.debug("\n" + "=" * 60)
    logger.debug("TEST 3: Sections Detected")
    logger.debug("=" * 60)
    
    sections = doc.get_sections()
    logger.debug(f"   Total sections found: {len(sections)}")
    
    if sections:
        logger.debug("\n   First 15 sections:")
        for section, page_num in sections[:15]:
            logger.debug(f"      [Page {page_num:2d}] {section}")
    
    return len(sections) > 0


def test_tables(doc):
    """Test table detection."""
    logger.debug("\n" + "=" * 60)
    logger.debug("TEST 4: Tables Detected")
    logger.debug("=" * 60)
    
    pages_with_tables = [p for p in doc.pages if p.has_tables]
    logger.debug(f"   Pages with tables: {len(pages_with_tables)}")
    
    # Find actual markdown tables
    table_count = 0
    for page in doc.pages:
        if '|' in page.markdown and '---' in page.markdown:
            table_count += 1
    
    logger.debug(f"   Pages with markdown tables: {table_count}")
    
    # Show a sample table
    for page in doc.get_content_pages():
        lines = page.markdown.split('\n')
        table_lines = [l for l in lines if l.strip().startswith('|')]
        if len(table_lines) >= 3:
            logger.debug(f"\n   Sample table from page {page.page_number}:")
            logger.debug("   " + "-" * 50)
            for line in table_lines[:8]:
                logger.debug(f"   {line}")
            if len(table_lines) > 8:
                logger.debug(f"   ... ({len(table_lines) - 8} more rows)")
            break
    
    return True


def test_clinical_content(doc):
    """Test for clinical content quality."""
    logger.debug("\n" + "=" * 60)
    logger.debug("TEST 5: Clinical Content Quality")
    logger.debug("=" * 60)
    
    full_text = doc.full_markdown.lower()
    
    # Key terms that should be in NG12
    key_terms = {
        "lung cancer": 0,
        "breast cancer": 0,
        "hemoptysis": 0,
        "haemoptysis": 0,  # UK spelling
        "urgent referral": 0,
        "2 week": 0,
        "two week": 0,
        "suspected cancer": 0,
        "refer": 0,
    }
    
    for term in key_terms:
        key_terms[term] = full_text.count(term)
    
    logger.debug("   Key clinical terms found:")
    for term, count in sorted(key_terms.items(), key=lambda x: -x[1]):
        status = "Yes" if count > 0 else "No"
        logger.debug(f"      {status} '{term}': {count}")
    
    return sum(key_terms.values()) > 50


def test_sample_content(doc):
    """Show sample content from different pages."""
    logger.debug("\n" + "=" * 60)
    logger.debug("TEST 6: Sample Content")
    logger.debug("=" * 60)
    
    content_pages = doc.get_content_pages()
    
    # Show samples from beginning, middle, end
    samples = []
    if len(content_pages) >= 3:
        samples = [
            content_pages[0],                          # First
            content_pages[len(content_pages) // 2],    # Middle
            content_pages[-1]                          # Last
        ]
    else:
        samples = content_pages[:3]
    
    for page in samples:
        logger.debug(f"\n   --- Page {page.page_number} ({page.content_type}) ---")
        logger.debug(f"   Sections: {page.sections if page.sections else 'None'}")
        preview = page.markdown[:400].replace('\n', '\n   ')
        logger.debug(f"   {preview}")
        if len(page.markdown) > 400:
            logger.debug(f"   ... [truncated, {len(page.markdown)} chars total]")
    
    return True


def test_ready_for_chunking(doc):
    """Verify document is ready for chunking."""
    logger.debug("\n" + "=" * 60)
    logger.debug("TEST 7: Ready for Chunking?")
    logger.debug("=" * 60)
    
    checks = {
        "Has content pages": len(doc.get_content_pages()) > 0,
        "Has sections": len(doc.get_sections()) > 0,
        "Has substantial text": len(doc.full_markdown) > 10000,
        "Has clinical terms": "refer" in doc.full_markdown.lower(),
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "Yes" if passed else "No"
        logger.debug(f"   {status} {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.debug("\n Document is ready for chunking!")
    else:
        logger.debug("\n Some checks failed. Review the output.!!!!!!!!")
    
    return all_passed


def main():
    logger.debug("   TESTING CACHED MARKER OUTPUT")
    logger.debug("=" * 60 + "\n")    
    # Load cached document
    doc = load_cached_document()
    if doc is None:
        return 1
    
    # Run all tests
    results = []
    results.append(("Basic Info", test_basic_info(doc)))
    results.append(("Page Structure", test_page_structure(doc)))
    results.append(("Sections", test_sections(doc)))
    results.append(("Tables", test_tables(doc)))
    results.append(("Clinical Content", test_clinical_content(doc)))
    results.append(("Sample Content", test_sample_content(doc)))
    results.append(("Ready for Chunking", test_ready_for_chunking(doc)))
    
    # Summary
    logger.debug("\n" + "=" * 60)
    logger.debug("SUMMARY")
    logger.debug("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "yes" if result else "No"
        logger.debug(f"   {status} {name}")
    
    logger.debug(f"\n   Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.debug("\n  All tests passed! Ready for next step: chunker.py")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
