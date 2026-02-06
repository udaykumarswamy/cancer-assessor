#!/usr/bin/env python3
"""
Export Cached Markdown to Readable Files

Exports the cached Marker output to markdown files you can open and read.

"""

import sys
from pathlib import Path
import pickle
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger('export_markdown', level=logging.DEBUG)

def main():
    # Load cached document
    cache_file = settings.DATA_DIR / "cache" / "ng12_parsed.pkl"
    
    if not cache_file.exists():
        logger.error(f"Cache file not found: {cache_file}")
        logger.error("Run first: python scripts/parse_and_cache.py")
        return 1
    
    logger.debug(f"Loading from cached document: {cache_file}")
    with open(cache_file, 'rb') as f:
        doc = pickle.load(f)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    pages_dir = output_dir / "ng12_pages"
    pages_dir.mkdir(exist_ok=True)
    
    # Export full markdown
    full_md_path = output_dir / "ng12_full.md"
    with open(full_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {doc.title}\n\n")
        f.write(f"**Source:** {doc.source}\n\n")
        f.write(f"**Total Pages:** {doc.total_pages}\n\n")
        f.write("---\n\n")
        f.write(doc.full_markdown)
    
    logger.info(f"   Full document: {full_md_path}")
    logger.info(f"   Size: {full_md_path.stat().st_size:,} bytes")
    
    # Export page-by-page
    for page in doc.pages:
        page_path = pages_dir / f"page_{page.page_number:02d}.md"
        with open(page_path, 'w', encoding='utf-8') as f:
            f.write(f"# Page {page.page_number}\n\n")
            f.write(f"**Type:** {page.content_type}\n\n")
            if page.sections:
                f.write(f"**Sections:** {', '.join(page.sections)}\n\n")
            f.write(f"**Has Tables:** {page.has_tables}\n\n")
            f.write("---\n\n")
            f.write(page.markdown)
        
    logger.info(f" Individual pages: {pages_dir}/")
    logger.info(f" Files: page_01.md to page_{doc.total_pages:02d}.md")
    logger.info(" Export complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
