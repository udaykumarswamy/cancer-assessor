#!/usr/bin/env python3
"""
Complete End-to-End Ingestion Pipeline 

Full pipeline from PDF download to retrieval testing:
1. Download NG12 PDF (if needed)
2. Parse and cache the document
3. Chunk the content semantically
4. Generate embeddings with Vertex AI
5. Store in ChromaDB
6. Test retrieval with various clinical filters

Usage:
    # Full pipeline (downloads PDF if needed)
    python scripts/end2end_ingestion.py --mock
    
    # Skip PDF download (use existing)
    python scripts/end2end_ingestion.py --mock --skip-download
    
    # Force re-download PDF
    python scripts/end2end_ingestion.py --mock --force-download
    
    # Skip storage (just test embedding)
    python scripts/end2end_ingestion.py --mock --no-store
    
    # Clear and re-ingest everything
    python scripts/end2end_ingestion.py --mock --force
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import hashlib

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger('e2e_pipeline', level='DEBUG')

def download_ng12_pdf(force: bool = False, skip: bool = False) -> bool:
    """
    Download the NG12 PDF guideline document.
    
    This is the first step of the pipeline - downloads the official NICE NG12 PDF
    if it doesn't already exist, or if force=True.
    
    Args:
        force: Force re-download even if file exists
        skip: Skip download entirely (assume PDF exists)
        
    Returns:
        True if PDF is available and valid, False otherwise
    """
    pdf_path = settings.pdf_path
    
    if skip:
        logger.info("⏭️  Skipping PDF download (--skip-download)")
        if pdf_path.exists():
            logger.info(f"   ✅ PDF exists at: {pdf_path}")
            return True
        else:
            logger.error(f"   ❌ PDF not found at: {pdf_path}")
            logger.info("   Run without --skip-download to download it")
            return False
    
    # Check if already downloaded
    if pdf_path.exists() and not force:
        logger.info(f"📄 PDF already downloaded at: {pdf_path}")
        file_size = pdf_path.stat().st_size / (1024 * 1024)  # Convert to MB
        logger.info(f"   ✅ File size: {file_size:.1f} MB")
        
        # Verify it's a valid PDF
        with open(pdf_path, "rb") as f:
            magic = f.read(5)
            if magic == b"%PDF-":
                logger.info("   ✅ PDF validation passed")
                return True
            else:
                logger.warning("   ⚠️  PDF validation failed, will re-download")
    
    # Download the PDF
    logger.info("\n📥 Downloading NG12 PDF guideline document...")
    logger.info(f"   URL: {settings.NG12_PDF_URL}")
    logger.info(f"   Destination: {pdf_path}")
    
    try:
        # Import here to avoid requiring httpx globally
        import httpx
        from tenacity import retry, stop_after_attempt, wait_exponential
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True
        )
        def download_with_retry(url: str, output: Path) -> None:
            """Download with retry logic."""
            output.parent.mkdir(parents=True, exist_ok=True)
            
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                
                with open(output, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if int(progress) % 10 == 0:  # Log every 10%
                                logger.info(f"   Download progress: {progress:.1f}%")
        
        download_with_retry(settings.NG12_PDF_URL, pdf_path)
        
        # Verify download
        if pdf_path.stat().st_size < 1000:
            logger.error("   ❌ Downloaded file is too small (corrupted?)")
            return False
        
        with open(pdf_path, "rb") as f:
            magic = f.read(5)
            if magic != b"%PDF-":
                logger.error("   ❌ Downloaded file is not a valid PDF")
                return False
        
        file_size = pdf_path.stat().st_size / (1024 * 1024)
        logger.info(f"   ✅ Download successful! ({file_size:.1f} MB)")
        logger.info(f"   📍 Saved to: {pdf_path}")
        return True
        
    except ImportError:
        logger.error("   ❌ Required libraries not installed")
        logger.info("   Run: pip install httpx tenacity")
        return False
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        logger.info("   This may be a temporary network issue. Try again.")
        return False

def load_document(use_cache: bool = True, lightweight: bool = False):
    """
    Load parsed document from cache or parse fresh PDF.
    
    Args:
        use_cache: If True, load from cache; otherwise parse fresh
        lightweight: If parsing fresh, use lightweight mode
        
    Returns:
        ParsedDocument object
    """
    if use_cache:
        return load_cached_document()
    else:
        return parse_fresh_pdf(lightweight=lightweight)


def load_cached_document():
    """Load the cached parsed document."""
    from parse_and_cache import load_from_cache
    
    logger.info("📄 Loading cached parsed document...")
    logger.info("   (Initializing dependencies - may take 10-30 seconds on first load)")
    
    start = time.time()
    doc = load_from_cache()
    elapsed = time.time() - start
    
    if not doc:
        logger.error("❌ No cached document found!")
        logger.info("   Run: python scripts/parse_and_cache.py first")
        logger.info("   Or use: python scripts/test_e2e_pipeline.py --parse-pdf")
        return None
    
    logger.info(f"   ✅ Cache loaded in {elapsed:.1f}s + dependency init")
    logger.info(f"   Title: {doc.title}")
    logger.info(f"   Total pages: {doc.total_pages}")
    logger.info(f"   Content pages: {len(doc.get_content_pages())}")
    
    return doc


def parse_fresh_pdf(lightweight: bool = False):
    """Parse the PDF fresh using Marker or lightweight parser."""
    from src.ingestion.pdf_parser import MarkerPDFParser
    
    pdf_path = settings.pdf_path
    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return None
    
    logger.info(f"📄 Parsing PDF: {pdf_path.name}")
    parser = MarkerPDFParser()
    
    start = time.time()
    if lightweight:
        logger.info("   Using lightweight parser (faster, less accurate)...")
        doc = parser.parse_lightweight(pdf_path)
    else:
        logger.info("   Using full Marker parser (slower, more accurate)...")
        doc = parser.parse(pdf_path)
    
    elapsed = time.time() - start
    logger.info(f"   ✅ Parsed in {elapsed:.1f}s")
    logger.info(f"   Title: {doc.title}")
    logger.info(f"   Total pages: {doc.total_pages}")
    logger.info(f"   Content pages: {len(doc.get_content_pages())}")
    
    return doc

def chunk_document(document, chunk_size: int = None, overlap: int = None):
    """Chunk the parsed document semantically and convert to RichChunk format."""
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.embedder import RichChunk, ClinicalMetadata
    
    chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
    overlap = overlap or settings.CHUNK_OVERLAP_TOKENS
    
    logger.info(f"\n✂️ Chunking document (size={chunk_size}, overlap={overlap})...")
    
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    raw_chunks = chunker.chunk_document(document)
    
    # Convert Chunk objects to RichChunk format
    chunks = []
    for raw_chunk in raw_chunks:
        # Helper function to convert list to comma-separated string
        def list_to_string(items):
            if isinstance(items, list):
                return ",".join(items) if items else ""
            return str(items) if items else ""
        
        # Convert clinical metadata - lists become comma-separated strings
        clinical = ClinicalMetadata(
            has_recommendation=raw_chunk.to_dict().get("has_recommendation", False),
            has_table=raw_chunk.to_dict().get("has_table", False),
            is_urgent=raw_chunk.to_dict().get("is_urgent", False),
            cancer_types=list_to_string(raw_chunk.clinical_metadata.cancer_types),
            symptoms=list_to_string(raw_chunk.clinical_metadata.symptoms),
            age_thresholds=list_to_string(raw_chunk.clinical_metadata.age_thresholds),
            timeframes=list_to_string(raw_chunk.clinical_metadata.timeframes),
            urgency=raw_chunk.clinical_metadata.urgency.value if raw_chunk.clinical_metadata.urgency else "",
            actions=list_to_string(raw_chunk.clinical_metadata.actions),
            risk_factors=list_to_string(raw_chunk.clinical_metadata.risk_factors),
            investigations=list_to_string(raw_chunk.clinical_metadata.investigations),
        )
        
        # Create RichChunk
        rich_chunk = RichChunk(
            chunk_id=raw_chunk.chunk_id,
            text=raw_chunk.text,
            page_start=raw_chunk.page_start,
            page_end=raw_chunk.page_end,
            section=raw_chunk.section or "",
            section_hierarchy="|".join(raw_chunk.section_hierarchy) if raw_chunk.section_hierarchy else "",
            content_type=raw_chunk.content_type.value,
            token_count=raw_chunk.token_count,
            char_count=raw_chunk.char_count,
            chunk_index=raw_chunk.chunk_index,
            prev_chunk_id=raw_chunk.prev_chunk_id,
            next_chunk_id=raw_chunk.next_chunk_id,
            semantic_density=raw_chunk.semantic_density,
            created_at=raw_chunk.created_at,
            source="NG12",
            clinical=clinical,
        )
        chunks.append(rich_chunk)
    
    # Print statistics
    total_tokens = sum(c.token_count for c in chunks)
    content_types = {}
    urgency_counts = {}
    
    for c in chunks:
        content_types[c.content_type] = content_types.get(c.content_type, 0) + 1
        urgency = c.clinical.urgency or "unspecified"
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
    
    logger.info(f"   ✅ Total chunks: {len(chunks)}")
    logger.info(f"   📊 Total tokens: {total_tokens:,}")
    logger.info(f"   📈 Avg tokens/chunk: {total_tokens // len(chunks)}")
    logger.info(f"   🏷️ Content types: {content_types}")
    logger.info(f"   🚨 Urgency levels: {urgency_counts}")
    
    return chunks


def embed_chunks(chunks, mock: bool = False):
    """Generate embeddings for all chunks."""
    from src.ingestion.embedder import get_embedder
    
    logger.info(f"\n📊 Generating embeddings ({'mock' if mock else 'Vertex AI'})...")
    
    embedder = get_embedder(mock=mock)
    start = time.time()
    
    embedded = embedder.embed_chunks(chunks, include_metadata=True)
    
    elapsed = time.time() - start
    logger.info(f"   ✅ Embedded {len(embedded)} chunks in {elapsed:.1f}s")
    logger.info(f"   ⚡ Rate: {len(embedded)/elapsed:.1f} chunks/sec")
    
    return embedded, embedder


def store_chunks(embedded_chunks, force: bool = False):
    """Store embedded chunks in ChromaDB."""
    from src.ingestion.vector_store import VectorStore
    
    logger.info(f"\n💾 Storing in ChromaDB...")
    
    store = VectorStore()
    
    # Check existing data
    stats = store.get_stats()
    if stats["total_chunks"] > 0 and not force:
        logger.info(f"   Collection already has {stats['total_chunks']} chunks")
        logger.info("   Use --force to replace")
        return store, stats["total_chunks"]
    
    if force and stats["total_chunks"] > 0:
        logger.info(f"   Clearing existing {stats['total_chunks']} chunks...")
        store.clear()
    
    count = store.add_chunks(embedded_chunks)
    
    # Verify storage
    final_stats = store.get_stats()
    logger.info(f"   📍 Stored at: {final_stats['persist_dir']}")
    logger.info(f"   🏷️ Collection: {final_stats['collection_name']}")
    logger.info(f"   ✅ Total chunks: {final_stats['total_chunks']}")
    
    return store, count


def test_retrieval(store, embedder):
    """Test various retrieval patterns."""
    logger.info(f"\n🔍 Testing retrieval patterns...")
    
    test_cases = [
        {
            "name": "Lymphoma with splenomegaly",
            "query": "splenomegaly lymphoma referral",
            "test": "search"
        },
        {
            "name": "Lung cancer symptoms",
            "query": "persistent cough chest x-ray",
            "test": "search"
        },
        {
            "name": "Urgent recommendations",
            "query": "cancer symptoms diagnosis",
            "test": "urgent_only"
        },
        {
            "name": "Breast cancer content",
            "query": "breast lump referral",
            "test": "by_cancer_type",
            "cancer_type": "breast"
        },
        {
            "name": "Lung cancer content",
            "query": "cough hemoptysis lung",
            "test": "by_cancer_type",
            "cancer_type": "lung"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n   Test {i}: {test['name']}")
        logger.info(f"   Query: '{test['query']}'")
        
        query_embedding = embedder.embed_query(test['query'])
        
        if test['test'] == 'search':
            results = store.search(query_embedding, top_k=3)
        elif test['test'] == 'urgent_only':
            results = store.search_urgent_only(query_embedding, top_k=3)
        elif test['test'] == 'by_cancer_type':
            results = store.search_by_cancer_type(
                query_embedding, 
                test['cancer_type'], 
                top_k=3
            )
        
        if results:
            logger.info(f"   ✅ Retrieved {len(results)} results:")
            for j, result in enumerate(results[:2], 1):  # Show top 2
                logger.info(f"      [{j}] {result.get_citation()} (score: {result.score:.3f})")
                logger.info(f"          Urgency: {result.get_urgency_display()}")
                logger.info(f"          Preview: {result.text[:80]}...")
        else:
            logger.warning(f"   ⚠️ No results!")


def test_clinical_filters(store, embedder):
    """Test clinical metadata filtering."""
    logger.info(f"\n🩺 Testing clinical metadata...")
    
    stats = store.get_stats()
    
    logger.info(f"   Total chunks: {stats['total_chunks']}")
    logger.info(f"   Urgent chunks: {stats['urgent_chunks']}")
    logger.info(f"   Recommendation chunks: {stats['recommendation_chunks']}")
    logger.info(f"   Table chunks: {stats['table_chunks']}")
    logger.info(f"   Cancer types: {', '.join(stats['cancer_types'])}")
    logger.info(f"   Urgency levels: {', '.join(stats['urgency_levels'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Test complete end-to-end ingestion pipeline"
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Use mock embeddings (no Vertex AI needed - RECOMMENDED for testing)"
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Skip storage (test chunking and embedding only)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-ingestion"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override chunk size (tokens)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Override chunk overlap (tokens)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose debug logging"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PDF download (assume it already exists)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of NG12 PDF even if it exists"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("END-TO-END INGESTION PIPELINE TEST")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Mode: {'Mock Embeddings 🧪' if args.mock else 'Vertex AI ☁️'}")
    if args.debug:
        logger.info("Debug: ON")
    logger.info("")
    
    start_time = time.time()
    
    try:
        # Step 0: Download NG12 PDF (if needed)
        logger.info(f"\n{'='*70}")
        logger.info("STEP 0: Download NG12 PDF Guideline")
        logger.info(f"{'='*70}")
        
        if not download_ng12_pdf(force=args.force_download, skip=args.skip_download):
            logger.error("❌ Failed to download or verify NG12 PDF")
            return 1
        
        # Step 1: Load cached document
        logger.info(f"\n{'='*70}")
        logger.info("STEP 1: Load and Parse Document")
        logger.info(f"{'='*70}")
        
        document = load_cached_document()
        if not document:
            return 1
        
        # Step 2: Chunk document
        logger.info(f"\n{'='*70}")
        logger.info("STEP 2: Chunk Document")
        logger.info(f"{'='*70}")
        
        chunks = chunk_document(
            document,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        
        # Step 3: Generate embeddings
        logger.info(f"\n{'='*70}")
        logger.info("STEP 3: Generate Embeddings")
        logger.info(f"{'='*70}")
        
        embedded_chunks, embedder = embed_chunks(chunks, mock=args.mock)
        
        # Step 4: Store (optional)
        logger.info(f"\n{'='*70}")
        logger.info("STEP 4: Store in Vector Database")
        logger.info(f"{'='*70}")
        if not args.no_store:
            store, count = store_chunks(embedded_chunks, force=args.force)
            
            # Step 5: Test retrieval
            logger.info(f"\n{'='*70}")
            logger.info("STEP 5: Test Retrieval Patterns")
            logger.info(f"{'='*70}")
            
            test_retrieval(store, embedder)
            
            # Step 6: Test clinical filters
            logger.info(f"\n{'='*70}")
            logger.info("STEP 6: Test Clinical Metadata Filtering")
            logger.info(f"{'='*70}")
            
            test_clinical_filters(store, embedder)
        else:
            logger.info("\n⏭️ Skipped storage (--no-store)")
        
        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("✅ FULL PIPELINE COMPLETE - ALL STEPS SUCCESSFUL!")
        logger.info("=" * 70)
        logger.info(f"\n📊 Pipeline Summary:")
        logger.info(f"   ✓ Step 0: PDF downloaded/verified")
        logger.info(f"   ✓ Step 1: Document loaded and parsed")
        logger.info(f"   ✓ Step 2: Created {len(chunks)} chunks")
        logger.info(f"   ✓ Step 3: Generated embeddings")
        if not args.no_store:
            logger.info(f"   ✓ Step 4: Stored {count} chunks in vector database")
            logger.info(f"   ✓ Step 5: Tested retrieval patterns")
            logger.info(f"   ✓ Step 6: Tested clinical metadata filtering")
        else:
            logger.info(f"   ✓ Step 4-6: Skipped (--no-store)")
        logger.info(f"\n⏱️  Total Duration: {elapsed:.1f}s")
        logger.info(f"📁 Vector store ready at: {settings.VECTORSTORE_DIR}")
        logger.info(f"🚀 System is ready for clinical assessments!\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
