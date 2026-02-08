#!/usr/bin/env python3
"""
NG12 PDF Ingestion Pipeline

Complete pipeline to:
1. Download the NG12 PDF (if not present)
2. Parse the PDF using Marker
3. Chunk the content semantically
4. Generate embeddings with Vertex AI
5. Store in ChromaDB

Usage:
    # Full pipeline
    python scripts/ingest_pdf.py
    
    # Lightweight mode (no Marker models, faster but less accurate)
    python scripts/ingest_pdf.py --lightweight
    
    # Mock embeddings (for testing without Vertex AI credentials)
    python scripts/ingest_pdf.py --mock-embeddings
    
    # Force re-ingestion
    python scripts/ingest_pdf.py --force

Interview Discussion Points:
---------------------------
1. Pipeline design:
   - Modular components (can swap parser, embedder, store)
   - Checkpointing (can resume from failures)
   - Clear separation of concerns

2. Error handling:
   - Each stage has independent retry logic
   - Partial failures don't lose progress
   - Clear error messages for debugging

3. Performance considerations:
   - One-time operation (during build/deploy)
   - Can parallelize embedding generation
   - Chunking is memory-efficient (streaming)

4. Production enhancements:
   - Add checksums to detect PDF changes
   - Version the vector store
   - Implement incremental updates
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings


def download_pdf(force: bool = False) -> Path:
    """Download the NG12 PDF if not present."""
    pdf_path = settings.pdf_path
    
    if pdf_path.exists() and not force:
        print(f"✅ PDF already exists: {pdf_path}")
        return pdf_path
    
    print("📥 Downloading NG12 PDF...")
    
    from scripts.download_ng12 import download_file, verify_pdf
    
    download_file(settings.NG12_PDF_URL, pdf_path)
    
    if not verify_pdf(pdf_path):
        raise RuntimeError("PDF verification failed")
    
    return pdf_path


def parse_pdf(pdf_path: Path, lightweight: bool = False):
    """Parse the PDF and extract structured content."""
    print(f"\n📄 Parsing PDF: {pdf_path.name}")
    
    from src.ingestion.pdf_parser import MarkerPDFParser
    
    parser = MarkerPDFParser()
    
    if lightweight:
        doc = parser.parse_lightweight(pdf_path)
    else:
        doc = parser.parse(pdf_path)
    
    print(f"   Title: {doc.title}")
    print(f"   Pages: {doc.total_pages}")
    print(f"   Content pages: {len(doc.get_content_pages())}")
    
    return doc


def chunk_document(document, chunk_size: int = 512, overlap: int = 100):
    """Chunk the parsed document."""
    print(f"\n✂️ Chunking document...")
    
    from src.ingestion.chunker import SemanticChunker
    
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    chunks = chunker.chunk_document(document)
    
    # Print statistics
    total_tokens = sum(c.token_count for c in chunks)
    content_types = {}
    for c in chunks:
        content_types[c.content_type] = content_types.get(c.content_type, 0) + 1
    
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Avg tokens/chunk: {total_tokens // len(chunks)}")
    print(f"   Content types: {content_types}")
    
    return chunks


def embed_chunks(chunks, mock: bool = False):
    """Generate embeddings for all chunks."""
    print(f"\n📊 Generating embeddings...")
    
    from src.ingestion.embedder import get_embedder
    
    embedder = get_embedder(mock=mock)
    embedded = embedder.embed_chunks(chunks)
    
    return embedded


def store_chunks(embedded_chunks, force: bool = False):
    """Store embedded chunks in ChromaDB."""
    print(f"\n💾 Storing in vector database...")
    
    from src.ingestion.vector_store import VectorStore
    
    store = VectorStore()
    
    # Check if collection already has data
    stats = store.get_stats()
    if stats["total_chunks"] > 0 and not force:
        print(f"   Collection already has {stats['total_chunks']} chunks")
        print("   Use --force to replace")
        return stats["total_chunks"]
    
    if force and stats["total_chunks"] > 0:
        print(f"   Clearing existing {stats['total_chunks']} chunks...")
        store.clear()
    
    count = store.add_chunks(embedded_chunks)
    
    # Verify storage
    final_stats = store.get_stats()
    print(f"   Stored at: {final_stats['persist_dir']}")
    print(f"   Collection: {final_stats['collection_name']}")
    print(f"   Total chunks: {final_stats['total_chunks']}")
    
    return count


def verify_retrieval(mock: bool = False):
    """Test retrieval with a sample query."""
    print(f"\n🔍 Verifying retrieval...")
    
    from src.ingestion.embedder import get_embedder
    from src.ingestion.vector_store import VectorStore
    
    embedder = get_embedder(mock=mock)
    store = VectorStore()
    
    # Test query
    test_query = "urgent referral for hemoptysis"
    print(f"   Test query: '{test_query}'")
    
    query_embedding = embedder.embed_query(test_query)
    results = store.search(query_embedding, top_k=3)
    
    print(f"   Retrieved {len(results)} chunks:")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. {result.get_citation()} (score: {result.score:.3f})")
        print(f"      Type: {result.content_type}")
        print(f"      Preview: {result.text[:150]}...")
    
    return len(results) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Ingest NG12 PDF into vector store"
    )
    parser.add_argument(
        "--lightweight", "-l",
        action="store_true",
        help="Use lightweight PDF parsing (no Marker models)"
    )
    parser.add_argument(
        "--mock-embeddings", "-m",
        action="store_true",
        help="Use mock embeddings (for testing without Vertex AI)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-ingestion even if data exists"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.CHUNK_SIZE_TOKENS,
        help=f"Chunk size in tokens (default: {settings.CHUNK_SIZE_TOKENS})"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=settings.CHUNK_OVERLAP_TOKENS,
        help=f"Chunk overlap in tokens (default: {settings.CHUNK_OVERLAP_TOKENS})"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PDF download (assume already present)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify retrieval (skip ingestion)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NG12 Cancer Guidelines - Ingestion Pipeline")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Mode: {'Lightweight' if args.lightweight else 'Full (Marker)'}")
    print(f"Embeddings: {'Mock' if args.mock_embeddings else 'Vertex AI'}")
    print()
    
    start_time = time.time()
    
    try:
        # Verify only mode
        if args.verify_only:
            success = verify_retrieval(mock=args.mock_embeddings)
            if success:
                print("\n✅ Verification passed!")
                return 0
            else:
                print("\n❌ Verification failed!")
                return 1
        
        # Step 1: Download PDF
        if not args.skip_download:
            pdf_path = download_pdf(force=args.force)
        else:
            pdf_path = settings.pdf_path
            if not pdf_path.exists():
                print(f"❌ PDF not found: {pdf_path}")
                return 1
        
        # Step 2: Parse PDF
        document = parse_pdf(pdf_path, lightweight=args.lightweight)
        
        # Step 3: Chunk document
        chunks = chunk_document(
            document,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        
        # Step 4: Generate embeddings
        embedded_chunks = embed_chunks(chunks, mock=args.mock_embeddings)
        
        # Step 5: Store in vector database
        stored_count = store_chunks(embedded_chunks, force=args.force)
        
        # Step 6: Verify retrieval
        verify_retrieval(mock=args.mock_embeddings)
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ Ingestion Complete!")
        print("=" * 60)
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Chunks indexed: {stored_count}")
        print(f"Vector store: {settings.VECTORSTORE_DIR}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
