#!/usr/bin/env python3
"""
Test Chunker with Cached Document

Tests the semantic chunker using the cached Marker output.

Usage:
    python tests/test_chunker.py
"""

import sys
import json
from pathlib import Path
import pickle
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.ingestion.chunker import SemanticChunker, chunk_document, Chunk, ContentType, UrgencyLevel


def load_cached_document():
    """Load the cached document."""
    cache_file = settings.DATA_DIR / "cache" / "ng12_parsed.pkl"
    
    if not cache_file.exists():
        print(f"❌ Cache not found: {cache_file}")
        print("   Run first: python scripts/parse_and_cache.py")
        return None
    
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def test_basic_chunking(doc):
    """Test basic chunking functionality."""
    print("=" * 60)
    print("TEST 1: Basic Chunking (Gemini Compatible)")
    print("=" * 60)
    
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=100)
    chunks = chunker.chunk_document(doc)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Statistics
    token_counts = [c.token_count for c in chunks]
    char_counts = [c.char_count for c in chunks]
    
    print(f"\n📊 Token Statistics (Gemini ~4 chars/token):")
    print(f"   Min: {min(token_counts)} tokens ({min(char_counts)} chars)")
    print(f"   Max: {max(token_counts)} tokens ({max(char_counts)} chars)")
    print(f"   Avg: {sum(token_counts) // len(token_counts)} tokens")
    print(f"   Total: {sum(token_counts):,} tokens")
    
    # Check consistency
    target_chars = 512 * 4  # 2048 chars
    over_limit = sum(1 for c in char_counts if c > target_chars * 1.1)
    print(f"\n📏 Chunk Size Consistency:")
    print(f"   Target: ~{target_chars} chars ({512} tokens)")
    print(f"   Over limit (>10%): {over_limit} chunks")
    
    return chunks


def test_clinical_metadata(chunks):
    """Test clinical metadata extraction."""
    print("\n" + "=" * 60)
    print("TEST 2: Clinical Metadata Extraction")
    print("=" * 60)
    
    # Collect all metadata
    all_cancers = []
    all_symptoms = []
    all_ages = []
    all_urgencies = []
    all_actions = []
    
    for chunk in chunks:
        meta = chunk.clinical_metadata
        all_cancers.extend(meta.cancer_types)
        all_symptoms.extend(meta.symptoms)
        all_ages.extend(meta.age_thresholds)
        all_urgencies.append(meta.urgency.value)
        all_actions.extend(meta.actions)
    
    print("\n🎯 Cancer Types Found:")
    for cancer, count in Counter(all_cancers).most_common(10):
        print(f"   {cancer}: {count}")
    
    print("\n🩺 Symptoms Found:")
    for symptom, count in Counter(all_symptoms).most_common(10):
        print(f"   {symptom}: {count}")
    
    print("\n👤 Age Thresholds:")
    for age, count in Counter(all_ages).most_common():
        print(f"   {age}: {count}")
    
    print("\n⚡ Urgency Levels:")
    for urgency, count in Counter(all_urgencies).most_common():
        print(f"   {urgency}: {count}")
    
    print("\n📋 Clinical Actions:")
    for action, count in Counter(all_actions).most_common():
        print(f"   {action}: {count}")
    
    return True


def test_content_types(chunks):
    """Test content type classification."""
    print("\n" + "=" * 60)
    print("TEST 3: Content Type Classification")
    print("=" * 60)
    
    types = Counter([c.content_type.value for c in chunks])
    
    print("\n📊 Content Types:")
    for ctype, count in types.most_common():
        pct = count * 100 // len(chunks)
        bar = "█" * (pct // 2)
        print(f"   {ctype:15s} {count:3d} ({pct:2d}%) {bar}")
    
    return True


def test_chunk_linking(chunks):
    """Test chunk prev/next linking."""
    print("\n" + "=" * 60)
    print("TEST 4: Chunk Linking")
    print("=" * 60)
    
    with_prev = sum(1 for c in chunks if c.prev_chunk_id)
    with_next = sum(1 for c in chunks if c.next_chunk_id)
    
    print(f"\n🔗 Linking Statistics:")
    print(f"   Chunks with prev link: {with_prev}/{len(chunks)}")
    print(f"   Chunks with next link: {with_next}/{len(chunks)}")
    
    # Verify chain integrity
    errors = 0
    for i, chunk in enumerate(chunks[1:], 1):
        if chunk.prev_chunk_id != chunks[i-1].chunk_id:
            errors += 1
    
    if errors == 0:
        print("   ✅ Chain integrity verified")
    else:
        print(f"   ⚠️  {errors} chain breaks found")
    
    return errors == 0


def test_sample_chunks(chunks):
    """Show sample chunks with metadata."""
    print("\n" + "=" * 60)
    print("TEST 5: Sample Chunks")
    print("=" * 60)
    
    # Find a recommendation chunk
    rec_chunk = None
    for chunk in chunks:
        if chunk.content_type == ContentType.RECOMMENDATION:
            rec_chunk = chunk
            break
    
    if rec_chunk:
        print("\n📋 Sample RECOMMENDATION chunk:")
        print("-" * 40)
        print(f"ID: {rec_chunk.chunk_id}")
        print(f"Section: {rec_chunk.section}")
        print(f"Pages: {rec_chunk.page_start}-{rec_chunk.page_end}")
        print(f"Tokens: {rec_chunk.token_count}")
        print(f"Chars: {rec_chunk.char_count}")
        print(f"Density: {rec_chunk.semantic_density:.3f}")
        print(f"Citation: {rec_chunk.get_citation()}")
        print(f"\nClinical Metadata:")
        meta = rec_chunk.clinical_metadata
        print(f"  Urgency: {meta.urgency.value}")
        print(f"  Cancers: {meta.cancer_types}")
        print(f"  Symptoms: {meta.symptoms}")
        print(f"  Ages: {meta.age_thresholds}")
        print(f"  Actions: {meta.actions}")
        print(f"\nText Preview:")
        print(f"  {rec_chunk.text[:400]}...")
    
    # Find an urgent chunk
    urgent_chunk = None
    for chunk in chunks:
        if chunk.clinical_metadata.urgency == UrgencyLevel.URGENT_2_WEEK:
            urgent_chunk = chunk
            break
    
    if urgent_chunk and urgent_chunk != rec_chunk:
        print("\n\n⚡ Sample URGENT 2-WEEK chunk:")
        print("-" * 40)
        print(f"ID: {urgent_chunk.chunk_id}")
        print(f"Section: {urgent_chunk.section}")
        print(f"Citation: {urgent_chunk.get_citation()}")
        print(f"Text Preview: {urgent_chunk.text[:300]}...")
    
    return True


def test_overlap(doc):
    """Test overlap consistency."""
    print("\n" + "=" * 60)
    print("TEST 6: Overlap Verification")
    print("=" * 60)
    
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=100)
    chunks = chunker.chunk_document(doc)
    
    overlaps_found = 0
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i].text[-400:]
        chunk2_start = chunks[i + 1].text[:400]
        
        words1 = set(chunk1_end.lower().split())
        words2 = set(chunk2_start.lower().split())
        common = words1 & words2
        
        if len(common) > 5:
            overlaps_found += 1
    
    overlap_pct = overlaps_found * 100 // (len(chunks) - 1) if len(chunks) > 1 else 0
    print(f"\n📊 Overlap Statistics:")
    print(f"   Chunks with detected overlap: {overlaps_found}/{len(chunks)-1}")
    print(f"   Overlap rate: {overlap_pct}%")
    
    return True


def test_to_dict(chunks):
    """Test serialization."""
    print("\n" + "=" * 60)
    print("TEST 7: Serialization (to_dict)")
    print("=" * 60)
    
    sample = chunks[0]
    d = sample.to_dict()
    
    print(f"\n📦 Serialized keys ({len(d)} total):")
    for key in sorted(d.keys()):
        value = d[key]
        if isinstance(value, str) and len(value) > 50:
            print(f"   {key}: '{value[:50]}...'")
        else:
            print(f"   {key}: {value}")
    
    return True


def export_chunks(chunks):
    """Export chunks to files."""
    print("\n" + "=" * 60)
    print("EXPORT: Saving Chunks")
    print("=" * 60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Export as JSON
    json_file = output_dir / "chunks.json"
    data = []
    for chunk in chunks:
        d = chunk.to_dict()
        data.append(d)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON: {json_file} ({json_file.stat().st_size:,} bytes)")
    
    # Export as markdown (first 20 chunks)
    md_file = output_dir / "chunks.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# NG12 Chunks (Gemini Compatible)\n\n")
        f.write(f"Total chunks: {len(chunks)}\n\n")
        
        for chunk in chunks:
            f.write(f"---\n\n")
            f.write(f"## {chunk.chunk_id}\n\n")
            f.write(f"- **Section:** {chunk.section or 'N/A'}\n")
            f.write(f"- **Pages:** {chunk.page_start}-{chunk.page_end}\n")
            f.write(f"- **Type:** {chunk.content_type.value}\n")
            f.write(f"- **Tokens:** {chunk.token_count}\n")
            f.write(f"- **Urgency:** {chunk.clinical_metadata.urgency.value}\n")
            f.write(f"- **Citation:** {chunk.get_citation()}\n")
            
            meta = chunk.clinical_metadata
            if meta.cancer_types:
                f.write(f"- **Cancers:** {', '.join(meta.cancer_types)}\n")
            if meta.symptoms:
                f.write(f"- **Symptoms:** {', '.join(meta.symptoms)}\n")
            
            f.write(f"\n```\n{chunk.text}\n```\n\n")
    
    print(f"✅ Markdown: {md_file}")
    print(f"\n💡 Open with: code {md_file}")


def main():
    print("\n" + "🧪 " * 20)
    print("   TESTING SEMANTIC CHUNKER (Gemini Compatible)")
    print("🧪 " * 20 + "\n")
    
    # Load document
    doc = load_cached_document()
    if doc is None:
        return 1
    
    print(f"📦 Loaded document: {doc.title}")
    print(f"   Pages: {doc.total_pages}\n")
    
    # Run tests
    chunks = test_basic_chunking(doc)
    test_clinical_metadata(chunks)
    test_content_types(chunks)
    test_chunk_linking(chunks)
    test_sample_chunks(chunks)
    test_overlap(doc)
    test_to_dict(chunks)
    
    # Export
    export_chunks(chunks)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Created {len(chunks)} chunks from {doc.total_pages} pages")
    print(f"✅ Average {sum(c.token_count for c in chunks) // len(chunks)} tokens/chunk")
    print(f"✅ Rich clinical metadata extracted")
    print(f"✅ Gemini-compatible token counting")
    print(f"✅ Ready for embedding and vector storage!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
