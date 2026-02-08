"""
Test Suite for Vector Store Module

Tests the ChromaDB vector store with rich clinical metadata.
Covers:
- RetrievedChunk creation and citations
- Adding chunks to store
- Semantic search
- Clinical metadata filtering
- Page/section retrieval
- Statistics and management

Run with: python test_vector_store.py
Or with pytest: pytest test_vector_store.py -v
"""

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules
from src.ingestion.embedder import (
    RichChunk,
    EmbeddedChunk,
    ClinicalMetadata,
    MockEmbedder,
    get_embedder,
    cosine_similarity,
)
from src.ingestion.vector_store import (
    RetrievedChunk,
    VectorStore,
    MockVectorStore,
    get_vector_store,
    CHROMADB_AVAILABLE,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_CHUNKS = [
    {
        "chunk_id": "ng12_lymphoma_001",
        "text": "Consider a suspected cancer pathway referral for adults with unexplained splenomegaly. Take into account associated symptoms: fever, night sweats, shortness of breath, pruritus, or weight loss.",
        "page_start": 22,
        "page_end": 22,
        "section": "1.10 Haematological cancers",
        "section_hierarchy": "1|1.10|1.10.6",
        "content_type": "text",
        "token_count": 45,
        "char_count": 200,
        "chunk_index": 35,
        "source": "NG12",
        "has_recommendation": True,
        "has_table": False,
        "is_urgent": True,
        "cancer_types": "lymphoma,leukemia",
        "symptoms": "splenomegaly,fever,night_sweats,weight_loss",
        "urgency": "urgent_2_week",
        "actions": "refer,consider",
        "investigations": "",
    },
    {
        "chunk_id": "ng12_lung_001",
        "text": "Consider an urgent chest X-ray to be performed within 2 weeks for patients aged 40 and over with unexplained persistent cough.",
        "page_start": 15,
        "page_end": 15,
        "section": "1.4 Lung and pleural cancers",
        "section_hierarchy": "1|1.4|1.4.1",
        "content_type": "text",
        "token_count": 30,
        "char_count": 120,
        "chunk_index": 20,
        "source": "NG12",
        "has_recommendation": True,
        "has_table": False,
        "is_urgent": False,
        "cancer_types": "lung",
        "symptoms": "cough,persistent_cough",
        "urgency": "routine",
        "actions": "consider",
        "investigations": "chest_xray",
        "age_thresholds": "40",
    },
    {
        "chunk_id": "ng12_breast_001",
        "text": "Refer people using a suspected cancer pathway referral for breast cancer if they are aged 30 and over and have an unexplained breast lump.",
        "page_start": 8,
        "page_end": 8,
        "section": "1.2 Breast cancer",
        "section_hierarchy": "1|1.2|1.2.1",
        "content_type": "text",
        "token_count": 35,
        "char_count": 140,
        "chunk_index": 10,
        "source": "NG12",
        "has_recommendation": True,
        "has_table": False,
        "is_urgent": True,
        "cancer_types": "breast",
        "symptoms": "lump,breast_lump",
        "urgency": "urgent_2_week",
        "actions": "refer",
        "age_thresholds": "30",
    },
    {
        "chunk_id": "ng12_colorectal_001",
        "text": "Offer faecal immunochemical testing (FIT) to assess for colorectal cancer in adults without rectal bleeding who have unexplained symptoms.",
        "page_start": 12,
        "page_end": 13,
        "section": "1.3 Colorectal cancers",
        "section_hierarchy": "1|1.3|1.3.1",
        "content_type": "text",
        "token_count": 28,
        "char_count": 130,
        "chunk_index": 15,
        "source": "NG12",
        "has_recommendation": True,
        "has_table": True,
        "is_urgent": False,
        "cancer_types": "colorectal",
        "symptoms": "abdominal_pain,bowel_changes",
        "urgency": "routine",
        "actions": "offer",
        "investigations": "fit",
    },
    {
        "chunk_id": "ng12_ovarian_001",
        "text": "Consider a suspected cancer pathway referral for ovarian cancer in women with ascites and/or pelvic or abdominal mass.",
        "page_start": 18,
        "page_end": 18,
        "section": "1.5 Gynaecological cancers",
        "section_hierarchy": "1|1.5|1.5.1",
        "content_type": "text",
        "token_count": 25,
        "char_count": 115,
        "chunk_index": 25,
        "source": "NG12",
        "has_recommendation": True,
        "has_table": False,
        "is_urgent": True,
        "cancer_types": "ovarian",
        "symptoms": "ascites,abdominal_mass,pelvic_mass",
        "urgency": "urgent_2_week",
        "actions": "refer,consider",
        "investigations": "ca125,ultrasound",
    },
]


def create_test_chunks() -> list[RichChunk]:
    """Create RichChunk objects from test data."""
    return [RichChunk.from_dict(c) for c in SAMPLE_CHUNKS]


def create_embedded_chunks(embedder: MockEmbedder = None) -> list[EmbeddedChunk]:
    """Create embedded chunks for testing."""
    if embedder is None:
        embedder = MockEmbedder()
    chunks = create_test_chunks()
    return embedder.embed_chunks(chunks, include_metadata=True, show_progress=False)


# =============================================================================
# Test Classes
# =============================================================================

class TestRetrievedChunk:
    """Tests for RetrievedChunk."""
    
    def test_from_chroma_result(self):
        """Test creating from ChromaDB result format."""
        metadata = {
            "page_start": 22,
            "page_end": 22,
            "section": "1.10 Haematological cancers",
            "section_hierarchy": "1|1.10|1.10.6",
            "content_type": "text",
            "is_urgent": True,
            "cancer_types": "lymphoma",
            "symptoms": "splenomegaly,fever",
            "urgency": "urgent_2_week",
            "has_recommendation": True,
            "has_table": False,
        }
        
        retrieved = RetrievedChunk.from_chroma_result(
            chunk_id="test_001",
            text="Sample text",
            score=0.85,
            metadata=metadata
        )
        
        assert retrieved.chunk_id == "test_001"
        assert retrieved.score == 0.85
        assert retrieved.page_start == 22
        assert retrieved.section_hierarchy == "1|1.10|1.10.6"
        assert retrieved.clinical.is_urgent is True
        assert retrieved.clinical.cancer_types == "lymphoma"
        
        print("✅ test_from_chroma_result passed")
    
    def test_get_citation_single_page(self):
        """Test citation generation for single page."""
        retrieved = RetrievedChunk(
            chunk_id="test",
            text="text",
            score=0.9,
            page_start=15,
            page_end=15,
            section="1.4 Lung cancer",
            section_hierarchy="1|1.4",
        )
        
        citation = retrieved.get_citation()
        assert "p.15" in citation
        assert "NG12" in citation
        assert "1.4" in citation  # Should extract correct section number
        
        print(f"✅ test_get_citation_single_page passed: {citation}")
    
    def test_get_citation_multi_page(self):
        """Test citation generation for multi-page chunk."""
        retrieved = RetrievedChunk(
            chunk_id="test",
            text="text",
            score=0.9,
            page_start=12,
            page_end=14,
            section="",
            section_hierarchy="1|1.3|1.3.1",  # Should extract "1.3.1"
        )
        
        citation = retrieved.get_citation()
        assert "pp.12-14" in citation
        assert "1.3.1" in citation  # Most specific section
        
        print(f"✅ test_get_citation_multi_page passed: {citation}")
    
    def test_get_urgency_display(self):
        """Test urgency display formatting."""
        clinical = ClinicalMetadata(urgency="urgent_2_week")
        retrieved = RetrievedChunk(
            chunk_id="test",
            text="text",
            score=0.9,
            page_start=1,
            page_end=1,
            section="",
            clinical=clinical,
        )
        
        display = retrieved.get_urgency_display()
        assert "🔴" in display
        assert "2-week" in display
        
        print(f"✅ test_get_urgency_display passed: {display}")
    
    def test_to_dict(self):
        """Test serialization to dict."""
        clinical = ClinicalMetadata(
            cancer_types="breast",
            urgency="urgent_2_week",
            is_urgent=True,
        )
        retrieved = RetrievedChunk(
            chunk_id="test",
            text="Sample text",
            score=0.9,
            page_start=8,
            page_end=8,
            section="1.2 Breast",
            section_hierarchy="1|1.2",
            clinical=clinical,
        )
        
        data = retrieved.to_dict()
        
        assert data["chunk_id"] == "test"
        assert data["score"] == 0.9
        assert data["cancer_types"] == "breast"
        assert data["is_urgent"] is True
        assert "citation" in data
        assert "urgency_display" in data
        
        print("✅ test_to_dict passed")


class TestMockVectorStore:
    """Tests for MockVectorStore (in-memory)."""
    
    def test_add_chunks(self):
        """Test adding chunks to mock store."""
        store = MockVectorStore()
        embedded = create_embedded_chunks()
        
        count = store.add_chunks(embedded, show_progress=False)
        
        assert count == len(SAMPLE_CHUNKS)
        assert store.get_stats()["total_chunks"] == len(SAMPLE_CHUNKS)
        
        print("✅ test_add_chunks passed")
    
    def test_search_basic(self):
        """Test basic semantic search."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        # Query for lymphoma
        query = "splenomegaly lymphoma referral"
        query_emb = embedder.embed_query(query)
        
        results = store.search(query_emb, top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, RetrievedChunk) for r in results)
        # Scores should be sorted descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        
        print("✅ test_search_basic passed")
        print(f"   Top result: {results[0].chunk_id} ({results[0].score:.4f})")
    
    def test_search_with_min_score(self):
        """Test search with minimum score filter."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        query_emb = embedder.embed_query("test query")
        
        # High threshold should return fewer results
        results_all = store.search(query_emb, top_k=10)
        results_filtered = store.search(query_emb, top_k=10, min_score=0.5)
        
        assert len(results_filtered) <= len(results_all)
        assert all(r.score >= 0.5 for r in results_filtered)
        
        print("✅ test_search_with_min_score passed")
    
    def test_search_urgent_only(self):
        """Test filtering for urgent chunks."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        query_emb = embedder.embed_query("cancer referral")
        results = store.search_urgent_only(query_emb, top_k=10)
        
        # All results should be urgent
        assert all(r.clinical.is_urgent for r in results)
        
        # Count expected urgent chunks
        expected_urgent = sum(1 for c in SAMPLE_CHUNKS if c.get("is_urgent"))
        assert len(results) == expected_urgent
        
        print(f"✅ test_search_urgent_only passed ({len(results)} urgent chunks)")
    
    def test_search_by_cancer_type(self):
        """Test filtering by cancer type."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        query_emb = embedder.embed_query("symptoms")
        
        # Search for lung cancer
        results = store.search_by_cancer_type(query_emb, "lung", top_k=10)
        
        assert len(results) >= 1
        assert all("lung" in r.clinical.cancer_types for r in results)
        
        print(f"✅ test_search_by_cancer_type passed ({len(results)} lung chunks)")
    
    def test_search_recommendations(self):
        """Test filtering for recommendation chunks."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        query_emb = embedder.embed_query("referral pathway")
        results = store.search_recommendations(query_emb, top_k=10)
        
        # All results should have recommendations
        assert all(r.clinical.has_recommendation for r in results)
        
        print(f"✅ test_search_recommendations passed ({len(results)} recommendations)")
    
    def test_clear(self):
        """Test clearing the store."""
        store = MockVectorStore()
        embedded = create_embedded_chunks()
        store.add_chunks(embedded, show_progress=False)
        
        assert store.get_stats()["total_chunks"] > 0
        
        store.clear()
        
        assert store.get_stats()["total_chunks"] == 0
        
        print("✅ test_clear passed")


class TestVectorStoreWithChroma:
    """Tests for actual ChromaDB vector store (if available)."""
    
    def test_chromadb_available(self):
        """Check if ChromaDB is available."""
        print(f"   ChromaDB available: {CHROMADB_AVAILABLE}")
        print("✅ test_chromadb_available passed")
    
    def test_add_and_search(self):
        """Test adding chunks and searching with ChromaDB."""
        if not CHROMADB_AVAILABLE:
            print("⏭️ test_add_and_search skipped (ChromaDB not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = MockEmbedder()
            store = VectorStore(
                persist_dir=Path(tmpdir),
                collection_name="test_collection"
            )
            
            # Add chunks
            embedded = create_embedded_chunks(embedder)
            store.add_chunks(embedded, show_progress=False)
            
            # Search
            query_emb = embedder.embed_query("lymphoma splenomegaly")
            results = store.search(query_emb, top_k=3)
            
            assert len(results) == 3
            assert all(isinstance(r, RetrievedChunk) for r in results)
            
            print("✅ test_add_and_search passed")
    
    def test_clinical_filters(self):
        """Test clinical metadata filters with ChromaDB."""
        if not CHROMADB_AVAILABLE:
            print("⏭️ test_clinical_filters skipped (ChromaDB not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = MockEmbedder()
            store = VectorStore(
                persist_dir=Path(tmpdir),
                collection_name="test_clinical"
            )
            
            embedded = create_embedded_chunks(embedder)
            store.add_chunks(embedded, show_progress=False)
            
            query_emb = embedder.embed_query("cancer referral")
            
            # Test urgent filter
            urgent_results = store.search_urgent_only(query_emb, top_k=10)
            assert all(r.clinical.is_urgent for r in urgent_results)
            
            # Test cancer type filter
            breast_results = store.search_by_cancer_type(query_emb, "breast", top_k=10)
            assert all("breast" in r.clinical.cancer_types for r in breast_results)
            
            # Test investigation filter
            fit_results = store.search_by_investigation(query_emb, "fit", top_k=10)
            assert all("fit" in r.clinical.investigations for r in fit_results)
            
            print("✅ test_clinical_filters passed")
    
    def test_combined_filters(self):
        """Test combining multiple clinical filters."""
        if not CHROMADB_AVAILABLE:
            print("⏭️ test_combined_filters skipped (ChromaDB not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = MockEmbedder()
            store = VectorStore(
                persist_dir=Path(tmpdir),
                collection_name="test_combined"
            )
            
            embedded = create_embedded_chunks(embedder)
            store.add_chunks(embedded, show_progress=False)
            
            query_emb = embedder.embed_query("referral")
            
            # Search with multiple filters
            results = store.search_with_clinical_filter(
                query_emb,
                top_k=10,
                is_urgent=True,
                has_recommendation=True,
            )
            
            for r in results:
                assert r.clinical.is_urgent
                assert r.clinical.has_recommendation
            
            print(f"✅ test_combined_filters passed ({len(results)} results)")
    
    def test_get_by_page(self):
        """Test retrieving chunks by page number."""
        if not CHROMADB_AVAILABLE:
            print("⏭️ test_get_by_page skipped (ChromaDB not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = MockEmbedder()
            store = VectorStore(
                persist_dir=Path(tmpdir),
                collection_name="test_page"
            )
            
            embedded = create_embedded_chunks(embedder)
            store.add_chunks(embedded, show_progress=False)
            
            # Get page 22 (lymphoma chunk)
            results = store.get_by_page(22)
            
            assert len(results) >= 1
            assert any(r.page_start <= 22 <= r.page_end for r in results)
            
            print(f"✅ test_get_by_page passed ({len(results)} chunks on page 22)")
    
    def test_get_chunk_by_id(self):
        """Test retrieving specific chunk by ID."""
        if not CHROMADB_AVAILABLE:
            print("⏭️ test_get_chunk_by_id skipped (ChromaDB not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = MockEmbedder()
            store = VectorStore(
                persist_dir=Path(tmpdir),
                collection_name="test_byid"
            )
            
            embedded = create_embedded_chunks(embedder)
            store.add_chunks(embedded, show_progress=False)
            
            # Get specific chunk
            result = store.get_chunk_by_id("ng12_breast_001")
            
            assert result is not None
            assert result.chunk_id == "ng12_breast_001"
            assert "breast" in result.clinical.cancer_types
            
            # Try non-existent ID
            missing = store.get_chunk_by_id("nonexistent_id")
            assert missing is None
            
            print("✅ test_get_chunk_by_id passed")
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        if not CHROMADB_AVAILABLE:
            print("⏭️ test_get_stats skipped (ChromaDB not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = MockEmbedder()
            store = VectorStore(
                persist_dir=Path(tmpdir),
                collection_name="test_stats"
            )
            
            embedded = create_embedded_chunks(embedder)
            store.add_chunks(embedded, show_progress=False)
            
            stats = store.get_stats()
            
            assert stats["total_chunks"] == len(SAMPLE_CHUNKS)
            assert stats["urgent_chunks"] > 0
            assert stats["recommendation_chunks"] > 0
            assert len(stats["cancer_types"]) > 0
            
            print("✅ test_get_stats passed")
            print(f"   Stats: {stats}")
    
    def test_clear_and_reset(self):
        """Test clearing and resetting collection."""
        if not CHROMADB_AVAILABLE:
            print("⏭️ test_clear_and_reset skipped (ChromaDB not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = MockEmbedder()
            store = VectorStore(
                persist_dir=Path(tmpdir),
                collection_name="test_reset"
            )
            
            embedded = create_embedded_chunks(embedder)
            store.add_chunks(embedded, show_progress=False)
            
            assert store.get_stats()["total_chunks"] > 0
            
            # Clear
            store.clear()
            assert store.get_stats()["total_chunks"] == 0
            
            # Add again
            store.add_chunks(embedded, show_progress=False)
            assert store.get_stats()["total_chunks"] > 0
            
            # Reset
            store.reset()
            # After reset, collection is recreated empty
            assert store.get_stats()["total_chunks"] == 0
            
            print("✅ test_clear_and_reset passed")


class TestFactoryFunction:
    """Tests for get_vector_store factory."""
    
    def test_get_mock_store(self):
        """Test getting mock store."""
        store = get_vector_store(mock=True)
        assert isinstance(store, MockVectorStore)
        
        print("✅ test_get_mock_store passed")
    
    def test_get_real_store(self):
        """Test getting real store (if ChromaDB available)."""
        if not CHROMADB_AVAILABLE:
            # Should fall back to mock
            store = get_vector_store(mock=False)
            assert isinstance(store, MockVectorStore)
            print("✅ test_get_real_store passed (fallback to mock)")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                store = get_vector_store(
                    persist_dir=Path(tmpdir),
                    collection_name="test_factory"
                )
                assert isinstance(store, VectorStore)
                print("✅ test_get_real_store passed (ChromaDB)")


class TestRetrievalScenarios:
    """Test realistic clinical retrieval scenarios."""
    
    def test_symptom_based_retrieval(self):
        """Test retrieving based on patient symptoms."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        # Patient presents with night sweats and weight loss
        query = "patient with night sweats and unexplained weight loss"
        query_emb = embedder.embed_query(query)
        
        results = store.search(query_emb, top_k=3)
        
        print("✅ test_symptom_based_retrieval passed")
        print(f"   Query: '{query}'")
        for r in results:
            print(f"   - [{r.score:.4f}] {r.chunk_id}: {r.clinical.symptoms[:30]}...")
    
    def test_urgent_pathway_retrieval(self):
        """Test finding urgent referral pathways."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        query = "urgent suspected cancer pathway referral"
        query_emb = embedder.embed_query(query)
        
        # Get only urgent chunks
        results = store.search_urgent_only(query_emb, top_k=5)
        
        print("✅ test_urgent_pathway_retrieval passed")
        print(f"   Found {len(results)} urgent pathways:")
        for r in results:
            print(f"   - {r.get_citation()}: {r.get_urgency_display()}")
    
    def test_investigation_lookup(self):
        """Test finding chunks mentioning specific investigations."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        # Find chunks mentioning FIT test
        query = "faecal immunochemical test colorectal"
        query_emb = embedder.embed_query(query)
        
        results = store.search(query_emb, top_k=3, where={"investigations": {"$contains": "fit"}})
        
        print("✅ test_investigation_lookup passed")
        print(f"   Found {len(results)} chunks mentioning FIT:")
        for r in results:
            print(f"   - {r.chunk_id}: {r.clinical.investigations}")
    
    def test_citation_generation(self):
        """Test generating citations for results."""
        embedder = MockEmbedder()
        store = MockVectorStore()
        embedded = create_embedded_chunks(embedder)
        store.add_chunks(embedded, show_progress=False)
        
        query_emb = embedder.embed_query("breast cancer referral")
        results = store.search(query_emb, top_k=3)
        
        print("✅ test_citation_generation passed")
        print("   Citations:")
        for r in results:
            citation = r.get_citation()
            print(f"   - {citation}")
            assert "[NG12" in citation
            assert "p." in citation or "pp." in citation


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all test classes."""
    print("=" * 70)
    print("VECTOR STORE TEST SUITE")
    print("=" * 70)
    
    test_classes = [
        TestRetrievedChunk,
        TestMockVectorStore,
        TestVectorStoreWithChroma,
        TestFactoryFunction,
        TestRetrievalScenarios,
    ]
    
    total_tests = 0
    passed_tests = 0
    skipped_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'─' * 50}")
        print(f"Running {test_class.__name__}")
        print('─' * 50)
        
        instance = test_class()
        
        # Find all test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                error_msg = str(e)
                if "skipped" in error_msg.lower():
                    skipped_tests += 1
                else:
                    failed_tests.append((test_class.__name__, method_name, error_msg))
                    print(f"❌ {method_name} FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Skipped: {skipped_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for class_name, method, error in failed_tests:
            print(f"  - {class_name}.{method}: {error}")
    else:
        print("\n🎉 All tests passed!")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)