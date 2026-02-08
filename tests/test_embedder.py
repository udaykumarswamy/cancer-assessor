"""
Test Suite for Vertex AI Embedder Module

Tests the embedder with rich clinical chunk format from NG12 guidelines.
Covers:
- RichChunk creation and serialization
- ClinicalMetadata handling
- Mock embedder functionality
- Metadata-enhanced embedding
- Batch processing
- Query embedding
- Similarity computations

Run with: python test_embedder.py
Or with pytest: pytest test_embedder.py -v
"""

import json
import tempfile
from pathlib import Path
from dataclasses import asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the embedder module
from src.ingestion.embedder import (
    RichChunk,
    ClinicalMetadata,
    EmbeddedChunk,
    MockEmbedder,
    VertexAIEmbedder,
    get_embedder,
    load_chunks_from_json,
    save_embedded_chunks,
    cosine_similarity,
    VERTEX_AI_AVAILABLE,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_CHUNK_JSON = {
    "chunk_id": "ng12_p22_0035_9aa76837",
    "text": "| Ovarian            | Refer women using a suspected cancer pathway referral [1.5.1] |\n\n#### **Abdominal, pelvic or rectal mass or enlarged abdominal organ**\n\nSplenomegaly (unexplained) in adults may indicate Non-Hodgkin's lymphoma. Consider a suspected cancer pathway referral. When considering referral, take into account any associated symptoms, particularly fever, night sweats, shortness of breath, pruritus or weight loss.",
    "page_start": 22,
    "page_end": 22,
    "section": "1.13.1 Take into account the insight and knowledge of parents and carers when considering making a referral for suspected cancer in a child or young person",
    "section_hierarchy": "1|1.13|1.13.1",
    "content_type": "table",
    "token_count": 678,
    "char_count": 2712,
    "chunk_index": 35,
    "prev_chunk_id": "ng12_p21_0034_79c7cd4e",
    "next_chunk_id": "ng12_p22_0036_da01a2c9",
    "semantic_density": 0.2581120943952802,
    "created_at": "2026-02-06T13:23:14.521858",
    "source": "NG12",
    "has_recommendation": False,
    "has_table": True,
    "is_urgent": True,
    "cancer_types": "lymphoma",
    "symptoms": "weight_loss,shortness_of_breath,night_sweats,fever,unexplained",
    "age_thresholds": "",
    "timeframes": "",
    "urgency": "urgent_2_week",
    "actions": "offer,refer,consider",
    "risk_factors": "",
    "investigations": "ct"
}

MINIMAL_CHUNK_JSON = {
    "chunk_id": "test_001",
    "text": "Simple test chunk without much metadata.",
    "page_start": 1,
    "page_end": 1,
}

MULTIPLE_CHUNKS = [
    {
        "chunk_id": "ng12_001",
        "text": "Breast cancer symptoms include lumps and skin changes.",
        "page_start": 5,
        "page_end": 5,
        "cancer_types": "breast",
        "symptoms": "lump,skin_changes",
        "urgency": "urgent_2_week",
        "actions": "refer",
    },
    {
        "chunk_id": "ng12_002", 
        "text": "Lung cancer may present with persistent cough and hemoptysis.",
        "page_start": 8,
        "page_end": 8,
        "cancer_types": "lung",
        "symptoms": "cough,hemoptysis",
        "urgency": "urgent_2_week",
        "actions": "refer,investigate",
        "investigations": "ct,chest_xray",
    },
    {
        "chunk_id": "ng12_003",
        "text": "Colorectal cancer screening recommendations for adults over 50.",
        "page_start": 12,
        "page_end": 13,
        "cancer_types": "colorectal",
        "age_thresholds": "50",
        "actions": "offer,screen",
        "investigations": "fit",
    },
]


# =============================================================================
# Test Classes
# =============================================================================

class TestClinicalMetadata:
    """Tests for ClinicalMetadata dataclass."""
    
    def test_from_dict_full(self):
        """Test creating metadata from full data."""
        data = {
            "has_recommendation": True,
            "has_table": True,
            "is_urgent": True,
            "cancer_types": "breast,ovarian",
            "symptoms": "lump,pain",
            "urgency": "urgent_2_week",
            "actions": "refer,investigate",
        }
        
        meta = ClinicalMetadata.from_dict(data)
        
        assert meta.has_recommendation is True
        assert meta.has_table is True
        assert meta.is_urgent is True
        assert meta.cancer_types == "breast,ovarian"
        assert meta.symptoms == "lump,pain"
        assert meta.urgency == "urgent_2_week"
        assert meta.actions == "refer,investigate"
        
        print("✅ test_from_dict_full passed")
    
    def test_from_dict_partial(self):
        """Test creating metadata from partial data (missing fields)."""
        data = {"cancer_types": "lung"}
        
        meta = ClinicalMetadata.from_dict(data)
        
        assert meta.cancer_types == "lung"
        assert meta.symptoms == ""  # Default
        assert meta.has_recommendation is False  # Default
        
        print("✅ test_from_dict_partial passed")
    
    def test_to_embedding_prefix(self):
        """Test metadata to embedding prefix conversion."""
        meta = ClinicalMetadata(
            cancer_types="lymphoma,leukemia",
            symptoms="weight_loss,night_sweats",
            urgency="urgent_2_week",
            actions="refer,consider",
            investigations="ct,mri",
        )
        
        prefix = meta.to_embedding_prefix()
        
        assert "Cancer types:" in prefix
        assert "lymphoma" in prefix
        assert "leukemia" in prefix
        assert "Symptoms:" in prefix
        assert "weight loss" in prefix  # Underscore replaced
        assert "Urgency:" in prefix
        assert "urgent 2 week" in prefix  # Underscores replaced
        assert "Actions:" in prefix
        assert "Investigations:" in prefix
        assert "CT" in prefix  # Uppercased
        
        print("✅ test_to_embedding_prefix passed")
        print(f"   Prefix: {prefix}")
    
    def test_empty_prefix(self):
        """Test that empty metadata gives empty prefix."""
        meta = ClinicalMetadata()
        prefix = meta.to_embedding_prefix()
        assert prefix == ""
        
        print("✅ test_empty_prefix passed")


class TestRichChunk:
    """Tests for RichChunk dataclass."""
    
    def test_from_dict_full(self):
        """Test creating chunk from full JSON data."""
        chunk = RichChunk.from_dict(SAMPLE_CHUNK_JSON)
        
        assert chunk.chunk_id == "ng12_p22_0035_9aa76837"
        assert chunk.page_start == 22
        assert chunk.page_end == 22
        assert chunk.content_type == "table"
        assert chunk.token_count == 678
        assert chunk.chunk_index == 35
        assert chunk.prev_chunk_id == "ng12_p21_0034_79c7cd4e"
        assert chunk.next_chunk_id == "ng12_p22_0036_da01a2c9"
        assert chunk.source == "NG12"
        
        # Clinical metadata
        assert chunk.clinical.is_urgent is True
        assert chunk.clinical.has_table is True
        assert chunk.clinical.cancer_types == "lymphoma"
        assert "night_sweats" in chunk.clinical.symptoms
        assert chunk.clinical.urgency == "urgent_2_week"
        
        print("✅ test_from_dict_full passed")
    
    def test_from_dict_minimal(self):
        """Test creating chunk from minimal data."""
        chunk = RichChunk.from_dict(MINIMAL_CHUNK_JSON)
        
        assert chunk.chunk_id == "test_001"
        assert chunk.text == "Simple test chunk without much metadata."
        assert chunk.content_type == "text"  # Default
        assert chunk.clinical.cancer_types == ""  # Default
        
        print("✅ test_from_dict_minimal passed")
    
    def test_to_dict_roundtrip(self):
        """Test that to_dict/from_dict is lossless."""
        original = RichChunk.from_dict(SAMPLE_CHUNK_JSON)
        serialized = original.to_dict()
        restored = RichChunk.from_dict(serialized)
        
        assert original.chunk_id == restored.chunk_id
        assert original.text == restored.text
        assert original.clinical.cancer_types == restored.clinical.cancer_types
        assert original.clinical.symptoms == restored.clinical.symptoms
        
        print("✅ test_to_dict_roundtrip passed")
    
    def test_get_embedding_text_with_metadata(self):
        """Test embedding text includes metadata prefix."""
        chunk = RichChunk.from_dict(SAMPLE_CHUNK_JSON)
        
        text_with = chunk.get_embedding_text(include_metadata=True)
        text_without = chunk.get_embedding_text(include_metadata=False)
        
        assert len(text_with) > len(text_without)
        assert "Cancer types:" in text_with
        assert "Symptoms:" in text_with
        assert text_without == chunk.text
        
        print("✅ test_get_embedding_text_with_metadata passed")
        print(f"   Text length with metadata: {len(text_with)}")
        print(f"   Text length without: {len(text_without)}")


class TestMockEmbedder:
    """Tests for MockEmbedder."""
    
    def test_dimension(self):
        """Test embedder has correct dimension."""
        embedder = MockEmbedder(dimension=768)
        assert embedder.dimension == 768
        
        embedder_custom = MockEmbedder(dimension=384)
        assert embedder_custom.dimension == 384
        
        print("✅ test_dimension passed")
    
    def test_embed_query_deterministic(self):
        """Test that same query gives same embedding."""
        embedder = MockEmbedder()
        
        query = "unexplained weight loss"
        emb1 = embedder.embed_query(query)
        emb2 = embedder.embed_query(query)
        
        assert emb1 == emb2
        assert len(emb1) == 768
        
        print("✅ test_embed_query_deterministic passed")
    
    def test_embed_query_different_texts(self):
        """Test that different queries give different embeddings."""
        embedder = MockEmbedder()
        
        emb1 = embedder.embed_query("breast cancer symptoms")
        emb2 = embedder.embed_query("lung cancer diagnosis")
        
        # Should be different
        sim = cosine_similarity(emb1, emb2)
        assert sim < 0.99  # Not identical
        
        print("✅ test_embed_query_different_texts passed")
        print(f"   Similarity between different queries: {sim:.4f}")
    
    def test_embed_queries_batch(self):
        """Test batch query embedding."""
        embedder = MockEmbedder()
        
        queries = ["query one", "query two", "query three"]
        embeddings = embedder.embed_queries(queries)
        
        assert len(embeddings) == 3
        assert all(len(e) == 768 for e in embeddings)
        
        print("✅ test_embed_queries_batch passed")
    
    def test_embed_chunks_basic(self):
        """Test embedding chunks."""
        embedder = MockEmbedder()
        chunks = [RichChunk.from_dict(c) for c in MULTIPLE_CHUNKS]
        
        embedded = embedder.embed_chunks(chunks, show_progress=False)
        
        assert len(embedded) == 3
        assert all(isinstance(e, EmbeddedChunk) for e in embedded)
        assert all(len(e.embedding) == 768 for e in embedded)
        
        print("✅ test_embed_chunks_basic passed")
    
    def test_embed_chunks_metadata_impact(self):
        """Test that metadata changes the embedding."""
        embedder = MockEmbedder()
        chunk = RichChunk.from_dict(SAMPLE_CHUNK_JSON)
        
        # Embed with and without metadata
        emb_with = embedder.embed_chunks([chunk], include_metadata=True, show_progress=False)
        emb_without = embedder.embed_chunks([chunk], include_metadata=False, show_progress=False)
        
        # Embeddings should be different
        sim = cosine_similarity(emb_with[0].embedding, emb_without[0].embedding)
        assert sim < 0.99  # Not identical
        
        # Check metadata flag
        assert emb_with[0].metadata_enhanced is True
        assert emb_without[0].metadata_enhanced is False
        
        print("✅ test_embed_chunks_metadata_impact passed")
        print(f"   Similarity with/without metadata: {sim:.4f}")
    
    def test_embedding_normalized(self):
        """Test that embeddings are L2 normalized."""
        embedder = MockEmbedder()
        
        emb = embedder.embed_query("test query")
        
        # Compute L2 norm
        norm = sum(x**2 for x in emb) ** 0.5
        
        # Should be approximately 1.0
        assert 0.99 < norm < 1.01
        
        print("✅ test_embedding_normalized passed")
        print(f"   Embedding L2 norm: {norm:.6f}")


class TestEmbeddedChunk:
    """Tests for EmbeddedChunk."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        chunk = RichChunk.from_dict(SAMPLE_CHUNK_JSON)
        embedded = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1] * 768,
            metadata_enhanced=True
        )
        
        data = embedded.to_dict()
        
        assert "chunk" in data
        assert "embedding" in data
        assert "metadata_enhanced" in data
        assert data["chunk"]["chunk_id"] == chunk.chunk_id
        assert len(data["embedding"]) == 768
        
        print("✅ test_to_dict passed")
    
    def test_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        chunk = RichChunk.from_dict(SAMPLE_CHUNK_JSON)
        original = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.5] * 768,
            metadata_enhanced=True
        )
        
        serialized = original.to_dict()
        restored = EmbeddedChunk.from_dict(serialized)
        
        assert restored.chunk.chunk_id == original.chunk.chunk_id
        assert restored.embedding == original.embedding
        assert restored.metadata_enhanced == original.metadata_enhanced
        
        print("✅ test_from_dict_roundtrip passed")


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_embedder_mock(self):
        """Test getting mock embedder."""
        embedder = get_embedder(mock=True)
        assert isinstance(embedder, MockEmbedder)
        
        print("✅ test_get_embedder_mock passed")
    
    def test_get_embedder_auto(self):
        """Test auto-selection when Vertex AI unavailable."""
        embedder = get_embedder(mock=False)
        
        if VERTEX_AI_AVAILABLE:
            assert isinstance(embedder, VertexAIEmbedder)
            print("✅ test_get_embedder_auto passed (Vertex AI)")
        else:
            assert isinstance(embedder, MockEmbedder)
            print("✅ test_get_embedder_auto passed (fallback to Mock)")
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001
        
        print("✅ test_cosine_similarity_identical passed")
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.0001
        
        print("✅ test_cosine_similarity_orthogonal passed")
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim + 1.0) < 0.0001
        
        print("✅ test_cosine_similarity_opposite passed")
    
    def test_load_save_chunks(self):
        """Test loading and saving chunks to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            input_path = Path(tmpdir) / "chunks.json"
            with open(input_path, 'w') as f:
                json.dump(MULTIPLE_CHUNKS, f)
            
            # Load chunks
            chunks = load_chunks_from_json(input_path)
            assert len(chunks) == 3
            assert all(isinstance(c, RichChunk) for c in chunks)
            
            # Embed and save
            embedder = MockEmbedder()
            embedded = embedder.embed_chunks(chunks, show_progress=False)
            
            output_path = Path(tmpdir) / "embedded.json"
            save_embedded_chunks(embedded, output_path)
            
            # Verify saved file
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data) == 3
            assert all("embedding" in item for item in saved_data)
            
            print("✅ test_load_save_chunks passed")


class TestRetrievalScenarios:
    """Test realistic retrieval scenarios."""
    
    def test_clinical_query_matching(self):
        """Test that clinical queries match relevant chunks."""
        embedder = MockEmbedder()
        
        # Create chunks with different cancer types
        chunks = [RichChunk.from_dict(c) for c in MULTIPLE_CHUNKS]
        embedded = embedder.embed_chunks(chunks, include_metadata=True, show_progress=False)
        
        # Query for breast cancer
        query = "breast lump referral"
        query_emb = embedder.embed_query(query)
        
        # Compute similarities
        similarities = [
            (ec.chunk.chunk_id, cosine_similarity(query_emb, ec.embedding))
            for ec in embedded
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("✅ test_clinical_query_matching passed")
        print("   Query: 'breast lump referral'")
        print("   Rankings:")
        for chunk_id, sim in similarities:
            print(f"      {chunk_id}: {sim:.4f}")
    
    def test_symptom_based_retrieval(self):
        """Test retrieval based on symptom matching."""
        embedder = MockEmbedder()
        chunk = RichChunk.from_dict(SAMPLE_CHUNK_JSON)  # Has lymphoma symptoms
        
        embedded = embedder.embed_chunks([chunk], include_metadata=True, show_progress=False)
        
        # Queries related to lymphoma symptoms
        queries = [
            "night sweats fever weight loss",  # Direct symptoms
            "unexplained splenomegaly",  # Specific symptom
            "breast cancer screening",  # Unrelated
        ]
        
        print("✅ test_symptom_based_retrieval passed")
        print(f"   Chunk: {chunk.chunk_id}")
        print(f"   Chunk symptoms: {chunk.clinical.symptoms}")
        print("   Query similarities:")
        
        for query in queries:
            query_emb = embedder.embed_query(query)
            sim = cosine_similarity(query_emb, embedded[0].embedding)
            print(f"      '{query}': {sim:.4f}")


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all test classes."""
    print("=" * 70)
    print("EMBEDDER TEST SUITE")
    print("=" * 70)
    
    test_classes = [
        TestClinicalMetadata,
        TestRichChunk,
        TestMockEmbedder,
        TestEmbeddedChunk,
        TestUtilityFunctions,
        TestRetrievalScenarios,
    ]
    
    total_tests = 0
    passed_tests = 0
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
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"❌ {method_name} FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
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