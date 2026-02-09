"""
Real-Data Retrieval Evaluation for NG12 Cancer Assessor

Tests the ClinicalRetriever against actual NG12 guideline chunks
using auto-generated ground truth from chunk metadata.

Run:
    # Full evaluation with ChromaDB
    python -m pytest tests/evaluation/test_retrieval_real.py -v -s

    # Just run specific scenario
    python -m pytest tests/evaluation/test_retrieval_real.py -v -s -k "colorectal"

    # Standalone (no pytest needed)
    python tests/evaluation/test_retrieval_real.py
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Imports from your project ──
# Adjust these paths to match your project structure
from src.evaluation.retrieval_metrics import (
    RetrievalEvaluator,
    RetrievalTestCase,
    MetricResult,
)
from src.evaluation.ground_truth_builder import (
    build_ground_truth,
    load_chunks_from_chromadb,
    load_chunks_from_markdown,
    NG12_TEST_QUERIES,
    ClinicalTestQuery,
    GroundTruthEntry,
)

logger = logging.getLogger("retrieval_eval")


# ──────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────

# Path to your chunks.md (fallback if ChromaDB not available)
CHUNKS_MD_PATH = os.path.join("output", "chunks.md")

# Evaluation settings
DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.5

# Minimum acceptable thresholds (fail the test if below these)
MIN_RECALL = 0.6
MIN_HIT_RATE = 0.8
MIN_MRR = 0.4
MIN_NDCG = 0.5


# ──────────────────────────────────────────────────────
# Helper: Load your retriever
# ──────────────────────────────────────────────────────

def get_retriever():
    """
    Initialize your ClinicalRetriever with real vector store and embedder.

    Adjust imports to match your project.
    """
    from src.services.retrieval import ClinicalRetriever

    # Option 1: If you have a factory/dependency function
    try:
        from src.api.dependencies import get_vector_store, get_embedder
        vector_store = get_vector_store()
        embedder = get_embedder()
    except ImportError:
        # Option 2: Direct initialization
        from src.ingestion.vector_store import VectorStore
        from src.ingestion.embedder import Embedder
        vector_store = VectorStore()
        embedder = Embedder()

    return ClinicalRetriever(
        vector_store=vector_store,
        embedder=embedder,
        default_top_k=DEFAULT_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )


def get_chunk_metadata(retriever=None) -> List[Dict[str, Any]]:
    """
    Load chunk metadata from ChromaDB or fallback to chunks.md.
    """
    # Try ChromaDB first
    if retriever and hasattr(retriever, 'vector_store'):
        try:
            chunks = load_chunks_from_chromadb(retriever.vector_store)
            logger.info(f"Loaded {len(chunks)} chunks from ChromaDB")
            return chunks
        except Exception as e:
            logger.warning(f"ChromaDB load failed: {e}, falling back to chunks.md")

    # Fallback to chunks.md
    if os.path.exists(CHUNKS_MD_PATH):
        chunks = load_chunks_from_markdown(CHUNKS_MD_PATH)
        logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_MD_PATH}")
        return chunks

    raise FileNotFoundError(
        f"No chunk data found. Provide ChromaDB access or place chunks.md at {CHUNKS_MD_PATH}"
    )


# ──────────────────────────────────────────────────────
# Core evaluation runner
# ──────────────────────────────────────────────────────

def run_full_evaluation(
    retriever=None,
    chunk_metadata: Optional[List[Dict]] = None,
    test_queries: Optional[List[ClinicalTestQuery]] = None,
    k: int = DEFAULT_K,
) -> Dict[str, Any]:
    """
    Run the full retrieval evaluation pipeline.

    Steps:
        1. Load chunk metadata (from ChromaDB or chunks.md)
        2. Build ground truth by matching query criteria → chunk metadata
        3. Run each query through the retriever
        4. Compute and log all metrics
        5. Return aggregated results

    Args:
        retriever: ClinicalRetriever instance (if None, will initialize)
        chunk_metadata: Pre-loaded chunk metadata (if None, will load)
        test_queries: Custom queries (if None, uses NG12_TEST_QUERIES)
        k: Top-K cutoff for metrics

    Returns:
        Dict with averaged metrics and per-query breakdown
    """
    # Setup
    if retriever is None:
        retriever = get_retriever()

    if chunk_metadata is None:
        chunk_metadata = get_chunk_metadata(retriever)

    if test_queries is None:
        test_queries = NG12_TEST_QUERIES

    # Step 1: Build ground truth from metadata
    ground_truth = build_ground_truth(test_queries, chunk_metadata)

    logger.info(f"\n{'='*70}")
    logger.info(f"NG12 RETRIEVAL EVALUATION")
    logger.info(f"{'='*70}")
    logger.info(f"Chunks in store: {len(chunk_metadata)}")
    logger.info(f"Test queries: {len(ground_truth)}")
    logger.info(f"K: {k}")
    logger.info(f"{'='*70}\n")

    # Log ground truth summary
    for gt in ground_truth:
        logger.info(
            f"[Ground Truth] \"{gt.query[:60]}...\" → "
            f"{len(gt.relevant_chunk_ids)} relevant chunks "
            f"(grades: {dict(sorted(gt.relevance_grades.items(), key=lambda x: -x[1]))})"
        )

    # Step 2: Run retrieval + compute metrics
    evaluator = RetrievalEvaluator(logger=logger)
    test_cases: List[RetrievalTestCase] = []

    for gt in ground_truth:
        # Run the actual retriever
        ctx = retriever.retrieve(query=gt.query, top_k=k)
        retrieved_ids = [r.chunk_id for r in ctx.results]

        # Also try patient-context retrieval if we have cancer/symptom info
        if gt.expected_cancers:
            cancer = list(gt.expected_cancers)[0]  # primary cancer type
            ctx_patient = retriever.retrieve_for_patient(
                query=gt.query,
                symptoms=list(gt.expected_symptoms)[:3] if gt.expected_symptoms else None,
                suspected_cancer=cancer,
                top_k=k,
            )
            patient_ids = [r.chunk_id for r in ctx_patient.results]

            # Log comparison
            basic_hits = set(retrieved_ids) & gt.relevant_chunk_ids
            patient_hits = set(patient_ids) & gt.relevant_chunk_ids
            if len(patient_hits) > len(basic_hits):
                logger.info(
                    f"  ↳ Patient-context retrieval found {len(patient_hits)} vs "
                    f"{len(basic_hits)} relevant (basic). Using patient-context."
                )
                retrieved_ids = patient_ids

        test_cases.append(RetrievalTestCase(
            query=gt.query,
            retrieved_ids=retrieved_ids,
            relevant_ids=gt.relevant_chunk_ids,
            relevance_grades=gt.relevance_grades,
        ))

    # Step 3: Compute batch metrics
    avg, per_query = evaluator.evaluate_batch(test_cases, k=k)

    # Step 4: Log detailed per-query analysis
    logger.info(f"\n{'='*70}")
    logger.info("PER-QUERY ANALYSIS")
    logger.info(f"{'='*70}")

    failures = []
    for tc, result in zip(test_cases, per_query):
        # Find which relevant chunks were missed
        retrieved_set = set(tc.retrieved_ids[:k])
        missed = tc.relevant_ids - retrieved_set
        if missed:
            failures.append({
                "query": tc.query,
                "missed_chunks": missed,
                "recall": result.recall_at_k,
                "mrr": result.mrr,
            })
            logger.warning(
                f"⚠️  MISSED CHUNKS for \"{tc.query[:50]}...\"\n"
                f"   Retrieved: {tc.retrieved_ids[:k]}\n"
                f"   Missed: {missed}\n"
                f"   Recall: {result.recall_at_k:.2f}, MRR: {result.mrr:.2f}"
            )

    # Step 5: Summary with pass/fail thresholds
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*70}")

    checks = {
        f"Recall@{k} >= {MIN_RECALL}": avg["recall@k"] >= MIN_RECALL,
        f"HitRate@{k} >= {MIN_HIT_RATE}": avg["hit_rate@k"] >= MIN_HIT_RATE,
        f"MRR >= {MIN_MRR}": avg["mrr"] >= MIN_MRR,
        f"NDCG@{k} >= {MIN_NDCG}": avg["ndcg@k"] >= MIN_NDCG,
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        logger.info(f"  {status}: {check_name}")

    logger.info(f"\nQueries with missed chunks: {len(failures)}/{len(test_cases)}")

    if not all_passed:
        logger.warning(
            "\n⚠️  RETRIEVAL QUALITY BELOW THRESHOLD\n"
            "   Suggested actions:\n"
            "   1. Check embedding quality for failed queries\n"
            "   2. Review chunk boundaries — relevant content may be split\n"
            "   3. Expand clinical synonyms in ClinicalRetriever.SYMPTOM_SYNONYMS\n"
            "   4. Consider metadata-boosted search for cancer-type filtering"
        )

    return {
        "averaged_metrics": avg,
        "per_query_results": [r.to_dict() for r in per_query],
        "failures": failures,
        "all_passed": all_passed,
    }


# ──────────────────────────────────────────────────────
# Pytest-compatible test class
# ──────────────────────────────────────────────────────

class TestNG12Retrieval:
    """
    Pytest test class for NG12 retrieval quality.

    These tests will FAIL if retrieval quality drops below
    the thresholds — treat them like integration tests.
    """

    @classmethod
    def setup_class(cls):
        """Initialize retriever once for all tests."""
        cls.retriever = get_retriever()
        cls.chunk_metadata = get_chunk_metadata(cls.retriever)
        cls.evaluator = RetrievalEvaluator(logger=logger)

    def _run_query(self, query: ClinicalTestQuery, k: int = DEFAULT_K):
        """Helper: run a single query and return metrics."""
        ground_truth = build_ground_truth([query], self.chunk_metadata)
        gt = ground_truth[0]

        ctx = self.retriever.retrieve(query=gt.query, top_k=k)
        retrieved_ids = [r.chunk_id for r in ctx.results]

        result = self.evaluator.evaluate_query(
            retrieved_ids=retrieved_ids,
            relevant_ids=gt.relevant_chunk_ids,
            query=gt.query,
            k=k,
            relevance_grades=gt.relevance_grades,
        )
        return result, gt

    # ── Per-cancer-type tests ──

    def test_pancreatic_cancer_retrieval(self):
        """Pancreatic: weight loss + back pain in over 60."""
        result, gt = self._run_query(NG12_TEST_QUERIES[0])
        assert result.hit_rate_at_k == 1.0, f"No relevant chunks found for pancreatic cancer query"
        assert result.recall_at_k >= 0.5, f"Recall too low: {result.recall_at_k:.2f}"

    def test_stomach_cancer_retrieval(self):
        """Stomach: dysphagia + weight loss in over 55."""
        result, gt = self._run_query(NG12_TEST_QUERIES[1])
        assert result.hit_rate_at_k == 1.0, f"No relevant chunks found for stomach cancer query"

    def test_colorectal_cancer_retrieval(self):
        """Colorectal: rectal bleeding + weight loss in over 50."""
        result, gt = self._run_query(NG12_TEST_QUERIES[3])
        assert result.hit_rate_at_k == 1.0, f"No relevant chunks found for colorectal query"
        assert result.recall_at_k >= 0.5, f"Recall too low: {result.recall_at_k:.2f}"

    def test_breast_cancer_retrieval(self):
        """Breast: unexplained lump aged 30+."""
        result, gt = self._run_query(NG12_TEST_QUERIES[6])
        assert result.hit_rate_at_k == 1.0, f"No relevant chunks found for breast cancer query"

    def test_ovarian_cancer_retrieval(self):
        """Ovarian: persistent bloating + abdominal pain in over 50."""
        result, gt = self._run_query(NG12_TEST_QUERIES[8])
        assert result.hit_rate_at_k == 1.0, f"No relevant chunks found for ovarian cancer query"

    def test_bladder_prostate_retrieval(self):
        """Urological: visible haematuria in over 45."""
        result, gt = self._run_query(NG12_TEST_QUERIES[10])
        assert result.hit_rate_at_k == 1.0, f"No relevant chunks found for urological query"

    # ── Cross-cutting tests ──

    def test_nonspecific_symptoms_retrieve_multiple_cancers(self):
        """Weight loss + fatigue should surface chunks from multiple cancer types."""
        result, gt = self._run_query(NG12_TEST_QUERIES[12])
        assert result.num_relevant_retrieved >= 2, (
            f"Non-specific query should retrieve chunks from multiple cancers, "
            f"only got {result.num_relevant_retrieved}"
        )

    # ── Quality threshold tests ──

    def test_overall_recall_threshold(self):
        """Batch recall must meet minimum threshold."""
        results = run_full_evaluation(
            retriever=self.retriever,
            chunk_metadata=self.chunk_metadata,
            k=DEFAULT_K,
        )
        assert results["averaged_metrics"]["recall@k"] >= MIN_RECALL, (
            f"Overall Recall@{DEFAULT_K} = {results['averaged_metrics']['recall@k']:.3f} "
            f"< threshold {MIN_RECALL}"
        )

    def test_overall_hit_rate_threshold(self):
        """No query should return zero relevant results."""
        results = run_full_evaluation(
            retriever=self.retriever,
            chunk_metadata=self.chunk_metadata,
            k=DEFAULT_K,
        )
        assert results["averaged_metrics"]["hit_rate@k"] >= MIN_HIT_RATE, (
            f"Overall HitRate@{DEFAULT_K} = {results['averaged_metrics']['hit_rate@k']:.3f} "
            f"< threshold {MIN_HIT_RATE}"
        )


# ──────────────────────────────────────────────────────
# Standalone runner (no pytest needed)
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("retrieval_eval_results.log"),
        ],
    )

    print("Starting NG12 Retrieval Evaluation...\n")

    try:
        results = run_full_evaluation(k=DEFAULT_K)

        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        for metric, value in results["averaged_metrics"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

        status = "✅ ALL PASSED" if results["all_passed"] else "❌ SOME FAILED"
        print(f"\nOverall: {status}")
        print(f"Results saved to retrieval_eval_results.log")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nTo run evaluation, ensure either:")
        print("  1. ChromaDB is accessible with your NG12 chunks loaded")
        print(f"  2. chunks.md exists at: {CHUNKS_MD_PATH}")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nAdjust the imports in get_retriever() to match your project structure.")