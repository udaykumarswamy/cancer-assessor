"""
Retrieval Evaluation Metrics for NG12 Clinical Retriever

Metrics implemented:
────────────────────
1. Recall@K       – Of all relevant docs, how many did we retrieve in top K?
2. Precision@K    – Of the K docs retrieved, how many are relevant?
3. MRR            – How high is the *first* relevant result ranked?
4. NDCG@K         – How good is the *ordering* of results (graded relevance)?
5. MAP            – Mean of precision values at each relevant doc's rank.
6. Hit Rate@K     – Binary: did *any* relevant doc appear in top K?

Usage:
    evaluator = RetrievalEvaluator(logger)

    # Single query evaluation
    scores = evaluator.evaluate_query(
        retrieved_ids=["c1", "c2", "c3", "c4", "c5"],
        relevant_ids={"c2", "c4", "c7"},
        k=5,
    )

    # Batch evaluation over a test set
    avg_scores = evaluator.evaluate_batch(test_cases, k=5)
"""

import math
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class RetrievalTestCase:
    """
    One evaluation example.

    Attributes:
        query:           The clinical query string.
        retrieved_ids:   Ordered list of chunk IDs returned by the retriever.
        relevant_ids:    Ground-truth set of chunk IDs that ARE relevant.
        relevance_grades: Optional graded relevance (chunk_id → int score)
                          used by NDCG. If absent, binary relevance is assumed.
    """
    query: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]
    relevance_grades: Optional[Dict[str, int]] = None


@dataclass
class MetricResult:
    """Holds all metric values for a single query."""
    query: str
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    mean_avg_precision: float = 0.0
    hit_rate_at_k: float = 0.0
    k: int = 5
    num_retrieved: int = 0
    num_relevant: int = 0
    num_relevant_retrieved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "recall@k": round(self.recall_at_k, 4),
            "precision@k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@k": round(self.ndcg_at_k, 4),
            "map": round(self.mean_avg_precision, 4),
            "hit_rate@k": round(self.hit_rate_at_k, 4),
            "k": self.k,
            "num_retrieved": self.num_retrieved,
            "num_relevant": self.num_relevant,
            "num_relevant_retrieved": self.num_relevant_retrieved,
        }


# ──────────────────────────────────────────────
# Core metric functions (stateless, testable)
# ──────────────────────────────────────────────

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K — what fraction of relevant docs appear in the top K results?

    Formula:  |relevant ∩ retrieved[:k]| / |relevant|

    Why it matters for clinical retrieval:
        A missed relevant guideline chunk could mean a missed cancer referral.
        High recall ensures we surface all pertinent clinical evidence.

    Returns 0.0 if there are no relevant documents.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K — what fraction of the top K results are relevant?

    Formula:  |relevant ∩ retrieved[:k]| / k

    Why it matters:
        Clinicians reviewing results need high signal-to-noise. Low precision
        means wading through irrelevant chunks, eroding trust in the system.

    Returns 0.0 if k is 0.
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    MRR — reciprocal of the rank of the *first* relevant result.

    Formula:  1 / rank_of_first_relevant_doc

    Why it matters:
        In clinical workflows, the first relevant result often drives the
        initial assessment. MRR measures how quickly the system surfaces
        something useful. An MRR of 1.0 means the top result is always relevant.

    Returns 0.0 if no relevant document is retrieved.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
    relevance_grades: Optional[Dict[str, int]] = None,
) -> float:
    """
    NDCG@K — Normalised Discounted Cumulative Gain.

    Measures ranking quality accounting for *position* and *graded* relevance.

    Steps:
        1. DCG@K  = Σ (relevance_i / log2(i + 1))  for i in 1..K
        2. IDCG@K = DCG of the ideal (perfect) ranking
        3. NDCG   = DCG / IDCG

    Graded relevance example for clinical retrieval:
        3 = exact guideline match (e.g. "2-week-wait referral for lung")
        2 = strongly related (e.g. general lung cancer section)
        1 = tangentially relevant (e.g. mentions lung in passing)
        0 = irrelevant

    If `relevance_grades` is None, binary relevance is used (1 if relevant, 0 otherwise).

    Returns 0.0 if IDCG is 0 (no relevant docs).
    """
    def _dcg(scores: List[float]) -> float:
        return sum(
            score / math.log2(rank + 1)
            for rank, score in enumerate(scores, start=1)
        )

    # Build gain vector for retrieved docs
    gains = []
    for doc_id in retrieved[:k]:
        if relevance_grades and doc_id in relevance_grades:
            gains.append(float(relevance_grades[doc_id]))
        elif doc_id in relevant:
            gains.append(1.0)
        else:
            gains.append(0.0)

    dcg = _dcg(gains)

    # Ideal gains: all relevant docs sorted by grade descending
    if relevance_grades:
        ideal_gains = sorted(
            [float(relevance_grades.get(rid, 1.0)) for rid in relevant],
            reverse=True,
        )[:k]
    else:
        ideal_gains = [1.0] * min(len(relevant), k)

    idcg = _dcg(ideal_gains)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Average Precision (AP) — mean of precision values computed at each
    position where a relevant document is found.

    Formula:  (1/|relevant|) * Σ  Precision@i * rel(i)   for i in 1..K
              where rel(i) = 1 if retrieved[i] is relevant

    Why it matters:
        Rewards systems that rank relevant results higher. Unlike MRR which
        only cares about the *first* hit, AP considers *all* relevant results
        and their positions — important when multiple guideline sections are
        relevant (e.g. both referral criteria AND red flags for a symptom).

    Returns 0.0 if there are no relevant documents.
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / rank

    return sum_precisions / len(relevant)


def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Hit Rate@K — binary: is there *at least one* relevant result in the top K?

    Returns 1.0 if yes, 0.0 if no.

    Why it matters:
        The most basic check — did the retriever find *anything* useful?
        A hit rate < 1.0 across your eval set means some queries return
        zero relevant results, which is a critical failure in clinical use.
    """
    top_k = retrieved[:k]
    return 1.0 if any(doc_id in relevant for doc_id in top_k) else 0.0


# ──────────────────────────────────────────────
# Evaluator class (orchestrates + logs)
# ──────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Orchestrates metric computation and logging for retrieval evaluation.

    Usage:
        evaluator = RetrievalEvaluator(logger)

        # Evaluate a single query
        result = evaluator.evaluate_query(
            retrieved_ids=["c1", "c2", "c3"],
            relevant_ids={"c1", "c5"},
            query="persistent cough with haemoptysis",
            k=5,
        )

        # Evaluate a batch of test cases
        avg_scores, per_query = evaluator.evaluate_batch(test_cases, k=5)
    """

    def __init__(self, logger=None):
        self.logger = logger

    def _log(self, level: str, msg: str):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(msg)

    def evaluate_query(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        query: str = "",
        k: int = 5,
        relevance_grades: Optional[Dict[str, int]] = None,
    ) -> MetricResult:
        """Compute all metrics for a single query and log them."""
        relevant_retrieved = set(retrieved_ids[:k]) & relevant_ids

        result = MetricResult(
            query=query,
            k=k,
            num_retrieved=min(len(retrieved_ids), k),
            num_relevant=len(relevant_ids),
            num_relevant_retrieved=len(relevant_retrieved),
            recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
            precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
            mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
            ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k, relevance_grades),
            mean_avg_precision=average_precision(retrieved_ids, relevant_ids, k),
            hit_rate_at_k=hit_rate_at_k(retrieved_ids, relevant_ids, k),
        )

        self._log_single_result(result)
        return result

    def evaluate_batch(
        self,
        test_cases: List[RetrievalTestCase],
        k: int = 5,
    ) -> tuple:
        """
        Evaluate over multiple queries. Returns (averaged_metrics_dict, per_query_results).
        """
        per_query: List[MetricResult] = []

        for case in test_cases:
            result = self.evaluate_query(
                retrieved_ids=case.retrieved_ids,
                relevant_ids=case.relevant_ids,
                query=case.query,
                k=k,
                relevance_grades=case.relevance_grades,
            )
            per_query.append(result)

        # Aggregate averages
        n = len(per_query)
        avg = {
            "recall@k": sum(r.recall_at_k for r in per_query) / n,
            "precision@k": sum(r.precision_at_k for r in per_query) / n,
            "mrr": sum(r.mrr for r in per_query) / n,
            "ndcg@k": sum(r.ndcg_at_k for r in per_query) / n,
            "map": sum(r.mean_avg_precision for r in per_query) / n,
            "hit_rate@k": sum(r.hit_rate_at_k for r in per_query) / n,
            "k": k,
            "num_queries": n,
        }

        self._log_batch_summary(avg)
        return avg, per_query

    # ── Logging helpers ──────────────────────

    def _log_single_result(self, r: MetricResult):
        self._log("info",
            f"[Retrieval Metrics] query=\"{r.query[:80]}\" | "
            f"Recall@{r.k}={r.recall_at_k:.4f} | "
            f"Precision@{r.k}={r.precision_at_k:.4f} | "
            f"MRR={r.mrr:.4f} | "
            f"NDCG@{r.k}={r.ndcg_at_k:.4f} | "
            f"MAP={r.mean_avg_precision:.4f} | "
            f"HitRate@{r.k}={r.hit_rate_at_k:.4f} | "
            f"relevant_retrieved={r.num_relevant_retrieved}/{r.num_relevant}"
        )

    def _log_batch_summary(self, avg: Dict[str, Any]):
        self._log("info",
            f"[Retrieval Metrics — Batch Summary] "
            f"n={avg['num_queries']} queries, k={avg['k']} │ "
            f"Recall@K={avg['recall@k']:.4f} │ "
            f"Precision@K={avg['precision@k']:.4f} │ "
            f"MRR={avg['mrr']:.4f} │ "
            f"NDCG@K={avg['ndcg@k']:.4f} │ "
            f"MAP={avg['map']:.4f} │ "
            f"HitRate@K={avg['hit_rate@k']:.4f}"
        )

        # Log interpretation guidance
        self._log("info",
            "[Metric Interpretation] "
            "Recall@K < 0.8 → missing relevant guidelines, risk of incomplete assessment | "
            "MRR < 0.5 → relevant results buried below rank 2, slows clinical workflow | "
            "HitRate@K < 1.0 → some queries return zero relevant results (critical failure) | "
            "NDCG@K < 0.7 → ranking order is poor, high-relevance chunks not prioritised"
        )