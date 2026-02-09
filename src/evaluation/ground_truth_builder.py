"""
Ground Truth Builder for NG12 Retrieval Evaluation

Generates query-relevance pairs from actual chunk metadata.
No manual labeling needed — uses the cancer_types and symptoms
fields already present in your ChromaDB chunks.

Strategy:
---------
Each chunk has metadata like:
    cancers: "breast_cancer, colorectal_cancer"
    symptoms: "weight_loss, abdominal_pain"
    urgency: "urgent_2_week"
    section: "1.3.6 Consider a"

We define clinical test queries with expected cancer types and symptoms,
then match them against the chunk metadata to build ground truth.

Graded relevance:
    3 = exact match (correct cancer type AND matching symptoms)
    2 = strong match (correct cancer type OR multiple symptom matches)
    1 = partial match (single symptom overlap)
    0 = irrelevant
"""

from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ClinicalTestQuery:
    """
    A test query with expected relevance criteria.

    Attributes:
        query:              Natural language clinical query
        expected_cancers:   Cancer types this query is about
        expected_symptoms:  Symptoms mentioned in the query
        expected_urgency:   Expected urgency level (optional filter)
        expected_sections:  Specific section numbers expected (optional)
        description:        Human-readable description of the scenario
    """
    query: str
    expected_cancers: Set[str]
    expected_symptoms: Set[str]
    expected_urgency: Optional[str] = None
    expected_sections: Set[str] = field(default_factory=set)
    description: str = ""


@dataclass
class GroundTruthEntry:
    """One query with its relevant chunk IDs and grades."""
    query: str
    relevant_chunk_ids: Set[str]
    relevance_grades: Dict[str, int]
    description: str = ""
    expected_cancers: Set[str] = field(default_factory=set)
    expected_symptoms: Set[str] = field(default_factory=set)


def build_ground_truth(
    test_queries: List[ClinicalTestQuery],
    chunk_metadata: List[Dict[str, Any]],
) -> List[GroundTruthEntry]:
    """
    Match test queries against chunk metadata to build ground truth.

    Args:
        test_queries: Clinical queries with expected cancer/symptom criteria
        chunk_metadata: List of dicts from ChromaDB, each with:
            - chunk_id: str
            - cancer_types: str (comma-separated) or list
            - symptoms: str (comma-separated) or list
            - urgency: str
            - section: str

    Returns:
        List of GroundTruthEntry with relevance grades
    """
    entries = []

    for tq in test_queries:
        relevant_ids = set()
        grades = {}

        for chunk in chunk_metadata:
            chunk_id = chunk["chunk_id"]

            # Parse chunk metadata (handle both comma-separated strings and lists)
            chunk_cancers = _parse_field(chunk.get("cancer_types", ""))
            chunk_symptoms = _parse_field(chunk.get("symptoms", ""))
            chunk_urgency = chunk.get("urgency", "")

            # Calculate relevance grade
            grade = _calculate_relevance_grade(
                query_cancers=tq.expected_cancers,
                query_symptoms=tq.expected_symptoms,
                query_urgency=tq.expected_urgency,
                chunk_cancers=chunk_cancers,
                chunk_symptoms=chunk_symptoms,
                chunk_urgency=chunk_urgency,
            )

            if grade > 0:
                relevant_ids.add(chunk_id)
                grades[chunk_id] = grade

        entries.append(GroundTruthEntry(
            query=tq.query,
            relevant_chunk_ids=relevant_ids,
            relevance_grades=grades,
            description=tq.description,
            expected_cancers=tq.expected_cancers,
            expected_symptoms=tq.expected_symptoms,
        ))

    return entries


def _parse_field(value) -> Set[str]:
    """Parse a metadata field into a set of lowercase strings."""
    if isinstance(value, list):
        return {v.strip().lower() for v in value if v.strip()}
    if isinstance(value, str) and value:
        return {v.strip().lower() for v in value.split(",") if v.strip()}
    return set()


def _calculate_relevance_grade(
    query_cancers: Set[str],
    query_symptoms: Set[str],
    query_urgency: Optional[str],
    chunk_cancers: Set[str],
    chunk_symptoms: Set[str],
    chunk_urgency: str,
) -> int:
    """
    Calculate graded relevance (0-3) for a chunk against a query.

    Grading logic:
        3 = Cancer type matches AND 2+ symptom matches
        2 = Cancer type matches OR (1 cancer match + 1 symptom match)
        1 = At least 1 symptom match (no cancer match)
        0 = No match
    """
    cancer_overlap = query_cancers & chunk_cancers
    symptom_overlap = query_symptoms & chunk_symptoms

    cancer_match = len(cancer_overlap) > 0
    symptom_count = len(symptom_overlap)

    # Urgency filter: if specified, non-matching urgency drops grade by 1
    urgency_penalty = 0
    if query_urgency and chunk_urgency and query_urgency != chunk_urgency:
        urgency_penalty = 1

    # Grading
    if cancer_match and symptom_count >= 2:
        grade = 3
    elif cancer_match and symptom_count >= 1:
        grade = 2
    elif cancer_match:
        grade = 2
    elif symptom_count >= 2:
        grade = 1
    elif symptom_count >= 1:
        grade = 1
    else:
        grade = 0

    return max(0, grade - urgency_penalty)


# ──────────────────────────────────────────────────────
# Clinical Test Queries — based on real NG12 scenarios
# ──────────────────────────────────────────────────────

NG12_TEST_QUERIES = [
    # --- Upper GI ---
    ClinicalTestQuery(
        query="60 year old with weight loss and back pain",
        expected_cancers={"pancreatic_cancer"},
        expected_symptoms={"weight_loss", "back_pain", "abdominal_pain"},
        expected_urgency="urgent_2_week",
        description="Pancreatic cancer: weight loss + back pain in over 60",
    ),
    ClinicalTestQuery(
        query="55 year old with dysphagia and weight loss",
        expected_cancers={"stomach_cancer"},
        expected_symptoms={"dysphagia", "weight_loss"},
        expected_urgency="urgent_2_week",
        description="Stomach cancer: dysphagia + weight loss in over 55",
    ),
    ClinicalTestQuery(
        query="patient with upper abdominal mass",
        expected_cancers={"stomach_cancer", "liver_cancer"},
        expected_symptoms={"abdominal_pain"},
        description="Upper abdominal mass — could be stomach, liver, or gall bladder",
    ),

    # --- Colorectal ---
    ClinicalTestQuery(
        query="50 year old with rectal bleeding and weight loss",
        expected_cancers={"colorectal_cancer"},
        expected_symptoms={"rectal_bleeding", "weight_loss", "abdominal_pain"},
        expected_urgency="urgent_2_week",
        description="Colorectal cancer: rectal bleeding + weight loss in over 50",
    ),
    ClinicalTestQuery(
        query="45 year old with change in bowel habit and abdominal pain",
        expected_cancers={"colorectal_cancer"},
        expected_symptoms={"abdominal_pain", "weight_loss"},
        description="Colorectal: change in bowel habit + abdominal pain",
    ),
    ClinicalTestQuery(
        query="patient with unexplained anal mass",
        expected_cancers={"colorectal_cancer"},
        expected_symptoms={"unexplained"},
        description="Anal cancer: unexplained anal mass — section 1.3.6",
    ),

    # --- Breast ---
    ClinicalTestQuery(
        query="35 year old woman with unexplained breast lump",
        expected_cancers={"breast_cancer"},
        expected_symptoms={"breast_lump", "lump"},
        expected_urgency="urgent_2_week",
        description="Breast cancer: unexplained lump aged 30+, 2-week referral",
    ),
    ClinicalTestQuery(
        query="50 year old with nipple discharge",
        expected_cancers={"breast_cancer"},
        expected_symptoms={"breast_lump", "lump"},
        description="Breast cancer: nipple symptoms in over 50",
    ),

    # --- Gynaecological ---
    ClinicalTestQuery(
        query="55 year old woman with persistent bloating and abdominal pain",
        expected_cancers={"ovarian_cancer"},
        expected_symptoms={"bloating", "abdominal_pain", "persistent", "weight_loss"},
        description="Ovarian cancer: persistent symptoms in woman over 50",
    ),
    ClinicalTestQuery(
        query="58 year old woman with post-menopausal bleeding",
        expected_cancers={"endometrial_cancer"},
        expected_symptoms={"unexplained", "persistent"},
        expected_urgency="urgent_2_week",
        description="Endometrial cancer: post-menopausal bleeding over 55",
    ),

    # --- Urological ---
    ClinicalTestQuery(
        query="65 year old man with visible haematuria",
        expected_cancers={"bladder_cancer", "prostate_cancer"},
        expected_symptoms={"haematuria"},
        expected_urgency="urgent_2_week",
        description="Bladder/renal cancer: visible haematuria over 45",
    ),
    ClinicalTestQuery(
        query="55 year old man with raised PSA and urinary symptoms",
        expected_cancers={"prostate_cancer"},
        expected_symptoms={"haematuria"},
        description="Prostate cancer: PSA threshold + lower urinary tract symptoms",
    ),

    # --- Cross-cutting / non-specific ---
    ClinicalTestQuery(
        query="patient with unexplained weight loss and fatigue",
        expected_cancers={"ovarian_cancer", "pancreatic_cancer", "stomach_cancer", "colorectal_cancer"},
        expected_symptoms={"weight_loss", "fatigue", "unexplained"},
        description="Non-specific: weight loss + fatigue, multiple cancers possible",
    ),
    ClinicalTestQuery(
        query="unexplained lump in the axilla",
        expected_cancers={"breast_cancer"},
        expected_symptoms={"lump", "unexplained"},
        description="Breast cancer: axillary lump in over 30",
    ),

    # --- Edge cases ---
    ClinicalTestQuery(
        query="testicular enlargement in young man",
        expected_cancers={"testicular_cancer"},
        expected_symptoms={"persistent", "unexplained"},
        description="Testicular cancer: non-painful enlargement",
    ),
    ClinicalTestQuery(
        query="woman with vulval lump and ulceration",
        expected_cancers={"vulval_cancer"},
        expected_symptoms={"unexplained", "lump"},
        description="Vulval cancer: unexplained vulval lump",
    ),
]


def load_chunks_from_chromadb(vector_store) -> List[Dict[str, Any]]:
    """
    Extract chunk metadata from your ChromaDB vector store.

    Adapts to your VectorStore's API. Adjust field names
    if your metadata keys differ.
    """
    collection = vector_store.collection
    all_data = collection.get(include=["metadatas"])

    chunks = []
    for chunk_id, metadata in zip(all_data["ids"], all_data["metadatas"]):
        chunks.append({
            "chunk_id": chunk_id,
            "cancer_types": metadata.get("cancer_types", ""),
            "symptoms": metadata.get("symptoms", ""),
            "urgency": metadata.get("urgency", ""),
            "section": metadata.get("section", ""),
        })

    return chunks


def load_chunks_from_markdown(md_path: str) -> List[Dict[str, Any]]:
    """
    Parse chunk metadata from your chunks.md file.

    Extracts chunk_id, cancers, symptoms, urgency from the
    markdown headers in each chunk block.
    """
    import re

    with open(md_path, "r") as f:
        content = f.read()

    chunks = []
    # Split on chunk headers (## ng12_pXX_XXXX_...)
    blocks = re.split(r"^## (ng12_\S+)", content, flags=re.MULTILINE)

    # blocks alternates: [preamble, id1, block1, id2, block2, ...]
    for i in range(1, len(blocks) - 1, 2):
        chunk_id = blocks[i].strip()
        block = blocks[i + 1]

        # Extract metadata from markdown list items
        cancers = _extract_md_field(block, "Cancers")
        symptoms = _extract_md_field(block, "Symptoms")
        urgency = _extract_md_field(block, "Urgency")
        section = _extract_md_field(block, "Section")

        chunks.append({
            "chunk_id": chunk_id,
            "cancer_types": cancers,
            "symptoms": symptoms,
            "urgency": urgency,
            "section": section,
        })

    return chunks


def _extract_md_field(block: str, field_name: str) -> str:
    """Extract a field value from markdown like '- **Cancers:** val1, val2'."""
    import re
    pattern = rf"\*\*{field_name}:\*\*\s*(.+)"
    match = re.search(pattern, block)
    if match:
        return match.group(1).strip()
    return ""