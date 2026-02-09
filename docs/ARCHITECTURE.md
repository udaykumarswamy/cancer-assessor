# NG12 Cancer Risk Assessor — Architecture Document

## Executive Summary

This document outlines the architecture for a Clinical Decision Support System that combines structured patient data with unstructured clinical guidelines (NICE NG12) to provide:
1. **Automated Risk Assessment** — ReAct agent with specialised clinical tools evaluates cancer referral criteria with full reasoning traces
2. **Conversational Querying** — Natural language Q&A over clinical guidelines with conversation state management
3. **Retrieval Evaluation** — Automated metrics pipeline (Recall@K, MRR, NDCG) with metadata-driven ground truth

The core architectural principles are:
- **RAG Pipeline Reuse** — a single vector store and retrieval mechanism serves both assessment and chat
- **Auditability** — every clinical decision produces an explicit reasoning trace clinicians can follow and verify
- **Transparency over accuracy** — a wrong-but-traceable answer is safer than a right-but-opaque one

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │   Risk Assessment   │              │    Chat Interface   │               │
│  │        Tab          │              │         Tab         │               │
│  │  [Patient ID Input] │              │  [Message Window]   │               │
│  │  [Submit Button]    │              │  [Input Box]        │               │
│  └──────────┬──────────┘              └──────────┬──────────┘               │
└─────────────┼────────────────────────────────────┼──────────────────────────┘
              │                                    │
              ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             | 
│  POST /assess/{patient_id}    POST /chat    GET /search    GET /system      │
│           │                       │              │              │           │
│           ▼                       ▼              ▼              ▼           │
│  ┌─────────────────┐    ┌──────────────┐  ┌──────────┐  ┌──────────┐        │
│  │ ClinicalAgent   │    │  ChatAgent   │  │  Search  │  │  System  │        │
│  │ (ReAct Agent)   │    │  Controller  │  │  Routes  │  │  Routes  │        │
│  └────────┬────────┘    └──────┬───────┘  └────┬─────┘  └──────────┘        │
│           │                    │               │                            │
│           └────────────┬───────┴───────────────┘                            │
│                        ▼                                                    │
│          ┌──────────────────────────┐                                       │
│          │   dependencies.py        │◄── Dependency injection               │
│          │   (VectorStore, Embedder,│    for all shared components          │
│          │    Agent instances)       │                                      │
│          └────────────┬─────────────┘                                       │
│                       ▼                                                     │
│          ┌──────────────────────────┐                                       │
│          │   SHARED RAG LAYER       │◄── Key Design Decision                │
│          │  ┌────────────────────┐  │                                       │
│          │  │ ClinicalRetriever  │  │                                       │
│          │  │ • Query expansion  │  │                                       │
│          │  │ • Metadata filters │  │                                       │
│          │  │ • Score ranking    │  │                                       │
│          │  └────────────────────┘  │                                       │
│          └──────────────────────────┘                                       │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────────────────────────┐
│   TOOL LAYER (7 Tools)   │  │              KNOWLEDGE LAYER                 │
├──────────────────────────┤  ├──────────────────────────────────────────────┤
│                          │  │                                              │
│ ┌──────────────────────┐ │  │  ┌──────────────────┐  ┌──────────────────┐  │
│ │ search_guidelines    │ │  │  │  ChromaDB        │  │  Vertex AI       │  │
│ │ check_red_flags      │ │  │  │  Vector Store    │  │  Embeddings      │  │
│ │ calculate_risk       │ │  │  │                  │  │  (text-embedding │  │
│ │ get_referral_pathway │ │  │  │  • Chunks        │  │   -004)          │  │
│ │ extract_symptoms     │ │  │  │  • Metadata      │  └──────────────────┘  │
│ │ lookup_cancer_criteria │  │  │  • Embeddings    │                        │
│ │ get_section          │ │  │  │  • Clinical      │                        │
│ └──────────────────────┘ │  │  │    search methods│                        │
│                          │  │  └──────────────────┘                        │
│ ┌──────────────────────┐ │  │           ▲                                  │
│ │ PatientDataTool      │ │  │           │                                  │
│ │ (patients.json /     │ │  │  ┌────────┴─────────┐                        │
│ │  Mock BigQuery)      │ │  │  │ PDF Ingestion    │                        │
│ └──────────────────────┘ │  │  │ Pipeline         │                        │
└──────────────────────────┘  │  └────────┬─────────┘                        │
                              │           │                                  │
                              │  ┌────────┴─────────┐                        │
                              │  │ NG12 PDF (Source)│                        │
                              │  └──────────────────┘                        │
                              └──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Google Vertex AI — Gemini 2.5 Pro                │    │
│  │                                                                     │    │
│  │  • ReAct Orchestration (Thought → Action → Observation loops)       │    │
│  │  • Function Calling (7 clinical tools + patient data)               │    │
│  │  • Structured Output (risk assessment JSON)                         │    │
│  │  • Conversational (chat with grounding + citations)                 │    │
│  │  • Patient Info Extraction (handles negations, clinical nuance)     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐   │
│  │  retrieval_metrics   │  │  ground_truth_builder│  │  test_retrieval  │   │
│  │  • Recall@K          │  │  • Metadata-based    │  │  _real.py        │   │
│  │  • Precision@K       │  │    auto-labeling     │  │  • Per-cancer    │   │
│  │  • MRR               │  │  • Graded relevance  │  │    tests         │   │
│  │  • NDCG@K            │  │    (0–3)             │  │  • Threshold     │   │
│  │  • MAP               │  │  • ChromaDB + chunks │  │    checks        │   │
│  │  • Hit Rate@K        │  │    .md loaders       │  │  • Failure       │   │
│  └──────────────────────┘  └──────────────────────┘  │    analysis      │   │
│                                                      └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. PDF Ingestion Pipeline

**Purpose**: Transform the 90+ page NG12 PDF into searchable, citable chunks with rich clinical metadata.

**Design Decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Marker | `Marker` | Better table extraction, page-level metadata, but slower than PyPDF2 |
| Chunking Strategy | Semantic + Page-aware | Preserves clinical context; avoids splitting criteria mid-sentence |
| Chunk Size | 500–800 tokens | Balances context richness vs. retrieval precision |
| Overlap | 100 tokens | Ensures criteria spanning chunk boundaries aren't lost |
| Metadata | `{page, section, chunk_id, cancers, symptoms, urgency, type}` | Rich metadata enables filtered retrieval and auto-evaluation |

**Metadata-Enhanced Embeddings**:

A key finding during development — prepending clinical context to text before embedding significantly improves retrieval accuracy:

```python
# Standard embedding (lower accuracy)
embedding = embed("Refer people using a suspected cancer pathway...")

# Metadata-enhanced embedding (higher accuracy)
context_prefix = "Cancer: colorectal_cancer | Symptoms: rectal_bleeding, weight_loss | Urgency: urgent_2_week"
embedding = embed(f"{context_prefix}\n{chunk_text}")
```

This ensures the embedding captures clinical semantics, not just surface-level text similarity.

**Chunk Metadata Structure**:

Each chunk carries structured metadata stored as comma-separated strings in ChromaDB, parsed into lists for processing:

```
# Example chunk from chunks.md
## ng12_p7_0007_360f2e5d
- Section: 1.3.6 Consider a
- Pages: 7-7
- Type: criteria
- Tokens: 731
- Urgency: urgent_2_week
- Cancers: colorectal_cancer, bladder_cancer, liver_cancer, stomach_cancer
- Symptoms: weight_loss, abdominal_pain, rectal_bleeding, unexplained
```

**Chunking Strategy Detail**:

```python
# Naive chunking (BAD — loses context)
chunks = text.split_every_n_chars(500)

# Our approach (GOOD — preserves clinical meaning)
chunks = semantic_chunker(
    text,
    boundaries=["1.1", "1.2", "Recommendation"],  # Section markers
    max_tokens=800,
    preserve_tables=True,  # NG12 has important threshold tables (e.g. PSA by age)
    page_tracking=True     # For citations
)
```

---

### 2. Vector Store Design

**Choice**: ChromaDB (over FAISS)

| Criteria | ChromaDB | FAISS |
|----------|----------|-------|
| Metadata filtering | ✅ Native support | ❌ Requires wrapper |
| Persistence | ✅ Built-in | ⚠️ Manual save/load |
| Docker-friendly | ✅ Simple volume mount | ✅ Yes |
| Clinical search methods | ✅ Custom (cancer type, urgency, section) | ❌ Must build from scratch |
| Production-ready | ⚠️ Good for MVP | ✅ Battle-tested |

**Clinical-Specific Search Methods**:

The vector store exposes specialised search methods beyond basic similarity:

```python
class VectorStore:
    def search(self, query_embedding, top_k) -> List[SearchResult]
    def search_by_cancer_type(self, query_embedding, cancer_type, top_k) -> List[SearchResult]
    def search_urgent_only(self, query_embedding, top_k) -> List[SearchResult]
    def search_by_section(self, section_prefix, top_k) -> List[SearchResult]
```

**Collection Schema**:

```python
collection.add(
    ids=["ng12_p7_0007_360f2e5d"],
    documents=["Refer people using a suspected cancer pathway..."],
    embeddings=[...],  # 768-dim from Vertex AI text-embedding-004
    metadatas=[{
        "page_start": 7,
        "page_end": 7,
        "section": "1.3.6 Consider a",
        "chunk_id": 7,
        "next_chunk_id":7:1,
        "prev_chunk_id":7:0,
        "source": "NG12",
        "content_type": "criteria",        # criteria | content | table
        "cancer_types": "colorectal_cancer, bladder_cancer, liver_cancer, stomach_cancer",
        "symptoms": "weight_loss, abdominal_pain, rectal_bleeding, unexplained",
        "urgency": "urgent_2_week",
        "is_recommendation": True,
        "token_count": 731
    }]
)
```

---

### 3. Clinical Retriever (The Shared RAG Layer)

This is the **key architectural component** that enables pipeline reuse between assessment and chat.

```python
class ClinicalRetriever:
    """
    Retrieves relevant NG12 chunks for clinical queries.

    Features:
    - Query expansion with clinical synonyms (haemoptysis → coughing blood)
    - Patient-context-aware retrieval (age, symptoms, suspected cancer)
    - Metadata filtering (urgency, cancer type, section)
    - Score-based ranking with similarity threshold
    - Section-specific retrieval for guideline browsing

    Design Pattern: Strategy + Facade
    - Facade: Single interface hiding ChromaDB + embedding complexity
    - Strategy: Different retrieval paths for different use cases
    """

    # Clinical synonym expansions
    SYMPTOM_SYNONYMS = {
        "haemoptysis": ["hemoptysis", "coughing blood", "blood in sputum"],
        "dysphagia": ["difficulty swallowing", "swallowing problems"],
        "breathlessness": ["shortness of breath", "dyspnea", "breathing difficulty"],
        # ... more mappings
    }

    def retrieve(self, query, top_k, expand_query, mode) -> RetrievalContext
    def retrieve_for_patient(self, query, patient_age, symptoms, suspected_cancer, urgent_only) -> RetrievalContext
    def retrieve_by_section(self, section_number, top_k) -> RetrievalContext
```

**Three retrieval paths**:

| Method | Used By | Strategy |
|--------|---------|----------|
| `retrieve()` | Chat agent, general search | Semantic + query expansion |
| `retrieve_for_patient()` | Risk assessment agent | Patient-context-enhanced query + cancer type / urgency filters |
| `retrieve_by_section()` | `get_section` tool | Direct section lookup, no embedding needed |

**Query Expansion Example**:

```
Input:  "patient with haemoptysis"
Output: "patient with haemoptysis hemoptysis coughing blood blood in sputum"
```

This catches guideline chunks regardless of whether they use British or American spelling, medical or lay terminology.

---

### 4. ReAct Agent — Risk Assessment

**Architecture**: ReAct (Reasoning + Acting) with 7 specialised clinical tools.

The ReAct pattern was chosen over basic RAG because clinical decision-making requires explicit reasoning traces that clinicians can follow and verify — a critical requirement for healthcare applications.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ReAct Agent Loop                             │
└─────────────────────────────────────────────────────────────────┘

Input: patient_id="PT-101"
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ THOUGHT: I need to get the patient's data first to understand   │
│          their symptoms and demographics.                       │
│                                                                 │
│ ACTION: extract_symptoms(patient_id="PT-101")                   │
│                                                                 │
│ OBSERVATION: {                                                  │
│   "age": 55, "sex": "male",                                     │
│   "symptoms": ["unexplained hemoptysis", "fatigue"],            │
│   "negated": ["no chest pain", "no weight loss"],               │
│   "smoking_history": "Current Smoker",                          │
│   "duration_days": 14                                           │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ THOUGHT: Patient has hemoptysis. I should check red flags and   │
│          look up the specific NG12 criteria for lung cancer.    │
│                                                                 │
│ ACTION: check_red_flags(symptoms=["hemoptysis"], age=55)        │
│                                                                 │
│ OBSERVATION: 🚩 RED FLAG — hemoptysis is an urgent referral     │
│ trigger for suspected lung cancer (NG12 Section 1.8)            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ THOUGHT: Red flag confirmed. Let me retrieve the specific       │
│          referral criteria from NG12.                           │
│                                                                 │
│ ACTION: search_guidelines("hemoptysis referral criteria")       │
│ ACTION: lookup_cancer_criteria(cancer_type="lung")              │
│                                                                 │
│ OBSERVATION: [Retrieved NG12 chunks with referral pathways]     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ THOUGHT: NG12 states urgent referral for CXR within 2 weeks     │
│          if unexplained hemoptysis. Patient meets criteria.     │
│                                                                 │
│ ACTION: get_referral_pathway(cancer_type="lung", urgency="2ww") │
│ ACTION: calculate_risk(patient_data=..., matched_criteria=...)  │
│                                                                 │
│ OBSERVATION: Risk=HIGH, Pathway=2-week-wait referral            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ FINAL ANSWER (Structured Output):                               │
│ {                                                               │
│   "patient_id": "PT-101",                                       │
│   "risk_level": "HIGH",                                         │
│   "recommendation": "URGENT_REFERRAL",                          │
│   "reasoning": "Patient (55M, current smoker) presents with     │
│     unexplained hemoptysis for 14 days. Per NG12 Section 1.8,   │
│     unexplained hemoptysis triggers urgent CXR referral within  │
│     2 weeks. Smoking history adds additional risk.",            │
│   "citations": [{"page": 15, "section": "1.8", "excerpt": "..."}],│
│   "reasoning_trace": [                                           │
│     "🔍 Extracted symptoms: hemoptysis, fatigue",                │
│     "🚩 Red flag identified: hemoptysis",                        │
│     "📋 NG12 criteria matched: Section 1.8.1",                   │
│     "⚡ Urgency: 2-week-wait referral"                            │
│   ]                                                              │
│ }                                                                │
└───────────────────────────────────────────────────────────────── ┘
```

**The 7 Clinical Tools**:

| Tool | Purpose | Uses RAG? |
|------|---------|-----------|
| `search_guidelines` | Semantic search over NG12 chunks | ✅ ClinicalRetriever.retrieve() |
| `check_red_flags` | Identify urgent clinical red flags | ✅ ClinicalRetriever.retrieve() with urgency filter |
| `calculate_risk` | Deterministic risk scoring | ❌ Rule-based logic |
| `get_referral_pathway` | Look up the correct referral route | ✅ ClinicalRetriever.retrieve_for_patient() |
| `extract_symptoms` | LLM-based patient info extraction | ❌ LLM call (handles negations) |
| `lookup_cancer_criteria` | Retrieve criteria for a specific cancer type | ✅ ClinicalRetriever.retrieve_for_patient() with cancer filter |
| `get_section` | Retrieve a specific NG12 section | ✅ ClinicalRetriever.retrieve_by_section() |

**Why ReAct over basic RAG?**

| Aspect | Basic RAG | ReAct Agent |
|--------|-----------|-------------|
| Reasoning visibility | Black box | Explicit Thought → Action → Observation trace |
| Multi-step queries | Single retrieval | Iterative — can refine search based on findings |
| Clinical auditability | Low | High — clinician can follow each reasoning step |
| Tool composition | Fixed pipeline | Dynamic — agent decides which tools to call and in what order |
| Error handling | Fail silently | Agent can recognise insufficient evidence and search again |

**LLM-Based Symptom Extraction**:

The `extract_symptoms` tool uses the LLM to parse patient records, handling clinical nuances that regex cannot:

```python
# Handles negations
"Patient denies chest pain, reports persistent cough"
→ symptoms: ["persistent cough"], negated: ["chest pain"]

# Handles temporal qualifiers
"Cough for 3 weeks, weight loss over past 2 months"
→ symptoms: [("cough", 21 days), ("weight_loss", 60 days)]
```

---

### 5. Conversation Flow Management

**Architecture**: Stateful chat with context-aware query classification.

A critical architectural decision — the system must distinguish between three types of user input to avoid triggering unnecessary clinical evaluations:

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│ Context Question Detector       │
│                                 │
│ Is this a:                      │
│ 1. Information gathering?       │──▶ Store answer, continue conversation
│ 2. Context/follow-up question?  │──▶ Answer from session context, no RAG
│ 3. Assessment request?          │──▶ Trigger full ReAct assessment loop
└─────────────────────────────────┘
```

**Examples**:
- "My patient is 55 years old" → **Information gathering** (store, don't assess)
- "What did you mean by 2-week-wait?" → **Context question** (answer from history)
- "Please assess this patient" → **Assessment request** (trigger ReAct agent)

**Session Memory Design**:

```python
@dataclass
class ConversationSession:
    session_id: str
    messages: List[Message]
    patient_context: Dict[str, Any]     # Accumulated patient info
    created_at: datetime
    last_active: datetime

    def get_context_window(self, max_turns: int = 5) -> List[Message]:
        """Return recent messages for context, avoiding token overflow."""
        return self.messages[-max_turns * 2:]
```

---

### 6. Retrieval Evaluation Pipeline

**Purpose**: Measure and track retrieval quality using automated metrics, without manual labeling.

**Architecture**:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Chunk Metadata   │────▶│ Ground Truth     │────▶│ Retrieval        │
│ (ChromaDB /      │     │ Builder          │     │ Evaluator        │
│  chunks.md)      │     │                  │     │                  │
│                  │     │ Matches queries  │     │ • Recall@K       │
│ • cancer_types   │     │ against metadata │     │ • Precision@K    │
│ • symptoms       │     │ to auto-label    │     │ • MRR            │
│ • urgency        │     │ relevance (0–3)  │     │ • NDCG@K         │
└──────────────────┘     └──────────────────┘     │ • MAP            │
                                                  │ • Hit Rate@K     │
                                                  └──────────────────┘
```

**Graded Relevance** (used by NDCG for ranking quality):

| Grade | Meaning | Matching Rule |
|-------|---------|---------------|
| 3 — EXACT | Correct cancer type AND 2+ symptom matches | Highest clinical relevance |
| 2 — STRONG | Correct cancer type OR cancer + 1 symptom | Related guideline section |
| 1 — PARTIAL | Symptom overlap only, no cancer type match | Tangentially relevant |
| 0 — IRRELEVANT | No overlap | Should not be retrieved |

**Auto-Generated Ground Truth**:

Rather than manual labeling, the system generates ground truth from chunk metadata. Each test query has expected cancer types and symptoms; these are matched against the metadata already present in every chunk:

```python
# Test query
ClinicalTestQuery(
    query="50 year old with rectal bleeding and weight loss",
    expected_cancers={"colorectal_cancer"},
    expected_symptoms={"rectal_bleeding", "weight_loss", "abdominal_pain"},
)

# Auto-matched against chunks:
# ng12_p7_0007 → grade 3 (colorectal + rectal_bleeding + weight_loss)
# ng12_p8_0008 → grade 2 (colorectal, no symptom overlap)
# ng12_p6_0006 → grade 1 (weight_loss only, wrong cancer)
```

**Quality Thresholds** (tests fail if below these):

| Metric | Threshold | Clinical Rationale |
|--------|-----------|-------------------|
| Recall@K | ≥ 0.6 | Missing guidelines = missed referrals |
| Hit Rate@K | ≥ 0.8 | Queries with zero results = critical failure |
| MRR | ≥ 0.4 | First relevant result should be in top 2–3 |
| NDCG@K | ≥ 0.5 | Highest-relevance chunks should rank first |

**Logged Interpretation Guidance**:

```
[Metric Interpretation]
  Recall@K < 0.8 → missing relevant guidelines, risk of incomplete assessment
  MRR < 0.5 → relevant results buried below rank 2, slows clinical workflow
  HitRate@K < 1.0 → some queries return zero relevant results (critical failure)
  NDCG@K < 0.7 → ranking order is poor, high-relevance chunks not prioritised
```

---

### 7. Grounding and Guardrails

**Critical for clinical applications** — the system must not hallucinate.

```python
class GroundingGuardrails:
    """Ensures responses are grounded in retrieved evidence."""

    CONFIDENCE_THRESHOLD = 0.7
    MIN_SUPPORTING_CHUNKS = 1

    def validate_response(self, response, retrieved_chunks, query) -> GroundedResponse:

        # Check 1: Were relevant chunks retrieved?
        if not retrieved_chunks or max(c.score for c in retrieved_chunks) < self.CONFIDENCE_THRESHOLD:
            return GroundedResponse(
                answer="I couldn't find sufficient evidence in the NG12 guidelines "
                       "to answer this question confidently.",
                is_grounded=False,
                citations=[]
            )

        # Check 2: Does response align with retrieved content?

        # Check 3: Extract and validate citations
        citations = self.extract_citations(response, retrieved_chunks)

        return GroundedResponse(
            answer=response,
            is_grounded=True,
            citations=citations
        )
```

---

## Data Flow Diagrams

### Risk Assessment Data Flow (ReAct)

```
Patient ID (PT-101)
       │
       ▼
┌──────────────┐     ┌──────────────────────────────────────────────┐
│   FastAPI    │────▶│          ReAct Agent Loop                    │
│   /assess    │     │                                              │
└──────────────┘     │  ┌────────┐   ┌────────┐   ┌────────────┐    │
                     │  │THOUGHT │──▶│ ACTION │──▶│OBSERVATION │    │
                     │  └────────┘   └───┬────┘   └─────┬──────┘    │
                     │       ▲           │              │           │
                     │       └───────────┴──────────────┘           │
                     │                                  (loops)     │
                     └────────────────────┬──────────────────────── ┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     │    Tool Calls      │                    │
                     ▼                    ▼                    ▼
              ┌────────────┐     ┌────────────────┐   ┌────────────┐
              │ extract_   │     │ search_        │   │ check_     │
              │ symptoms   │     │ guidelines     │   │ red_flags  │
              │            │     │                │   │            │
              │ LLM-based  │     │ ClinicalRe-    │   │ ClinicalRe-│
              │ extraction │     │ triever.       │   │ triever +  │
              │ (negations)│     │ retrieve()     │   │ urgency    │
              └────────────┘     └────────────────┘   └────────────┘
                     │                    │                    │
                     └────────────────────┼────────────────────┘
                                          │
                                          ▼
                                ┌─────────────────┐
                                │ Structured JSON │
                                │ +Reasoning Trace│
                                │ + Citations     │
                                └─────────────────┘
```

### Chat Data Flow

```
User Message + Session ID
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│  Context     │
│   /chat      │     │  Question    │
└──────────────┘     │  Detector    │
                     └──────┬───────┘
                            │
               ┌────────────┼────────────────┐
               ▼            ▼                ▼
        ┌──────────┐ ┌──────────┐    ┌──────────────┐
        │ Info     │ │ Context  │    │ Assessment   │
        │ Gather   │ │ Query    │    │ Request      │
        │          │ │          │    │              │
        │ Store in │ │ Answer   │    │ Trigger full │
        │ session  │ │ from     │    │ ReAct loop   │
        └──────────┘ │ history  │    └──────────────┘
                     └──────────┘
                            │
                            ▼
               ┌──────────────────────┐
               │  ClinicalRetriever   │
               │  .retrieve()         │
               └───────────┬──────────┘
                           │
                           ▼
               ┌──────────────────────┐
               │    Gemini 2.5 Pro    │
               │  (Grounded Response) │
               └───────────┬──────────┘
                           │
                           ▼
               ┌──────────────────────┐
               │  Chat Response       │
               │  + Citations         │
               └──────────────────────┘
```

---

## Project Structure

```
CANCER-ASSESSOR/
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
│
├── output/
│   ├── chunks.md                    # Processed NG12 chunks with metadata
│   └── ng12_full.md                 # Full extracted NG12 text
│
├── docs/
│   ├── ARCHITECTURE.md              # This document
│   ├── PROMPTS.md                   # System prompt documentation
│   └── CHAT_PROMPTS.md              # Chat-specific prompts
│
├── data/
│   ├── patients.json                # Mock patient database
│   └── ng12/                        # PDF storage
│
├── scripts/
│   ├── download_ng12.py             # Downloads PDF
│   └── ingest_pdf.py                # Runs ingestion pipeline
│
├── src/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app entry point
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py          # Dependency injection (vector store, embedder, agents)
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── assessment.py        # POST /assess/{patient_id}
│   │   │   ├── chat.py              # POST /chat, GET /chat/{id}/history
│   │   │   ├── search.py            # GET /search (direct retrieval)
│   │   │   └── system.py            # GET /system (health, stats)
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── requests.py          # Pydantic request models
│   │       └── responses.py         # Pydantic response models
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── react_agent.py           # Core ReAct loop (Thought → Action → Observation)
│   │   ├── clinical_agent.py        # Clinical assessment orchestrator
│   │   └── tools.py                 # 7 clinical tools (search_guidelines, check_red_flags, etc.)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── clinical_retriever.py    # ClinicalRetriever (query expansion, patient-context search)
│   │   ├── vector_store.py          # ChromaDB wrapper (clinical search methods)
│   │   └── embedder.py              # Vertex AI embeddings (metadata-enhanced)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py            # PDF extraction with marker
│   │   └── chunker.py               # Semantic + page-aware chunking
│   │
│   ├── evaluation/                  # Retrieval quality metrics
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py     # Recall@K, MRR, NDCG, Precision, MAP, Hit Rate
│   │   └── ground_truth_builder.py  # Auto-labels relevance from chunk metadata
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── gemini_client.py         # Vertex AI Gemini wrapper
│   │   └── prompts.py               # Prompt templates
│   │
│   └── config/
│       ├── __init__.py
│       ├── settings.py              # Pydantic settings
│       └── logging_config.py        # Structured logging
│
├── frontend/
│   ├── index.html                   # HTML/JS frontend
│   ├── styles.css
│   └── app.js
│
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py            # Unit tests for ClinicalRetriever
│   ├── test_risk_assessment.py      # Agent integration tests
│   ├── test_chat.py                 # Chat flow tests   
│   
│
└── vectorstore/                     # ChromaDB persistent storage (gitignored)
```

---

## Key Design Decisions Summary

| Decision | Choice | Alternatives Considered | Why This Choice |
|----------|--------|------------------------|-----------------|
| Agent Pattern | ReAct | Basic RAG, LangChain agents | Explicit reasoning traces critical for clinical auditability |
| Vector DB | ChromaDB | FAISS, Pinecone | Metadata filtering, clinical search methods, easy persistence |
| Embeddings | Vertex AI text-embedding-004 | OpenAI ada-002 | Project requires Vertex AI; 768-dim is efficient |
| Embedding Strategy | Metadata-enhanced | Raw text only | Prepending clinical context improves retrieval accuracy |
| LLM | Gemini 2.5 Pro | Gemini 2.5 Flash, GPT 4 | Better reasoning for multi-step clinical decisions |
| Symptom Extraction | LLM-based | Regex, NER | Handles negations and clinical nuance that rule-based methods miss |
| PDF Parser | Marker / PyMuPDF | Dockling/Surya, PyPDF2, pdfplumber | Better table handling, page metadata |
| Chunking | Semantic + Page-aware | Fixed-size, sentence | Preserves clinical recommendation integrity |
| Evaluation Ground Truth | Auto-generated from metadata | Manual labeling, LLM-as-judge | Zero manual effort, clinically meaningful, repeatable |
| Session Memory | In-memory dict | Redis, SQLite | Acceptable for MVP; noted production path |
| Frontend | Vanilla HTML/JS | React, Vue | Minimal requirement; fast to implement |
| API Framework | FastAPI | Flask | Async support, auto-docs, Pydantic integration |

---

## Interview Discussion Points

### 1. "Why ReAct over basic RAG for risk assessment?"

Basic RAG does a single retrieve-then-generate step. For clinical decisions this is insufficient because:
- **Multi-step reasoning**: A patient with hemoptysis needs: symptom extraction → red flag check → guideline lookup → risk calculation → referral pathway. Each step informs the next.
- **Auditability**: The Thought → Action → Observation trace gives clinicians a complete reasoning chain they can verify step by step.
- **Dynamic tool selection**: The agent decides which tools to call based on what it finds. If the first search is insufficient, it can refine and search again.
- **Negation handling**: The LLM-based `extract_symptoms` tool correctly handles "patient denies chest pain" — a regex approach would flag chest pain as present.

### 2. "Why not fine-tune instead of RAG?"

RAG is preferred here because:
- **Updatability**: NG12 gets revised (most recently 2025); RAG just needs re-ingestion
- **Auditability**: Can trace every answer to source chunks with page numbers
- **Compliance**: Clinical decisions need citations; fine-tuning is a black box
- **Cost**: No training costs; faster iteration

### 3. "How do you evaluate retrieval quality without manual labels?"

The chunks already carry rich metadata (`cancer_types`, `symptoms`, `urgency`). We define clinical test queries with expected cancer types and symptoms, then automatically match them against chunk metadata to produce graded relevance labels (0–3). This gives us a fully automated evaluation pipeline that runs in CI. With 16 test queries covering all major cancer types in NG12, we can detect retrieval regressions immediately.

### 4. "What are the failure modes?"

| Failure Mode | Detection | Mitigation |
|---|---|---|
| Retrieval miss (relevant chunk not in top-K) | Recall@K drops | Expand clinical synonyms, improve chunking |
| Wrong ranking (relevant chunk buried) | MRR/NDCG drops | Metadata-boosted search, re-ranking |
| Hallucination (LLM ignores context) | Grounding guardrails | Confidence threshold, min supporting chunks |
| Citation mismatch (wrong page cited) | Manual spot checks | Improve metadata pipeline |
| Negation failure ("no chest pain" → "chest pain") | Symptom extraction tests | LLM-based extraction (not regex) |
| Context confusion (multi-turn chat) | Session isolation tests | Context question detector, session cleanup |

### 5. "How would you scale this?"

- **Vector DB**: Move to managed service (Pinecone, Vertex AI Matching Engine)
- **Sessions**: Redis cluster with TTL for auto-expiry
- **LLM**: Request batching, caching common queries
- **API**: Kubernetes horizontal pod autoscaling
- **Evaluation**: Expand ground truth to 100+ queries, add LLM-as-judge for answer quality

---
