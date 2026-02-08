# NG12 Cancer Risk Assessor - Architecture Document

## Executive Summary

This document outlines the architecture for a Clinical Decision Support System that combines structured patient data with unstructured clinical guidelines (NICE NG12) to provide:
1. **Automated Risk Assessment** - Deterministic evaluation of cancer referral criteria
2. **Conversational Querying** - Natural language Q&A over clinical guidelines

The core architectural principle is **RAG Pipeline Reuse** - a single vector store and retrieval mechanism serves both use cases, demonstrating efficient resource utilization and consistent knowledge grounding.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
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
│                              API LAYER (FastAPI)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POST /assess/{patient_id}          POST /chat                              │
│           │                              │                                   │
│           ▼                              ▼                                   │
│  ┌─────────────────┐            ┌─────────────────┐                         │
│  │ RiskAssessment  │            │   ChatAgent     │                         │
│  │     Agent       │            │   Controller    │                         │
│  └────────┬────────┘            └────────┬────────┘                         │
│           │                              │                                   │
│           └──────────────┬───────────────┘                                   │
│                          ▼                                                   │
│              ┌───────────────────────┐                                       │
│              │   SHARED RAG LAYER    │◄─── Key Design Decision              │
│              │  ┌─────────────────┐  │                                       │
│              │  │ RetrievalService│  │                                       │
│              │  └────────┬────────┘  │                                       │
│              └───────────┼───────────┘                                       │
└──────────────────────────┼──────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐    ┌─────────────────────────────────────────────────┐
│   TOOL LAYER        │    │              KNOWLEDGE LAYER                     │
├─────────────────────┤    ├─────────────────────────────────────────────────┤
│                     │    │                                                  │
│ ┌─────────────────┐ │    │  ┌─────────────────┐    ┌─────────────────────┐ │
│ │ PatientDataTool │ │    │  │  ChromaDB       │    │  Vertex AI          │ │
│ │                 │ │    │  │  Vector Store   │    │  Embeddings         │ │
│ │ patients.json   │ │    │  │                 │    │  (text-embedding-   │ │
│ │ (Mock BigQuery) │ │    │  │  - Chunks       │    │   004)              │ │
│ └─────────────────┘ │    │  │  - Metadata     │    └─────────────────────┘ │
│                     │    │  │  - Embeddings   │                             │
└─────────────────────┘    │  └─────────────────┘                             │
                           │           ▲                                      │
                           │           │                                      │
                           │  ┌────────┴────────┐                             │
                           │  │ PDF Ingestion   │                             │
                           │  │ Pipeline        │                             │
                           │  │ (One-time)      │                             │
                           │  └─────────────────┘                             │
                           │           ▲                                      │
                           │           │                                      │
                           │  ┌────────┴────────┐                             │
                           │  │   NG12 PDF      │                             │
                           │  │   (Source)      │                             │
                           │  └─────────────────┘                             │
                           └─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Google Vertex AI - Gemini 1.5 Pro                │    │
│  │                                                                      │    │
│  │  • Function Calling (for Patient Data Tool)                         │    │
│  │  • Structured Output (for Risk Assessment JSON)                     │    │
│  │  • Conversational (for Chat with grounding)                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. PDF Ingestion Pipeline

**Purpose**: Transform the 50+ page NG12 PDF into searchable, citable chunks.

**Design Decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PDF Parser | `PyMuPDF (fitz)` | Better table extraction, page-level metadata, faster than PyPDF2 |
| Chunking Strategy | Semantic + Page-aware | Preserves clinical context; avoids splitting criteria mid-sentence |
| Chunk Size | 500-800 tokens | Balances context richness vs. retrieval precision |
| Overlap | 100 tokens | Ensures criteria spanning chunk boundaries aren't lost |
| Metadata | `{page, section, chunk_id}` | Critical for citations - interview talking point |

**Chunking Strategy Detail**:

```python
# Naive chunking (BAD - loses context)
chunks = text.split_every_n_chars(500)

# Our approach (GOOD - preserves clinical meaning)
chunks = semantic_chunker(
    text,
    boundaries=["1.1", "1.2", "Recommendation"],  # Section markers
    max_tokens=800,
    preserve_tables=True,  # NG12 has important threshold tables
    page_tracking=True     # For citations
)
```

**Interview Question**: "Why not just chunk by fixed character count?"
**Answer**: Clinical guidelines have structured recommendations (e.g., "Refer urgently if X AND Y"). Naive chunking could split "Refer urgently if patient has hemoptysis" from "AND is over 40 years old", leading to incorrect risk assessments.

---

### 2. Vector Store Design

**Choice**: ChromaDB (over FAISS)

| Criteria | ChromaDB | FAISS |
|----------|----------|-------|
| Metadata filtering | ✅ Native support | ❌ Requires wrapper |
| Persistence | ✅ Built-in | ⚠️ Manual save/load |
| Docker-friendly | ✅ Simple volume mount | ✅ Yes |
| Production-ready | ⚠️ Good for MVP | ✅ Battle-tested |

**Why ChromaDB for this project**:
- Native metadata filtering lets us do things like `where={"page": {"$gte": 10}}` 
- Simpler Docker setup with persistent volumes
- Good enough for the document size (~50 pages)

**Collection Schema**:

```python
collection.add(
    ids=["ng12_p15_chunk_03"],
    documents=["Refer people using a suspected cancer pathway..."],
    embeddings=[...],  # 768-dim from Vertex AI
    metadatas=[{
        "page": 15,
        "section": "1.3 Lung Cancer",
        "chunk_index": 3,
        "source": "NG12",
        "content_type": "recommendation"  # vs "background", "table"
    }]
)
```

---

### 3. Retrieval Service (The Shared Layer)

This is the **key architectural component** that enables pipeline reuse.

```python
class RetrievalService:
    """
    Unified retrieval interface for both Risk Assessment and Chat.
    
    Design Pattern: Strategy + Facade
    - Facade: Single interface hiding ChromaDB complexity
    - Strategy: Different retrieval strategies for different use cases
    """
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict = None,
        strategy: Literal["semantic", "hybrid", "symptom_match"] = "semantic"
    ) -> List[RetrievedChunk]:
        """
        Core retrieval method used by BOTH agents.
        
        For Risk Assessment: strategy="symptom_match", filters by content_type
        For Chat: strategy="semantic", broader search
        """
        pass
    
    def retrieve_for_symptoms(
        self,
        symptoms: List[str],
        patient_context: dict
    ) -> List[RetrievedChunk]:
        """
        Specialized method for risk assessment.
        Performs multiple queries (one per symptom) and merges results.
        """
        pass
```

**Interview Question**: "Why abstract the retrieval into a service?"
**Answer**: 
1. **Single source of truth** - Both agents use identical retrieval logic
2. **Testability** - Can unit test retrieval independently
3. **Flexibility** - Can swap ChromaDB for Pinecone/Weaviate without changing agents
4. **Consistency** - Citation format is uniform across both features

---

### 4. Risk Assessment Agent

**Architecture**: ReAct-style Agent with Function Calling

```
┌─────────────────────────────────────────────────────────────────┐
│                    Risk Assessment Flow                          │
└─────────────────────────────────────────────────────────────────┘

Input: patient_id="PT-101"
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: TOOL CALL - get_patient_data(patient_id)                │
│                                                                  │
│ Returns: {                                                       │
│   "name": "John Doe",                                           │
│   "age": 55,                                                    │
│   "symptoms": ["unexplained hemoptysis", "fatigue"],            │
│   "smoking_history": "Current Smoker",                          │
│   "symptom_duration_days": 14                                   │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: RAG RETRIEVAL                                           │
│                                                                  │
│ Query 1: "hemoptysis referral criteria"                         │
│ Query 2: "fatigue cancer referral"                              │
│ Query 3: "lung cancer smoking age criteria"                     │
│                                                                  │
│ Retrieved chunks contain NG12 recommendations                   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: REASONING + SYNTHESIS                                   │
│                                                                  │
│ LLM evaluates patient data against retrieved criteria:          │
│ - "NG12 states: Refer urgently for CXR within 2 weeks if        │
│    unexplained hemoptysis" [Page 15]                            │
│ - Patient HAS hemoptysis → Criteria MET                         │
│ - Age 55 + Smoker → Additional risk factors                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: STRUCTURED OUTPUT                                       │
│                                                                  │
│ {                                                                │
│   "patient_id": "PT-101",                                       │
│   "risk_level": "HIGH",                                         │
│   "recommendation": "URGENT_REFERRAL",                          │
│   "reasoning": "...",                                           │
│   "citations": [{"page": 15, "excerpt": "..."}]                 │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

**Function Calling Schema** (for Gemini):

```python
get_patient_data_tool = {
    "name": "get_patient_data",
    "description": "Retrieves patient record from the clinical database",
    "parameters": {
        "type": "object",
        "properties": {
            "patient_id": {
                "type": "string",
                "description": "The unique patient identifier (e.g., PT-101)"
            }
        },
        "required": ["patient_id"]
    }
}
```

---

### 5. Chat Agent

**Architecture**: RAG + Conversation Memory

```
┌─────────────────────────────────────────────────────────────────┐
│                       Chat Agent Flow                            │
└─────────────────────────────────────────────────────────────────┘

Turn 1: "What symptoms trigger urgent referral for lung cancer?"
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Retrieve from Vector Store                                   │
│    Query: "lung cancer urgent referral symptoms"                │
│    Returns: 5 relevant chunks with page numbers                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Generate Grounded Response                                   │
│    "According to NG12, urgent referral for suspected lung       │
│     cancer is indicated for: unexplained hemoptysis [p.15],     │
│     or chest X-ray findings suggestive of lung cancer [p.16]"  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Store in Session Memory                                      │
│    session_id: "abc123"                                         │
│    history: [{"role": "user", "content": "..."},                │
│              {"role": "assistant", "content": "..."}]           │
└─────────────────────────────────────────────────────────────────┘

Turn 2: "What about for patients under 40?"
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Session History                                         │
│ 2. Context-aware retrieval (understands "lung cancer" context)  │
│ 3. Generate response referencing age-specific criteria          │
└─────────────────────────────────────────────────────────────────┘
```

**Session Memory Design**:

```python
# In-memory store (acceptable for take-home)
sessions: Dict[str, ConversationSession] = {}

@dataclass
class ConversationSession:
    session_id: str
    messages: List[Message]
    created_at: datetime
    last_active: datetime
    
    def get_context_window(self, max_turns: int = 5) -> List[Message]:
        """Return recent messages for context, avoiding token overflow"""
        return self.messages[-max_turns * 2:]  # User + Assistant pairs
```

**Interview Question**: "How would you handle session memory at scale?"
**Answer**: 
- Move from in-memory dict to Redis (TTL for auto-expiry)
- Consider session summarization for long conversations
- Implement proper session cleanup cron job

---

### 6. Grounding & Guardrails

**Critical for clinical applications** - the system must not hallucinate.

```python
class GroundingGuardrails:
    """
    Ensures responses are grounded in retrieved evidence.
    """
    
    CONFIDENCE_THRESHOLD = 0.7
    MIN_SUPPORTING_CHUNKS = 1
    
    def validate_response(
        self,
        response: str,
        retrieved_chunks: List[RetrievedChunk],
        query: str
    ) -> GroundedResponse:
        
        # Check 1: Were relevant chunks retrieved?
        if not retrieved_chunks or max(c.score for c in retrieved_chunks) < self.CONFIDENCE_THRESHOLD:
            return GroundedResponse(
                answer="I couldn't find sufficient evidence in the NG12 guidelines to answer this question confidently.",
                is_grounded=False,
                citations=[]
            )
        
        # Check 2: Does response align with retrieved content?
        # (Could use NLI model or keyword overlap as proxy)
        
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

### Risk Assessment Data Flow

```
Patient ID (PT-101)
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│    Agent     │────▶│   Gemini     │
│   Endpoint   │     │  Orchestrator│     │   1.5 Pro    │
└──────────────┘     └──────┬───────┘     └──────────────┘
                           │                      │
              ┌────────────┴────────────┐         │
              ▼                         ▼         │
       ┌────────────┐           ┌────────────┐    │
       │  Patient   │           │  Retrieval │    │
       │  Data Tool │           │  Service   │    │
       │            │           │            │    │
       │ JSON Mock  │           │  ChromaDB  │    │
       └────────────┘           └────────────┘    │
              │                         │         │
              └─────────┬───────────────┘         │
                        │                         │
                        ▼                         │
              ┌─────────────────┐                 │
              │ Context Package │                 │
              │ - Patient Data  │─────────────────┘
              │ - NG12 Chunks   │
              │ - Metadata      │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Risk Assessment │
              │     JSON        │
              └─────────────────┘
```

### Chat Data Flow

```
User Message + Session ID
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│   Session    │
│   /chat      │     │   Manager    │
└──────────────┘     └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   History    │
                    │   Context    │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌────────────┐           ┌────────────┐
       │  Context-  │           │  Retrieval │
       │  Aware     │           │  Service   │
       │  Query     │           │            │
       └────────────┘           └────────────┘
              │                         │
              └─────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │    Gemini       │
              │    1.5 Pro      │
              │  (Grounded Gen) │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Chat Response  │
              │  + Citations    │
              └─────────────────┘
```

---

## Project Structure

```
ng12-cancer-assessor/
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
│
├── docs/
│   ├── ARCHITECTURE.md          # This document
│   ├── PROMPTS.md               # System prompt documentation
│   └── CHAT_PROMPTS.md          # Chat-specific prompts
│
├── data/
│   ├── patients.json            # Mock patient database
│   └── ng12/                    # PDF storage (gitignored, downloaded at build)
│
├── scripts/
│   ├── download_ng12.py         # Downloads PDF
│   └── ingest_pdf.py            # Runs ingestion pipeline
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── assessment.py    # POST /assess/{patient_id}
│   │   │   └── chat.py          # POST /chat, GET /chat/{id}/history
│   │   └── models/
│   │       ├── requests.py      # Pydantic request models
│   │       └── responses.py     # Pydantic response models
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── risk_assessment.py   # Risk assessment agent
│   │   └── chat_agent.py        # Conversational agent
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── retrieval.py         # Shared RAG service
│   │   ├── patient_data.py      # Patient data tool
│   │   └── session_manager.py   # Chat session management
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py        # PDF extraction
│   │   ├── chunker.py           # Text chunking
│   │   └── embedder.py          # Vertex AI embeddings
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── gemini_client.py     # Vertex AI Gemini wrapper
│   │   └── prompts.py           # Prompt templates
│   │
│   └── config/
│       ├── __init__.py
│       └── settings.py          # Pydantic settings
│
├── frontend/
│   ├── index.html               # Simple HTML/JS frontend
│   ├── styles.css
│   └── app.js
│
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py
│   ├── test_risk_assessment.py
│   └── test_chat.py
│
└── vectorstore/                 # ChromaDB persistent storage (gitignored)
```

---

## Key Design Decisions Summary

| Decision | Choice | Alternatives Considered | Why This Choice |
|----------|--------|------------------------|-----------------|
| Vector DB | ChromaDB | FAISS, Pinecone | Metadata filtering, easy persistence, good for MVP |
| Embeddings | Vertex AI text-embedding-004 | OpenAI ada-002 | Project requires Vertex AI; 768-dim is efficient |
| LLM | Gemini 1.5 Pro | Gemini 1.5 Flash | Better reasoning for clinical decisions |
| PDF Parser | PyMuPDF | PyPDF2, pdfplumber | Better table handling, page metadata |
| Chunking | Semantic + Page-aware | Fixed-size, sentence | Preserves clinical recommendations integrity |
| Session Memory | In-memory dict | Redis, SQLite | Acceptable for take-home; noted production path |
| Frontend | Vanilla HTML/JS | React, Vue | Minimal requirement; fast to implement |
| API Framework | FastAPI | Flask | Async support, auto-docs, Pydantic integration |

---

## Interview Discussion Points

### 1. "Why not fine-tune instead of RAG?"

RAG is preferred here because:
- **Updatability**: NG12 gets revised; RAG just needs re-ingestion
- **Auditability**: Can trace every answer to source chunks
- **Compliance**: Clinical decisions need citations; fine-tuning is a black box
- **Cost**: No training costs; faster iteration

### 2. "How would you evaluate this system?"

- **Retrieval Quality**: Recall@K for known symptom-to-guideline mappings
- **Assessment Accuracy**: Gold-standard patient cases with expected outcomes
- **Chat Groundedness**: Human evaluation of citation accuracy
- **Latency**: P95 response time targets

### 3. "What are the failure modes?"

- **Retrieval Miss**: Relevant chunk not in top-K → increase K, improve chunking
- **Hallucination**: LLM ignores retrieved context → stronger grounding prompts
- **Citation Mismatch**: Wrong page cited → improve metadata pipeline
- **Session Confusion**: Wrong context in multi-turn → better session isolation

### 4. "How would you scale this?"

- Vector DB: Move to managed service (Pinecone, Vertex AI Matching Engine)
- Sessions: Redis cluster with TTL
- LLM: Request batching, caching common queries
- API: Kubernetes horizontal pod autoscaling

---

## Next Steps

1. ✅ Architecture documented
2. ⬜ Implement PDF ingestion pipeline
3. ⬜ Set up ChromaDB and embeddings
4. ⬜ Build retrieval service
5. ⬜ Implement risk assessment agent
6. ⬜ Implement chat agent
7. ⬜ Create FastAPI endpoints
8. ⬜ Build minimal frontend
9. ⬜ Dockerize
10. ⬜ Write tests
11. ⬜ Document prompts
