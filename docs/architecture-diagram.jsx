import React, { useState } from 'react';

const ArchitectureDiagram = () => {
  const [activeLayer, setActiveLayer] = useState(null);
  const [activeFlow, setActiveFlow] = useState('assessment');
  const [hoveredTool, setHoveredTool] = useState(null);
  const [showReactLoop, setShowReactLoop] = useState(false);

  const layers = {
    client: {
      title: "Client Layer",
      color: "#6366F1",
      icon: "🖥",
      description: "Minimal HTML/JS frontend with two tabs: Risk Assessment (patient ID input + structured results with reasoning traces) and Chat (conversational interface with citation links).",
      components: ["Risk Assessment Tab", "Chat Interface Tab"]
    },
    api: {
      title: "API Layer — FastAPI",
      color: "#818CF8",
      icon: "⚡",
      description: "RESTful endpoints with async support, automatic OpenAPI docs, Pydantic validation, and dependency injection for shared components (vector store, embedder, agent instances).",
      components: ["POST /assess/{patient_id}", "POST /chat", "GET /search", "GET /system"]
    },
    agents: {
      title: "Agent Layer — ReAct Pattern",
      color: "#EC4899",
      icon: "🧠",
      description: "ReAct agent (Thought → Action → Observation loop) orchestrates 7 clinical tools. Provides explicit reasoning traces for clinical auditability. Context question detector classifies input as info-gathering, follow-up, or assessment request.",
      components: ["ReAct Agent", "Clinical Agent", "Context Question Detector", "Chat Agent"]
    },
    tools: {
      title: "Tool Layer — 7 Clinical Tools",
      color: "#F59E0B",
      icon: "🔧",
      description: "Specialised tools called by the ReAct agent. 5 use the shared RAG layer, 1 uses LLM-based extraction (handles negations), 1 uses deterministic rules.",
      components: [
        "search_guidelines",
        "check_red_flags",
        "calculate_risk",
        "get_referral_pathway",
        "extract_symptoms",
        "lookup_cancer_criteria",
        "get_section"
      ]
    },
    retrieval: {
      title: "Shared RAG Layer — ClinicalRetriever",
      color: "#14B8A6",
      icon: "🔍",
      description: "Query expansion with clinical synonyms (haemoptysis → coughing blood), patient-context-aware retrieval (age, symptoms, cancer type filters), metadata-enhanced embeddings, and three retrieval paths: semantic, patient-context, and section-based.",
      components: ["Query Expansion", "Patient-Context Search", "Metadata Filtering", "Section Lookup"]
    },
    knowledge: {
      title: "Knowledge Layer",
      color: "#10B981",
      icon: "📚",
      description: "ChromaDB with clinical-specific search methods (search_by_cancer_type, search_urgent_only, search_by_section). Chunks carry rich metadata: cancer_types, symptoms, urgency, section, type. Vertex AI text-embedding-004 with metadata-enhanced embeddings.",
      components: ["ChromaDB Vector Store", "Vertex AI Embeddings", "NG12 Chunks + Metadata"]
    },
    llm: {
      title: "LLM Layer — Gemini 2.5 Pro",
      color: "#EF4444",
      icon: "🤖",
      description: "Google Vertex AI Gemini 2.5 Pro for ReAct orchestration, function calling across 7 tools, structured JSON output for assessments, patient info extraction with negation handling, and grounded conversational generation with citations.",
      components: ["ReAct Orchestration", "Function Calling", "Structured Output", "Negation-Aware Extraction"]
    },
    evaluation: {
      title: "Evaluation Layer",
      color: "#A855F7",
      icon: "📊",
      description: "Automated retrieval quality metrics with metadata-driven ground truth (no manual labeling). Graded relevance (0–3) auto-generated from chunk metadata. Quality thresholds trigger test failures if retrieval degrades.",
      components: ["Recall@K", "Precision@K", "MRR", "NDCG@K", "MAP", "Hit Rate@K", "Ground Truth Builder"]
    }
  };

  const toolDetails = {
    search_guidelines: { desc: "Semantic search over NG12 chunks", usesRag: true, icon: "🔎" },
    check_red_flags: { desc: "Identify urgent clinical red flags", usesRag: true, icon: "🚩" },
    calculate_risk: { desc: "Deterministic risk scoring", usesRag: false, icon: "📊" },
    get_referral_pathway: { desc: "Look up correct referral route", usesRag: true, icon: "🏥" },
    extract_symptoms: { desc: "LLM-based extraction (handles negations)", usesRag: false, icon: "💊" },
    lookup_cancer_criteria: { desc: "Criteria for specific cancer type", usesRag: true, icon: "📋" },
    get_section: { desc: "Retrieve specific NG12 section", usesRag: true, icon: "📄" }
  };

  const assessmentFlow = [
    { from: "client", label: "Patient ID", detail: "PT-101" },
    { from: "api", label: "Orchestrate", detail: "dependencies.py injects agent" },
    { from: "agents", label: "ReAct Loop", detail: "Thought → Action → Observation" },
    { from: "tools", label: "Tool Calls", detail: "extract_symptoms → check_red_flags → search_guidelines" },
    { from: "retrieval", label: "RAG Query", detail: "Patient-context search with cancer type filter" },
    { from: "knowledge", label: "Retrieve", detail: "Metadata-filtered chunks + scores" },
    { from: "llm", label: "Synthesize", detail: "Structured JSON + reasoning trace + citations" },
  ];

  const chatFlow = [
    { from: "client", label: "Message + Session", detail: "User query with session context" },
    { from: "api", label: "Classify Input", detail: "Info gathering? Follow-up? Assessment?" },
    { from: "agents", label: "Context Detection", detail: "Skip RAG if simple follow-up" },
    { from: "retrieval", label: "RAG Query", detail: "Query expansion + semantic search" },
    { from: "knowledge", label: "Retrieve", detail: "Top-K chunks with citations" },
    { from: "llm", label: "Generate", detail: "Grounded response with page references" },
  ];

  const reactSteps = [
    { step: "THOUGHT", text: "I need to get the patient's symptoms and check for red flags.", color: "#818CF8" },
    { step: "ACTION", text: "extract_symptoms(patient_id=\"PT-101\")", color: "#F59E0B" },
    { step: "OBSERVATION", text: "Symptoms: [hemoptysis, fatigue]. Negated: [chest pain]. Age: 55.", color: "#10B981" },
    { step: "THOUGHT", text: "Hemoptysis is a red flag. Let me check NG12 criteria.", color: "#818CF8" },
    { step: "ACTION", text: "check_red_flags(symptoms=[\"hemoptysis\"], age=55)", color: "#F59E0B" },
    { step: "OBSERVATION", text: "🚩 RED FLAG — hemoptysis triggers urgent lung cancer referral.", color: "#10B981" },
    { step: "THOUGHT", text: "Confirmed. Retrieving specific referral pathway.", color: "#818CF8" },
    { step: "ACTION", text: "get_referral_pathway(cancer_type=\"lung\", urgency=\"2ww\")", color: "#F59E0B" },
    { step: "FINAL", text: "Risk: HIGH. 2-week-wait CXR referral. [NG12 §1.8, p.15]", color: "#EF4444" },
  ];

  const activeFlowSteps = activeFlow === 'assessment' ? assessmentFlow : chatFlow;

  return (
    <div style={{ fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace", background: '#0B0F1A' }} className="min-h-screen p-4 md:p-8">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        
        .layer-card {
          backdrop-filter: blur(12px);
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          border: 1px solid rgba(255,255,255,0.06);
        }
        .layer-card:hover {
          transform: translateY(-2px);
          border-color: rgba(255,255,255,0.15);
        }
        .layer-card.active {
          transform: translateY(-2px);
          box-shadow: 0 0 30px rgba(255,255,255,0.05);
        }
        .flow-step {
          animation: fadeSlideIn 0.3s ease forwards;
          opacity: 0;
        }
        @keyframes fadeSlideIn {
          to { opacity: 1; transform: translateX(0); }
          from { opacity: 0; transform: translateX(-8px); }
        }
        .react-step {
          animation: stepReveal 0.4s ease forwards;
          opacity: 0;
        }
        @keyframes stepReveal {
          to { opacity: 1; transform: translateY(0); }
          from { opacity: 0; transform: translateY(6px); }
        }
        .tool-chip {
          transition: all 0.2s ease;
          cursor: default;
        }
        .tool-chip:hover {
          transform: scale(1.05);
        }
        .glow-line {
          height: 1px;
          background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
        }
        .pulse-dot {
          animation: pulse 2s infinite;
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        .section-title {
          font-family: 'Space Grotesk', sans-serif;
        }
      `}</style>

      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-2 h-2 rounded-full bg-green-400 pulse-dot"></div>
            <span style={{ color: '#4ADE80', fontSize: '11px', letterSpacing: '2px', textTransform: 'uppercase' }}>System Architecture</span>
          </div>
          <h1 className="section-title" style={{ fontSize: '2rem', fontWeight: 700, color: '#E2E8F0', letterSpacing: '-0.02em', lineHeight: 1.1 }}>
            NG12 Cancer Assessor
          </h1>
          <p style={{ color: '#64748B', fontSize: '13px', marginTop: '8px', maxWidth: '600px', lineHeight: 1.6 }}>
            ReAct-based clinical decision support with 7 specialised tools, metadata-enhanced RAG, and automated evaluation. Click layers to explore.
          </p>
        </div>

        {/* Flow Toggle */}
        <div className="flex gap-2 mb-6">
          {[
            { id: 'assessment', label: '📋 Risk Assessment Flow', color: '#6366F1' },
            { id: 'chat', label: '💬 Chat Flow', color: '#A855F7' },
            { id: 'react', label: '🧠 ReAct Loop', color: '#EC4899' }
          ].map(f => (
            <button
              key={f.id}
              onClick={() => {
                if (f.id === 'react') {
                  setShowReactLoop(!showReactLoop);
                  setActiveFlow('assessment');
                } else {
                  setActiveFlow(f.id);
                  setShowReactLoop(false);
                }
              }}
              style={{
                background: (activeFlow === f.id || (f.id === 'react' && showReactLoop))
                  ? f.color + '25' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${(activeFlow === f.id || (f.id === 'react' && showReactLoop)) ? f.color + '60' : 'rgba(255,255,255,0.06)'}`,
                color: (activeFlow === f.id || (f.id === 'react' && showReactLoop)) ? '#E2E8F0' : '#64748B',
                fontSize: '12px',
                padding: '8px 16px',
                borderRadius: '8px',
                cursor: 'pointer',
                fontFamily: 'inherit',
                transition: 'all 0.2s ease',
              }}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* ReAct Loop Visualization */}
        {showReactLoop && (
          <div style={{ background: 'rgba(236,72,153,0.05)', border: '1px solid rgba(236,72,153,0.15)', borderRadius: '12px', padding: '20px', marginBottom: '24px' }}>
            <div className="flex items-center gap-2 mb-4">
              <span style={{ color: '#EC4899', fontSize: '13px', fontWeight: 600 }}>ReAct Agent Loop — Patient PT-101</span>
              <span style={{ color: '#64748B', fontSize: '11px' }}>Thought → Action → Observation (repeats until final answer)</span>
            </div>
            <div className="space-y-2">
              {reactSteps.map((s, i) => (
                <div
                  key={i}
                  className="react-step flex items-start gap-3"
                  style={{ animationDelay: `${i * 0.08}s` }}
                >
                  <span style={{
                    color: s.color,
                    fontSize: '10px',
                    fontWeight: 700,
                    letterSpacing: '1px',
                    minWidth: '100px',
                    paddingTop: '2px',
                  }}>
                    {s.step}
                  </span>
                  <span style={{
                    color: s.step === 'FINAL' ? '#F9A8D4' : '#94A3B8',
                    fontSize: '12px',
                    lineHeight: 1.5,
                    fontWeight: s.step === 'FINAL' ? 500 : 400,
                  }}>
                    {s.text}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Architecture Grid */}
        <div className="space-y-3 mb-6">
          {/* Client */}
          <LayerCard layer={layers.client} id="client" activeLayer={activeLayer} setActiveLayer={setActiveLayer} />

          {/* API */}
          <LayerCard layer={layers.api} id="api" activeLayer={activeLayer} setActiveLayer={setActiveLayer} />

          {/* Agent + LLM side by side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <LayerCard layer={layers.agents} id="agents" activeLayer={activeLayer} setActiveLayer={setActiveLayer} />
            <LayerCard layer={layers.llm} id="llm" activeLayer={activeLayer} setActiveLayer={setActiveLayer} />
          </div>

          {/* Tools - expanded view */}
          <div
            className={`layer-card rounded-xl p-4 cursor-pointer ${activeLayer === 'tools' ? 'active' : ''}`}
            style={{
              background: activeLayer === 'tools' ? 'rgba(245,158,11,0.08)' : 'rgba(255,255,255,0.02)',
              borderColor: activeLayer === 'tools' ? '#F59E0B40' : undefined,
            }}
            onClick={() => setActiveLayer(activeLayer === 'tools' ? null : 'tools')}
          >
            <div className="flex items-center gap-2 mb-3">
              <span style={{ fontSize: '14px' }}>{layers.tools.icon}</span>
              <span style={{ color: layers.tools.color, fontSize: '13px', fontWeight: 600 }}>{layers.tools.title}</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
              {Object.entries(toolDetails).map(([name, detail]) => (
                <div
                  key={name}
                  className="tool-chip"
                  onMouseEnter={() => setHoveredTool(name)}
                  onMouseLeave={() => setHoveredTool(null)}
                  style={{
                    background: hoveredTool === name ? 'rgba(245,158,11,0.15)' : 'rgba(255,255,255,0.03)',
                    border: `1px solid ${hoveredTool === name ? '#F59E0B40' : 'rgba(255,255,255,0.04)'}`,
                    borderRadius: '8px',
                    padding: '10px',
                  }}
                >
                  <div className="flex items-center gap-1.5 mb-1">
                    <span style={{ fontSize: '12px' }}>{detail.icon}</span>
                    <span style={{ color: '#E2E8F0', fontSize: '11px', fontWeight: 500 }}>{name}</span>
                  </div>
                  <div style={{ color: '#64748B', fontSize: '10px', lineHeight: 1.4 }}>{detail.desc}</div>
                  <div className="mt-1.5">
                    <span style={{
                      fontSize: '9px',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      background: detail.usesRag ? 'rgba(20,184,166,0.15)' : 'rgba(239,68,68,0.12)',
                      color: detail.usesRag ? '#5EEAD4' : '#FCA5A5',
                      letterSpacing: '0.5px',
                    }}>
                      {detail.usesRag ? 'USES RAG' : 'NO RAG'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Retrieval */}
          <LayerCard layer={layers.retrieval} id="retrieval" activeLayer={activeLayer} setActiveLayer={setActiveLayer} />

          {/* Knowledge */}
          <LayerCard layer={layers.knowledge} id="knowledge" activeLayer={activeLayer} setActiveLayer={setActiveLayer} />

          <div className="glow-line w-full my-2"></div>

          {/* Evaluation */}
          <LayerCard layer={layers.evaluation} id="evaluation" activeLayer={activeLayer} setActiveLayer={setActiveLayer} />
        </div>

        {/* Active Layer Detail Panel */}
        {activeLayer && (
          <div style={{
            background: layers[activeLayer].color + '0A',
            border: `1px solid ${layers[activeLayer].color}25`,
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '24px',
          }}>
            <div className="flex items-center gap-2 mb-2">
              <span style={{ fontSize: '16px' }}>{layers[activeLayer].icon}</span>
              <span style={{ color: layers[activeLayer].color, fontSize: '14px', fontWeight: 600 }} className="section-title">
                {layers[activeLayer].title}
              </span>
            </div>
            <p style={{ color: '#94A3B8', fontSize: '12px', lineHeight: 1.7 }}>
              {layers[activeLayer].description}
            </p>
          </div>
        )}

        {/* Data Flow */}
        {!showReactLoop && (
          <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '12px', padding: '20px', marginBottom: '24px' }}>
            <span style={{ color: '#94A3B8', fontSize: '12px', fontWeight: 600, letterSpacing: '1px', textTransform: 'uppercase' }}>
              {activeFlow === 'assessment' ? '📋 Risk Assessment' : '💬 Chat'} — Data Flow
            </span>
            <div className="mt-4 space-y-2">
              {activeFlowSteps.map((step, i) => (
                <div
                  key={`${activeFlow}-${i}`}
                  className="flow-step flex items-center gap-3"
                  style={{ animationDelay: `${i * 0.06}s` }}
                >
                  <div style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '6px',
                    background: layers[step.from].color + '20',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '11px',
                    color: layers[step.from].color,
                    fontWeight: 700,
                    flexShrink: 0,
                  }}>
                    {i + 1}
                  </div>
                  <div className="flex-1 flex items-center gap-2">
                    <span style={{ color: layers[step.from].color, fontSize: '12px', fontWeight: 600, minWidth: '110px' }}>
                      {step.label}
                    </span>
                    <span style={{ color: '#475569', fontSize: '11px' }}>→</span>
                    <span style={{ color: '#64748B', fontSize: '11px' }}>{step.detail}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Key Architecture Decisions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
          <div style={{ background: 'rgba(99,102,241,0.05)', border: '1px solid rgba(99,102,241,0.12)', borderRadius: '12px', padding: '16px' }}>
            <span style={{ color: '#818CF8', fontSize: '12px', fontWeight: 600 }}>🎯 Shared RAG Layer</span>
            <div className="mt-3 space-y-2">
              {[
                { label: 'Assessment', text: 'Patient-context search + cancer type filters' },
                { label: 'Chat', text: 'Semantic search + query expansion' },
                { label: 'Tools', text: '5 of 7 tools route through same retriever' },
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <span style={{ color: '#6366F1', fontSize: '10px', minWidth: '80px', fontWeight: 500 }}>{item.label}</span>
                  <span style={{ color: '#64748B', fontSize: '11px' }}>{item.text}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ background: 'rgba(236,72,153,0.05)', border: '1px solid rgba(236,72,153,0.12)', borderRadius: '12px', padding: '16px' }}>
            <span style={{ color: '#F472B6', fontSize: '12px', fontWeight: 600 }}>🧠 Why ReAct over Basic RAG?</span>
            <div className="mt-3 space-y-2">
              {[
                { label: 'Auditability', text: 'Explicit reasoning trace clinicians can follow' },
                { label: 'Multi-step', text: 'Iterative search — refine based on findings' },
                { label: 'Composition', text: 'Agent decides which tools to call and when' },
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <span style={{ color: '#EC4899', fontSize: '10px', minWidth: '80px', fontWeight: 500 }}>{item.label}</span>
                  <span style={{ color: '#64748B', fontSize: '11px' }}>{item.text}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Tech Stack */}
        <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '12px', padding: '16px' }}>
          <span style={{ color: '#64748B', fontSize: '11px', fontWeight: 600, letterSpacing: '1px', textTransform: 'uppercase' }}>Tech Stack</span>
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2 mt-3">
            {[
              { name: "FastAPI", role: "API", color: "#14B8A6" },
              { name: "Gemini 2.5 Pro", role: "LLM", color: "#EF4444" },
              { name: "ChromaDB", role: "Vector Store", color: "#10B981" },
              { name: "Vertex AI", role: "Embeddings", color: "#F59E0B" },
              { name: "PyMuPDF", role: "PDF", color: "#6366F1" },
              { name: "Pydantic", role: "Validation", color: "#818CF8" },
              { name: "Docker", role: "Container", color: "#3B82F6" },
              { name: "Python 3.11", role: "Runtime", color: "#A855F7" },
              { name: "ReAct", role: "Agent Pattern", color: "#EC4899" },
              { name: "NDCG/MRR", role: "Eval Metrics", color: "#F472B6" },
            ].map((tech, i) => (
              <div key={i} style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.04)',
                borderRadius: '8px',
                padding: '8px 10px',
                textAlign: 'center',
              }}>
                <div style={{ color: tech.color, fontSize: '12px', fontWeight: 600 }}>{tech.name}</div>
                <div style={{ color: '#475569', fontSize: '9px', marginTop: '2px', letterSpacing: '0.5px' }}>{tech.role}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const LayerCard = ({ layer, id, activeLayer, setActiveLayer }) => (
  <div
    className={`layer-card rounded-xl p-4 cursor-pointer ${activeLayer === id ? 'active' : ''}`}
    style={{
      background: activeLayer === id ? layer.color + '0A' : 'rgba(255,255,255,0.02)',
      borderColor: activeLayer === id ? layer.color + '30' : undefined,
    }}
    onClick={() => setActiveLayer(activeLayer === id ? null : id)}
  >
    <div className="flex items-center gap-2 mb-2">
      <span style={{ fontSize: '14px' }}>{layer.icon}</span>
      <span style={{ color: layer.color, fontSize: '13px', fontWeight: 600 }}>{layer.title}</span>
    </div>
    <div className="flex flex-wrap gap-1.5">
      {layer.components.map((comp, i) => (
        <span
          key={i}
          style={{
            fontSize: '10px',
            padding: '3px 8px',
            borderRadius: '5px',
            background: layer.color + '12',
            color: layer.color,
            border: `1px solid ${layer.color}20`,
            fontWeight: 500,
          }}
        >
          {comp}
        </span>
      ))}
    </div>
  </div>
);

export default ArchitectureDiagram;