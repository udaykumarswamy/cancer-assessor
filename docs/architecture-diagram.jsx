import React, { useState } from 'react';

const ArchitectureDiagram = () => {
  const [activeLayer, setActiveLayer] = useState(null);
  const [activeFlow, setActiveFlow] = useState('assessment');
  
  const layers = {
    client: {
      title: "Client Layer",
      color: "#3B82F6",
      description: "Minimal UI with two tabs: Risk Assessment (patient ID input) and Chat (message window)",
      components: ["Risk Assessment Tab", "Chat Interface Tab"]
    },
    api: {
      title: "API Layer (FastAPI)",
      color: "#8B5CF6",
      description: "RESTful endpoints with async support, automatic OpenAPI docs, and Pydantic validation",
      components: ["POST /assess/{patient_id}", "POST /chat", "GET /chat/{session_id}/history"]
    },
    agents: {
      title: "Agent Layer",
      color: "#EC4899",
      description: "LLM-powered reasoning engines that orchestrate tools and RAG retrieval",
      components: ["RiskAssessmentAgent", "ChatAgent", "Shared RAG Layer"]
    },
    services: {
      title: "Services Layer",
      color: "#F59E0B",
      description: "Core business logic: retrieval, patient data, session management",
      components: ["RetrievalService", "PatientDataTool", "SessionManager"]
    },
    knowledge: {
      title: "Knowledge Layer",
      color: "#10B981",
      description: "Vector store with embedded NG12 chunks and metadata for citations",
      components: ["ChromaDB", "Vertex AI Embeddings", "PDF Ingestion Pipeline"]
    },
    llm: {
      title: "LLM Layer",
      color: "#EF4444",
      description: "Gemini 1.5 Pro for reasoning, function calling, and grounded generation",
      components: ["Function Calling", "Structured Output", "Conversational Generation"]
    }
  };

  const assessmentFlow = [
    { from: "client", to: "api", label: "Patient ID" },
    { from: "api", to: "agents", label: "Orchestrate" },
    { from: "agents", to: "services", label: "Tool Call" },
    { from: "services", to: "knowledge", label: "RAG Query" },
    { from: "agents", to: "llm", label: "Synthesize" },
    { from: "llm", to: "api", label: "JSON Response" }
  ];

  const chatFlow = [
    { from: "client", to: "api", label: "Message + Session ID" },
    { from: "api", to: "agents", label: "Load History" },
    { from: "agents", to: "knowledge", label: "RAG Query" },
    { from: "agents", to: "llm", label: "Generate" },
    { from: "llm", to: "api", label: "Grounded Response" }
  ];

  const LayerBox = ({ id, layer, index }) => (
    <div
      className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
        activeLayer === id 
          ? 'border-white shadow-lg scale-105' 
          : 'border-transparent hover:border-white/50'
      }`}
      style={{ backgroundColor: layer.color + '20', borderColor: activeLayer === id ? layer.color : undefined }}
      onClick={() => setActiveLayer(activeLayer === id ? null : id)}
    >
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: layer.color }}></div>
        <h3 className="font-semibold text-white">{layer.title}</h3>
      </div>
      <div className="flex flex-wrap gap-1">
        {layer.components.map((comp, i) => (
          <span 
            key={i} 
            className="text-xs px-2 py-1 rounded"
            style={{ backgroundColor: layer.color + '40', color: 'white' }}
          >
            {comp}
          </span>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-2">NG12 Cancer Assessor Architecture</h1>
        <p className="text-gray-400 mb-6">Click on layers to see details. Toggle between data flows.</p>
        
        {/* Flow Toggle */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveFlow('assessment')}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              activeFlow === 'assessment'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Risk Assessment Flow
          </button>
          <button
            onClick={() => setActiveFlow('chat')}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              activeFlow === 'chat'
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Chat Flow
          </button>
        </div>

        {/* Main Architecture Grid */}
        <div className="grid gap-4 mb-6">
          <LayerBox id="client" layer={layers.client} index={0} />
          <LayerBox id="api" layer={layers.api} index={1} />
          
          <div className="grid grid-cols-2 gap-4">
            <LayerBox id="agents" layer={layers.agents} index={2} />
            <LayerBox id="llm" layer={layers.llm} index={3} />
          </div>
          
          <LayerBox id="services" layer={layers.services} index={4} />
          <LayerBox id="knowledge" layer={layers.knowledge} index={5} />
        </div>

        {/* Active Layer Details */}
        {activeLayer && (
          <div 
            className="p-4 rounded-lg mb-6"
            style={{ backgroundColor: layers[activeLayer].color + '20' }}
          >
            <h3 className="font-semibold text-white mb-2">{layers[activeLayer].title}</h3>
            <p className="text-gray-300">{layers[activeLayer].description}</p>
          </div>
        )}

        {/* Data Flow Visualization */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-semibold mb-4">
            {activeFlow === 'assessment' ? '📋 Risk Assessment' : '💬 Chat'} Data Flow
          </h3>
          <div className="flex flex-wrap items-center gap-2">
            {(activeFlow === 'assessment' ? assessmentFlow : chatFlow).map((step, i) => (
              <React.Fragment key={i}>
                <div 
                  className="px-3 py-2 rounded-lg text-sm"
                  style={{ backgroundColor: layers[step.from].color + '40', color: 'white' }}
                >
                  {step.label}
                </div>
                {i < (activeFlow === 'assessment' ? assessmentFlow : chatFlow).length - 1 && (
                  <span className="text-gray-500">→</span>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Key Design Decisions */}
        <div className="mt-6 bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-semibold mb-4">🎯 Key Design Decision: Shared RAG Layer</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-blue-900/30 p-3 rounded-lg">
              <h4 className="text-blue-400 font-medium mb-2">Risk Assessment Uses RAG For:</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• Symptom → Guideline matching</li>
                <li>• Criteria verification</li>
                <li>• Citation extraction</li>
              </ul>
            </div>
            <div className="bg-purple-900/30 p-3 rounded-lg">
              <h4 className="text-purple-400 font-medium mb-2">Chat Uses Same RAG For:</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• Semantic search over guidelines</li>
                <li>• Grounded response generation</li>
                <li>• Citation linking</li>
              </ul>
            </div>
          </div>
          <p className="text-gray-400 text-sm mt-3">
            ✅ Same vector store, same retrieval service, same embeddings = consistent behavior & efficient resources
          </p>
        </div>

        {/* Tech Stack Summary */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { name: "FastAPI", role: "API Framework" },
            { name: "Gemini 1.5 Pro", role: "LLM" },
            { name: "ChromaDB", role: "Vector Store" },
            { name: "Vertex AI", role: "Embeddings" },
            { name: "PyMuPDF", role: "PDF Parsing" },
            { name: "Pydantic", role: "Validation" },
            { name: "Docker", role: "Containerization" },
            { name: "Python 3.11", role: "Runtime" }
          ].map((tech, i) => (
            <div key={i} className="bg-gray-800 p-3 rounded-lg text-center">
              <div className="text-white font-medium">{tech.name}</div>
              <div className="text-gray-500 text-xs">{tech.role}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ArchitectureDiagram;
