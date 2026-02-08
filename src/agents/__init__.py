"""
Agents Module

ReAct-based agentic framework for clinical assessment.

Components:
- tools.py: Tool definitions (search, extract, calculate)
- react_agent.py: Core ReAct reasoning loop
- clinical_agent.py: High-level clinical assessment interface
"""

from src.agents.tools import (
    Tool,
    ToolParameter,
    ToolResult,
    ClinicalTools,
)

from src.agents.react_agent import (
    ReActAgent,
    StreamingReActAgent,
    AgentResult,
    AgentTrace,
    ThoughtStep,
    AgentState,
)

from src.agents.clinical_agent import (
    ClinicalAgent,
    ConversationalClinicalAgent,
    ClinicalAssessment,
    PatientInfo,
    RiskLevel,
    UrgencyLevel,
    ConversationSession,
    ConversationContext,
    ConversationState,
    ConversationTurn,
    create_clinical_agent,
)

__all__ = [
    # Tools
    "Tool",
    "ToolParameter", 
    "ToolResult",
    "ClinicalTools",
    # ReAct Agent
    "ReActAgent",
    "StreamingReActAgent",
    "AgentResult",
    "AgentTrace",
    "ThoughtStep",
    "AgentState",
    # Clinical Agent
    "ClinicalAgent",
    "ConversationalClinicalAgent",
    "ClinicalAssessment",
    "PatientInfo",
    "RiskLevel",
    "UrgencyLevel",
    "ConversationSession",
    "ConversationContext",
    "ConversationState",
    "ConversationTurn",
    "create_clinical_agent",
]
