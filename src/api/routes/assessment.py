"""
Assessment API Routes (ReAct-based)

Endpoints for clinical risk assessment using agentic framework.

Routes:
- POST /assess - Full assessment with ReAct reasoning
- POST /assess/quick - Quick assessment from symptoms
- POST /assess/stream - Streaming assessment with real-time steps
- GET /assess/trace/{id} - Get reasoning trace for assessment
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import json

from src.api.models.requests import (
    AssessmentRequest,
    QuickAssessmentRequest,
    ExtractSymptomsRequest,
)
from src.api.models.responses import (
    AssessmentResponse,
    ExtractedSymptomsResponse,
    ErrorResponse,
)
from src.agents.clinical_agent import (
    ClinicalAgent,
    PatientInfo,
)
from src.config.logging_config import get_logger

logger = get_logger("api.assessment")

router = APIRouter(prefix="/assess", tags=["Assessment"])

# Store for reasoning traces (in production, use proper storage)
_trace_store: Dict[str, Any] = {}


def get_clinical_agent() -> ClinicalAgent:
    """Get clinical agent instance."""
    from src.api.dependencies import get_clinical_agent as _get_agent
    return _get_agent()


@router.post(
    "",
    response_model=AssessmentResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Assessment failed"},
    },
    summary="Full clinical assessment (ReAct-based)",
    description="""
    Perform a comprehensive clinical risk assessment using ReAct agentic framework.
    
    The agent:
    1. Reasons about the patient's symptoms and risk factors
    2. Uses tools to search NG12 guidelines
    3. Checks for red flags
    4. Calculates risk level
    5. Provides recommendations with citations
    
    Returns detailed assessment with full reasoning trace.
    
    **Key Feature:** The assessment includes a reasoning trace showing each step
    the agent took, making the decision process transparent and auditable.
    """
)
async def assess_patient(
    request: AssessmentRequest,
    agent: ClinicalAgent = Depends(get_clinical_agent),
) -> AssessmentResponse:
    """Perform full clinical assessment with ReAct reasoning."""
    
    logger.info(f"Assessment request for patient {request.patient.patient_id or 'anonymous'}")
    
    try:
        # Convert request to PatientInfo
        patient = PatientInfo(
            age=request.patient.age,
            sex=request.patient.sex,
            presenting_complaint=request.patient.presenting_complaint or "",
            symptoms=request.patient.symptoms,
            symptom_duration=request.patient.symptom_duration,
            medical_history=request.patient.medical_history,
            risk_factors=request.patient.risk_factors,
            family_history=request.patient.family_history,
            additional_notes=request.patient.additional_notes or "",
        )
        
        # Perform assessment using ReAct agent
        result = agent.assess(patient)
        
        # Store trace for later retrieval
        if result.reasoning_trace:
            _trace_store[result.assessment_id] = result.reasoning_trace.to_dict()
        
        # Build reasoning summary
        reasoning = ""
        if request.include_reasoning and result.reasoning_trace:
            steps = result.reasoning_trace.steps
            reasoning = f"Assessment completed in {len(steps)} reasoning steps.\n\n"
            for step in steps:
                reasoning += f"Step {step.step_number}: {step.thought[:100]}...\n"
                if step.action:
                    reasoning += f"  → Used tool: {step.action}\n"
        
        # Convert to response
        return AssessmentResponse(
            assessment_id=result.assessment_id,
            risk_level=result.risk_level.value,
            urgency=result.urgency.value,
            urgency_display=result.get_urgency_display(),
            summary=result.summary,
            recommended_actions=result.recommended_actions,
            investigations=result.investigations,
            referral_pathway=result.referral_pathway,
            reasoning=reasoning,
            citations=result.citations,
            matched_criteria=result.matched_criteria,
            differential_considerations=result.differential_considerations,
            red_flags=result.red_flags,
            confidence=result.confidence,
            timestamp=result.timestamp,
            model_used=f"ReAct Agent ({result.steps_taken} steps, tools: {', '.join(result.tools_used)})",
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@router.post(
    "/quick",
    response_model=AssessmentResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Assessment failed"},
    },
    summary="Quick assessment from symptoms",
    description="""
    Perform a quick clinical assessment based on symptoms alone.
    
    Uses the same ReAct framework but with minimal patient information.
    """
)
async def quick_assess(
    request: QuickAssessmentRequest,
    agent: ClinicalAgent = Depends(get_clinical_agent),
) -> AssessmentResponse:
    """Quick assessment from symptoms only."""
    
    logger.info(f"Quick assessment for symptoms: {request.symptoms[:3]}...")
    
    try:
        result = agent.assess_quick(
            symptoms=request.symptoms,
            age=request.age,
            sex=request.sex,
        )
        
        return AssessmentResponse(
            assessment_id=result.assessment_id,
            risk_level=result.risk_level.value,
            urgency=result.urgency.value,
            urgency_display=result.get_urgency_display(),
            summary=result.summary,
            recommended_actions=result.recommended_actions,
            investigations=result.investigations,
            referral_pathway=result.referral_pathway,
            reasoning="",
            citations=result.citations,
            matched_criteria=result.matched_criteria,
            red_flags=result.red_flags,
            confidence=result.confidence,
            timestamp=result.timestamp,
            model_used=f"ReAct Agent ({result.steps_taken} steps)",
        )
        
    except Exception as e:
        logger.error(f"Quick assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@router.post(
    "/stream",
    summary="Streaming assessment",
    description="""
    Perform assessment with real-time streaming of reasoning steps.
    
    Returns Server-Sent Events (SSE) with each step as it happens:
    - thinking: Agent is reasoning
    - thought: Agent's thought for this step
    - action: Tool being used
    - observation: Tool result
    - final_answer: Assessment complete
    
    Useful for showing progress in UI.
    """
)
async def stream_assess(
    request: AssessmentRequest,
    agent: ClinicalAgent = Depends(get_clinical_agent),
):
    """Stream assessment with real-time steps."""
    
    async def generate():
        patient = PatientInfo(
            age=request.patient.age,
            sex=request.patient.sex,
            presenting_complaint=request.patient.presenting_complaint or "",
            symptoms=request.patient.symptoms,
            symptom_duration=request.patient.symptom_duration,
            risk_factors=request.patient.risk_factors,
        )
        
        for step in agent.assess_streaming(patient):
            yield f"data: {json.dumps(step)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.get(
    "/trace/{assessment_id}",
    summary="Get reasoning trace",
    description="""
    Retrieve the full reasoning trace for a previous assessment.
    
    Shows each step the agent took, including:
    - Thoughts (reasoning)
    - Actions (tools used)
    - Observations (tool results)
    
    Useful for auditing and understanding how the assessment was made.
    """
)
async def get_trace(assessment_id: str):
    """Get reasoning trace for an assessment."""
    
    trace = _trace_store.get(assessment_id)
    if not trace:
        raise HTTPException(
            status_code=404,
            detail=f"Trace for assessment {assessment_id} not found"
        )
    
    return trace


@router.get(
    "/tools",
    summary="List available tools",
    description="Get list of tools available to the ReAct agent."
)
async def list_tools(
    agent: ClinicalAgent = Depends(get_clinical_agent),
):
    """List available agent tools."""
    
    tools = agent.get_available_tools()
    
    return {
        "tools": [
            {
                "name": t["name"],
                "description": t["description"][:200],
                "parameters": list(t["parameters"]["properties"].keys())
            }
            for t in tools
        ],
        "total": len(tools)
    }


@router.post(
    "/extract",
    response_model=ExtractedSymptomsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Extraction failed"},
    },
    summary="Extract symptoms from free text",
    description="""
    Extract structured symptom information from free-text clinical notes.
    
    Uses the agent's extract_symptoms tool to identify:
    - Symptoms mentioned
    - Duration of symptoms
    - Risk factors
    - Red flags
    - Patient demographics
    """
)
async def extract_symptoms(
    request: ExtractSymptomsRequest,
    agent: ClinicalAgent = Depends(get_clinical_agent),
) -> ExtractedSymptomsResponse:
    """Extract symptoms from free text."""
    
    logger.info(f"Extracting symptoms from text: {request.text[:50]}...")
    
    try:
        # Use the extract_symptoms tool directly
        result = agent.tools.execute_tool(
            "extract_symptoms",
            clinical_text=request.text
        )
        
        if not result.success:
            raise ValueError(result.error)
        
        data = result.result or {}
        
        return ExtractedSymptomsResponse(
            symptoms=data.get("symptoms", []),
            duration=data.get("duration"),
            risk_factors=data.get("risk_factors", []),
            red_flags=data.get("red_flags", []),
            age=data.get("age"),
            sex=data.get("sex"),
            raw_text=request.text,
        )
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
