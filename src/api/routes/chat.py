"""
Chat API Routes — Guideline Q&A Mode

Endpoints for conversational Q&A over NICE NG12 guidelines.
Users ask questions like "What symptoms trigger urgent referral for lung cancer?"
and get grounded answers with citations.

This is SEPARATE from the Assessment tab which handles patient risk evaluation.

Routes:
- POST /chat       - Send a guideline question
- POST /chat/start - Explicitly start a new Q&A session
- GET  /chat/{session_id} - Get conversation history
- DELETE /chat/{session_id} - End conversation
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends

from src.api.models.requests import ChatRequest
from src.api.models.responses import (
    ChatResponse,
    ErrorResponse,
)
from src.agents.clinical_agent import (
    ClinicalAgent,
    GuidelineChatSession,
)
from src.config.logging_config import get_logger

logger = get_logger("api.chat")

router = APIRouter(prefix="/chat", tags=["Chat"])


# Dependency injection placeholder
def get_agent() -> ClinicalAgent:
    """Get clinical agent instance."""
    from src.api.dependencies import get_clinical_agent as _get_agent
    return _get_agent()


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Chat failed"},
    },
    summary="Ask a guideline question",
    description="""
    Ask a question about the NICE NG12 guidelines for suspected cancer 
    recognition and referral.
    
    The agent will:
    1. Retrieve relevant guideline passages from the NG12 knowledge base
    2. Generate a grounded answer using only retrieved evidence
    3. Cite specific guideline sections and page numbers
    4. Maintain conversation context for follow-up questions
    
    **Example questions:**
    - "What symptoms trigger an urgent referral for lung cancer?"
    - "Does persistent hoarseness require urgent referral, and at what age?"
    - "What does NG12 say about dyspepsia and thresholds for investigation?"
    - "Summarize the referral criteria for visible haematuria."
    
    **Starting a new conversation:**
    - Omit session_id or pass null
    
    **Follow-up questions:**
    - Include session_id from previous response
    - The agent uses conversation history for context
      (e.g., "What about for patients under 40?" after a lung cancer question)
    
    **For patient risk assessment**, use the /assess endpoints instead.
    """,
)
async def chat(
    request: ChatRequest,
    agent: ClinicalAgent = Depends(get_agent),
) -> ChatResponse:
    """Process a guideline Q&A message."""
    
    try:
        # Get or create guideline session
        if request.session_id:
            session = agent.get_guideline_session(request.session_id)
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {request.session_id} not found",
                )
        else:
            # Start new guideline Q&A session
            session = agent.start_guideline_session()
            logger.info(f"Started new guideline Q&A session: {session.session_id}")
        
        # Process message through guideline Q&A
        result = agent.chat_guideline(session, request.message)
        
        # Build response
        response = ChatResponse(
            session_id=session.session_id,
            message=result["message"],
            state="active",
            assessment_complete=False,
            gathered_info={},
        )
        
        # Add citations if the response model supports it
        if hasattr(response, 'citations'):
            response.citations = result.get("citations", [])
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Guideline chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post(
    "/start",
    response_model=ChatResponse,
    summary="Start new guideline Q&A session",
    description="""
    Explicitly start a new guideline Q&A conversation.
    Returns the initial greeting and session ID.
    
    Alternative to POST /chat without session_id.
    """,
)
async def start_conversation(
    agent: ClinicalAgent = Depends(get_agent),
) -> ChatResponse:
    """Start a new guideline Q&A session."""
    
    session = agent.start_guideline_session()
    
    # Get greeting message
    greeting = session.history[0].content if session.history else ""
    
    return ChatResponse(
        session_id=session.session_id,
        message=greeting,
        state="active",
        assessment_complete=False,
        gathered_info={},
    )


@router.get(
    "/{session_id}",
    response_model=ChatResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
    summary="Get conversation state",
    description="Get the current state of a guideline Q&A conversation.",
)
async def get_conversation(
    session_id: str,
    agent: ClinicalAgent = Depends(get_agent),
) -> ChatResponse:
    """Get guideline Q&A conversation state."""
    
    session = agent.get_guideline_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )
    
    # Get last assistant message
    last_message = ""
    last_citations = []
    for turn in reversed(session.history):
        if turn.role == "assistant":
            last_message = turn.content
            last_citations = turn.citations
            break
    
    response = ChatResponse(
        session_id=session.session_id,
        message=last_message,
        state="active",
        assessment_complete=False,
        gathered_info={},
    )
    
    if hasattr(response, 'citations'):
        response.citations = last_citations
    
    return response


@router.delete(
    "/{session_id}",
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
    summary="End conversation",
    description="End and clean up a guideline Q&A session.",
)
async def end_conversation(
    session_id: str,
    agent: ClinicalAgent = Depends(get_agent),
):
    """End a guideline Q&A session."""
    
    session = agent.get_guideline_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )
    
    # Remove session
    if hasattr(agent, '_guideline_sessions') and session_id in agent._guideline_sessions:
        del agent._guideline_sessions[session_id]
    
    logger.info(f"Ended guideline session: {session_id}")
    
    return {"message": f"Session {session_id} ended"}
