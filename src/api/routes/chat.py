"""
Chat API Routes

Endpoints for conversational clinical assessment.

Routes:
- POST /chat - Send message in conversation
- GET /chat/{session_id} - Get conversation history
- DELETE /chat/{session_id} - End conversation
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from src.api.models.requests import ChatRequest
from src.api.models.responses import (
    ChatResponse,
    AssessmentResponse,
    ErrorResponse,
)
from src.agents.clinical_agent import (
    ClinicalAgent,
    ConversationContext,
    ConversationState,
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
    summary="Send chat message",
    description="""
    Send a message in a clinical assessment conversation.
    
    The agent will:
    1. Process your message
    2. Extract relevant clinical information
    3. Ask clarifying questions if needed
    4. Perform assessment when sufficient information is gathered
    
    **Starting a new conversation:**
    - Omit session_id or pass null
    - Agent will greet and guide information gathering
    
    **Continuing conversation:**
    - Include session_id from previous response
    - Conversation maintains context across messages
    
    **Conversation states:**
    - greeting: Initial state
    - gathering: Collecting patient information
    - clarifying: Asking follow-up questions
    - assessing: Performing assessment
    - complete: Assessment finished
    """
)
async def chat(
    request: ChatRequest,
    agent: ClinicalAgent = Depends(get_agent),
) -> ChatResponse:
    """Process a chat message."""
    
    try:
        # Get or create session
        if request.session_id:
            context = agent.get_session(request.session_id)
            if not context:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {request.session_id} not found"
                )
        else:
            # Start new session
            context = agent.start_session()
            logger.info(f"Started new session: {context.session_id}")
        
        # Process message
        response_text = agent.chat(context, request.message)
        
        # Build response
        response = ChatResponse(
            session_id=context.session_id,
            message=response_text,
            state=context.state.value,
            assessment_complete=context.state == ConversationState.COMPLETE,
            gathered_info=context.info_gathered,
        )
        
        # Include assessment if complete
        if context.state == ConversationState.COMPLETE and context.assessment_result:
            result = context.assessment_result
            
            # Extract reasoning from trace if available
            reasoning = ""
            if result.reasoning_trace:
                if hasattr(result.reasoning_trace, 'get_reasoning_chain'):
                    reasoning = result.reasoning_trace.get_reasoning_chain()
                else:
                    reasoning = f"Completed in {result.steps_taken} reasoning steps"
            
            response.assessment = AssessmentResponse(
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
                model_used=f"ReAct Agent ({result.steps_taken} steps)",
            )
        
        return response
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get(
    "/{session_id}",
    response_model=ChatResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
    summary="Get conversation state",
    description="""
    Get the current state of a conversation.
    
    Returns:
    - Current conversation state
    - What information has been gathered
    - Assessment if complete
    """
)
async def get_conversation(
    session_id: str,
    agent: ClinicalAgent = Depends(get_agent),
) -> ChatResponse:
    """Get conversation state."""
    
    context = agent.get_session(session_id)
    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    # Get last assistant message
    last_message = ""
    for turn in reversed(context.history):
        if turn.role == "assistant":
            last_message = turn.content
            break
    
    response = ChatResponse(
        session_id=context.session_id,
        message=last_message,
        state=context.state.value,
        assessment_complete=context.state == ConversationState.COMPLETE,
        gathered_info=context.info_gathered,
    )
    
    # Include assessment if complete
    if context.state == ConversationState.COMPLETE and context.assessment_result:
        result = context.assessment_result
        response.assessment = AssessmentResponse(
            assessment_id=result.assessment_id,
            risk_level=result.risk_level.value,
            urgency=result.urgency.value,
            urgency_display=result.get_urgency_display(),
            summary=result.summary,
            recommended_actions=result.recommended_actions,
            investigations=result.investigations,
            referral_pathway=result.referral_pathway,
            reasoning=result.reasoning,
            citations=result.citations,
            matched_criteria=result.matched_criteria,
            differential_considerations=result.differential_considerations,
            red_flags=result.red_flags,
            confidence=result.confidence,
            timestamp=result.timestamp,
            model_used=result.model_used,
        )
    
    return response


@router.delete(
    "/{session_id}",
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
    summary="End conversation",
    description="End and clean up a conversation session."
)
async def end_conversation(
    session_id: str,
    agent: ClinicalAgent = Depends(get_agent),
):
    """End a conversation session."""
    
    context = agent.get_session(session_id)
    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    # Remove session
    if hasattr(agent, '_sessions') and session_id in agent._sessions:
        del agent._sessions[session_id]
    
    logger.info(f"Ended session: {session_id}")
    
    return {"message": f"Session {session_id} ended"}


@router.post(
    "/start",
    response_model=ChatResponse,
    summary="Start new conversation",
    description="""
    Explicitly start a new conversation session.
    
    Alternative to POST /chat without session_id.
    Returns the initial greeting and session ID.
    """
)
async def start_conversation(
    agent: ClinicalAgent = Depends(get_agent),
) -> ChatResponse:
    """Start a new conversation."""
    
    context = agent.start_session()
    
    # Get greeting message
    greeting = context.history[0].content if context.history else ""
    
    return ChatResponse(
        session_id=context.session_id,
        message=greeting,
        state=context.state.value,
        assessment_complete=False,
        gathered_info=context.info_gathered,
    )
