"""
API Response Models

Pydantic models for API responses.
Provides consistent response structure across all endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ClinicalMetadataResponse(BaseModel):
    """Clinical metadata for a guideline chunk."""
    urgency: str = ""
    cancer_types: List[str] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)
    investigations: List[str] = Field(default_factory=list)
    age_thresholds: List[str] = Field(default_factory=list)


class GuidelineChunkResponse(BaseModel):
    """A retrieved guideline chunk."""
    chunk_id: str
    text: str
    page: int
    section: str = ""
    citation: str = ""
    score: float = 0.0
    clinical_metadata: ClinicalMetadataResponse = Field(default_factory=ClinicalMetadataResponse)


class AssessmentResponse(BaseModel):
    """
    Full clinical assessment response.
    
    Returned by: POST /assess, POST /assess/quick
    """
    assessment_id: str = Field(
        ...,
        description="Unique assessment identifier"
    )
    risk_level: str = Field(
        ...,
        description="Risk level: critical, high, moderate, low, insufficient_info"
    )
    urgency: str = Field(
        ...,
        description="Urgency level for action"
    )
    urgency_display: str = Field(
        default="",
        description="Human-readable urgency with emoji"
    )
    summary: str = Field(
        ...,
        description="Brief assessment summary"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="List of recommended actions"
    )
    investigations: List[str] = Field(
        default_factory=list,
        description="Recommended investigations/tests"
    )
    referral_pathway: Optional[str] = Field(
        default=None,
        description="Specific referral pathway if applicable"
    )
    reasoning: str = Field(
        default="",
        description="Clinical reasoning explanation"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="NG12 guideline citations"
    )
    matched_criteria: List[str] = Field(
        default_factory=list,
        description="NG12 criteria that were matched"
    )
    differential_considerations: List[str] = Field(
        default_factory=list,
        description="Possible diagnoses to consider"
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="Identified red flags"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score 0-1"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Assessment timestamp"
    )
    model_used: str = Field(
        default="",
        description="LLM model used"
    )
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "examples": [
                {
                    "assessment_id": "a1b2c3d4",
                    "risk_level": "high",
                    "urgency": "urgent_2_week",
                    "urgency_display": "🔴 2-week suspected cancer pathway",
                    "summary": "Patient meets criteria for urgent lung cancer referral",
                    "recommended_actions": [
                        "Refer using suspected cancer pathway",
                        "Request chest X-ray within 2 weeks"
                    ],
                    "investigations": ["Chest X-ray", "Full blood count"],
                    "referral_pathway": "Suspected cancer pathway - lung",
                    "citations": ["[NG12 Section 1.1.1, p.10]"],
                    "red_flags": ["Haemoptysis in smoker over 40"],
                    "confidence": 0.85,
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """
    Chat response for conversational assessment.
    
    Returned by: POST /chat
    """
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    message: str = Field(
        ...,
        description="Agent's response message"
    )
    state: str = Field(
        ...,
        description="Current conversation state"
    )
    assessment_complete: bool = Field(
        default=False,
        description="Whether assessment is complete"
    )
    assessment: Optional[AssessmentResponse] = Field(
        default=None,
        description="Assessment result if complete"
    )
    gathered_info: Dict[str, bool] = Field(
        default_factory=dict,
        description="What information has been gathered"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "sess_abc123",
                    "message": "Thank you. What is the patient's age?",
                    "state": "gathering",
                    "assessment_complete": False,
                    "gathered_info": {
                        "symptoms": True,
                        "age": False,
                        "duration": False,
                        "risk_factors": False
                    }
                }
            ]
        }
    }


class SearchResponse(BaseModel):
    """
    Search response for guideline retrieval.
    
    Returned by: POST /search
    """
    query: str = Field(
        ...,
        description="Original search query"
    )
    expanded_query: Optional[str] = Field(
        default=None,
        description="Expanded query with synonyms"
    )
    total_results: int = Field(
        ...,
        description="Number of results returned"
    )
    results: List[GuidelineChunkResponse] = Field(
        default_factory=list,
        description="Retrieved guideline chunks"
    )
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters that were applied"
    )


class ExtractedSymptomsResponse(BaseModel):
    """
    Response from symptom extraction.
    
    Returned by: POST /extract
    """
    symptoms: List[str] = Field(
        default_factory=list,
        description="Extracted symptoms"
    )
    duration: Optional[str] = Field(
        default=None,
        description="Extracted symptom duration"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Extracted risk factors"
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="Identified red flags"
    )
    age: Optional[int] = Field(
        default=None,
        description="Age if mentioned"
    )
    sex: Optional[str] = Field(
        default=None,
        description="Sex if mentioned"
    )
    raw_text: str = Field(
        default="",
        description="Original input text"
    )


class HealthResponse(BaseModel):
    """
    Health check response.
    
    Returned by: GET /health
    """
    status: str = Field(
        ...,
        description="Service status: healthy, degraded, unhealthy"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Component health status"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


class StatsResponse(BaseModel):
    """
    System statistics response.
    
    Returned by: GET /stats
    """
    total_chunks: int = Field(
        default=0,
        description="Total guideline chunks indexed"
    )
    urgent_chunks: int = Field(
        default=0,
        description="Number of urgent recommendation chunks"
    )
    cancer_types: List[str] = Field(
        default_factory=list,
        description="Cancer types covered"
    )
    urgency_levels: List[str] = Field(
        default_factory=list,
        description="Urgency levels present"
    )
    model_info: Dict[str, str] = Field(
        default_factory=dict,
        description="LLM and embedding model info"
    )
    
    model_config = {
        "protected_namespaces": ()
    }


class ErrorResponse(BaseModel):
    """
    Standard error response.
    
    Returned on any error.
    """
    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ValidationError",
                    "message": "Invalid patient age",
                    "detail": "Age must be between 0 and 120",
                }
            ]
        }
    }
