"""
API Models

Pydantic models for request/response validation.
"""

from src.api.models.requests import (
    PatientRequest,
    AssessmentRequest,
    QuickAssessmentRequest,
    ChatRequest,
    SearchRequest,
    ExtractSymptomsRequest,
    FeedbackRequest,
)

from src.api.models.responses import (
    AssessmentResponse,
    ChatResponse,
    SearchResponse,
    GuidelineChunkResponse,
    ClinicalMetadataResponse,
    ExtractedSymptomsResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
)

__all__ = [
    # Requests
    "PatientRequest",
    "AssessmentRequest",
    "QuickAssessmentRequest",
    "ChatRequest",
    "SearchRequest",
    "ExtractSymptomsRequest",
    "FeedbackRequest",
    # Responses
    "AssessmentResponse",
    "ChatResponse",
    "SearchResponse",
    "GuidelineChunkResponse",
    "ClinicalMetadataResponse",
    "ExtractedSymptomsResponse",
    "HealthResponse",
    "StatsResponse",
    "ErrorResponse",
]
