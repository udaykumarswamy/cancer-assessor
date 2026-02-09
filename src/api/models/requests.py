"""
API Request Models

Pydantic models for API request validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class PatientRequest(BaseModel):
    """
    Patient information for assessment.
    
    Used in: POST /assess, POST /chat
    """
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional patient identifier for tracking"
    )
    age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Patient age in years"
    )
    sex: Optional[str] = Field(
        default=None,
        description="Patient sex (male/female/other)"
    )
    presenting_complaint: Optional[str] = Field(
        default=None,
        description="Main reason for consultation"
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="List of symptoms"
    )
    symptom_duration: Optional[str] = Field(
        default=None,
        description="Duration of symptoms (e.g., '2 weeks', '3 months')"
    )
    medical_history: List[str] = Field(
        default_factory=list,
        description="Relevant medical history"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Known risk factors (smoking, family history, etc.)"
    )
    family_history: List[str] = Field(
        default_factory=list,
        description="Relevant family history of cancer"
    )
    medications: List[str] = Field(
        default_factory=list,
        description="Current medications"
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional clinical notes"
    )
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v is not None:
            v_lower = v.lower()
            if v_lower not in ['male', 'female', 'other', 'm', 'f']:
                raise ValueError("Sex must be male, female, or other")
            return v_lower
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 55,
                    "sex": "male",
                    "symptoms": ["persistent cough", "weight loss", "fatigue"],
                    "symptom_duration": "3 weeks",
                    "risk_factors": ["smoker", "family history of lung cancer"],
                }
            ]
        }
    }


class AssessmentRequest(BaseModel):
    """
    Full assessment request.
    
    Used in: POST /assess
    """
    patient: PatientRequest
    include_reasoning: bool = Field(
        default=True,
        description="Include detailed reasoning in response"
    )
    include_differential: bool = Field(
        default=True,
        description="Include differential considerations"
    )


class QuickAssessmentRequest(BaseModel):
    """
    Quick assessment from symptoms only.
    
    Used in: POST /assess/quick
    """
    symptoms: List[str] = Field(
        ...,
        min_length=1,
        description="List of symptoms (at least one required)"
    )
    age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Patient age if known"
    )
    sex: Optional[str] = Field(
        default=None,
        description="Patient sex if known"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symptoms": ["haemoptysis", "weight loss"],
                    "age": 50
                }
            ]
        }
    }


class ChatRequest(BaseModel):
    """
    Chat message request.
    
    Used in: POST /chat
    """
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for continuing conversation (omit to start new)"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User message"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "abc123",
                    "message": "I have a 55 year old male patient with persistent cough"
                }
            ]
        }
    }


class SearchRequest(BaseModel):
    """
    Search request for guidelines.
    
    Used in: POST /search
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    cancer_type: Optional[str] = Field(
        default=None,
        description="Filter by cancer type"
    )
    urgent_only: bool = Field(
        default=False,
        description="Only return urgent recommendations"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "lung cancer symptoms over 40",
                    "top_k": 5,
                    "cancer_type": "lung"
                }
            ]
        }
    }


class ExtractSymptomsRequest(BaseModel):
    """
    Request to extract symptoms from free text.
    
    Used in: POST /extract
    """
    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Free text clinical notes"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "55yo male presents with 3-week history of persistent cough, unintentional weight loss of 5kg, and increasing fatigue. Former smoker (30 pack years)."
                }
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """
    Feedback on assessment quality.
    
    Used in: POST /feedback
    """
    assessment_id: str = Field(
        ...,
        description="Assessment ID being rated"
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating 1-5 (5 = most helpful)"
    )
    feedback_type: str = Field(
        default="general",
        description="Type: general, accuracy, relevance, clarity"
    )
    comments: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Additional comments"
    )
