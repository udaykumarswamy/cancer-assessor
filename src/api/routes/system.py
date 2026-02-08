"""
System API Routes

Health checks, statistics, and system information.

Routes:
- GET /health - Health check
- GET /stats - System statistics
- GET /patients - List test patients
- GET /patients/{id} - Get test patient
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
import json
from pathlib import Path

from src.api.models.responses import (
    HealthResponse,
    StatsResponse,
    ErrorResponse,
)
from src.config.logging_config import get_logger

logger = get_logger("api.system")

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy and all components are working."
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    
    components = {}
    overall_status = "healthy"
    
    # Check vector store
    try:
        from src.api.dependencies import get_retriever
        retriever = get_retriever()
        stats = retriever.get_stats()
        if stats.get("vector_store", {}).get("total_chunks", 0) > 0:
            components["vector_store"] = "healthy"
        else:
            components["vector_store"] = "empty"
            overall_status = "degraded"
    except Exception as e:
        components["vector_store"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
    
    # Check LLM availability
    try:
        from src.llm.gemini import VERTEX_AI_AVAILABLE
        if VERTEX_AI_AVAILABLE:
            components["llm"] = "healthy"
        else:
            components["llm"] = "mock_only"
            overall_status = "degraded"
    except Exception as e:
        components["llm"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
    
    # Check embedder
    try:
        from src.ingestion.embedder import VERTEX_AI_AVAILABLE as EMBED_AVAILABLE
        if EMBED_AVAILABLE:
            components["embedder"] = "healthy"
        else:
            components["embedder"] = "mock_only"
    except Exception as e:
        components["embedder"] = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components,
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System statistics",
    description="Get statistics about the indexed guidelines and system configuration."
)
async def get_stats() -> StatsResponse:
    """Get system statistics."""
    
    try:
        from src.api.dependencies import get_retriever
        retriever = get_retriever()
        stats = retriever.get_stats()
        
        vs_stats = stats.get("vector_store", {})
        
        return StatsResponse(
            total_chunks=vs_stats.get("total_chunks", 0),
            urgent_chunks=vs_stats.get("urgent_chunks", 0),
            cancer_types=vs_stats.get("cancer_types", []),
            urgency_levels=vs_stats.get("urgency_levels", []),
            model_info={
                "embedding_model": "text-embedding-004",
                "llm_model": "gemini-1.5-pro-002",
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/patients",
    summary="List test patients",
    description="Get list of test patient cases for demo/testing."
)
async def list_patients():
    """List available test patients."""
    
    try:
        # Try to load patients.json
        patients_path = Path(__file__).parent.parent.parent.parent / "data" / "patients.json"
        
        if not patients_path.exists():
            return {"patients": [], "message": "No test patients file found"}
        
        with open(patients_path) as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        patients_list = data if isinstance(data, list) else data.get("patients", [])
        
        # Return summary of each patient
        summaries = []
        for patient in patients_list:
            # Build symptoms string from symptoms list if present
            symptoms_str = ", ".join(patient.get("symptoms", [])) if patient.get("symptoms") else ""
            
            summaries.append({
                "id": patient.get("patient_id") or patient.get("id"),
                "name": patient.get("name", "Unknown"),
                "age": patient.get("age"),
                "presenting_complaint": symptoms_str[:100],
            })
        
        return {
            "patients": summaries,
            "total": len(summaries),
        }
        
    except Exception as e:
        logger.error(f"Failed to load patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/patients/{patient_id}",
    summary="Get test patient",
    description="Get full details of a test patient case."
)
async def get_patient(patient_id: str):
    """Get a specific test patient."""
    
    try:
        patients_path = Path(__file__).parent.parent.parent.parent / "data" / "patients.json"
        
        if not patients_path.exists():
            raise HTTPException(status_code=404, detail="Patients file not found")
        
        with open(patients_path) as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        patients_list = data if isinstance(data, list) else data.get("patients", [])
        
        for patient in patients_list:
            # Match by either 'id' or 'patient_id' field
            if patient.get("patient_id") == patient_id or patient.get("id") == patient_id:
                return patient
        
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Failed to load patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/",
    summary="API Info",
    description="Basic API information."
)
async def api_info():
    """API information."""
    return {
        "name": "NG12 Cancer Risk Assessment API",
        "version": "1.0.0",
        "description": "Clinical decision support for cancer referral based on NICE NG12 guidelines",
        "endpoints": {
            "assessment": "/assess - Full clinical assessment",
            "quick_assess": "/assess/quick - Quick symptom-based assessment",
            "chat": "/chat - Conversational assessment",
            "search": "/search - Search guidelines",
            "health": "/health - Health check",
            "docs": "/docs - API documentation",
        }
    }
