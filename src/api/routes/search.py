"""
Search API Routes

Endpoints for searching NG12 guidelines.

Routes:
- POST /search - Search guidelines
- GET /search/section/{section} - Get section content
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from src.api.models.requests import SearchRequest
from src.api.models.responses import (
    SearchResponse,
    GuidelineChunkResponse,
    ClinicalMetadataResponse,
    ErrorResponse,
)
from src.services.retrieval import ClinicalRetriever
from src.config.logging_config import get_logger

logger = get_logger("api.search")

router = APIRouter(prefix="/search", tags=["Search"])


# Dependency injection placeholder
def get_retriever() -> ClinicalRetriever:
    """Get retriever instance."""
    from src.api.dependencies import get_retriever as _get_retriever
    return _get_retriever()


@router.post(
    "",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Search failed"},
    },
    summary="Search NG12 guidelines",
    description="""
    Search the NG12 clinical guidelines database.
    
    Features:
    - Semantic search using embeddings
    - Query expansion with clinical synonyms
    - Metadata filtering (cancer type, urgency)
    
    **Filters:**
    - cancer_type: Filter by specific cancer (lung, breast, etc.)
    - urgent_only: Only return urgent recommendations
    
    **Returns:**
    - Relevant guideline chunks ranked by relevance
    - Original and expanded query
    - Applied filters
    """
)
async def search_guidelines(
    request: SearchRequest,
    retriever: ClinicalRetriever = Depends(get_retriever),
) -> SearchResponse:
    """Search NG12 guidelines."""
    
    logger.info(f"Search request: '{request.query[:50]}...'")
    
    try:
        # Determine search method based on filters
        if request.urgent_only:
            context = retriever.retrieve_for_patient(
                query=request.query,
                urgent_only=True,
                top_k=request.top_k,
            )
        elif request.cancer_type:
            context = retriever.retrieve_for_patient(
                query=request.query,
                suspected_cancer=request.cancer_type,
                top_k=request.top_k,
            )
        else:
            context = retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
            )
        
        # Convert results
        results = []
        for r in context.results:
            results.append(GuidelineChunkResponse(
                chunk_id=r.chunk_id,
                text=r.text,
                page=r.page,
                section=r.section,
                citation=r.citation,
                score=r.score,
                clinical_metadata=ClinicalMetadataResponse(
                    urgency=r.urgency,
                    cancer_types=r.cancer_types,
                ),
            ))
        
        return SearchResponse(
            query=context.query,
            expanded_query=context.expanded_query,
            total_results=len(results),
            results=results,
            filters_applied=context.filters_applied,
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/section/{section_number}",
    response_model=SearchResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Section not found"},
        500: {"model": ErrorResponse, "description": "Search failed"},
    },
    summary="Get section content",
    description="""
    Get all guideline content from a specific section.
    
    **Examples:**
    - /search/section/1.1 - Lung cancer section
    - /search/section/1.3 - Upper GI cancer section
    - /search/section/1.4 - Breast cancer section
    
    Returns all chunks from the specified section prefix.
    """
)
async def get_section(
    section_number: str,
    top_k: int = Query(default=10, ge=1, le=50),
    retriever: ClinicalRetriever = Depends(get_retriever),
) -> SearchResponse:
    """Get content from a specific section."""
    
    logger.info(f"Section request: {section_number}")
    
    try:
        context = retriever.retrieve_by_section(
            section_number=section_number,
            top_k=top_k,
        )
        
        if context.is_empty:
            raise HTTPException(
                status_code=404,
                detail=f"Section {section_number} not found"
            )
        
        # Convert results
        results = []
        for r in context.results:
            results.append(GuidelineChunkResponse(
                chunk_id=r.chunk_id,
                text=r.text,
                page=r.page,
                section=r.section,
                citation=r.citation,
                score=r.score,
                clinical_metadata=ClinicalMetadataResponse(
                    urgency=r.urgency,
                    cancer_types=r.cancer_types,
                ),
            ))
        
        return SearchResponse(
            query=f"Section {section_number}",
            total_results=len(results),
            results=results,
            filters_applied={"section": section_number},
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Section retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@router.get(
    "/cancer-types",
    summary="List cancer types",
    description="Get list of cancer types covered in guidelines."
)
async def list_cancer_types(
    retriever: ClinicalRetriever = Depends(get_retriever),
):
    """List available cancer types."""
    
    stats = retriever.get_stats()
    cancer_types = stats.get("vector_store", {}).get("cancer_types", [])
    
    return {
        "cancer_types": cancer_types,
        "total": len(cancer_types),
    }


@router.get(
    "/urgency-levels",
    summary="List urgency levels",
    description="Get list of urgency levels in guidelines."
)
async def list_urgency_levels(
    retriever: ClinicalRetriever = Depends(get_retriever),
):
    """List urgency levels."""
    
    stats = retriever.get_stats()
    urgency_levels = stats.get("vector_store", {}).get("urgency_levels", [])
    
    return {
        "urgency_levels": urgency_levels,
        "descriptions": {
            "urgent_2_week": "Suspected cancer pathway - 2 week referral",
            "urgent": "Urgent referral (not 2-week pathway)",
            "consider": "Consider referral",
            "routine": "Routine referral",
            "unknown": "Urgency not specified",
        }
    }
