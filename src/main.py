"""
NG12 Cancer Risk Assessment API

FastAPI application for clinical decision support.

Usage:
    # Development
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
    
    # Production
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
    
    # With mock services (no GCP needed)
    USE_MOCK=true uvicorn src.main:app --reload

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - OpenAPI: http://localhost:8000/openapi.json
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import warnings
warnings.filterwarnings("ignore")

from src.config.logging_config import get_logger
from src.api.routes import (
    assessment_router,
    chat_router,
    search_router,
    system_router,
)
from src.api.dependencies import initialize_services

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Initializes services on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("=" * 50)
    logger.info("Starting NG12 Cancer Risk Assessment API")
    logger.info("=" * 50)
    
    # Check for mock mode
    use_mock = os.environ.get("USE_MOCK", "false").lower() == "true"
    if use_mock:
        logger.info("Running in MOCK mode (no GCP credentials needed)")
    
    try:
        # Configure services for lazy loading (on first request)
        initialize_services(mock=use_mock)
        logger.info("✓ API startup complete - ready to accept requests")
        logger.info("Services will initialize on first use (faster startup)")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.warning("Some endpoints may not work correctly")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="NG12 Cancer Risk Assessment API",
    description="""
    Clinical decision support system for cancer referral based on NICE NG12 guidelines.
    
    ## Features
    
    - **Clinical Assessment**: Analyze patient symptoms against NG12 criteria
    - **Conversational Interface**: Multi-turn chat for guided assessment
    - **Guideline Search**: Search and retrieve relevant guidelines
    - **Risk Stratification**: Urgency-based recommendations
    
    ## Usage
    
    1. **Quick Assessment**: POST /assess/quick with symptoms list
    2. **Full Assessment**: POST /assess with complete patient data
    3. **Chat Interface**: POST /chat for conversational assessment
    4. **Search Guidelines**: POST /search to find relevant recommendations
    
    ## Important Notes
    
    - This is a decision SUPPORT tool - final decisions must be made by qualified clinicians
    - All recommendations include citations to NG12 guidelines
    - Urgency levels align with NHS referral pathways
    """,
    version="1.0.0",
    license_info={
        "name": "MIT",
    },
    contact={
        "name": "uday kumar swamy, (udaykumar.swamy007@gmail.com)",
    },
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if os.environ.get("DEBUG") else None,
        }
    )


# Include routers
app.include_router(system_router)
app.include_router(assessment_router)
app.include_router(chat_router)
app.include_router(search_router)


# Additional endpoints

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    
    Returns 200 if API is running.
    Services will be initialized on first request to an actual endpoint.
    """
    return {"status": "ok", "service": "NG12 Cancer Risk Assessment API"}


@app.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    
    Warms up services on first call. Subsequent calls return immediately.
    Useful for Kubernetes readiness probes.
    """
    from src.api.dependencies import get_vector_store, get_embedder, get_llm
    
    try:
        # Warm up critical services
        get_vector_store()
        get_embedder()
        get_llm()
        
        return {
            "status": "ready",
            "service": "NG12 Cancer Risk Assessment API",
            "services_initialized": True
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "not_ready",
            "service": "NG12 Cancer Risk Assessment API",
            "error": str(e)
        }, 503



@app.get("/version")
async def get_version():
    """Get API version information."""
    return {
        "api_version": "1.0.0",
        "guideline": "NICE NG12",
        "guideline_version": "2023",
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    reload = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
    )
