"""
Invisible Feedback - Main API Application

FastAPI application that serves:
- Chat endpoints for conversation management
- Analytics endpoints for dashboard data
- Static files for the frontend
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from api.routes.chat import router as chat_router
from api.routes.analytics import router as analytics_router

# Create FastAPI app
app = FastAPI(
    title="Invisible Feedback",
    description="Surveys that feel like DMs - Real-time conversational feedback collection",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(analytics_router)

# Mount static files (frontend)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Serve the main chat UI."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Invisible Feedback API",
        "docs": "/docs",
        "chat_ui": "/static/index.html",
        "dashboard": "/static/dashboard.html",
    }


@app.get("/dashboard")
async def dashboard():
    """Serve the admin dashboard."""
    dashboard_path = os.path.join(frontend_dir, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return {"error": "Dashboard not found"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "invisible-feedback",
        "version": "0.1.0",
    }


@app.get("/config")
async def get_config():
    """Get public configuration."""
    return {
        "survey_id": "apparel_post_purchase_v1",
        "brand_name": "ThreadCraft",
        "features": {
            "persona_simulation": True,
            "real_time_metrics": True,
            "field_extraction": True,
        },
    }


# =============================================================================
# STARTUP/SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("🚀 Invisible Feedback API starting...")
    print(f"📁 Frontend directory: {frontend_dir}")

    # Check for OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key configured")
    else:
        print("⚠️  OPENAI_API_KEY not set - LLM features will use fallbacks")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Invisible Feedback API shutting down...")


# =============================================================================
# RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
