"""
Analytics Routes - Dashboard and metrics endpoints.

Endpoints:
- GET /analytics - Get aggregate analytics
- GET /analytics/dashboard - Get full dashboard data
- GET /analytics/sessions/{session_id} - Get session details
- GET /analytics/export/csv - Export data as CSV
"""

import os
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analytics_tools import (
    compute_aggregate_metrics,
    get_dashboard_data,
    export_sessions_csv,
    export_extractions_csv,
)
from src.orchestrator_tools import get_session, get_session_summary

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("")
async def get_analytics():
    """Get aggregate analytics across all sessions."""
    metrics = compute_aggregate_metrics()
    return metrics.to_dict()


@router.get("/dashboard")
async def get_dashboard():
    """Get complete dashboard data including live sessions."""
    return get_dashboard_data()


@router.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get detailed analytics for a specific session."""
    session = get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    return {
        "summary": get_session_summary(session),
        "full_state": session.to_dict(),
    }


@router.get("/export/sessions", response_class=PlainTextResponse)
async def export_sessions():
    """Export all session data as CSV."""
    csv_content = export_sessions_csv()
    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sessions.csv"}
    )


@router.get("/export/extractions", response_class=PlainTextResponse)
async def export_extractions():
    """Export all extractions as CSV."""
    csv_content = export_extractions_csv()
    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=extractions.csv"}
    )


@router.get("/metrics/definitions")
async def get_metric_definitions():
    """Get definitions and thresholds for all metrics."""
    return {
        "efi": {
            "name": "Engagement Friction Index",
            "range": [0, 1],
            "description": "Measures user disengagement/resistance. Higher = more friction.",
            "components": {
                "latency": "Response time (60s = max friction)",
                "brevity": "Short responses indicate disengagement",
                "sentiment": "Negative sentiment = friction",
                "deflection": "Patterns like 'idk', 'whatever', 'busy'",
                "fatigue": "Increases with conversation length",
            },
            "weights": {
                "latency": 0.15,
                "brevity": 0.25,
                "sentiment": 0.20,
                "deflection": 0.30,
                "fatigue": 0.10,
            },
            "thresholds": {
                "engaged": {"range": [0, 0.3], "action": "continue_normal"},
                "mild_friction": {"range": [0.3, 0.5], "action": "simplify"},
                "moderate_friction": {"range": [0.5, 0.7], "action": "quick_reply_options"},
                "high_friction": {"range": [0.7, 0.85], "action": "one_last_question"},
                "disengaged": {"range": [0.85, 1.0], "action": "exit_graceful"},
            },
        },
        "ids": {
            "name": "Insight Depth Score",
            "range": [0, 1],
            "description": "Measures response quality/actionability. Higher = better insights.",
            "components": {
                "attributes": "Product-specific mentions (fit, fabric, size)",
                "causal": "Explanatory language (because, since, after)",
                "nouns": "Concrete noun/adjective density",
                "specificity": "Numbers, comparisons, specific details",
                "vagueness_penalty": "Deduction for generic words (fine, okay)",
            },
            "weights": {
                "attributes": 0.30,
                "causal": 0.20,
                "nouns": 0.25,
                "specificity": 0.25,
            },
            "threshold": {
                "low_quality": {"range": [0, 0.3], "action": "probe for more detail"},
                "acceptable": {"range": [0.3, 1.0], "action": "accept response"},
            },
        },
        "nps": {
            "name": "Inferred Net Promoter Score",
            "range": [-100, 100],
            "description": "Predicted NPS based on conversation signals.",
            "buckets": {
                "promoter": {"p_advocacy": [0.7, 1.0], "contribution": "+1"},
                "passive": {"p_advocacy": [0.4, 0.7], "contribution": "0"},
                "detractor": {"p_advocacy": [0, 0.4], "contribution": "-1"},
            },
            "calculation": "(promoters - detractors) / total * 100",
            "signals": {
                "positive": ["love", "amazing", "recommend", "best", "perfect"],
                "negative": ["terrible", "returning", "waste", "never again"],
            },
        },
    }
