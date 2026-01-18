"""
Analytics Tools - Aggregation and theme extraction for dashboard.

Keeps it simple:
- Rule-based theme extraction (no clustering/embeddings)
- Basic aggregation functions
- Dashboard-ready JSON output
"""

import re
from typing import Dict, List, Optional, Any
from collections import Counter
from dataclasses import dataclass
from .orchestrator_tools import (
    get_all_sessions,
    get_session_summary,
    ConversationState,
    ConversationStatus,
)


# =============================================================================
# THEME EXTRACTION (Keyword-based)
# =============================================================================

# Theme patterns for improvement suggestions
THEME_PATTERNS = {
    "sizing": [
        r"\b(size|sizing|fit|fits|fitting|small|large|big|tight|loose)\b",
        r"\b(size\s*(chart|guide)|true\s*to\s*size)\b",
    ],
    "colors": [
        r"\b(color|colour|colors|colours|shade|dye)\b",
        r"\b(more\s*colors|different\s*colors|color\s*options)\b",
    ],
    "price": [
        r"\b(price|pricing|cost|expensive|cheap|affordable|value|worth)\b",
        r"\b(discount|sale|deal)\b",
    ],
    "quality": [
        r"\b(quality|stitching|seams|hem|construction|durable|durability)\b",
        r"\b(cheap\s*feeling|flimsy|well\s*made)\b",
    ],
    "shipping": [
        r"\b(shipping|delivery|ship|shipped|arrive|arrived|package)\b",
        r"\b(fast|slow|delayed|tracking)\b",
    ],
    "fabric": [
        r"\b(fabric|material|cotton|polyester|soft|rough|texture)\b",
        r"\b(breathable|comfortable|comfort)\b",
    ],
    "photos": [
        r"\b(photo|photos|picture|pictures|image|images)\b",
        r"\b(accurate|match|different|misleading)\b",
    ],
    "returns": [
        r"\b(return|returns|exchange|refund|policy)\b",
    ],
    "variety": [
        r"\b(variety|options|selection|more|different|styles)\b",
    ],
}


def extract_themes_from_text(text: str) -> List[str]:
    """Extract themes from a single text response."""
    text_lower = text.lower()
    themes = []

    for theme, patterns in THEME_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                themes.append(theme)
                break  # Only add each theme once

    return themes


def extract_themes_from_sessions(sessions: List[ConversationState]) -> Dict[str, int]:
    """Extract and count themes across all sessions."""
    theme_counts = Counter()

    for session in sessions:
        # Get improvement suggestions
        if "improvement_suggestion" in session.extracted_fields:
            suggestion = session.extracted_fields["improvement_suggestion"]
            if suggestion.value and suggestion.value != "none":
                themes = extract_themes_from_text(suggestion.value)
                theme_counts.update(themes)

        # Also scan all user messages for theme content
        for message in session.messages:
            if message.role == "user":
                themes = extract_themes_from_text(message.text)
                theme_counts.update(themes)

    return dict(theme_counts)


# =============================================================================
# INSIGHT CLUSTERS + HEALTH METRICS
# =============================================================================

STOPWORDS = {
    "the", "and", "but", "with", "that", "this", "for", "are", "was", "were",
    "your", "you", "our", "just", "like", "really", "have", "has", "had",
    "from", "they", "them", "their", "when", "what", "how", "its",
    "into", "about", "would", "could", "should", "been", "than", "then",
    "also", "there", "here", "more", "less", "much", "very", "feel", "felt",
}


def _sentiment_from_nps(p_advocacy: Optional[float]) -> float:
    if p_advocacy is None:
        return 0.0
    return max(-1.0, min(1.0, (p_advocacy - 0.5) * 2.0))


def _extract_keywords(texts: List[str], limit: int = 5) -> List[str]:
    tokens: List[str] = []
    for text in texts:
        tokens.extend(re.findall(r"[a-zA-Z']{3,}", text.lower()))
    tokens = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(limit)]


def compute_topic_clusters(sessions: List[ConversationState]) -> List[Dict[str, Any]]:
    """Compute topic clusters with NPS and engagement quality positioning.

    Uses NPS for X-axis positioning and inverted EFI (engagement quality) for Y-axis.
    """
    theme_sessions: Dict[str, List[ConversationState]] = {k: [] for k in THEME_PATTERNS}
    theme_texts: Dict[str, List[str]] = {k: [] for k in THEME_PATTERNS}

    for session in sessions:
        user_texts = [m.text for m in session.messages if m.role == "user"]
        suggestion = session.extracted_fields.get("improvement_suggestion")
        if suggestion and suggestion.value:
            user_texts.append(str(suggestion.value))
        combined = " ".join(user_texts)
        for theme, patterns in THEME_PATTERNS.items():
            if any(re.search(pat, combined.lower()) for pat in patterns):
                theme_sessions[theme].append(session)
                theme_texts[theme].extend(user_texts)

    clusters: List[Dict[str, Any]] = []
    for theme, matched in theme_sessions.items():
        if not matched:
            continue

        nps_vals = []
        engagement_vals = []
        for s in matched:
            nps_info = (s.last_metrics or {}).get("nps") or {}
            p_adv = nps_info.get("p_advocacy")
            if isinstance(p_adv, (int, float)):
                nps_vals.append(float(p_adv))

            # Use inverted EFI as engagement quality (low friction = high quality)
            efi = s.current_efi if s.efi_history else 0.5
            engagement_vals.append(1.0 - efi)

        avg_nps = sum(nps_vals) / len(nps_vals) if nps_vals else 0.5
        avg_engagement = sum(engagement_vals) / len(engagement_vals) if engagement_vals else 0.5

        words = _extract_keywords(theme_texts[theme], limit=4)
        clusters.append(
            {
                "topic": theme,
                "nps_score": round(avg_nps * 100, 1),
                "sentiment": round(avg_engagement, 2),  # Now represents engagement quality
                "volume": len(matched),
                "words": words,
            }
        )

    clusters.sort(key=lambda c: (-c["volume"], c["nps_score"]))
    return clusters


def compute_session_points(sessions: List[ConversationState]) -> List[Dict[str, Any]]:
    """Create per-session points for the perception map.

    Uses NPS for X-axis and inverted EFI (engagement quality) for Y-axis.
    Low EFI = high engagement quality = higher Y position.
    """
    points: List[Dict[str, Any]] = []
    for session in sessions:
        user_texts = [m.text for m in session.messages if m.role == "user"]
        suggestion = session.extracted_fields.get("improvement_suggestion")
        if suggestion and suggestion.value:
            user_texts.append(str(suggestion.value))

        combined = " ".join(user_texts).lower()
        # Match ALL themes for this session, not just the first one
        matched_themes = []
        for theme, patterns in THEME_PATTERNS.items():
            if any(re.search(pat, combined) for pat in patterns):
                matched_themes.append(theme)

        if not matched_themes:
            continue

        nps_info = (session.last_metrics or {}).get("nps") or {}
        p_adv = nps_info.get("p_advocacy")
        if not isinstance(p_adv, (int, float)):
            p_adv = 0.5  # Default to middle if no NPS

        # Use EFI as an independent Y-axis metric
        # Invert it: low friction (low EFI) = high quality = high Y position
        efi = session.current_efi if session.efi_history else 0.5
        engagement_quality = 1.0 - efi  # Invert: 0 friction -> 1.0, 1 friction -> 0.0

        # Create a point for each matched theme
        for theme in matched_themes:
            points.append(
                {
                    "session_id": session.session_id,
                    "topic": theme,
                    "nps_score": round(float(p_adv) * 100, 1),
                    "sentiment": round(engagement_quality, 2),  # Now uses EFI-based quality
                }
            )

    return points


def compute_conversation_health(sessions: List[ConversationState]) -> Dict[str, Any]:
    """Compute aggregate health signals for conversations."""
    if not sessions:
        return {
            "avg_turns_before_dropoff": 0.0,
            "left_on_read_count": 0,
            "left_on_read_rate": 0.0,
            "avg_turns_completed": 0.0,
            "avg_turns_in_progress": 0.0,
        }

    dropoff = [
        s for s in sessions
        if s.status in [ConversationStatus.EXITED, ConversationStatus.ABANDONED]
    ]
    in_progress = [s for s in sessions if s.status == ConversationStatus.IN_PROGRESS]
    completed = [s for s in sessions if s.status == ConversationStatus.COMPLETED]

    avg_dropoff = sum(s.turn_count for s in dropoff) / len(dropoff) if dropoff else 0.0
    avg_completed = sum(s.turn_count for s in completed) / len(completed) if completed else 0.0
    avg_in_progress = sum(s.turn_count for s in in_progress) / len(in_progress) if in_progress else 0.0

    left_on_read = [
        s for s in in_progress
        if s.messages and s.messages[-1].role == "bot"
    ]
    left_on_read_count = len(left_on_read)
    left_on_read_rate = left_on_read_count / len(in_progress) if in_progress else 0.0

    return {
        "avg_turns_before_dropoff": round(avg_dropoff, 1),
        "left_on_read_count": left_on_read_count,
        "left_on_read_rate": round(left_on_read_rate, 2),
        "avg_turns_completed": round(avg_completed, 1),
        "avg_turns_in_progress": round(avg_in_progress, 1),
    }

# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

@dataclass
class AggregateMetrics:
    """Aggregate metrics across sessions."""
    total_sessions: int
    completed_sessions: int
    exited_sessions: int
    in_progress_sessions: int

    completion_rate: float
    avg_turns: float
    avg_duration_seconds: float

    avg_efi: float
    avg_ids: float
    nps_score: float

    promoter_count: int
    passive_count: int
    detractor_count: int

    top_themes: List[Dict[str, Any]]
    field_completion_rates: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "total_sessions": self.total_sessions,
            "completed_sessions": self.completed_sessions,
            "exited_sessions": self.exited_sessions,
            "in_progress_sessions": self.in_progress_sessions,
            "completion_rate": round(self.completion_rate, 2),
            "avg_turns": round(self.avg_turns, 1),
            "avg_duration_seconds": round(self.avg_duration_seconds, 1),
            "avg_efi": round(self.avg_efi, 3),
            "avg_ids": round(self.avg_ids, 3),
            "nps_score": round(self.nps_score, 1),
            "promoter_count": self.promoter_count,
            "passive_count": self.passive_count,
            "detractor_count": self.detractor_count,
            "top_themes": self.top_themes,
            "field_completion_rates": {
                k: round(v, 2) for k, v in self.field_completion_rates.items()
            },
        }


def compute_aggregate_metrics() -> AggregateMetrics:
    """Compute aggregate metrics across all sessions."""
    sessions = get_all_sessions()

    if not sessions:
        return AggregateMetrics(
            total_sessions=0,
            completed_sessions=0,
            exited_sessions=0,
            in_progress_sessions=0,
            completion_rate=0.0,
            avg_turns=0.0,
            avg_duration_seconds=0.0,
            avg_efi=0.0,
            avg_ids=0.0,
            nps_score=0.0,
            promoter_count=0,
            passive_count=0,
            detractor_count=0,
            top_themes=[],
            field_completion_rates={},
        )

    # Count by status
    completed = sum(1 for s in sessions if s.status == ConversationStatus.COMPLETED)
    exited = sum(1 for s in sessions if s.status == ConversationStatus.EXITED)
    in_progress = sum(1 for s in sessions if s.status == ConversationStatus.IN_PROGRESS)

    # Completion rate (of finished sessions)
    finished = completed + exited
    completion_rate = completed / finished if finished > 0 else 0.0

    # Average turns and duration
    turn_counts = [s.turn_count for s in sessions]
    durations = [(s.last_activity - s.started_at).total_seconds() for s in sessions]

    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0.0
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    # Average EFI and IDS
    efi_values = [s.current_efi for s in sessions if s.efi_history]
    ids_values = [s.current_ids for s in sessions if s.ids_history]

    avg_efi = sum(efi_values) / len(efi_values) if efi_values else 0.0
    avg_ids = sum(ids_values) / len(ids_values) if ids_values else 0.0

    # NPS calculation
    promoters = 0
    passives = 0
    detractors = 0

    for s in sessions:
        if s.last_metrics and "nps" in s.last_metrics:
            bucket = s.last_metrics["nps"]["bucket"]
            if bucket == "promoter":
                promoters += 1
            elif bucket == "passive":
                passives += 1
            elif bucket == "detractor":
                detractors += 1

    total_nps_responses = promoters + passives + detractors
    nps_score = ((promoters - detractors) / total_nps_responses * 100) if total_nps_responses > 0 else 0.0

    # Theme extraction
    theme_counts = extract_themes_from_sessions(sessions)
    top_themes = [
        {"theme": theme, "count": count}
        for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1])[:5]
    ]

    # Field completion rates
    field_counts = Counter()
    total_finished = len([s for s in sessions if s.status in [ConversationStatus.COMPLETED, ConversationStatus.EXITED]])

    for s in sessions:
        for field_id in s.extracted_fields.keys():
            field_counts[field_id] += 1

    field_completion_rates = {
        field_id: count / total_finished if total_finished > 0 else 0.0
        for field_id, count in field_counts.items()
    }

    return AggregateMetrics(
        total_sessions=len(sessions),
        completed_sessions=completed,
        exited_sessions=exited,
        in_progress_sessions=in_progress,
        completion_rate=completion_rate,
        avg_turns=avg_turns,
        avg_duration_seconds=avg_duration,
        avg_efi=avg_efi,
        avg_ids=avg_ids,
        nps_score=nps_score,
        promoter_count=promoters,
        passive_count=passives,
        detractor_count=detractors,
        top_themes=top_themes,
        field_completion_rates=field_completion_rates,
    )


# =============================================================================
# DASHBOARD DATA
# =============================================================================

def get_dashboard_data() -> Dict:
    """Get all data needed for the admin dashboard."""
    metrics = compute_aggregate_metrics()
    sessions = get_all_sessions()
    topic_clusters = compute_topic_clusters(sessions)
    session_points = compute_session_points(sessions)
    conversation_health = compute_conversation_health(sessions)

    # Session summaries
    session_summaries = [get_session_summary(s) for s in sessions]

    # Live sessions (in progress)
    live_sessions = [
        {
            "session_id": s.session_id,
            "user_id": s.user_id,
            "persona_id": s.persona_id,
            "turn_count": s.turn_count,
            "current_efi": round(s.current_efi, 2),
            "current_ids": round(s.current_ids, 2) if s.ids_history else 0.0,
            "fields_completed": len(s.extracted_fields),
            "last_message": s.messages[-1].text[:50] if s.messages else "",
        }
        for s in sessions
        if s.status == ConversationStatus.IN_PROGRESS
    ]

    # Recent extractions (last 10)
    recent_extractions = []
    for s in sorted(sessions, key=lambda x: x.last_activity, reverse=True)[:10]:
        for field_id, extraction in s.extracted_fields.items():
            recent_extractions.append({
                "session_id": s.session_id,
                "field_id": field_id,
                "value": extraction.value,
                "confidence": extraction.confidence,
                "quote": extraction.quote[:80],
            })

    return {
        "aggregate_metrics": metrics.to_dict(),
        "session_summaries": session_summaries,
        "live_sessions": live_sessions,
        "recent_extractions": recent_extractions[:20],  # Limit to 20
        "topic_clusters": topic_clusters,
        "session_points": session_points,
        "conversation_health": conversation_health,
    }


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_sessions_csv() -> str:
    """Export session data as CSV string."""
    sessions = get_all_sessions()

    headers = [
        "session_id", "user_id", "persona_id", "status",
        "turn_count", "completion_rate", "final_efi", "final_ids",
        "nps_bucket", "duration_seconds"
    ]

    rows = [",".join(headers)]

    for s in sessions:
        summary = get_session_summary(s)
        row = [
            summary["session_id"],
            summary["user_id"],
            summary.get("persona_id", ""),
            summary["status"],
            str(summary["turn_count"]),
            str(summary["completion_rate"]),
            str(round(summary["final_efi"], 3)),
            str(round(summary["final_ids"], 3)),
            summary["nps_bucket"],
            str(round(summary["duration_seconds"], 1)),
        ]
        rows.append(",".join(row))

    return "\n".join(rows)


def export_extractions_csv() -> str:
    """Export all extractions as CSV string."""
    sessions = get_all_sessions()

    headers = ["session_id", "field_id", "value", "confidence", "source_turn", "quote"]
    rows = [",".join(headers)]

    for s in sessions:
        for field_id, extraction in s.extracted_fields.items():
            # Escape quotes in the quote field
            quote = extraction.quote.replace('"', '""')
            row = [
                s.session_id,
                field_id,
                str(extraction.value),
                str(round(extraction.confidence, 2)),
                str(extraction.source_turn),
                f'"{quote}"',
            ]
            rows.append(",".join(row))

    return "\n".join(rows)
