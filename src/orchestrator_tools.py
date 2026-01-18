"""
Orchestrator Tools - State machine and conversation flow control.

The orchestrator manages:
- Conversation state across turns
- Field tracking (completed, pending, probed)
- Strategy selection based on metrics
- Coordination between agents
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
USER_CONTEXT_PATH = BASE_DIR / "user_context.txt"

from .metrics_tools import (
    compute_all_metrics,
    choose_strategy,
    MetricsState,
    EFI_BASELINE,
)
from .survey_questions import (
    get_question_map,
    get_brand_name,
    get_opening_message,
    get_closing_message,
)


class ConversationStatus(str, Enum):
    """Status of a conversation session."""
    PENDING = "pending"  # Not started
    IN_PROGRESS = "in_progress"  # Active conversation
    COMPLETED = "completed"  # All fields collected
    EXITED = "exited"  # User disengaged, graceful exit
    ABANDONED = "abandoned"  # User stopped responding


@dataclass
class Message:
    """A single message in the conversation."""
    turn: int
    role: str  # "bot" or "user"
    text: str
    timestamp: datetime
    latency_ms: Optional[int] = None  # Only for user messages

    def to_dict(self) -> Dict:
        return {
            "turn": self.turn,
            "role": self.role,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
        }


@dataclass
class ExtractedField:
    """A field value extracted from conversation."""
    field_id: str
    value: Any
    confidence: float
    source_turn: int
    quote: str

    def to_dict(self) -> Dict:
        return {
            "field_id": self.field_id,
            "value": self.value,
            "confidence": self.confidence,
            "source_turn": self.source_turn,
            "quote": self.quote,
        }


@dataclass
class ConversationState:
    """Complete state of a conversation session."""
    session_id: str
    survey_id: str
    user_id: str
    persona_id: Optional[str]  # For demo simulation

    status: ConversationStatus
    turn_count: int
    started_at: datetime
    last_activity: datetime

    messages: List[Message]
    extracted_fields: Dict[str, ExtractedField]
    fields_pending: List[str]
    fields_probed: List[str]  # Fields we've asked follow-up for

    current_strategy: str
    next_field_target: Optional[str]

    # Metrics history
    efi_history: List[float]
    ids_history: List[float]
    nps_logit: float
    last_metrics: Optional[Dict]

    user_context: str
    questions: Dict[str, str]

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "survey_id": self.survey_id,
            "user_id": self.user_id,
            "persona_id": self.persona_id,
            "status": self.status.value,
            "turn_count": self.turn_count,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "extracted_fields": {k: v.to_dict() for k, v in self.extracted_fields.items()},
            "fields_pending": self.fields_pending,
            "fields_probed": self.fields_probed,
            "current_strategy": self.current_strategy,
            "next_field_target": self.next_field_target,
            "efi_history": [round(e, 3) for e in self.efi_history],
            "ids_history": [round(i, 3) for i in self.ids_history],
            "nps_logit": round(self.nps_logit, 3),
            "last_metrics": self.last_metrics,
            "user_context": self.user_context,
            "questions": self.questions,
        }

    @property
    def current_efi(self) -> float:
        return self.efi_history[-1] if self.efi_history else EFI_BASELINE

    @property
    def current_ids(self) -> float:
        return self.ids_history[-1] if self.ids_history else 0.0

    @property
    def fields_completed(self) -> List[str]:
        return list(self.extracted_fields.keys())

    @property
    def completion_rate(self) -> float:
        total = len(self.questions)
        completed = len(self.extracted_fields)
        return completed / total if total > 0 else 0.0


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

# In-memory session store (replace with Redis/DB in production)
_sessions: Dict[str, ConversationState] = {}


def create_session(
    survey_id: str,
    user_id: str,
    persona_id: Optional[str] = None,
) -> ConversationState:
    """Create a new conversation session."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    question_map = get_question_map()

    now = datetime.utcnow()

    with open(USER_CONTEXT_PATH, "r", encoding="utf-8") as f:
        user_context_from_txt = f.read()


    state = ConversationState(
        session_id=session_id,
        survey_id=survey_id,
        user_id=user_id,
        persona_id=persona_id,
        status=ConversationStatus.PENDING,
        turn_count=0,
        started_at=now,
        last_activity=now,
        messages=[],
        extracted_fields={},
        fields_pending=list(question_map.keys()),
        fields_probed=[],
        current_strategy="opening",
        next_field_target=list(question_map.keys())[0] if question_map else None,
        efi_history=[EFI_BASELINE],
        ids_history=[],
        nps_logit=0.0,
        last_metrics=None,
        user_context=user_context_from_txt,
        questions=question_map,
    )

    _sessions[session_id] = state
    return state


def get_session(session_id: str) -> Optional[ConversationState]:
    """Get a session by ID."""
    return _sessions.get(session_id)


def get_all_sessions() -> List[ConversationState]:
    """Get all sessions."""
    return list(_sessions.values())


def update_session(state: ConversationState) -> None:
    """Update a session in the store."""
    _sessions[state.session_id] = state


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


# =============================================================================
# CONVERSATION FLOW
# =============================================================================

def add_bot_message(state: ConversationState, text: str) -> ConversationState:
    """Add a bot message to the conversation."""
    state.turn_count += 1
    state.last_activity = datetime.utcnow()

    message = Message(
        turn=state.turn_count,
        role="bot",
        text=text,
        timestamp=state.last_activity,
    )

    state.messages.append(message)

    if state.status == ConversationStatus.PENDING:
        state.status = ConversationStatus.IN_PROGRESS

    update_session(state)
    return state


def process_user_message(
    state: ConversationState,
    text: str,
    latency_ms: int = 0,
) -> ConversationState:
    """
    Process an incoming user message.

    This is the main orchestration function that:
    1. Records the message
    2. Computes metrics
    3. Determines next strategy
    4. Prepares context for the writer agent
    """
    state.turn_count += 1
    now = datetime.utcnow()
    state.last_activity = now

    # Record the message
    message = Message(
        turn=state.turn_count,
        role="user",
        text=text,
        timestamp=now,
        latency_ms=latency_ms,
    )
    state.messages.append(message)

    # Compute metrics
    latency_seconds = latency_ms / 1000.0 if latency_ms > 0 else 5.0
    field_already_probed = state.next_field_target in state.fields_probed

    metrics = compute_all_metrics(
        text=text,
        latency_seconds=latency_seconds,
        turn=state.turn_count,
        previous_efi=state.current_efi,
        previous_nps_logit=state.nps_logit,
        field_already_probed=field_already_probed,
        session_id=state.session_id,
    )

    # Update metrics history
    state.efi_history.append(metrics.efi.current)
    state.ids_history.append(metrics.ids.current)
    state.nps_logit = metrics.nps.logit
    state.last_metrics = metrics.to_dict()

    # Determine strategy
    can_probe = (
        state.next_field_target is not None and
        state.next_field_target not in state.fields_probed and
        metrics.ids.should_probe
    )

    strategy = choose_strategy(
        efi=metrics.efi.current,
        ids=metrics.ids.current,
        fields_remaining=len(state.fields_pending),
        nps_should_ask=metrics.nps.should_ask_direct,
        can_probe=can_probe,
    )

    # Handle probing
    if strategy == "empathize_followup" and state.next_field_target:
        state.fields_probed.append(state.next_field_target)

    state.current_strategy = strategy

    # Check for conversation end conditions
    if strategy == "exit_graceful":
        state.status = ConversationStatus.EXITED

    update_session(state)
    return state


def mark_field_extracted(
    state: ConversationState,
    field_id: str,
    value: Any,
    confidence: float,
    quote: str,
) -> ConversationState:
    """Mark a field as extracted from conversation."""
    extraction = ExtractedField(
        field_id=field_id,
        value=value,
        confidence=confidence,
        source_turn=state.turn_count,
        quote=quote,
    )

    state.extracted_fields[field_id] = extraction

    # Remove from pending if high confidence
    if confidence >= 0.5 and field_id in state.fields_pending:
        state.fields_pending.remove(field_id)

    # Update next target
    if state.fields_pending:
        state.next_field_target = state.fields_pending[0]
    else:
        state.next_field_target = None
        if state.status == ConversationStatus.IN_PROGRESS:
            state.status = ConversationStatus.COMPLETED

    update_session(state)
    return state


def advance_to_next_field(state: ConversationState) -> ConversationState:
    """Advance to the next pending field without extraction."""
    if state.fields_pending:
        # Move current target to end (we'll try again later if needed)
        current = state.next_field_target
        if current and current in state.fields_pending:
            state.fields_pending.remove(current)
            state.fields_pending.append(current)

        state.next_field_target = state.fields_pending[0]

    update_session(state)
    return state


# =============================================================================
# CONTEXT BUILDING FOR WRITER AGENT
# =============================================================================

def get_writer_context(state: ConversationState) -> Dict:
    """
    Build context for the writer agent to generate the next message.
    """
    # Get last few messages for context
    recent_messages = state.messages[-4:] if len(state.messages) >= 4 else state.messages

    # Get current field info
    current_field = None
    if state.next_field_target:
        question_text = state.questions.get(state.next_field_target)
        current_field = {
            "field_id": state.next_field_target,
            "question_text": question_text,
        }

    return {
        "session_id": state.session_id,
        "brand_name": get_brand_name(),
        "strategy": state.current_strategy,
        "turn_count": state.turn_count,

        "recent_messages": [m.to_dict() for m in recent_messages],
        "last_user_message": state.messages[-1].text if state.messages and state.messages[-1].role == "user" else None,

        "current_field": current_field,
        "fields_completed": state.fields_completed,
        "fields_remaining": len(state.fields_pending),

        "current_efi": state.current_efi,
        "current_ids": state.current_ids,
        "nps_bucket": state.last_metrics["nps"]["bucket"] if state.last_metrics else "unknown",

        "opening_message": get_opening_message(),
        "closing_message": get_closing_message(),
        "user_context": state.user_context,
    }


# =============================================================================
# ANALYTICS HELPERS
# =============================================================================

def get_session_summary(state: ConversationState) -> Dict:
    """Get a summary of a session for analytics."""
    return {
        "session_id": state.session_id,
        "user_id": state.user_id,
        "persona_id": state.persona_id,
        "status": state.status.value,
        "turn_count": state.turn_count,
        "completion_rate": round(state.completion_rate, 2),
        "final_efi": state.current_efi,
        "final_ids": state.current_ids if state.ids_history else 0.0,
        "nps_bucket": state.last_metrics["nps"]["bucket"] if state.last_metrics else "unknown",
        "duration_seconds": (state.last_activity - state.started_at).total_seconds(),
        "fields_extracted": list(state.extracted_fields.keys()),
    }


def get_aggregate_analytics() -> Dict:
    """Get aggregate analytics across all sessions."""
    sessions = get_all_sessions()

    if not sessions:
        return {
            "total_sessions": 0,
            "completion_rate": 0.0,
            "avg_efi": 0.0,
            "avg_ids": 0.0,
            "nps_score": 0.0,
            "status_breakdown": {},
            "live_sessions": [],
        }

    # Count statuses
    status_counts = {}
    for s in sessions:
        status = s.status.value
        status_counts[status] = status_counts.get(status, 0) + 1

    # Calculate completion rate
    completed = status_counts.get("completed", 0)
    total_finished = completed + status_counts.get("exited", 0) + status_counts.get("abandoned", 0)
    completion_rate = completed / total_finished if total_finished > 0 else 0.0

    # Average metrics
    efi_values = [s.current_efi for s in sessions if s.efi_history]
    ids_values = [s.current_ids for s in sessions if s.ids_history]

    avg_efi = sum(efi_values) / len(efi_values) if efi_values else 0.0
    avg_ids = sum(ids_values) / len(ids_values) if ids_values else 0.0

    # NPS calculation
    promoters = 0
    detractors = 0
    total_nps = 0

    for s in sessions:
        if s.last_metrics and "nps" in s.last_metrics:
            bucket = s.last_metrics["nps"]["bucket"]
            if bucket == "promoter":
                promoters += 1
            elif bucket == "detractor":
                detractors += 1
            total_nps += 1

    nps_score = ((promoters - detractors) / total_nps * 100) if total_nps > 0 else 0.0

    # Live sessions
    live_sessions = [
        {
            "session_id": s.session_id,
            "status": s.status.value,
            "current_efi": round(s.current_efi, 2),
            "turn_count": s.turn_count,
        }
        for s in sessions
        if s.status == ConversationStatus.IN_PROGRESS
    ]

    return {
        "total_sessions": len(sessions),
        "completion_rate": round(completion_rate, 2),
        "avg_efi": round(avg_efi, 3),
        "avg_ids": round(avg_ids, 3),
        "nps_score": round(nps_score, 1),
        "status_breakdown": status_counts,
        "live_sessions": live_sessions,
    }
