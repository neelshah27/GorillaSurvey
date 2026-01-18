# orchestrator_tools.py
"""
Orchestrator Tools - State machine and conversation flow control.

The orchestrator manages:
- Conversation state across turns
- Field tracking (completed, pending, probed)
- Strategy selection based on metrics
- Coordination between agents

UPDATED:
- Hard limiter: track how many times each field/question has been asked
  (max 2 total asks: initial + one "expand on that" follow-up).
- Permission refusal detection: user says "no" to questions -> EXIT immediately.
- Skip exhausted fields when selecting next target.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import re

from .metrics_tools import (
    compute_all_metrics,
    choose_strategy,
    EFI_BASELINE,
)
from .survey_schema import get_survey


class ConversationStatus(str, Enum):
    """Status of a conversation session."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXITED = "exited"
    ABANDONED = "abandoned"


@dataclass
class Message:
    """A single message in the conversation."""
    turn: int
    role: str  # "bot" or "user"
    text: str
    timestamp: datetime
    latency_ms: Optional[int] = None

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


# Local completion thresholds (keeps this module independent)
FIELD_COMPLETE_THRESHOLDS = {
    "fit_rating": 0.35,
    "would_recommend": 0.40,
    "overall_satisfaction": 0.45,
    "fabric_quality": 0.45,
    "improvement_suggestion": 0.55,
}
DEFAULT_COMPLETE_THRESHOLD = 0.50


@dataclass
class ConversationState:
    """Complete state of a conversation session."""
    session_id: str
    survey_id: str
    user_id: str
    persona_id: Optional[str]

    status: ConversationStatus
    turn_count: int
    started_at: datetime
    last_activity: datetime

    messages: List[Message]
    extracted_fields: Dict[str, ExtractedField]
    fields_pending: List[str]
    fields_probed: List[str]

    current_strategy: str
    next_field_target: Optional[str]

    # Metrics history
    efi_history: List[float]
    ids_history: List[float]
    nps_logit: float
    last_metrics: Optional[Dict]

    # Hard limiter / anti-repeat
    question_ask_counts: Dict[str, int] = field(default_factory=dict)
    questions_exhausted: List[str] = field(default_factory=list)
    last_asked_field: Optional[str] = None

    # Optional: allow UI/prompt to know recent targets
    recent_question_fields: List[str] = field(default_factory=list)

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
            "question_ask_counts": dict(self.question_ask_counts),
            "questions_exhausted": list(self.questions_exhausted),
            "last_asked_field": self.last_asked_field,
            "recent_question_fields": list(self.recent_question_fields),
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
        survey = get_survey()
        total = len(survey.fields)
        completed = len(self.extracted_fields)
        return completed / total if total > 0 else 0.0


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

_sessions: Dict[str, ConversationState] = {}


def create_session(
    survey_id: str,
    user_id: str,
    persona_id: Optional[str] = None,
) -> ConversationState:
    """Create a new conversation session."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    survey = get_survey()
    now = datetime.utcnow()

    priority_fields = survey.get_fields_by_priority()
    first_target = priority_fields[0].field_id if priority_fields else None

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
        fields_pending=[f.field_id for f in priority_fields],
        fields_probed=[],
        current_strategy="opening",
        next_field_target=first_target,
        efi_history=[EFI_BASELINE],
        ids_history=[],
        nps_logit=0.0,
        last_metrics=None,
    )

    _sessions[session_id] = state
    return state


def get_session(session_id: str) -> Optional[ConversationState]:
    return _sessions.get(session_id)


def get_all_sessions() -> List[ConversationState]:
    return list(_sessions.values())


def update_session(state: ConversationState) -> None:
    _sessions[state.session_id] = state


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


# =============================================================================
# HARD LIMITER HELPERS
# =============================================================================

def can_ask_field(state: ConversationState, field_id: Optional[str]) -> bool:
    """Allowed to ask if not exhausted and ask count < 2."""
    if not field_id:
        return False
    if field_id in state.questions_exhausted:
        return False
    return state.question_ask_counts.get(field_id, 0) < 2


def record_field_asked(state: ConversationState, field_id: str) -> None:
    """Increment ask count; exhaust after 2."""
    state.question_ask_counts[field_id] = state.question_ask_counts.get(field_id, 0) + 1
    state.last_asked_field = field_id

    # keep a short list of recent asked fields (for prompt steering)
    state.recent_question_fields.append(field_id)
    state.recent_question_fields = state.recent_question_fields[-5:]

    if state.question_ask_counts[field_id] >= 2:
        if field_id not in state.questions_exhausted:
            state.questions_exhausted.append(field_id)


def _select_next_field_target(state: ConversationState) -> Optional[str]:
    """Pick next pending field that is not exhausted."""
    for f in state.fields_pending:
        if f not in state.questions_exhausted:
            return f
    return None


def note_attempted_question(state: ConversationState) -> ConversationState:
    """
    Called when the bot is about to ask the current target.
    Spends an ask token for state.next_field_target if allowed.
    """
    field_id = state.next_field_target
    if field_id and can_ask_field(state, field_id):
        record_field_asked(state, field_id)
        update_session(state)
    return state


# =============================================================================
# CONSENT REFUSAL DETECTION
# =============================================================================

def _is_permission_refusal(user_text: str, last_bot_text: Optional[str]) -> bool:
    """
    Detect when the user refuses to answer questions.
    Used to force EXIT early, so we don't keep chatting.
    """
    if not user_text:
        return False

    t = user_text.strip().lower()

    strong_phrases = [
        "not interested",
        "no thanks",
        "no thank you",
        "i'd rather not",
        "i would rather not",
        "id rather not",
        "don't want to answer",
        "dont want to answer",
        "please don't",
        "pls don't",
        "please stop",
        "pls stop",
        "stop messaging me",
        "leave me alone",
    ]
    for phrase in strong_phrases:
        if phrase in t:
            return True

    short_negatives = {"no", "no.", "nah", "nah.", "nope", "nope.", "not really", "rather not"}
    if t in short_negatives and last_bot_text:
        lb = last_bot_text.lower()
        # Only treat short "no" as refusal if last bot asked permission / questions
        if ("would you mind" in lb) or ("ask you" in lb) or ("questions" in lb):
            return True

    return False


# =============================================================================
# CONVERSATION FLOW
# =============================================================================

def add_bot_message(state: ConversationState, text: str) -> ConversationState:
    """Add a bot message to the conversation."""
    state.turn_count += 1
    state.last_activity = datetime.utcnow()

    state.messages.append(
        Message(
            turn=state.turn_count,
            role="bot",
            text=text,
            timestamp=state.last_activity,
        )
    )

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
    1) Records the message
    2) Checks permission refusal -> EXIT immediately
    3) Computes metrics and strategy
    """
    state.turn_count += 1
    now = datetime.utcnow()
    state.last_activity = now

    # Find previous bot message (for refusal context)
    last_bot_text: Optional[str] = None
    for m in reversed(state.messages):
        if m.role == "bot":
            last_bot_text = m.text
            break

    # Record user message
    state.messages.append(
        Message(
            turn=state.turn_count,
            role="user",
            text=text,
            timestamp=now,
            latency_ms=latency_ms,
        )
    )

    # Immediate refusal exit
    if _is_permission_refusal(text, last_bot_text):
        state.current_strategy = "exit_graceful"
        state.status = ConversationStatus.EXITED
        update_session(state)
        return state

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
        state.next_field_target is not None
        and state.next_field_target not in state.fields_probed
        and metrics.ids.should_probe
    )

    strategy = choose_strategy(
        efi=metrics.efi.current,
        ids=metrics.ids.current,
        fields_remaining=len(state.fields_pending),
        nps_should_ask=metrics.nps.should_ask_direct,
        can_probe=can_probe,
    )

    if strategy == "empathize_followup" and state.next_field_target:
        state.fields_probed.append(state.next_field_target)

    state.current_strategy = strategy

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
    state.extracted_fields[field_id] = ExtractedField(
        field_id=field_id,
        value=value,
        confidence=confidence,
        source_turn=state.turn_count,
        quote=quote,
    )

    # Remove from pending if confident enough (field-specific thresholds)
    threshold = FIELD_COMPLETE_THRESHOLDS.get(field_id, DEFAULT_COMPLETE_THRESHOLD)
    if confidence >= threshold and field_id in state.fields_pending:
        state.fields_pending.remove(field_id)

    # Update next target (skip exhausted)
    state.next_field_target = _select_next_field_target(state)

    # Completion condition: all required fields done
    if state.next_field_target is None:
        if state.status == ConversationStatus.IN_PROGRESS:
            survey = get_survey()
            required_fields = [f.field_id for f in survey.fields if f.required]
            all_required_done = all(f in state.extracted_fields for f in required_fields)
            if all_required_done:
                state.status = ConversationStatus.COMPLETED

    update_session(state)
    return state


def advance_to_next_field(state: ConversationState) -> ConversationState:
    """Advance to the next pending field without extraction."""
    if state.fields_pending:
        current = state.next_field_target
        if current and current in state.fields_pending:
            state.fields_pending.remove(current)
            state.fields_pending.append(current)

    state.next_field_target = _select_next_field_target(state)
    update_session(state)
    return state


# =============================================================================
# CONTEXT BUILDING FOR WRITER AGENT
# =============================================================================

def get_writer_context(state: ConversationState) -> Dict:
    """Build context for the writer agent."""
    survey = get_survey()

    recent_messages = state.messages[-4:] if len(state.messages) >= 4 else state.messages

    current_field = None
    if state.next_field_target:
        current_field = survey.get_field(state.next_field_target)

    return {
        "session_id": state.session_id,
        "brand_name": getattr(survey, "brand_name", "ThreadCraft"),
        "brand_voice": getattr(survey, "brand_voice", ""),
        "strategy": state.current_strategy,
        "turn_count": state.turn_count,

        "recent_messages": [m.to_dict() for m in recent_messages],
        "last_user_message": state.messages[-1].text if state.messages and state.messages[-1].role == "user" else None,

        "current_field": current_field.to_dict() if current_field else None,

        "fields_completed": list(state.fields_completed),
        "fields_pending": list(state.fields_pending),
        "fields_remaining": len(state.fields_pending),

        "current_efi": state.current_efi,
        "current_ids": state.current_ids,
        "nps_bucket": state.last_metrics["nps"]["bucket"] if state.last_metrics else "unknown",

        "opening_message": getattr(survey, "opening_message", ""),
        "closing_message": getattr(survey, "closing_message", ""),

        # Hard limiter context
        "question_ask_counts": dict(state.question_ask_counts),
        "questions_exhausted": list(state.questions_exhausted),
        "last_asked_field": state.last_asked_field,
        "recent_question_fields": list(state.recent_question_fields),
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
        "questions_exhausted": list(state.questions_exhausted),
        "question_ask_counts": dict(state.question_ask_counts),
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

    status_counts: Dict[str, int] = {}
    for s in sessions:
        status_counts[s.status.value] = status_counts.get(s.status.value, 0) + 1

    completed = status_counts.get("completed", 0)
    total_finished = completed + status_counts.get("exited", 0) + status_counts.get("abandoned", 0)
    completion_rate = completed / total_finished if total_finished > 0 else 0.0

    efi_values = [s.current_efi for s in sessions if s.efi_history]
    ids_values = [s.current_ids for s in sessions if s.ids_history]

    avg_efi = sum(efi_values) / len(efi_values) if efi_values else 0.0
    avg_ids = sum(ids_values) / len(ids_values) if ids_values else 0.0

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
