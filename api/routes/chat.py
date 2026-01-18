# chat.py
"""
Chat Routes - Handles conversation endpoints.

REVAMPED: Prompts engineered for natural, reactive DM conversations.
Context: Brand slides into DMs after seeing user's social post wearing their product.

Endpoints:
- POST /chat/start - Start a new conversation
- POST /chat/message - Send a user message
- GET /chat/state/{session_id} - Get conversation state
- GET /chat/sessions - List all sessions
- POST /chat/simulate - Simulate user response (demo)
- GET /chat/personas - List personas (demo)
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.orchestrator_tools import (
    create_session,
    get_session,
    get_all_sessions,
    add_bot_message,
    process_user_message,
    mark_field_extracted,
    get_writer_context,
    ConversationStatus,
    can_ask_field,
    note_attempted_question,
    advance_to_next_field,
)
from src.extraction_tools import extract_all_fields, is_confident_enough
from src.survey_schema import get_survey, get_graceful_pivot
from src.personas import get_persona, get_persona_ids, get_scripted_response, get_persona_system_prompt

router = APIRouter(prefix="/chat", tags=["chat"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class StartChatRequest(BaseModel):
    survey_id: str = "apparel_social_reactive_v1"
    user_id: str = "demo_user"
    persona_id: Optional[str] = None  # For simulated users


class StartChatResponse(BaseModel):
    session_id: str
    opening_message: str
    persona_id: Optional[str]


class SendMessageRequest(BaseModel):
    session_id: str
    text: str
    latency_ms: int = 5000  # Simulated latency for demo


class SendMessageResponse(BaseModel):
    bot_response: str
    metrics: dict
    extractions: list
    status: str
    fields_completed: list
    fields_remaining: int


class SimulateUserRequest(BaseModel):
    session_id: str
    field_hint: Optional[str] = None  # Which field to respond about


class SimulateUserResponse(BaseModel):
    user_message: str
    persona_id: str


# =============================================================================
# LLM INTEGRATION
# =============================================================================

def get_openai_client():
    """Get OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return openai.OpenAI(api_key=api_key)


def generate_opening_message(brand_name: str) -> str:
    """
    Generate the very first opening DM message.

    REQUIRED:
    - Human name + role ("Sarah from market research at <brand_name>")
    - Warm compliment based on gym photo + tee context
    - Ask permission to ask questions
    """
    try:
        client = get_openai_client()

        system_prompt = f"""Write ONLY the first DM message.

FORMAT (required):
- Start with a casual greeting and name+role:
  "Hey! I'm Sarah from market research at {brand_name}, ..."
- Include a warm compliment about their gym photo wearing the tee (1 clause).
- End with: asking if they'd mind answering a few questions about the shirt.
- 1–2 sentences, <= 35 words.

RULES:
- Do NOT say "survey", "questionnaire", "feedback form", or "1–10".
- Keep it human and simple.
- Output only the DM text.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}],
            max_tokens=80,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Opening message LLM error: {e}")
        return (
            f"Hey! I’m Sarah from market research at {brand_name}. "
            "Saw your gym pic in our tee — it looks great on you. "
            "Would you mind if I asked you a few questions about the shirt?"
        )


def generate_bot_message(context: dict) -> str:
    """
    Generate post-opener bot messages.

    HARD RULES:
    - NO compliments after the opener.
    - Consent handling:
      * If user agrees -> "Great! So first thing I want to know..."
      * If user refuses -> graceful exit (no more questions)
    - Hard limiter:
      * If a field has been asked once, only ONE follow-up is allowed,
        and it must be an "expand on that" prompt.
      * After 2 total asks, never ask that field again (unless clarifying).
    - Clarification mode:
      * If user is asking what we mean, explain plainly with no fluff.
    """
    strategy = context.get("strategy", "continue_normal")
    brand_name = context.get("brand_name", "ThreadCraft")
    current_field = context.get("current_field")
    last_user_message = context.get("last_user_message", "") or ""
    current_efi = context.get("current_efi", 0.2)

    fields_completed = context.get("fields_completed", [])
    fields_pending = context.get("fields_pending", [])
    recent_question_fields = context.get("recent_question_fields", [])
    question_ask_counts = context.get("question_ask_counts", {})
    questions_exhausted = context.get("questions_exhausted", [])
    last_asked_field = context.get("last_asked_field", None)

    # Context about target field
    target_field_id = None
    target_intent = ""
    if current_field:
        target_field_id = current_field.get("field_id")
        target_intent = current_field.get("question_intent", "")

    try:
        client = get_openai_client()

        system_prompt = f"""You are the DM voice for {brand_name}, continuing a conversation AFTER the opening DM.

POST-OPENER MODE (NON-NEGOTIABLE):
- Do NOT compliment them (no style/appearance/post praise).
- Do NOT re-thank them for the post.
- Keep it friendly but focused.
- Do NOT use the phrase "would you mind"

NO SURVEY VIBES:
- Do NOT say "survey", "questionnaire", "feedback form".
- Do NOT ask ratings/scales (no "1–10", no "rate").
- Ask ONE thing at a time.

GOALS (quietly):
1) Overall quality
2) Would recommend to friends/family
3) Likes
4) Dislikes / what they'd change

STATE / TRACKING:
- Completed fields: {fields_completed}
- Pending fields: {fields_pending}
- Exhausted (never ask again): {questions_exhausted}
- Ask counts by field: {question_ask_counts}
- Last asked field: {last_asked_field}
- Current target field: {target_field_id}

HARD LIMITER (CRITICAL):
- If current target field is in Exhausted -> DO NOT ask it. Ask about a different pending topic.
- If ask count for current target is 0 -> you may ask it normally.
- If ask count is 1 -> you may ask ONE follow-up ONLY as "expand on that" (not re-asking).
  Examples:
  - "Got it — could you say a bit more about what made it feel that way?"
  - "When you say it's 'fine', what part stood out most?"
- If ask count >= 2 -> never revisit that topic again (unless user asks what you meant).
If ask_count for current field is 1:
- You MUST NOT repeat the original question.
- You MUST reference what they already said and ask them to expand.
- Examples:
  - "You said you'd definitely recommend it — what makes it stand out for you?"
  - "You mentioned you'd probably recommend it — is that mainly for the fit, the fabric, or something else?"
- Do not start with "Great! So first thing I want to know..." on follow-ups.

CONSENT HANDLING (RIGHT AFTER OPENER):
If the user's last message clearly AGREES to questions (e.g. "yeah sure", "go ahead", "ok"):
- Reply with: "Great! So first thing I want to know..." then ask ONE question (normal ask count rules apply).

If the user's last message clearly REFUSES (e.g. "no", "not interested", "I'd rather not"):
- Reply with ONE graceful exit line. No questions. No extra fluff.
  Example:
  - "Totally fine — thanks for letting me know. Appreciate you wearing {brand_name}."

CLARIFICATION MODE:
If the user's last message is mainly asking what you meant (e.g. "what do you mean?", "clarify?"):
- No fluff. No compliments. No callbacks.
- Give a plain explanation of what you're asking, optionally ending with ONE clear question.

OUTPUT:
- <= 35 words.
- DM text only. No JSON. No labels."""

        goal_map = {
            "overall_satisfaction": "their overall vibe/quality feel",
            "fit_rating": "fit ONCE only (size/feel) then move on",
            "fabric_quality": "fabric feel/comfort/quality",
            "would_recommend": "whether they'd tell friends about it",
            "improvement_suggestion": "what they'd change / wishlist",
        }
        goal = goal_map.get(target_field_id, target_intent)

        user_prompt = f"""Last user message: "{last_user_message}"
Strategy: {strategy}
Engagement (EFI): {current_efi}
Natural goal this turn (if relevant): {goal}

Write the next DM message following all rules."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=80,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"LLM error: {e}")
        return get_fallback_message(strategy, current_field)


def get_fallback_message(strategy: str, current_field: Optional[dict]) -> str:
    """
    Fallback messages when LLM is unavailable.

    NOTE: Still should avoid compliments post-opener.
    """
    fallbacks = {
        "continue_normal": "Got it. What’s been your favorite part about the shirt so far?",
        "empathize_followup": "That makes sense. What part of it led you to feel that way?",
        "quick_reply_options": "Got it — would you say it’s been mostly good, mixed, or not great?",
        "summarize_confirm": "Got it — thanks for sharing that.",
        "one_last_question": "If you could change one thing about it, what would it be?",
        "ask_nps_direct": "Do you think you’d tell a friend to grab one, or probably not?",
        "exit_graceful": "Totally fine — thanks for letting me know. Appreciate you wearing it.",
    }
    return fallbacks.get(strategy, "Thanks — that helps.")


def simulate_user_response(persona_id: str, bot_message: str, field_hint: Optional[str] = None) -> str:
    """Simulate a user response based on persona (demo)."""
    if field_hint:
        scripted = get_scripted_response(persona_id, field_hint)
        if scripted:
            return scripted

    try:
        client = get_openai_client()
        persona = get_persona(persona_id)
        if not persona:
            return "yeah sure"

        system_prompt = get_persona_system_prompt(persona_id)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Brand DM: \"{bot_message}\"\n\nRespond naturally as this person would."},
            ],
            max_tokens=80,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Persona simulation error: {e}")
        return get_scripted_response(persona_id, "overall_satisfaction") or "thanks!"


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/start", response_model=StartChatResponse)
async def start_chat(request: StartChatRequest):
    """Start a new conversation session."""
    state = create_session(
        survey_id=request.survey_id,
        user_id=request.user_id,
        persona_id=request.persona_id,
    )

    survey = get_survey()
    brand_name = getattr(survey, "brand_name", "ThreadCraft")
    opening = generate_opening_message(brand_name)

    add_bot_message(state, opening)

    return StartChatResponse(
        session_id=state.session_id,
        opening_message=opening,
        persona_id=request.persona_id,
    )


@router.post("/message", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """Process a user message and get bot response."""
    state = get_session(request.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    # Hard stop after exit/completion (no more bot replies)
    if state.status in [ConversationStatus.COMPLETED, ConversationStatus.EXITED]:
        raise HTTPException(status_code=400, detail="Conversation has ended")

    # Process user message (updates metrics/state; may set EXITED on refusal)
    state = process_user_message(state, request.text, request.latency_ms)

    # Extract fields from the message
    extraction_result = extract_all_fields(
        request.text,
        state.session_id,
        state.turn_count,
    )

    # Mark extracted fields using improved thresholding
    for extraction in extraction_result.extractions:
        if is_confident_enough(extraction):
            state = mark_field_extracted(
                state,
                extraction.field_id,
                extraction.extracted_value,
                extraction.confidence,
                extraction.quote,
            )

    # If user refusal (or metrics) ended convo, respond once with graceful exit
    if state.status == ConversationStatus.COMPLETED:
        survey = get_survey()
        bot_response = survey.closing_message

    elif state.status == ConversationStatus.EXITED:
        bot_response = get_graceful_pivot("low_engagement")

    else:
        # Ensure we don't target exhausted fields
        # (advance until askable or no target)
        guard = 0
        while state.next_field_target and (not can_ask_field(state, state.next_field_target)):
            state = advance_to_next_field(state)
            guard += 1
            if guard > 10:
                break

        context = get_writer_context(state)
        bot_response = generate_bot_message(context)

        # Spend an "ask token" for the current target when we're in active questioning
        # (This is the hard limiter that prevents re-asking down the road.)
        state = note_attempted_question(state)

    add_bot_message(state, bot_response)

    return SendMessageResponse(
        bot_response=bot_response,
        metrics=state.last_metrics or {},
        extractions=[e.__dict__ for e in extraction_result.extractions],
        status=state.status.value,
        fields_completed=state.fields_completed,
        fields_remaining=len(state.fields_pending),
    )


@router.get("/state/{session_id}")
async def get_chat_state(session_id: str):
    """Get the full state of a conversation."""
    state = get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.to_dict()


@router.get("/sessions")
async def list_sessions():
    """List all conversation sessions."""
    sessions = get_all_sessions()
    return {
        "count": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "persona_id": s.persona_id,
                "status": s.status.value,
                "turn_count": s.turn_count,
                "completion_rate": round(s.completion_rate, 2),
            }
            for s in sessions
        ],
    }


@router.post("/simulate", response_model=SimulateUserResponse)
async def simulate_user(request: SimulateUserRequest):
    """Simulate a user response (demo)."""
    state = get_session(request.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    if not state.persona_id:
        raise HTTPException(status_code=400, detail="Session has no persona for simulation")

    last_bot_message = ""
    for msg in reversed(state.messages):
        if msg.role == "bot":
            last_bot_message = msg.text
            break

    field_hint = request.field_hint or state.next_field_target
    user_message = simulate_user_response(
        state.persona_id,
        last_bot_message,
        field_hint,
    )

    return SimulateUserResponse(
        user_message=user_message,
        persona_id=state.persona_id,
    )


@router.get("/personas")
async def list_personas():
    """List all available personas for simulation."""
    from src.personas import get_all_personas
    personas = get_all_personas()
    return {"count": len(personas), "personas": [p.to_dict() for p in personas]}
