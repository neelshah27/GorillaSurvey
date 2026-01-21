"""
Chat Routes - Handles conversation endpoints.

Endpoints:
- POST /chat/start - Start a new conversation
- POST /chat/message - Send a user message
- GET /chat/state/{session_id} - Get conversation state
- GET /chat/sessions - List all sessions
"""

import os
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
from typing import Dict, Any, List, Optional

from src.agent_config import get_writer_config


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
)
from src.extraction_tools import extract_all_fields
from src.personas import get_persona, get_persona_ids, get_scripted_response, get_persona_system_prompt

router = APIRouter(prefix="/chat", tags=["chat"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class StartChatRequest(BaseModel):
    survey_id: str = "apparel_post_purchase_v1"
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



def generate_bot_message(context: Dict[str, Any]) -> str:
    """
    Generate a bot message using:
    - orchestrator-chosen strategy (context["strategy"])
    - current field target (context["current_field"])
    - recent chat history (context["recent_messages"])
    - metrics snapshot (EFI/IDS/NPS bucket)
    - writer.yaml config (model + instruction)
    """
    brand_name = context.get("brand_name", "GymFish")
    strategy = context.get("strategy", "continue_normal")
    current_field = context.get("current_field")  # dict or None
    turn_count = int(context.get("turn_count", 0) or 0)

    opening_message = context.get(
        "opening_message",
        f"Hey! Thanks for shopping with {brand_name} Mind if I ask a couple quick things about your order?"
    )
    closing_message = context.get(
        "closing_message",
        "Thanks so much for the feedback! Really appreciate you taking the time 🙏"
    )

    user_context = context.get("user_context")

    # Short-circuit strategies that should not hit the LLM
    if strategy == "opening" or turn_count <= 1:
        client = get_openai_client()

        context_note = user_context.strip() if isinstance(user_context, str) and user_context.strip() else None
        if not context_note:
            context_note = "No extra context provided. Use a generic reason like a recent purchase or interaction."

        system_prompt = f"""
You are a friendly, human-sounding member of the {brand_name} marketing team.
Write ONE opening message that sounds natural and varies phrasing each time.

Must include:
- Your made up name with a genaric first name.
- "{brand_name} marketing team" (or close variant that clearly conveys the team).
- A brief, context-based reason for reaching out (use the context provided below).
- A polite ask for permission to ask a few questions.

Style:
- 1-2 sentences, casual, warm, under ~35 words.
- 0-1 emoji.
- Do NOT mention "survey", "questionnaire", "rate", or "scale".
- Do NOT copy a fixed template verbatim.
""".strip()

        user_prompt = f"""
Context to use for the reason we are reaching out:
{context_note}

Return only the message text.
""".strip()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=80,
        )

        return response.choices[0].message.content.strip()
    if strategy == "exit_graceful":
        return generate_exit_message(context)

    # Load writer agent config (yaml) if available
    model_name = "gpt-4o-mini"
    temperature = 0.7
    instruction = None
    try:
        cfg = get_writer_config()
        model_cfg = (cfg.get("model") or {}) if isinstance(cfg, dict) else {}
        model_name = model_cfg.get("name", model_name)
        temperature = model_cfg.get("temperature", temperature)
        instruction = cfg.get("instruction")
    except Exception as e:
        print(f"Writer config load error: {e}")

    if not instruction:
        # Safe default instruction if YAML missing
        instruction = f"You write DM messages for {brand_name}, a friendly apparel brand."

    # Pull small metrics snapshot (writer should FOLLOW strategy, metrics are only for tone)
    current_efi = context.get("current_efi", None)
    current_ids = context.get("current_ids", None)
    nps_bucket = context.get("nps_bucket", "unknown")
    fields_remaining = context.get("fields_remaining", None)

    # Build field prompt
    field_bits = ""
    if isinstance(current_field, dict):
        q_text = current_field.get("question_text") or current_field.get("question")
        q_intent = current_field.get("question_intent") or q_text or "general feedback"
        field_bits = (
            f"TARGET QUESTION INTENT: {q_intent}\n"
            f"TARGET QUESTION TEXT: {q_text or '(phrase naturally)'}\n"
        )

    # Convert recent messages to OpenAI message format
    recent_messages = context.get("recent_messages", []) or []
    history: List[Dict[str, str]] = []
    for m in recent_messages:
        role = m.get("role")
        text = m.get("text")    
        if not text:
            continue
        history.append({
            "role": "assistant" if role == "bot" else "user",
            "content": text
        })

    # Strategy-specific “hard requirements” (keeps policy consistent)
    strategy_requirements = {
        "continue_normal": "Naturally respond, then smoothly move toward the target field with ONE question.",
        "empathize_followup": "Acknowledge emotion/feedback first, then ask ONE specific follow-up question to get details.",
        "quick_reply_options": "Offer 3-5 quick reply options in one line. Keep it super short.",
        "summarize_confirm": "Briefly reflect what the user said (1 sentence), then ask ONE confirm question.",
        "one_last_question": "Frame it as the last quick thing, then ask ONE question.",
        "ask_nps_direct": "Ask recommendation in a casual way. Use a 1–5 framing (1 = no, 5 = definitely).",
    }
    requirement = strategy_requirements.get(strategy, "Be natural and ask at most one question.")

    # Prevent repeated greetings
    no_greeting_rule = "Do NOT greet again (no 'hey', 'hey there', 'hi' openings) unless the user greeted you first."

    # If the user asked a direct question, answer it before moving on
    last_user_message = context.get("last_user_message", None)
    user_question_rule = ""
    if isinstance(last_user_message, str) and "?" in last_user_message:
        user_question_rule = "If the user asked a clarifying question, answer it first, then continue with the strategy."

    system_prompt = f"""{instruction}

    NEVER SAY THINGS LIKE RATE ON A SCALE, SURVEY, DO NOT MAKE IT FEEL LIKE A SURVEY BUT LIKE A FRIEND ASKING FOR EXPLANATION/ADVICE ON SOMETHING SPECIFIC

CONTEXT (do not reveal):
- Brand: {brand_name}
- Strategy to implement: {strategy}
- Fields remaining: {fields_remaining}
- EFI: {current_efi} | IDS: {current_ids} | NPS bucket: {nps_bucket}

STYLE:
- Casual, DM-like, human.
- Short message (under ~35 words).
- 0-1 emoji usually.
- Engage directly with what the user just said when relevant (e.g., apologize for negative feedback, affirm positive feedback).
- Never mention "survey", "questionnaire", "rate", "scale from","scale of", or use "on a scale of..." / "would you say you are very satisfied..." phrasing.
- Use the target question as intent and ask in your own words (do not copy the target question verbatim).
- {no_greeting_rule}
- {user_question_rule}

STRATEGY REQUIREMENT:
- {requirement}

{field_bits}
OUTPUT: only the message text.
""".strip()

    # Director prompt: remind it what the user just said, and what to do next
    director = f"""Last user message: "{last_user_message}"

Write the next message now.""".strip()

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": director},
            ],
            max_tokens=120,
            temperature=temperature,
        )
        text = resp.choices[0].message.content.strip()

        # Simple guard: avoid multiple questions
        if strategy != "quick_reply_options" and text.count("?") > 1:
            first = text.find("?")
            text = text[: first + 1].strip()

        return text

    except Exception as e:
        print(f"LLM error: {e}")
        return get_fallback_message(strategy, current_field)


def generate_exit_message(context: Dict[str, Any]) -> str:
    """Generate a final, appreciative exit message based on NPS bucket."""
    brand_name = context.get("brand_name", "GymFish")
    nps_bucket = context.get("nps_bucket", "unknown")

    # No questions, no prompts for more info.
    if nps_bucket == "promoter":
        return (
            f"Thank you so much for the feedback and for supporting {brand_name},”"
            "it really means a lot to us."
        )
    if nps_bucket == "detractor":
        return (
            "I'm really sorry it didn't meet expectations. "
            "Thank you for sharing your feedback' we'll use it to improve."
        )
    if nps_bucket == "passive":
        return "Thanks so much for the feedback. We'll use it to make improvements."

    return "Thanks so much for the feedback, we really appreciate your time."




def get_fallback_message(strategy: str, current_field: Optional[dict]) -> str:
    """Fallback messages when LLM is unavailable."""
    fallbacks = {
        "continue_normal": "And how about the next thing?",
        "empathize_followup": "Got it! Can you tell me a bit more about that?",
        "quick_reply_options": "Quick one: 👍 Great / 😐 Okay / 👎 Not great",
        "summarize_confirm": "Just to make sure I got that right—did I understand correctly?",
        "one_last_question": "Last thing—any quick feedback for us?",
        "ask_nps_direct": "Would you recommend us to a friend?",
        "exit_graceful": "Thanks so much for the feedback—we really appreciate your time.",
    }

    return fallbacks.get(strategy, "Thanks for sharing! What else can you tell me?")


def simulate_user_response(persona_id: str, bot_message: str, field_hint: Optional[str] = None) -> str:
    """Simulate a user response based on persona."""
    # First try scripted responses
    if field_hint:
        scripted = get_scripted_response(persona_id, field_hint)
        if scripted:
            return scripted

    # Fall back to LLM-generated response
    try:
        client = get_openai_client()
        persona = get_persona(persona_id)

        if not persona:
            return "yeah it's fine"

        system_prompt = get_persona_system_prompt(persona_id)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Brand message: \"{bot_message}\"\n\nRespond as this customer:"}
            ],
            max_tokens=80,
            temperature=0.8,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Persona simulation error: {e}")
        return get_scripted_response(persona_id, "overall_satisfaction")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/start", response_model=StartChatResponse)
async def start_chat(request: StartChatRequest):
    """Start a new conversation session."""
    # Create session
    state = create_session(
        survey_id=request.survey_id,
        user_id=request.user_id,
        persona_id=request.persona_id,
    )

    # Get opening message (dynamic, human-like)
    context = get_writer_context(state)
    opening = generate_bot_message(context)

    # Add bot message to state
    add_bot_message(state, opening)

    return StartChatResponse(
        session_id=state.session_id,
        opening_message=opening,
        persona_id=request.persona_id,
    )


@router.post("/message", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """Process a user message and get bot response."""
    # Get session
    state = get_session(request.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status in [ConversationStatus.COMPLETED, ConversationStatus.EXITED]:
        raise HTTPException(status_code=400, detail="Conversation has ended")

    # Process user message (updates metrics and state)
    state = process_user_message(state, request.text, request.latency_ms)

    # Extract fields from the message
    extraction_result = extract_all_fields(
        request.text,
        state.session_id,
        state.turn_count,
    )

    # Mark extracted fields
    for extraction in extraction_result.extractions:
        if extraction.confidence >= 0.5:
            state = mark_field_extracted(
                state,
                extraction.field_id,
                extraction.extracted_value,
                extraction.confidence,
                extraction.quote,
            )

    # Check if conversation should end
    if state.status == ConversationStatus.COMPLETED:
        context = get_writer_context(state)
        bot_response = generate_exit_message(context)
    elif state.status == ConversationStatus.EXITED:
        context = get_writer_context(state)
        bot_response = generate_exit_message(context)
    else:
        # Generate bot response
        context = get_writer_context(state)
        bot_response = generate_bot_message(context)

    # Add bot message to state
    if state.status not in [ConversationStatus.COMPLETED, ConversationStatus.EXITED]:
        add_bot_message(state, bot_response)
    else:
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
        ]
    }


@router.post("/simulate", response_model=SimulateUserResponse)
async def simulate_user(request: SimulateUserRequest):
    """Simulate a user response (for demo purposes)."""
    state = get_session(request.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    if not state.persona_id:
        raise HTTPException(status_code=400, detail="Session has no persona for simulation")

    # Get the last bot message
    last_bot_message = ""
    for msg in reversed(state.messages):
        if msg.role == "bot":
            last_bot_message = msg.text
            break

    # Determine field hint
    field_hint = request.field_hint or state.next_field_target

    # Simulate response
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
    return {
        "count": len(personas),
        "personas": [p.to_dict() for p in personas]
    }



