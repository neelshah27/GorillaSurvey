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
)
from src.extraction_tools import extract_all_fields
from src.survey_schema import get_survey
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


def generate_bot_message(context: dict) -> str:
    """Generate a bot message using the Writer Agent logic."""
    strategy = context.get("strategy", "continue_normal")
    brand_name = context.get("brand_name", "ThreadCraft")
    current_field = context.get("current_field")
    last_user_message = context.get("last_user_message", "")

    # For simple strategies, use templates
    if strategy == "opening":
        return context.get("opening_message", f"Hey! Thanks for shopping with {brand_name} 🧵 Mind if I ask a few quick questions about your recent order?")

    if strategy == "exit_graceful":
        return context.get("closing_message", "Thanks so much for the feedback! Really appreciate you taking the time 🙏")

    # For other strategies, use LLM
    try:
        client = get_openai_client()

        system_prompt = f"""You write DM messages for {brand_name}, a friendly apparel brand.

BRAND VOICE:
- Casual, warm, like texting a friend
- Uses light emoji sparingly (1-2 max per message)
- Never sounds like a survey
- Short sentences, conversational rhythm
- Acknowledges what user said first

CURRENT STRATEGY: {strategy}
- continue_normal: Natural transition to next topic
- empathize_followup: Validate feeling, ask for specifics
- quick_reply_options: Offer 3-4 emoji-labeled choices
- summarize_confirm: Reflect back understanding
- one_last_question: Frame as final, respect their time
- ask_nps_direct: Ask recommendation question naturally

RULES:
- Keep messages under 30 words
- Match energy to user's tone
- Output ONLY the message text, nothing else"""

        field_info = ""
        if current_field:
            field_info = f"\nNEXT TOPIC TO ASK ABOUT: {current_field.get('question_intent', 'general feedback')}"

        user_prompt = f"""Last user message: "{last_user_message}"
{field_info}
Strategy to use: {strategy}

Write the next DM message:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        # Fallback to simple templates
        print(f"LLM error: {e}")
        return get_fallback_message(strategy, current_field)


def get_fallback_message(strategy: str, current_field: Optional[dict]) -> str:
    """Fallback messages when LLM is unavailable."""
    fallbacks = {
        "continue_normal": "And how about the next thing?",
        "empathize_followup": "Got it! Can you tell me a bit more about that?",
        "quick_reply_options": "Quick one: 👍 Great / 😐 Okay / 👎 Not great",
        "summarize_confirm": "Just to make sure I got that right—did I understand correctly?",
        "one_last_question": "Last thing—any quick feedback for us?",
        "ask_nps_direct": "Would you recommend us to a friend?",
        "exit_graceful": "Thanks so much! Really appreciate your time 🙏",
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

    # Get opening message
    survey = get_survey()
    opening = survey.opening_message

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
        survey = get_survey()
        bot_response = survey.closing_message
    elif state.status == ConversationStatus.EXITED:
        bot_response = "Thanks so much for the feedback! Really appreciate you 🙏"
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
