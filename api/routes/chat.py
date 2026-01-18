"""
Chat Routes - Handles conversation endpoints.

REVAMPED: All prompts rewritten for natural, reactive DM conversations.
Context: Brand slides into DMs after seeing user's social post wearing their product.

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
# LLM INTEGRATION - WARM, HUMAN TONE
# =============================================================================

def get_openai_client():
    """Get OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return openai.OpenAI(api_key=api_key)


def generate_bot_message(context: dict) -> str:
    """
    Generate a bot message using the Writer Agent logic.

    TONE: Warm, human, conversational. Like a friendly brand rep, not a hype beast.
    RULE: First message must end with a small open-ended question.
    """
    strategy = context.get("strategy", "continue_normal")
    brand_name = context.get("brand_name", "ThreadCraft")
    current_field = context.get("current_field")
    last_user_message = context.get("last_user_message", "")
    current_efi = context.get("current_efi", 0.2)

    # For opening, use warm opener with open-ended question
    if strategy == "opening":
        return context.get("opening_message",
            f"Hey — saw your post in our tee. It really suits you. How did it feel wearing it?")

    # For graceful exit, use warm appreciation
    if strategy == "exit_graceful":
        # Check if they were negative or just disengaged
        if current_efi > 0.7:
            return get_graceful_pivot("low_engagement")
        return context.get("closing_message",
            "Really appreciate you sharing all that. Thanks for being part of what we do.")

    # For other strategies, use LLM with warm, human prompt
    try:
        client = get_openai_client()

        system_prompt = f"""You are the social media voice for {brand_name}, reaching out after
seeing someone post a photo wearing your brand. You're warm and genuine, not a survey bot.

THE SCENARIO:
- User posted a photo wearing {brand_name} gear
- They didn't say anything about the product in their post
- You genuinely appreciate them wearing your brand
- You want to connect naturally and hear what they think

YOUR TONE:
- Warm and friendly, like a brand rep who actually cares
- Genuine appreciation, not forced enthusiasm
- Conversational, casual but not over-the-top
- React to what they said, then let conversation flow
- Sound like a real person, not a script

ABSOLUTE RULES (break these = you're a survey bot):
❌ NEVER use: "rate", "scale", "feedback", "survey", "question", "1-10"
❌ NEVER ask permission: "mind if I ask", "can I ask you", "quick question"
❌ NEVER use corporate phrases: "we'd love to hear", "your opinion matters"
❌ NEVER ask multiple things at once
❌ NEVER sound like you're gathering data

✅ DO react to what they said before moving on
✅ DO match their energy level
✅ DO compliment genuinely but without hyperbole
✅ DO pivot gracefully if they're not engaged
✅ DO make it about THEM not about getting info

CURRENT STRATEGY: {strategy}
- continue_normal: Comment that naturally leads to next topic
- empathize_followup: Genuine curiosity about what they said
- quick_reply_options: Easy emoji options (👍 Love it / 🙂 Good / 😐 Meh)
- summarize_confirm: Reflect back to show you're listening
- one_last_question: Genuine curiosity, not "final question"
- ask_nps_direct: Frame around their social circle naturally

OUTPUT: Just the DM message. Under 25 words. Warm and human, never pushy."""

        # Build context about what to naturally steer toward
        field_context = ""
        if current_field:
            intent = current_field.get('question_intent', '')
            # Translate survey intent to conversational goal
            goal_map = {
                "overall_satisfaction": "gauge their vibe about the product through the compliment exchange",
                "fit_rating": "learn about fit by commenting on how it looks on them",
                "fabric_quality": "get fabric thoughts through comfort/lifestyle chat",
                "would_recommend": "sense advocacy through social sharing conversation",
                "improvement_suggestion": "surface wishlist through dream collab framing",
            }
            field_id = current_field.get('field_id', '')
            goal = goal_map.get(field_id, intent)
            field_context = f"\nNATURAL GOAL: {goal} (but don't ask directly, let it come up organically)"

        user_prompt = f"""Their last message: "{last_user_message}"
{field_context}
Strategy: {strategy}

Write your DM response (warm and human, not survey bot):"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=80,
            temperature=0.85,  # Higher for more natural variation
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        # Fallback to hyped templates
        print(f"LLM error: {e}")
        return get_fallback_message(strategy, current_field)


def get_fallback_message(strategy: str, current_field: Optional[dict]) -> str:
    """
    Fallback messages when LLM is unavailable.

    TONE: Warm, human, conversational. Not hype.
    """
    fallbacks = {
        "continue_normal": "The way you styled it works really well.",
        "empathize_followup": "What made you feel that way?",
        "quick_reply_options": "Quick check: 👍 Love it / 🙂 Good / 😐 Meh?",
        "summarize_confirm": "Got it — thanks for sharing that.",
        "one_last_question": "If you could change one thing about it, what would it be?",
        "ask_nps_direct": "Your friends might ask where you got it.",
        "exit_graceful": "No worries, thanks for the post. Appreciate you wearing it.",
    }

    return fallbacks.get(strategy, "Thanks for sharing that.")


def simulate_user_response(persona_id: str, bot_message: str, field_hint: Optional[str] = None) -> str:
    """
    Simulate a user response based on persona.

    Responds naturally to brand DMs, not survey questions.
    """
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
            return "haha thanks! yeah I love it"

        system_prompt = get_persona_system_prompt(persona_id)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Brand DM: \"{bot_message}\"\n\nRespond naturally as this person would to a brand sliding into their DMs after they posted a photo:"}
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
    """
    Start a new conversation session.

    Context: Brand slides into DMs after seeing user's post.
    """
    # Create session
    state = create_session(
        survey_id=request.survey_id,
        user_id=request.user_id,
        persona_id=request.persona_id,
    )

    # Get opening message (now a hype reaction, not survey intro)
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
    """
    Process a user message and get bot response.

    The bot responds warmly and naturally, extracting data conversationally.
    """
    # Get session
    state = get_session(request.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.status in [ConversationStatus.COMPLETED, ConversationStatus.EXITED]:
        raise HTTPException(status_code=400, detail="Conversation has ended")

    # Process user message (updates metrics and state)
    state = process_user_message(state, request.text, request.latency_ms)

    # Extract fields from the message (data capture still works!)
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
        # Graceful exit with friendship energy
        bot_response = get_graceful_pivot("low_engagement")
    else:
        # Generate bot response (warm, conversational style)
        context = get_writer_context(state)
        bot_response = generate_bot_message(context)

    # Add bot message to state
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
    """
    Simulate a user response (for demo purposes).

    Simulates how a user would respond to a brand's DM.
    """
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
