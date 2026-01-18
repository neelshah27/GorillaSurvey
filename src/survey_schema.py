"""
Survey Schema - Defines the survey structure and brand configuration.

TONE: Warm, human, conversational. Like a friendly brand rep, not a hype beast.
RULE: First message must end with a small open-ended question.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class FieldType(str, Enum):
    RATING = "rating"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    OPEN_TEXT = "open_text"
    NUMERIC = "numeric"


@dataclass
class SurveyField:
    """A single field in the survey schema."""
    field_id: str
    field_type: FieldType
    question_intent: str
    valid_values: Optional[List[str]] = None
    priority: int = 1
    required: bool = True

    def to_dict(self) -> Dict:
        return {
            "field_id": self.field_id,
            "type": self.field_type.value,
            "question_intent": self.question_intent,
            "valid_values": self.valid_values,
            "priority": self.priority,
            "required": self.required,
        }


@dataclass
class SurveySchema:
    """Complete survey schema with brand configuration."""
    survey_id: str
    brand_name: str
    brand_voice: str
    fields: List[SurveyField]
    opening_message: str
    closing_message: str

    def to_dict(self) -> Dict:
        return {
            "survey_id": self.survey_id,
            "brand_name": self.brand_name,
            "brand_voice": self.brand_voice,
            "fields": [f.to_dict() for f in self.fields],
            "opening_message": self.opening_message,
            "closing_message": self.closing_message,
        }

    def get_field(self, field_id: str) -> Optional[SurveyField]:
        for f in self.fields:
            if f.field_id == field_id:
                return f
        return None

    def get_fields_by_priority(self) -> List[SurveyField]:
        return sorted(self.fields, key=lambda f: f.priority)


# =============================================================================
# HARDCODED DEMO SURVEY - WARM, HUMAN TONE
# =============================================================================

APPAREL_SURVEY = SurveySchema(
    survey_id="apparel_social_reactive_v1",
    brand_name="ThreadCraft",
    # UPDATED: Calmer, more genuine voice
    brand_voice="warm, genuine, conversational, like a friendly brand rep who actually cares",
    fields=[
        SurveyField(
            field_id="overall_satisfaction",
            field_type=FieldType.RATING,
            question_intent="Capture overall sentiment through their response to the compliment",
            valid_values=["loved", "liked", "neutral", "disappointed"],
            priority=1,
        ),
        SurveyField(
            field_id="fit_rating",
            field_type=FieldType.CATEGORICAL,
            question_intent="Learn about fit through natural style conversation",
            valid_values=["perfect", "slightly_small", "slightly_large", "wrong_size"],
            priority=2,
        ),
        SurveyField(
            field_id="fabric_quality",
            field_type=FieldType.RATING,
            question_intent="Understand fabric perception through wear experience",
            valid_values=["excellent", "good", "average", "poor"],
            priority=3,
        ),
        SurveyField(
            field_id="would_recommend",
            field_type=FieldType.BOOLEAN,
            question_intent="Gauge recommendation likelihood through social context",
            valid_values=["yes", "no", "maybe"],
            priority=4,
        ),
        SurveyField(
            field_id="improvement_suggestion",
            field_type=FieldType.OPEN_TEXT,
            question_intent="Extract improvement ideas through aspirational framing",
            valid_values=None,
            priority=5,
            required=False,
        ),
    ],
    # UPDATED: Warm opener that ends with open-ended question
    opening_message="Hey — saw your post in our tee. It really suits you. How did it feel wearing it?",
    # UPDATED: Genuine, warm close
    closing_message="Really appreciate you sharing all that. Thanks for being part of what we do.",
)


# =============================================================================
# CONVERSATIONAL HOOKS - WARM AND NATURAL
# =============================================================================
# Gentle nudges, not hype. Each naturally steers toward a topic.

CONVERSATION_HOOKS = {
    # OVERALL SATISFACTION: Warm acknowledgment, let them share
    "overall_satisfaction": [
        "It looks great on you, honestly.",
        "The way you styled it works really well.",
        "You wear it better than we imagined.",
    ],

    # FIT RATING: Comment on how it looks, they'll mention fit
    "fit_rating": [
        "Looks like it fits you well — did you go with your usual size?",
        "The fit looks right. How does it actually feel?",
        "It seems to sit nicely. True to size for you?",
    ],

    # FABRIC QUALITY: Talk about comfort naturally
    "fabric_quality": [
        "Curious — how's the fabric holding up?",
        "Is it as comfortable as it looks in the photo?",
        "Has it stayed soft after washing?",
    ],

    # WOULD RECOMMEND: Frame around their social reach
    "would_recommend": [
        "Your friends might ask where you got it.",
        "Would you mind if we shared your post?",
        "Think others in your circle would be into it?",
    ],

    # IMPROVEMENT SUGGESTION: Thoughtful framing
    "improvement_suggestion": [
        "If you could change one thing about it, what would it be?",
        "Anything you wish we'd done differently?",
        "What would make it even better for you?",
    ],
}

# Legacy alias for compatibility
QUESTION_TEMPLATES = CONVERSATION_HOOKS


# =============================================================================
# DEPTH PROBES - GENUINE CURIOSITY
# =============================================================================
# When responses are vague, ask with real interest

DEPTH_PROBES = {
    "overall_satisfaction": "What made you feel that way?",
    "fit_rating": "Was it the length, the shoulders, or something else?",
    "fabric_quality": "What specifically stood out about it?",
    "would_recommend": "What would make you more confident recommending it?",
    "improvement_suggestion": "Even small things — anything come to mind?",
}

# Legacy alias
FOLLOWUP_PROBES = DEPTH_PROBES


# =============================================================================
# QUICK REACTIONS - SIMPLE EMOJI OPTIONS
# =============================================================================
# For disengaging users, keep it easy

QUICK_VIBES = {
    "overall_satisfaction": ["👍 Love it", "🙂 It's good", "😐 Meh", "👎 Not great"],
    "fit_rating": ["👌 Perfect", "🤏 A bit snug", "📐 A bit loose", "❌ Way off"],
    "fabric_quality": ["👍 Great", "🙂 Good", "😐 Okay", "👎 Not good"],
    "would_recommend": ["👍 Yes", "🤷 Maybe", "👎 No"],
}

# Legacy alias
QUICK_REPLY_OPTIONS = QUICK_VIBES


# =============================================================================
# GRACEFUL PIVOTS - RESPECTFUL EXITS
# =============================================================================
# Human, warm exits that respect their time

GRACEFUL_PIVOTS = {
    "low_engagement": [
        "No worries, thanks for the post. Appreciate you wearing it.",
        "All good — just wanted to say thanks for sharing.",
        "Won't keep you. Thanks for rocking it.",
    ],
    "negative_sentiment": [
        "That's fair, and thanks for being honest with us.",
        "Sorry to hear that. We appreciate you letting us know.",
        "Noted — genuinely, thank you for the feedback.",
    ],
    "hostile": [
        "Totally understand. Thanks for wearing it either way.",
        "Fair enough. Appreciate the honesty.",
    ],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_survey() -> SurveySchema:
    """Get the demo survey schema."""
    return APPAREL_SURVEY


def get_conversation_hook(field_id: str, variant: int = 0) -> str:
    """Get a conversational hook for a field."""
    hooks = CONVERSATION_HOOKS.get(field_id, ["That's nice to hear."])
    return hooks[variant % len(hooks)]


def get_depth_probe(field_id: str) -> str:
    """Get a depth probe for shallow responses."""
    return DEPTH_PROBES.get(field_id, "Can you tell me a bit more?")


def get_quick_vibes(field_id: str) -> List[str]:
    """Get quick emoji reactions for disengaging users."""
    return QUICK_VIBES.get(field_id, ["👍 Yes", "👎 No", "🤷 Not sure"])


def get_graceful_pivot(situation: str = "low_engagement") -> str:
    """Get a graceful exit line."""
    import random
    pivots = GRACEFUL_PIVOTS.get(situation, GRACEFUL_PIVOTS["low_engagement"])
    return random.choice(pivots)


# Legacy compatibility
def get_question_template(field_id: str, variant: int = 0) -> str:
    return get_conversation_hook(field_id, variant)


def get_followup_probe(field_id: str) -> str:
    return get_depth_probe(field_id)


def get_quick_replies(field_id: str) -> List[str]:
    return get_quick_vibes(field_id)
