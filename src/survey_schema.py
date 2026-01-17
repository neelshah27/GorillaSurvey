"""
Survey Schema - Defines the survey structure and brand configuration.

For the hackathon demo, we use a hardcoded apparel post-purchase survey.
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
# HARDCODED DEMO SURVEY
# =============================================================================

APPAREL_SURVEY = SurveySchema(
    survey_id="apparel_post_purchase_v1",
    brand_name="ThreadCraft",
    brand_voice="friendly, casual, millennial-coded, warm but not overly enthusiastic",
    fields=[
        SurveyField(
            field_id="overall_satisfaction",
            field_type=FieldType.RATING,
            question_intent="How did they feel about the purchase overall?",
            valid_values=["loved", "liked", "neutral", "disappointed"],
            priority=1,
        ),
        SurveyField(
            field_id="fit_rating",
            field_type=FieldType.CATEGORICAL,
            question_intent="How was the fit/sizing?",
            valid_values=["perfect", "slightly_small", "slightly_large", "wrong_size"],
            priority=2,
        ),
        SurveyField(
            field_id="fabric_quality",
            field_type=FieldType.RATING,
            question_intent="What did they think of material/fabric?",
            valid_values=["excellent", "good", "average", "poor"],
            priority=3,
        ),
        SurveyField(
            field_id="would_recommend",
            field_type=FieldType.BOOLEAN,
            question_intent="Would they recommend to a friend?",
            valid_values=["yes", "no", "maybe"],
            priority=4,
        ),
        SurveyField(
            field_id="improvement_suggestion",
            field_type=FieldType.OPEN_TEXT,
            question_intent="Any suggestions for improvement?",
            valid_values=None,
            priority=5,
            required=False,
        ),
    ],
    opening_message="Hey! Thanks for shopping with ThreadCraft 🧵 Mind if I ask a few quick questions about your recent order?",
    closing_message="Thanks so much for the feedback! Really appreciate you taking the time 🙏",
)


# =============================================================================
# NATURAL LANGUAGE QUESTION TEMPLATES
# =============================================================================

# These are used by the Writer Agent to generate natural questions
QUESTION_TEMPLATES = {
    "overall_satisfaction": [
        "So how are you feeling about your order overall?",
        "What's your first impression of everything?",
        "How's the {product} treating you so far?",
    ],
    "fit_rating": [
        "And how did the sizing work out for you?",
        "Did the fit match what you expected?",
        "How's it fitting?",
    ],
    "fabric_quality": [
        "What do you think of the fabric/material?",
        "How does it feel quality-wise?",
        "And the material itself—thoughts?",
    ],
    "would_recommend": [
        "Would you recommend us to a friend?",
        "Think your friends would like this too?",
        "Would you tell others about us?",
    ],
    "improvement_suggestion": [
        "Any suggestions on what we could do better?",
        "Anything you wish was different?",
        "What would make it even better for you?",
    ],
}

# Follow-up probes when IDS is low
FOLLOWUP_PROBES = {
    "overall_satisfaction": "What specifically made you feel that way?",
    "fit_rating": "Was it the length, the width, or something else with the sizing?",
    "fabric_quality": "What about the fabric stood out to you?",
    "would_recommend": "What would make you more likely to recommend us?",
    "improvement_suggestion": "Even small things—anything come to mind?",
}

# Quick reply options for high-friction moments
QUICK_REPLY_OPTIONS = {
    "overall_satisfaction": ["😍 Loved it", "👍 Pretty good", "😐 It's okay", "👎 Disappointed"],
    "fit_rating": ["👌 Perfect fit", "📏 Bit small", "📐 Bit large", "❌ Way off"],
    "fabric_quality": ["⭐ Excellent", "👍 Good", "😐 Average", "👎 Poor"],
    "would_recommend": ["💯 Definitely", "🤔 Maybe", "❌ Probably not"],
}


def get_survey() -> SurveySchema:
    """Get the demo survey schema."""
    return APPAREL_SURVEY


def get_question_template(field_id: str, variant: int = 0) -> str:
    """Get a question template for a field."""
    templates = QUESTION_TEMPLATES.get(field_id, ["Tell me more about that?"])
    return templates[variant % len(templates)]


def get_followup_probe(field_id: str) -> str:
    """Get a follow-up probe for a field."""
    return FOLLOWUP_PROBES.get(field_id, "Can you tell me more about that?")


def get_quick_replies(field_id: str) -> List[str]:
    """Get quick reply options for a field."""
    return QUICK_REPLY_OPTIONS.get(field_id, ["👍 Yes", "👎 No", "🤷 Not sure"])
