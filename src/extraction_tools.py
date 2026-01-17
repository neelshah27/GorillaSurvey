"""
Extraction Tools - Maps user responses to survey fields.

Uses a combination of:
1. Rule-based keyword extraction (fast, no LLM)
2. LLM-based extraction (for complex/ambiguous cases)
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .survey_schema import get_survey, FieldType


@dataclass
class ExtractionResult:
    """Result of extracting a single field."""
    field_id: str
    extracted_value: Optional[str]
    confidence: float
    reasoning: str
    quote: str


@dataclass
class ExtractionResponse:
    """Complete extraction response for a user message."""
    session_id: str
    turn: int
    user_text: str
    extractions: List[ExtractionResult]
    no_match_fields: List[str]

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "user_text": self.user_text,
            "extractions": [
                {
                    "field_id": e.field_id,
                    "extracted_value": e.extracted_value,
                    "confidence": round(e.confidence, 2),
                    "reasoning": e.reasoning,
                    "quote": e.quote,
                }
                for e in self.extractions
            ],
            "no_match_fields": self.no_match_fields,
        }


# =============================================================================
# EXTRACTION PATTERNS (Rule-based)
# =============================================================================

# Overall satisfaction patterns
SATISFACTION_PATTERNS = {
    "loved": [
        r"\b(love[d]?|amazing|awesome|fantastic|perfect|excellent|obsessed|best)\b",
        r"(😍|❤️|🔥|💯)",
    ],
    "liked": [
        r"\b(like[d]?|good|great|nice|happy|pleased|satisfied)\b",
        r"(👍|😊|👌)",
    ],
    "neutral": [
        r"\b(okay|ok|fine|alright|decent|average|meh)\b",
        r"(😐|🤷)",
    ],
    "disappointed": [
        r"\b(disappoint|bad|terrible|awful|hate|worst|horrible|regret)\b",
        r"(👎|😡|😤)",
    ],
}

# Fit rating patterns
FIT_PATTERNS = {
    "perfect": [
        r"\b(perfect|spot\s*on|exactly|just\s*right|true\s*to\s*size)\b",
        r"\bfit[s]?\s*(great|perfect|well|good)\b",
    ],
    "slightly_small": [
        r"\b(small|tight|snug|short|narrow)\b",
        r"\b(size\s*up|bigger)\b",
        r"\bruns?\s*small\b",
    ],
    "slightly_large": [
        r"\b(large|big|loose|long|baggy|wide)\b",
        r"\b(size\s*down|smaller)\b",
        r"\bruns?\s*(large|big)\b",
    ],
    "wrong_size": [
        r"\b(way\s*(too|off)|completely|totally)\s*(small|big|wrong|off)\b",
        r"\b(exchange|return|wrong\s*size)\b",
    ],
}

# Fabric quality patterns
FABRIC_PATTERNS = {
    "excellent": [
        r"\b(excellent|amazing|fantastic|premium|luxurious|high\s*quality)\b",
        r"\b(love|loved)\s*(the\s*)?(fabric|material|quality)\b",
        r"\b(super|really|very|so)\s*(soft|comfortable|nice)\b",
    ],
    "good": [
        r"\b(good|nice|soft|comfortable|decent)\s*(fabric|material|quality)?\b",
        r"\b(fabric|material)\s*(is\s*)?(good|nice|soft)\b",
    ],
    "average": [
        r"\b(okay|ok|fine|average|normal|typical)\b",
        r"\b(nothing\s*special|expected)\b",
    ],
    "poor": [
        r"\b(poor|cheap|thin|flimsy|rough|scratchy|bad)\b",
        r"\b(low\s*quality|disappointing)\b",
    ],
}

# Would recommend patterns
RECOMMEND_PATTERNS = {
    "yes": [
        r"\b(yes|yeah|yep|definitely|absolutely|totally|100|for\s*sure)\b",
        r"\b(would|will)\s*(definitely\s*)?(recommend|tell)\b",
        r"\b(already\s*)?(told|recommend)\b",
        r"(💯|👍)",
    ],
    "no": [
        r"\b(no|nope|nah|never|wouldn'?t|won'?t)\b",
        r"\bnot\s*(really|likely|gonna)\b",
        r"(👎)",
    ],
    "maybe": [
        r"\b(maybe|perhaps|depends|possibly|might|unsure|not\s*sure)\b",
        r"\b(idk|don'?t\s*know)\b",
        r"(🤷|🤔)",
    ],
}


# =============================================================================
# RULE-BASED EXTRACTION
# =============================================================================

def match_patterns(text: str, patterns: Dict[str, List[str]]) -> Tuple[Optional[str], float, str]:
    """
    Match text against pattern dictionary.
    Returns (matched_value, confidence, matched_quote).
    """
    text_lower = text.lower()

    for value, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                # Higher confidence for more specific patterns
                confidence = 0.7 if len(pattern) > 20 else 0.6
                return value, confidence, match.group(0)

    return None, 0.0, ""


def extract_overall_satisfaction(text: str) -> ExtractionResult:
    """Extract overall satisfaction from text."""
    value, confidence, quote = match_patterns(text, SATISFACTION_PATTERNS)

    if value:
        return ExtractionResult(
            field_id="overall_satisfaction",
            extracted_value=value,
            confidence=confidence,
            reasoning=f"Matched pattern for '{value}'",
            quote=quote,
        )

    return ExtractionResult(
        field_id="overall_satisfaction",
        extracted_value=None,
        confidence=0.0,
        reasoning="No satisfaction keywords found",
        quote="",
    )


def extract_fit_rating(text: str) -> ExtractionResult:
    """Extract fit rating from text."""
    value, confidence, quote = match_patterns(text, FIT_PATTERNS)

    if value:
        return ExtractionResult(
            field_id="fit_rating",
            extracted_value=value,
            confidence=confidence,
            reasoning=f"Matched fit pattern for '{value}'",
            quote=quote,
        )

    return ExtractionResult(
        field_id="fit_rating",
        extracted_value=None,
        confidence=0.0,
        reasoning="No fit-related keywords found",
        quote="",
    )


def extract_fabric_quality(text: str) -> ExtractionResult:
    """Extract fabric quality from text."""
    value, confidence, quote = match_patterns(text, FABRIC_PATTERNS)

    if value:
        return ExtractionResult(
            field_id="fabric_quality",
            extracted_value=value,
            confidence=confidence,
            reasoning=f"Matched fabric pattern for '{value}'",
            quote=quote,
        )

    return ExtractionResult(
        field_id="fabric_quality",
        extracted_value=None,
        confidence=0.0,
        reasoning="No fabric-related keywords found",
        quote="",
    )


def extract_would_recommend(text: str) -> ExtractionResult:
    """Extract recommendation intent from text."""
    value, confidence, quote = match_patterns(text, RECOMMEND_PATTERNS)

    if value:
        return ExtractionResult(
            field_id="would_recommend",
            extracted_value=value,
            confidence=confidence,
            reasoning=f"Matched recommendation pattern for '{value}'",
            quote=quote,
        )

    return ExtractionResult(
        field_id="would_recommend",
        extracted_value=None,
        confidence=0.0,
        reasoning="No recommendation keywords found",
        quote="",
    )


def extract_improvement_suggestion(text: str) -> ExtractionResult:
    """
    Extract improvement suggestions.
    For open text, we capture the full response if it contains suggestion-like content.
    """
    suggestion_indicators = [
        r"\b(suggest|suggestion|improve|better|wish|would\s*be\s*nice|could|should)\b",
        r"\b(more|less|different|change|add|remove)\b",
        r"\b(one\s*thing|only\s*thing|feedback)\b",
    ]

    text_lower = text.lower()

    for pattern in suggestion_indicators:
        if re.search(pattern, text_lower):
            # For open text, capture a meaningful portion
            # Remove common filler phrases
            cleaned = re.sub(r"^(i\s*think|i\s*mean|like|um|uh)\s*", "", text_lower)
            cleaned = cleaned.strip()

            if len(cleaned) > 10:  # Meaningful content
                return ExtractionResult(
                    field_id="improvement_suggestion",
                    extracted_value=text,  # Keep original casing
                    confidence=0.8,
                    reasoning="Contains suggestion-like content",
                    quote=text[:100],
                )

    # Check if it's a negative response (no suggestion)
    no_suggestion_patterns = [
        r"\b(nothing|none|no|nope|nah|can'?t\s*think|idk|don'?t\s*know)\b",
    ]

    for pattern in no_suggestion_patterns:
        if re.search(pattern, text_lower):
            return ExtractionResult(
                field_id="improvement_suggestion",
                extracted_value="none",
                confidence=0.7,
                reasoning="User indicated no suggestions",
                quote=text[:50],
            )

    return ExtractionResult(
        field_id="improvement_suggestion",
        extracted_value=None,
        confidence=0.0,
        reasoning="No clear suggestion content found",
        quote="",
    )


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

EXTRACTORS = {
    "overall_satisfaction": extract_overall_satisfaction,
    "fit_rating": extract_fit_rating,
    "fabric_quality": extract_fabric_quality,
    "would_recommend": extract_would_recommend,
    "improvement_suggestion": extract_improvement_suggestion,
}


def extract_fields(
    text: str,
    target_fields: List[str],
    session_id: str = "",
    turn: int = 0,
) -> ExtractionResponse:
    """
    Extract multiple fields from user text.

    Args:
        text: User's message
        target_fields: List of field IDs to try extracting
        session_id: Session identifier
        turn: Current turn number

    Returns:
        ExtractionResponse with all extractions
    """
    extractions = []
    no_match_fields = []

    for field_id in target_fields:
        extractor = EXTRACTORS.get(field_id)
        if extractor:
            result = extractor(text)
            if result.extracted_value is not None and result.confidence > 0:
                extractions.append(result)
            else:
                no_match_fields.append(field_id)
        else:
            no_match_fields.append(field_id)

    return ExtractionResponse(
        session_id=session_id,
        turn=turn,
        user_text=text,
        extractions=extractions,
        no_match_fields=no_match_fields,
    )


def extract_all_fields(text: str, session_id: str = "", turn: int = 0) -> ExtractionResponse:
    """Extract all survey fields from text."""
    survey = get_survey()
    target_fields = [f.field_id for f in survey.fields]
    return extract_fields(text, target_fields, session_id, turn)


# =============================================================================
# LLM-BASED EXTRACTION (for complex cases)
# =============================================================================

def get_extraction_prompt(text: str, field_id: str) -> str:
    """
    Generate a prompt for LLM-based extraction.
    Use this when rule-based extraction fails or has low confidence.
    """
    survey = get_survey()
    field = survey.get_field(field_id)

    if not field:
        return ""

    valid_values_str = ", ".join(field.valid_values) if field.valid_values else "any text"

    return f"""Extract the following survey field from the user's message.

Field: {field_id}
Description: {field.question_intent}
Valid values: {valid_values_str}
Field type: {field.field_type.value}

User message: "{text}"

Respond with JSON:
{{
    "field_id": "{field_id}",
    "extracted_value": <value or null if not found>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>",
    "quote": "<relevant part of user message>"
}}

If the user's message doesn't contain information about this field, set extracted_value to null and confidence to 0.
Be conservative - only extract if you're reasonably confident."""


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    test_messages = [
        "loved it! fabric is amazing, fits perfect",
        "it's fine I guess",
        "The sizing was off - ordered M but it fits more like S. Material is nice though.",
        "terrible. returning it. waste of money",
        "would definitely recommend, already told my sister",
        "More color options would be nice, maybe forest green?",
        "👍 good stuff",
        "idk whatever",
    ]

    print("=" * 80)
    print("EXTRACTION TOOLS - TEST OUTPUT")
    print("=" * 80)

    for text in test_messages:
        print(f"\nMessage: \"{text}\"")
        response = extract_all_fields(text, "test", 1)

        if response.extractions:
            for e in response.extractions:
                print(f"  ✓ {e.field_id}: {e.extracted_value} (conf: {e.confidence:.2f})")
                print(f"    Quote: \"{e.quote}\"")
        else:
            print("  No extractions")

        if response.no_match_fields:
            print(f"  No match: {', '.join(response.no_match_fields)}")
