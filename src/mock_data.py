"""
Mock data seeding for dashboard demos.

Creates in-memory sessions with diverse themes and outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict
import random

from .orchestrator_tools import (
    create_session,
    update_session,
    ConversationStatus,
    Message,
    ExtractedField,
    get_all_sessions,
)
from .survey_questions import get_question_map


@dataclass
class MockTemplate:
    topic: str
    status: ConversationStatus
    nps_p: float
    sentiment: float
    user_lines: List[str]
    improvement: str
    turn_count: int


TOPIC_LIBRARY = {
    "sizing": {
        "lines": [
            "The fit runs a bit tight in the shoulders.",
            "Sleeves feel short after a wash.",
            "Size chart is close but not perfect.",
        ],
        "improvement": "Sizing runs small; a clearer size guide would help.",
    },
    "fabric": {
        "lines": [
            "Fabric feels soft and breathable during workouts.",
            "Material wicks sweat well but pills a little.",
            "Texture is comfy but could be smoother.",
        ],
        "improvement": "Love the fabric, but reduce pilling after washes.",
    },
    "shipping": {
        "lines": [
            "Shipping was slow and the tracking was vague.",
            "Package arrived two days late.",
            "Delivery updates were inconsistent.",
        ],
        "improvement": "Faster shipping or clearer tracking would be great.",
    },
    "price": {
        "lines": [
            "Price feels a bit high for a basic tee.",
            "Good quality but expensive compared to other brands.",
            "I'd buy again if there was a small discount.",
        ],
        "improvement": "A loyalty discount would make the price feel fair.",
    },
    "quality": {
        "lines": [
            "Stitching looks solid and the seams feel durable.",
            "Quality is great, no loose threads so far.",
            "Construction is strong but the hem is slightly uneven.",
        ],
        "improvement": "Quality is strong overall, just tighten QC on hems.",
    },
    "colors": {
        "lines": [
            "Color is vibrant and matches the photos.",
            "Would love more color options for this line.",
            "The shade is nice but it fades a bit.",
        ],
        "improvement": "Add more color options and improve fade resistance.",
    },
    "returns": {
        "lines": [
            "Return process was easy and quick.",
            "Exchange took longer than expected.",
            "Policy is clear but shipping cost was on me.",
        ],
        "improvement": "Prepaid return labels would make exchanges easier.",
    },
    "photos": {
        "lines": [
            "Product photos were accurate and helpful.",
            "The shirt looked thicker in the pictures.",
            "Lighting makes the color look different.",
        ],
        "improvement": "Use more consistent lighting in product photos.",
    },
    "variety": {
        "lines": [
            "Love the style, but there are limited options.",
            "Would be cool to see more cuts and fits.",
            "Selection feels small compared to other brands.",
        ],
        "improvement": "Expand styles and add more fit options.",
    },
}


def _bucket_from_nps(nps_p: float) -> str:
    if nps_p >= 0.7:
        return "promoter"
    if nps_p >= 0.4:
        return "passive"
    return "detractor"


def _build_messages(start: datetime, lines: List[str]) -> List[Message]:
    messages: List[Message] = []
    now = start
    turn = 0
    for line in lines:
        turn += 1
        messages.append(
            Message(
                turn=turn,
                role="bot" if turn % 2 == 1 else "user",
                text=line,
                timestamp=now,
                latency_ms=None if turn % 2 == 1 else random.randint(1500, 9000),
            )
        )
        now += timedelta(minutes=random.randint(1, 4))
    return messages


def seed_mock_sessions(count: int = 20) -> None:
    """Seed mock sessions once per process."""
    if get_all_sessions():
        return

    question_map = get_question_map()
    topics = list(TOPIC_LIBRARY.keys())

    templates: List[MockTemplate] = []
    for idx in range(count):
        topic = topics[idx % len(topics)]
        base = TOPIC_LIBRARY[topic]

        if idx % 5 == 0:
            status = ConversationStatus.EXITED
            nps_p = 0.25 + (idx % 3) * 0.05
            sentiment = -0.6
        elif idx % 4 == 0:
            status = ConversationStatus.IN_PROGRESS
            nps_p = 0.5
            sentiment = 0.0
        else:
            status = ConversationStatus.COMPLETED
            nps_p = 0.75 - (idx % 3) * 0.05
            sentiment = 0.6

        templates.append(
            MockTemplate(
                topic=topic,
                status=status,
                nps_p=nps_p,
                sentiment=sentiment,
                user_lines=base["lines"],
                improvement=base["improvement"],
                turn_count=random.randint(6, 12),
            )
        )

    now = datetime.utcnow()
    for idx, template in enumerate(templates):
        state = create_session(
            survey_id="apparel_post_purchase_v1",
            user_id=f"mock_user_{idx + 1}",
            persona_id=None,
        )

        start_time = now - timedelta(hours=random.randint(2, 48), minutes=random.randint(0, 59))
        message_lines = []
        for i in range(template.turn_count):
            if i % 2 == 0:
                message_lines.append(
                    "Thanks for sharing! A couple quick questions if that's cool."
                )
            else:
                message_lines.append(random.choice(template.user_lines))

        messages = _build_messages(start_time, message_lines)

        if template.status == ConversationStatus.IN_PROGRESS:
            messages.append(
                Message(
                    turn=len(messages) + 1,
                    role="bot",
                    text="Anything else you'd like to add?",
                    timestamp=messages[-1].timestamp + timedelta(minutes=2),
                    latency_ms=None,
                )
            )

        state.messages = messages
        state.turn_count = len(messages)
        state.started_at = start_time
        state.last_activity = messages[-1].timestamp
        state.status = template.status

        efi = round(random.uniform(0.2, 0.7), 2)
        ids = round(random.uniform(0.3, 0.8), 2)
        nps_bucket = _bucket_from_nps(template.nps_p)

        state.efi_history = [efi] * max(1, len(messages) // 2)
        state.ids_history = [ids] * max(1, len(messages) // 2)
        state.nps_logit = template.nps_p
        state.last_metrics = {
            "efi": {"current": efi, "action": "continue_normal"},
            "ids": {"current": ids, "should_probe": False},
            "nps": {
                "p_advocacy": template.nps_p,
                "bucket": nps_bucket,
                "confidence": 0.65,
            },
        }

        extracted_fields: Dict[str, ExtractedField] = {}
        extracted_fields["overall_satisfaction"] = ExtractedField(
            field_id="overall_satisfaction",
            value="positive" if template.sentiment > 0 else "negative",
            confidence=0.7,
            source_turn=max(1, len(messages) - 2),
            quote=template.user_lines[0],
        )
        extracted_fields["fit_rating"] = ExtractedField(
            field_id="fit_rating",
            value="true_to_size" if template.topic != "sizing" else "runs_small",
            confidence=0.65,
            source_turn=max(1, len(messages) - 3),
            quote=template.user_lines[1],
        )
        extracted_fields["fabric_quality"] = ExtractedField(
            field_id="fabric_quality",
            value="high" if template.sentiment > 0 else "low",
            confidence=0.6,
            source_turn=max(1, len(messages) - 4),
            quote=template.user_lines[2],
        )
        extracted_fields["would_recommend"] = ExtractedField(
            field_id="would_recommend",
            value="yes" if template.nps_p >= 0.7 else "maybe",
            confidence=0.6,
            source_turn=max(1, len(messages) - 2),
            quote="Would recommend to a friend.",
        )
        extracted_fields["improvement_suggestion"] = ExtractedField(
            field_id="improvement_suggestion",
            value=template.improvement,
            confidence=0.8,
            source_turn=max(1, len(messages) - 1),
            quote=template.improvement,
        )

        if template.status == ConversationStatus.IN_PROGRESS:
            extracted_fields.pop("improvement_suggestion")

        state.extracted_fields = extracted_fields

        completed_count = len(extracted_fields)
        pending = max(0, len(question_map) - completed_count)
        state.fields_pending = list(question_map.keys())[:pending]
        state.next_field_target = state.fields_pending[0] if state.fields_pending else None

        update_session(state)
