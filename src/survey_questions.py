"""
Survey Questions - Loads the survey questions from JSON.

JSON format (minimal):
{
  "questions": [
    "Question text 1",
    "Question text 2"
  ]
}

Optional keys:
- brand_name
- closing_message
- opening_message
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "survey_questions.json"

DEFAULT_BRAND_NAME = os.getenv("BRAND_NAME", "GymFish")
DEFAULT_OPENING_MESSAGE = "Hey! Thanks for shopping with us. Mind if I ask a couple quick things?"
DEFAULT_CLOSING_MESSAGE = "Thanks so much for the feedback! Really appreciate your time."


def _load_config() -> Dict[str, Any]:
    if not QUESTIONS_PATH.exists():
        return {"questions": []}
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"questions": data}


def get_questions() -> List[str]:
    data = _load_config()
    questions = data.get("questions", [])
    return [q for q in questions if isinstance(q, str) and q.strip()]


def get_question_map() -> Dict[str, str]:
    questions = get_questions()
    return {f"q{i + 1}": q for i, q in enumerate(questions)}


def get_question_ids() -> List[str]:
    return list(get_question_map().keys())


def get_brand_name() -> str:
    data = _load_config()
    name = data.get("brand_name")
    return name if isinstance(name, str) and name.strip() else DEFAULT_BRAND_NAME


def get_opening_message() -> str:
    data = _load_config()
    msg = data.get("opening_message")
    return msg if isinstance(msg, str) and msg.strip() else DEFAULT_OPENING_MESSAGE


def get_closing_message() -> str:
    data = _load_config()
    msg = data.get("closing_message")
    return msg if isinstance(msg, str) and msg.strip() else DEFAULT_CLOSING_MESSAGE
