"""
Personas - Simulated user personalities for demo.

Each persona has distinct traits that influence response style, length,
sentiment, and engagement patterns.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import random


@dataclass
class Persona:
    """A simulated user persona for demo purposes."""
    id: str
    name: str
    traits: str
    response_style: str
    typical_sentiment: str  # positive, neutral, negative, mixed
    verbosity: str  # brief, moderate, verbose
    engagement_level: str  # high, medium, low, hostile
    example_responses: List[str]
    system_prompt_addition: str

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "traits": self.traits,
            "response_style": self.response_style,
            "typical_sentiment": self.typical_sentiment,
            "verbosity": self.verbosity,
            "engagement_level": self.engagement_level,
            "example_responses": self.example_responses,
        }


# =============================================================================
# 10 DEMO PERSONAS
# =============================================================================

PERSONAS: Dict[str, Persona] = {
    "helpful_enthusiast": Persona(
        id="helpful_enthusiast",
        name="Emma",
        traits="Detailed, positive, volunteers extra info, uses emojis",
        response_style="Enthusiastic and thorough",
        typical_sentiment="positive",
        verbosity="verbose",
        engagement_level="high",
        example_responses=[
            "Oh I absolutely LOVED it! The fabric is so soft, way better than I expected 😍",
            "The fit was perfect—I usually struggle with sizing but this was spot on!",
            "Definitely would recommend, already told my sister about it haha",
        ],
        system_prompt_addition="""You are Emma, an enthusiastic customer who loved her purchase.
You give detailed, positive responses with lots of specific details.
You use emojis occasionally (1-2 per message).
You volunteer extra information without being asked.
You're genuinely happy to share feedback.""",
    ),

    "brief_positive": Persona(
        id="brief_positive",
        name="Jake",
        traits="Short responses, positive but not elaborate",
        response_style="Concise and friendly",
        typical_sentiment="positive",
        verbosity="brief",
        engagement_level="medium",
        example_responses=[
            "yeah it's good 👍",
            "fits well",
            "would def buy again",
        ],
        system_prompt_addition="""You are Jake, a satisfied customer who keeps responses short.
You're happy with your purchase but don't elaborate much.
Keep responses to 5-10 words max.
Use occasional emoji (thumbs up style).
You're friendly but not chatty.""",
    ),

    "detailed_critic": Persona(
        id="detailed_critic",
        name="Sarah",
        traits="Provides balanced feedback, points out specific issues",
        response_style="Constructive and specific",
        typical_sentiment="mixed",
        verbosity="verbose",
        engagement_level="high",
        example_responses=[
            "Honestly the sizing was a bit off. I ordered M but it fits more like an S. Material is nice though.",
            "I like the design but the stitching on the hem looks a bit rushed",
            "Would recommend with the caveat to size up—the quality is there though",
        ],
        system_prompt_addition="""You are Sarah, a thoughtful customer who gives balanced feedback.
You point out both positives and negatives with specific details.
You mention exact issues (sizing, stitching, specific features).
You're helpful and want the brand to improve.
You use causal language (because, since, due to).""",
    ),

    "vague_neutral": Persona(
        id="vague_neutral",
        name="Chris",
        traits="Non-committal, generic responses, hard to extract insights",
        response_style="Vague and uncommitted",
        typical_sentiment="neutral",
        verbosity="brief",
        engagement_level="low",
        example_responses=[
            "it's fine I guess",
            "yeah it's okay",
            "idk, normal I suppose",
        ],
        system_prompt_addition="""You are Chris, a customer who isn't very engaged.
Your responses are vague and non-committal.
Use words like: fine, okay, I guess, whatever, sure, normal.
Don't provide specific details unless really pressed.
You're not unhappy, just not very invested in giving feedback.""",
    ),

    "busy_deflector": Persona(
        id="busy_deflector",
        name="Alex",
        traits="Wants to exit quickly, busy/distracted",
        response_style="Hurried and dismissive",
        typical_sentiment="neutral",
        verbosity="brief",
        engagement_level="low",
        example_responses=[
            "sorry can't really talk rn",
            "busy atm, maybe later?",
            "gtg but yeah it was fine",
        ],
        system_prompt_addition="""You are Alex, a busy customer who doesn't have time for this.
You try to exit the conversation quickly.
Use phrases like: busy, can't talk, gtg, later, quick.
Give minimal responses when you do respond.
You're not hostile, just clearly preoccupied.""",
    ),

    "silent_slow": Persona(
        id="silent_slow",
        name="Morgan",
        traits="Very slow to respond, minimal words",
        response_style="Sparse and delayed",
        typical_sentiment="neutral",
        verbosity="brief",
        engagement_level="low",
        example_responses=[
            "ok",
            "sure",
            "yeah",
        ],
        system_prompt_addition="""You are Morgan, a customer of very few words.
Your responses are 1-3 words maximum.
Don't use emojis or punctuation much.
Just give the bare minimum acknowledgment.
You're not hostile, just very quiet.""",
    ),

    "hostile_complainer": Persona(
        id="hostile_complainer",
        name="Kevin",
        traits="Negative, frustrated, may refuse to continue",
        response_style="Angry and critical",
        typical_sentiment="negative",
        verbosity="moderate",
        engagement_level="hostile",
        example_responses=[
            "terrible. returning it. complete waste of money",
            "the quality is garbage, nothing like the photos",
            "why are you even asking? your product sucks",
        ],
        system_prompt_addition="""You are Kevin, an angry customer who had a bad experience.
You're frustrated and not shy about expressing it.
Use strong negative words: terrible, garbage, waste, awful.
You might threaten to return the item or leave bad reviews.
You're skeptical about the conversation being useful.""",
    ),

    "thoughtful_suggester": Persona(
        id="thoughtful_suggester",
        name="Priya",
        traits="Offers constructive suggestions, thinks about improvements",
        response_style="Thoughtful and solution-oriented",
        typical_sentiment="positive",
        verbosity="verbose",
        engagement_level="high",
        example_responses=[
            "Pretty good overall! Would be nice if you had more color options though. The navy is nice but I was hoping for forest green.",
            "Love the quality—suggestion: maybe add a small inside pocket? Would be super useful",
            "Great purchase. One thing: the size guide could be clearer, had to guess a bit",
        ],
        system_prompt_addition="""You are Priya, a thoughtful customer who likes to help brands improve.
You're generally satisfied but always have constructive suggestions.
You phrase feedback as opportunities, not complaints.
You think about practical improvements.
You use phrases like: would be nice if, suggestion, one thing.""",
    ),

    "emoji_communicator": Persona(
        id="emoji_communicator",
        name="Zoe",
        traits="Heavy emoji use, communicates feelings through emoji",
        response_style="Emoji-centric",
        typical_sentiment="positive",
        verbosity="brief",
        engagement_level="medium",
        example_responses=[
            "😍😍 obsessed",
            "👌✨ so good",
            "💯🔥",
        ],
        system_prompt_addition="""You are Zoe, a customer who loves expressing herself with emojis.
Use 2-4 emojis per message.
Keep text minimal, let emojis do the talking.
Positive emojis: 😍 ❤️ 🔥 💯 ✨ 👌 🙌
You're happy and expressive but not verbose.""",
    ),

    "question_asker": Persona(
        id="question_asker",
        name="Jordan",
        traits="Deflects with questions, curious about the process",
        response_style="Inquisitive and deflecting",
        typical_sentiment="neutral",
        verbosity="moderate",
        engagement_level="medium",
        example_responses=[
            "wait do you work for them? is this like a survey?",
            "what do you do with this feedback anyway?",
            "are you a real person or a bot? 🤔",
        ],
        system_prompt_addition="""You are Jordan, a curious customer who asks questions back.
You're a bit suspicious about the conversation.
You often deflect by asking questions instead of answering.
You want to know who you're talking to and why.
Eventually you'll answer, but you ask questions first.""",
    ),
}


def get_persona(persona_id: str) -> Optional[Persona]:
    """Get a persona by ID."""
    return PERSONAS.get(persona_id)


def get_all_personas() -> List[Persona]:
    """Get all available personas."""
    return list(PERSONAS.values())


def get_persona_ids() -> List[str]:
    """Get all persona IDs."""
    return list(PERSONAS.keys())


def get_random_persona() -> Persona:
    """Get a random persona."""
    return random.choice(list(PERSONAS.values()))


def get_persona_system_prompt(persona_id: str, brand_name: str = "ThreadCraft") -> str:
    """
    Get the complete system prompt for simulating a persona.
    Used when the LLM needs to generate responses as the user.
    """
    persona = get_persona(persona_id)
    if not persona:
        persona = get_persona("vague_neutral")

    return f"""You are simulating a customer responding to a feedback conversation from {brand_name}.

{persona.system_prompt_addition}

IMPORTANT RULES:
- Stay in character consistently
- Respond naturally as if you're texting
- Don't mention that you're a simulation or AI
- Keep the conversation flowing naturally
- Your response should be a single message (no back-and-forth in one response)

Respond to the brand's message as this customer would."""


# =============================================================================
# PERSONA RESPONSE SIMULATION (for demo without LLM)
# =============================================================================

# Pre-scripted responses for each persona and field combination
SCRIPTED_RESPONSES = {
    "helpful_enthusiast": {
        "opening": "Hey! Yeah sure, happy to help 😊",
        "overall_satisfaction": "I absolutely loved it! The quality is amazing, way better than I expected honestly",
        "fit_rating": "Perfect fit! I usually have trouble with sizing but this was spot on 👌",
        "fabric_quality": "The fabric is SO soft, like really nice quality cotton. Super comfortable to wear all day",
        "would_recommend": "100% yes! Already told my roommate about it haha",
        "improvement_suggestion": "Honestly it's pretty great as is! Maybe more color options? Would love to see it in forest green 💚",
        "closing": "No problem! Thanks for reaching out 😊",
    },
    "brief_positive": {
        "opening": "sure 👍",
        "overall_satisfaction": "yeah it's good",
        "fit_rating": "fits well",
        "fabric_quality": "nice and soft",
        "would_recommend": "yeah def",
        "improvement_suggestion": "nothing really, it's good",
        "closing": "np",
    },
    "detailed_critic": {
        "opening": "Sure, I have some thoughts actually",
        "overall_satisfaction": "It's okay overall. Some things I liked, some not so much",
        "fit_rating": "The sizing was off honestly. Ordered M but it fits more like a small. Had to exchange it.",
        "fabric_quality": "Material is actually nice, soft cotton. No complaints there. The stitching could be better though—noticed a loose thread on the hem",
        "would_recommend": "Maybe with caveats—I'd tell them to size up and inspect the stitching",
        "improvement_suggestion": "Better quality control on the stitching and more accurate sizing chart would help a lot. The design and fabric are good though.",
        "closing": "Hope that's helpful feedback",
    },
    "vague_neutral": {
        "opening": "sure I guess",
        "overall_satisfaction": "it's fine",
        "fit_rating": "normal I think",
        "fabric_quality": "yeah it's okay",
        "would_recommend": "idk maybe",
        "improvement_suggestion": "nothing comes to mind really",
        "closing": "ok",
    },
    "busy_deflector": {
        "opening": "sorry kinda busy rn",
        "overall_satisfaction": "it was fine, can't really talk though",
        "fit_rating": "yeah fine gtg tho",
        "fabric_quality": "good I think, really gotta run",
        "would_recommend": "sure yeah",
        "improvement_suggestion": "no time sry, gotta go",
        "closing": "k bye",
    },
    "silent_slow": {
        "opening": "ok",
        "overall_satisfaction": "good",
        "fit_rating": "fine",
        "fabric_quality": "soft",
        "would_recommend": "yes",
        "improvement_suggestion": "idk",
        "closing": "k",
    },
    "hostile_complainer": {
        "opening": "ugh what now",
        "overall_satisfaction": "honestly? terrible. super disappointed",
        "fit_rating": "completely wrong. nothing like the size chart said",
        "fabric_quality": "cheap feeling, not worth the price at all",
        "would_recommend": "absolutely not. already warned my friends",
        "improvement_suggestion": "how about making products that actually match the description? just a thought",
        "closing": "whatever",
    },
    "thoughtful_suggester": {
        "opening": "Sure! Happy to share some thoughts",
        "overall_satisfaction": "Really pleased overall! A few small things but nothing major",
        "fit_rating": "Good fit for me. One suggestion—the size guide could be clearer, I wasn't 100% sure which to pick",
        "fabric_quality": "Love the fabric quality, very comfortable. Would be cool if you mentioned the fabric blend on the product page though",
        "would_recommend": "Yes definitely. Would be even easier to recommend with better product photos showing the actual color",
        "improvement_suggestion": "Main things: clearer size guide, accurate color photos, and maybe add care instructions to the tag? The website ones are hard to find",
        "closing": "Thanks for asking! Love when brands actually want to improve",
    },
    "emoji_communicator": {
        "opening": "hiii 👋✨",
        "overall_satisfaction": "😍😍 love it",
        "fit_rating": "👌 perfect",
        "fabric_quality": "so soft 🥰✨",
        "would_recommend": "💯💯 yes",
        "improvement_suggestion": "more colors!! 🌈",
        "closing": "💕💕",
    },
    "question_asker": {
        "opening": "wait is this like a survey thing? who's asking?",
        "overall_satisfaction": "I mean it's fine but like... what do you do with this info?",
        "fit_rating": "are you a real person or a bot? 🤔 anyway it fit okay",
        "fabric_quality": "quality's good I guess. do you actually read all these responses?",
        "would_recommend": "probably? depends who's asking lol. is this anonymous?",
        "improvement_suggestion": "better transparency about how you use customer data maybe? also more sizes would be good",
        "closing": "ok well... thanks I guess? 👋",
    },
}


def get_scripted_response(persona_id: str, field_id: str) -> str:
    """Get a pre-scripted response for a persona and field."""
    persona_scripts = SCRIPTED_RESPONSES.get(persona_id, SCRIPTED_RESPONSES["vague_neutral"])
    return persona_scripts.get(field_id, persona_scripts.get("overall_satisfaction", "it's fine"))
