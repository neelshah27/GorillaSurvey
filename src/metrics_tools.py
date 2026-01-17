"""
Metrics Tools - Computes EFI, IDS, and NPS scores.

All formulas are implemented as pure Python with minimal dependencies.
LLM calls are optional and only used when heuristics are ambiguous.
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# EFI Weights
EFI_WEIGHTS = {
    "latency": 0.15,
    "brevity": 0.25,
    "sentiment": 0.20,
    "deflection": 0.30,
    "fatigue": 0.10,
}

# EFI Thresholds
EFI_THRESHOLDS = {
    "engaged": 0.3,
    "mild_friction": 0.5,
    "moderate_friction": 0.7,
    "high_friction": 0.85,
}

# EMA smoothing factor (higher = more reactive)
EFI_ALPHA = 0.4
EFI_BASELINE = 0.2

# IDS Weights
IDS_WEIGHTS = {
    "attributes": 0.30,
    "causal": 0.20,
    "nouns": 0.25,
    "specificity": 0.25,
}

# NPS Configuration
NPS_PRIOR = 0.5
NPS_LR_POSITIVE = 2.0
NPS_LR_NEGATIVE = 0.4

# Deflection patterns
DEFLECTION_PATTERNS = [
    r"\b(idk|dunno|don'?t\s*know|not\s*sure|whatever|fine|ok(ay)?|meh|nm|nvm)\b",
    r"\b(busy|later|gtg|gotta\s*go|can'?t\s*talk|brb|ttyl)\b",
    r"^.{0,5}$",  # Very short responses (5 chars or less)
    r"^(k|y|n|no|yes|yeah|nah|sure|maybe)\.?$",  # Single word responses
]

# Vagueness patterns (for IDS penalty)
VAGUENESS_PATTERNS = [
    r"\b(good|bad|fine|okay|ok|nice|cool|whatever|idk|meh|alright)\b",
]

# Product attribute keywords (apparel-specific)
PRODUCT_ATTRIBUTES = [
    r"\b(fit|fits|fitting|fitted|size|sized|sizing)\b",
    r"\b(fabric|material|cotton|polyester|soft|rough|smooth|texture)\b",
    r"\b(price|cost|expensive|cheap|worth|value|afford)\b",
    r"\b(color|colour|shade|dye|faded|bright|dark)\b",
    r"\b(durable|durability|lasting|wash|washed|washing|shrink|shrunk)\b",
    r"\b(quality|construction|stitching|seams|hem)\b",
    r"\b(comfortable|comfort|comfy|cozy|tight|loose|snug)\b",
    r"\b(style|look|looks|design|trendy|fashion)\b",
]

# Causal markers
CAUSAL_MARKERS = [
    r"\b(because|since|after|when|due\s*to|so\s*that|as\s*a\s*result)\b",
    r"\b(caused|reason|why|therefore|thus|hence)\b",
]

# Positive sentiment signals (for NPS)
POSITIVE_SIGNALS = [
    r"\b(love|loved|loving|amazing|awesome|fantastic|perfect|excellent)\b",
    r"\b(definitely|absolutely|highly|totally|completely)\b",
    r"\b(best|favorite|favourite|obsessed|hooked)\b",
    r"\b(recommend|tell\s*(my\s*)?(friend|everyone)|buy\s*again)\b",
    r"\b(happy|pleased|satisfied|delighted|thrilled)\b",
    r"(😍|❤️|🔥|💯|👍|😊|🙌|💕)",
]

# Negative sentiment signals (for NPS)
NEGATIVE_SIGNALS = [
    r"\b(hate|hated|terrible|awful|horrible|worst|disgusting)\b",
    r"\b(disappointed|disappointing|letdown|waste|regret)\b",
    r"\b(never\s*again|won'?t\s*buy|return(ing)?|refund)\b",
    r"\b(cheap|flimsy|poor|broke|broken|ripped|fell\s*apart)\b",
    r"\b(scam|fraud|rip\s*off|overpriced)\b",
    r"(😡|👎|😤|💔|😠|🤮)",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EFIComponents:
    """Individual components of the EFI score."""
    latency: float = 0.0
    brevity: float = 0.0
    sentiment: float = 0.0
    deflection: float = 0.0
    fatigue: float = 0.0


@dataclass
class EFIResult:
    """Result of EFI computation."""
    current: float
    raw: float
    previous: float
    components: EFIComponents
    action: str


@dataclass
class IDSFeatures:
    """Features extracted for IDS computation."""
    attribute_count: int = 0
    has_causal: bool = False
    noun_count: int = 0
    specificity_count: int = 0
    vagueness_penalty: float = 0.0


@dataclass
class IDSResult:
    """Result of IDS computation."""
    current: float
    features: IDSFeatures
    should_probe: bool
    probe_reason: Optional[str] = None


@dataclass
class NPSResult:
    """Result of NPS computation."""
    p_advocacy: float
    bucket: str  # "promoter", "passive", "detractor"
    confidence: float
    logit: float
    signals_positive: List[str]
    signals_negative: List[str]
    should_ask_direct: bool


@dataclass
class MetricsState:
    """Complete metrics state for a session."""
    session_id: str
    turn: int
    efi: EFIResult
    ids: IDSResult
    nps: NPSResult

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "efi": {
                "current": round(self.efi.current, 3),
                "raw": round(self.efi.raw, 3),
                "components": {
                    "latency": round(self.efi.components.latency, 3),
                    "brevity": round(self.efi.components.brevity, 3),
                    "sentiment": round(self.efi.components.sentiment, 3),
                    "deflection": round(self.efi.components.deflection, 3),
                    "fatigue": round(self.efi.components.fatigue, 3),
                },
                "action": self.efi.action,
            },
            "ids": {
                "current": round(self.ids.current, 3),
                "features": {
                    "attribute_count": self.ids.features.attribute_count,
                    "has_causal": self.ids.features.has_causal,
                    "noun_count": self.ids.features.noun_count,
                    "specificity_count": self.ids.features.specificity_count,
                    "vagueness_penalty": round(self.ids.features.vagueness_penalty, 3),
                },
                "should_probe": self.ids.should_probe,
                "probe_reason": self.ids.probe_reason,
            },
            "nps": {
                "p_advocacy": round(self.nps.p_advocacy, 3),
                "bucket": self.nps.bucket,
                "confidence": round(self.nps.confidence, 3),
                "signals_positive": self.nps.signals_positive,
                "signals_negative": self.nps.signals_negative,
                "should_ask_direct": self.nps.should_ask_direct,
            },
        }


# =============================================================================
# EFI COMPUTATION
# =============================================================================

def compute_deflection_score(text: str) -> float:
    """
    Compute deflection probability using regex patterns.
    Returns 1.0 if deflection detected, 0.0 otherwise.
    """
    text_lower = text.lower().strip()

    for pattern in DEFLECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return 1.0

    return 0.0


def compute_sentiment_heuristic(text: str) -> float:
    """
    Simple sentiment heuristic using keyword matching.
    Returns value in [-1, 1] where -1 is negative, 1 is positive.
    """
    text_lower = text.lower()

    positive_count = 0
    negative_count = 0

    for pattern in POSITIVE_SIGNALS:
        positive_count += len(re.findall(pattern, text_lower, re.IGNORECASE))

    for pattern in NEGATIVE_SIGNALS:
        negative_count += len(re.findall(pattern, text_lower, re.IGNORECASE))

    total = positive_count + negative_count
    if total == 0:
        return 0.0  # Neutral

    # Normalize to [-1, 1]
    return (positive_count - negative_count) / total


def compute_efi(
    text: str,
    latency_seconds: float,
    turn: int,
    previous_efi: float = EFI_BASELINE,
) -> EFIResult:
    """
    Compute Engagement Friction Index (EFI) for a user message.

    Args:
        text: User's message text
        latency_seconds: Time taken for user to respond (seconds)
        turn: Current conversation turn number
        previous_efi: EFI from previous turn (for smoothing)

    Returns:
        EFIResult with current score, components, and recommended action
    """
    # Compute individual components

    # 1. Latency score: 60s = max friction
    latency_score = min(1.0, latency_seconds / 60.0)

    # 2. Brevity score: <50 tokens = friction (approximate with chars/4)
    token_estimate = len(text) / 4
    brevity_score = max(0.0, 1.0 - token_estimate / 50.0)

    # 3. Sentiment score: map [-1,1] to [0,1] where negative = friction
    sentiment_raw = compute_sentiment_heuristic(text)
    sentiment_score = (1.0 - sentiment_raw) / 2.0

    # 4. Deflection score
    deflection_score = compute_deflection_score(text)

    # 5. Fatigue score: increases with turn count
    fatigue_score = min(1.0, turn / 15.0)

    # Combine with weights
    efi_raw = (
        EFI_WEIGHTS["latency"] * latency_score +
        EFI_WEIGHTS["brevity"] * brevity_score +
        EFI_WEIGHTS["sentiment"] * sentiment_score +
        EFI_WEIGHTS["deflection"] * deflection_score +
        EFI_WEIGHTS["fatigue"] * fatigue_score
    )

    # Apply EMA smoothing
    efi_smoothed = EFI_ALPHA * efi_raw + (1 - EFI_ALPHA) * previous_efi

    # Determine action based on thresholds
    if efi_smoothed >= EFI_THRESHOLDS["high_friction"]:
        action = "exit_graceful"
    elif efi_smoothed >= EFI_THRESHOLDS["moderate_friction"]:
        action = "one_last_question"
    elif efi_smoothed >= EFI_THRESHOLDS["mild_friction"]:
        action = "quick_reply_options"
    elif efi_smoothed >= EFI_THRESHOLDS["engaged"]:
        action = "simplify"
    else:
        action = "continue_normal"

    components = EFIComponents(
        latency=latency_score,
        brevity=brevity_score,
        sentiment=sentiment_score,
        deflection=deflection_score,
        fatigue=fatigue_score,
    )

    return EFIResult(
        current=efi_smoothed,
        raw=efi_raw,
        previous=previous_efi,
        components=components,
        action=action,
    )


# =============================================================================
# IDS COMPUTATION
# =============================================================================

def count_pattern_matches(text: str, patterns: List[str]) -> int:
    """Count total matches across all patterns."""
    text_lower = text.lower()
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower, re.IGNORECASE))
    return count


def extract_ids_features(text: str) -> IDSFeatures:
    """Extract all features needed for IDS computation."""
    text_lower = text.lower()

    # 1. Product attributes
    attribute_count = count_pattern_matches(text, PRODUCT_ATTRIBUTES)

    # 2. Causal markers
    has_causal = count_pattern_matches(text, CAUSAL_MARKERS) > 0

    # 3. Noun/adjective density (simplified: count words > 4 chars as proxy)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
    noun_count = len(words)

    # 4. Specificity (numbers, comparisons)
    specificity_patterns = [
        r'\d+',  # Numbers
        r'\b(too|very|really|slightly|much|more|less|better|worse)\b',
        r'\b(small|large|big|tight|loose|long|short)\b',
    ]
    specificity_count = count_pattern_matches(text, specificity_patterns)

    # 5. Vagueness penalty
    vague_count = count_pattern_matches(text, VAGUENESS_PATTERNS)
    vagueness_penalty = min(0.3, vague_count * 0.1)  # Max penalty of 0.3

    return IDSFeatures(
        attribute_count=attribute_count,
        has_causal=has_causal,
        noun_count=noun_count,
        specificity_count=specificity_count,
        vagueness_penalty=vagueness_penalty,
    )


def compute_ids(
    text: str,
    field_already_probed: bool = False,
) -> IDSResult:
    """
    Compute Insight Depth Score (IDS) for a user message.

    Args:
        text: User's message text
        field_already_probed: Whether we've already asked a follow-up for this field

    Returns:
        IDSResult with score and probing recommendation
    """
    features = extract_ids_features(text)

    # Compute IDS score
    ids_raw = (
        IDS_WEIGHTS["attributes"] * min(1.0, features.attribute_count / 5.0) +
        IDS_WEIGHTS["causal"] * (1.0 if features.has_causal else 0.0) +
        IDS_WEIGHTS["nouns"] * min(1.0, features.noun_count / 10.0) +
        IDS_WEIGHTS["specificity"] * min(1.0, features.specificity_count / 3.0) -
        features.vagueness_penalty
    )

    ids_score = max(0.0, min(1.0, ids_raw))

    # Determine if we should probe for more detail
    should_probe = ids_score < 0.3 and not field_already_probed

    probe_reason = None
    if should_probe:
        if features.attribute_count == 0:
            probe_reason = "no_specific_attributes"
        elif features.vagueness_penalty > 0.1:
            probe_reason = "vague_response"
        else:
            probe_reason = "low_detail"

    return IDSResult(
        current=ids_score,
        features=features,
        should_probe=should_probe,
        probe_reason=probe_reason,
    )


# =============================================================================
# NPS COMPUTATION
# =============================================================================

def sigmoid(x: float) -> float:
    """Sigmoid function for probability conversion."""
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    """Logit function (inverse sigmoid)."""
    p = max(0.001, min(0.999, p))  # Clamp to avoid inf
    return math.log(p / (1 - p))


def compute_nps(
    text: str,
    previous_logit: float = 0.0,  # logit(0.5) = 0
    turn: int = 1,
    current_efi: float = 0.3,
) -> NPSResult:
    """
    Compute inferred NPS (advocacy likelihood) using Bayesian updates.

    Args:
        text: User's message text
        previous_logit: Log-odds from previous turn
        turn: Current turn number
        current_efi: Current EFI score (used for direct ask decision)

    Returns:
        NPSResult with advocacy probability and bucket classification
    """
    text_lower = text.lower()

    # Find positive and negative signals
    signals_positive = []
    signals_negative = []

    for pattern in POSITIVE_SIGNALS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        signals_positive.extend(matches)

    for pattern in NEGATIVE_SIGNALS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        signals_negative.extend(matches)

    # Compute likelihood ratio updates
    log_lr_update = 0.0
    log_lr_update += len(signals_positive) * math.log(NPS_LR_POSITIVE)
    log_lr_update += len(signals_negative) * math.log(NPS_LR_NEGATIVE)

    # Update logit
    new_logit = previous_logit + log_lr_update

    # Convert to probability
    p_advocacy = sigmoid(new_logit)

    # Compute confidence (distance from 0.5)
    confidence = abs(p_advocacy - 0.5) * 2.0

    # Determine bucket
    if p_advocacy >= 0.7:
        bucket = "promoter"
    elif p_advocacy >= 0.4:
        bucket = "passive"
    else:
        bucket = "detractor"

    # Determine if we should ask directly
    should_ask_direct = (
        confidence < 0.4 and
        turn >= 4 and
        current_efi < 0.5
    )

    return NPSResult(
        p_advocacy=p_advocacy,
        bucket=bucket,
        confidence=confidence,
        logit=new_logit,
        signals_positive=signals_positive[:5],  # Limit for display
        signals_negative=signals_negative[:5],
        should_ask_direct=should_ask_direct,
    )


# =============================================================================
# COMBINED METRICS COMPUTATION
# =============================================================================

def compute_all_metrics(
    text: str,
    latency_seconds: float,
    turn: int,
    previous_efi: float = EFI_BASELINE,
    previous_nps_logit: float = 0.0,
    field_already_probed: bool = False,
    session_id: str = "",
) -> MetricsState:
    """
    Compute all metrics for a single user message.

    This is the main entry point for the Metrics Agent.
    """
    # Compute EFI
    efi_result = compute_efi(text, latency_seconds, turn, previous_efi)

    # Compute IDS
    ids_result = compute_ids(text, field_already_probed)

    # Compute NPS
    nps_result = compute_nps(text, previous_nps_logit, turn, efi_result.current)

    return MetricsState(
        session_id=session_id,
        turn=turn,
        efi=efi_result,
        ids=ids_result,
        nps=nps_result,
    )


# =============================================================================
# STRATEGY POLICY
# =============================================================================

def choose_strategy(
    efi: float,
    ids: float,
    fields_remaining: int,
    nps_should_ask: bool,
    can_probe: bool,
) -> str:
    """
    Choose the next message strategy based on current metrics.

    Returns one of:
    - continue_normal
    - empathize_followup
    - quick_reply_options
    - summarize_confirm
    - one_last_question
    - ask_nps_direct
    - exit_graceful
    """
    # Priority 1: Disengage if high friction
    if efi >= 0.85:
        return "exit_graceful"

    # Priority 2: Simplify if moderate friction
    if efi >= 0.5:
        if fields_remaining > 2:
            return "quick_reply_options"
        else:
            return "one_last_question"

    # Priority 3: Probe for depth if shallow
    if ids < 0.3 and can_probe:
        return "empathize_followup"

    # Priority 4: NPS direct ask if uncertain
    if nps_should_ask:
        return "ask_nps_direct"

    # Default: natural progression
    return "continue_normal"


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    # Test with sample messages
    test_messages = [
        ("loved it! fabric is amazing, fits perfect", 5.0, 2),
        ("it's fine I guess", 30.0, 3),
        ("idk whatever", 45.0, 4),
        ("The sizing was off - ordered M but fits like S. Material is nice though because it's soft cotton.", 10.0, 2),
        ("terrible. returning it. waste of money", 3.0, 2),
        ("busy rn can't talk", 60.0, 5),
    ]

    print("=" * 80)
    print("METRICS TOOLS - TEST OUTPUT")
    print("=" * 80)

    for text, latency, turn in test_messages:
        print(f"\nMessage: \"{text}\"")
        print(f"Latency: {latency}s, Turn: {turn}")

        metrics = compute_all_metrics(
            text=text,
            latency_seconds=latency,
            turn=turn,
            session_id="test",
        )

        print(f"  EFI: {metrics.efi.current:.3f} -> Action: {metrics.efi.action}")
        print(f"  IDS: {metrics.ids.current:.3f} -> Probe: {metrics.ids.should_probe}")
        print(f"  NPS: {metrics.nps.p_advocacy:.3f} ({metrics.nps.bucket})")

        strategy = choose_strategy(
            efi=metrics.efi.current,
            ids=metrics.ids.current,
            fields_remaining=3,
            nps_should_ask=metrics.nps.should_ask_direct,
            can_probe=not metrics.ids.should_probe,
        )
        print(f"  Strategy: {strategy}")
