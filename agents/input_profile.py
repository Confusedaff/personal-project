"""
Input Profiler — Classifies user input by richness and type.

Inspects title presence, text length, word/sentence count, and structure
to route the request through appropriate weighting in the orchestrator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class InputType(str, Enum):
    MINIMAL = "MINIMAL"   # single word, entity, or very short phrase
    SHORT = "SHORT"       # short phrase or brief headline
    MEDIUM = "MEDIUM"     # partial text, headline + short paragraph
    ARTICLE = "ARTICLE"   # full article body or title + long body


@dataclass
class InputProfile:
    """Result of input analysis."""
    input_type: InputType
    has_title: bool
    title_length: int
    body_length: int
    total_word_count: int
    sentence_count: int
    is_single_token: bool
    has_factual_claims: bool
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "input_type": self.input_type.value,
            "has_title": self.has_title,
            "title_length": self.title_length,
            "body_length": self.body_length,
            "total_word_count": self.total_word_count,
            "sentence_count": self.sentence_count,
            "is_single_token": self.is_single_token,
            "has_factual_claims": self.has_factual_claims,
            "reasoning": self.reasoning,
        }


# Patterns that suggest the input is an entity/topic rather than a claim
_ENTITY_PATTERNS = [
    re.compile(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$"),  # "Modi", "Bitcoin Price"
    re.compile(r"^[A-Z]{2,}$"),                         # "NASA", "FBI"
]

_CLAIM_INDICATORS = [
    "is", "are", "was", "were", "will", "has", "have", "had",
    "says", "said", "claims", "announces", "reveals", "reports",
    "crashes", "surges", "killed", "dead", "banned", "launches",
    "new", "major", "first", "last", "biggest", "breaking",
    "crash", "surge", "ban", "launch", "announce", "reveal",
    "confirm", "confirmed", "study", "research", "report",
    "according", "evidence", "officials", "government", "president",
]


def _count_sentences(text: str) -> int:
    """Estimate sentence count from text."""
    if not text:
        return 0
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'[.!?]+[\s\n]', text)
    # Filter empty strings from trailing splits
    return max(len([s for s in sentences if s.strip()]), 0)


def _contains_factual_claims(text: str) -> bool:
    """Heuristic: does the text contain verifiable factual claims?"""
    if not text:
        return False
    text_lower = text.lower()
    indicator_count = sum(1 for indicator in _CLAIM_INDICATORS if indicator in text_lower)
    # Require at least a couple claim indicators and sufficient length
    return indicator_count >= 2 and len(text.split()) >= 8


def _is_entity_or_topic(text: str) -> bool:
    """Check if input is just a named entity or topic keyword."""
    stripped = text.strip()
    words = stripped.split()
    if len(words) > 3:
        return False
    # Check against entity patterns
    for pattern in _ENTITY_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def analyze_input(
    title: str = "",
    text: str = "",
    url: str = "",
) -> InputProfile:
    """Classify user input into a profile for dynamic weighting.

    Uses multiple signals — not just character count — to determine input
    richness and route to appropriate weighting.

    Returns an InputProfile with the classification and characteristics.
    """
    title = title or ""
    text = text or ""
    full_body = text

    title_length = len(title)
    body_length = len(full_body)
    total_text = f"{title} {full_body}".strip()
    total_word_count = len(total_text.split()) if total_text else 0
    sentence_count = _count_sentences(full_body)
    is_single_token = total_word_count <= 1
    has_title = bool(title.strip())
    has_factual_claims = _contains_factual_claims(total_text)
    is_entity = _is_entity_or_topic(total_text) if total_word_count <= 3 else False

    # Classification logic
    reasoning_parts: list[str] = []

    if is_single_token or (total_word_count <= 2 and not has_title):
        profile_type = InputType.MINIMAL
        reasoning_parts.append(
            f"Input is {'a single token' if is_single_token else f'only {total_word_count} words'}"
            f" with no meaningful body text."
        )
        if has_title:
            reasoning_parts.append(f"Title present ({title_length} chars) but body is empty.")

    elif is_entity and body_length == 0:
        profile_type = InputType.MINIMAL
        reasoning_parts.append(
            f"Input appears to be a named entity or topic ({total_word_count} words, "
            f"{total_text}) rather than a factual claim."
        )

    elif total_word_count <= 20 and body_length < 150:
        profile_type = InputType.SHORT
        reasoning_parts.append(
            f"Short input: {total_word_count} words, {body_length} body chars. "
            f"Insufficient for reliable style-based classification."
        )

    elif total_word_count <= 80 or (body_length < 500 and sentence_count < 5):
        profile_type = InputType.MEDIUM
        reasoning_parts.append(
            f"Medium input: {total_word_count} words, {body_length} body chars, "
            f"{sentence_count} sentences. Some textual context available."
        )

    else:
        profile_type = InputType.ARTICLE
        reasoning_parts.append(
            f"Rich article input: {total_word_count} words, {body_length} body chars, "
            f"{sentence_count} sentences. Sufficient content for TF-IDF analysis."
        )

    # Add context notes
    if has_title:
        reasoning_parts.append(f"Title present ({title_length} chars).")
    if has_factual_claims:
        reasoning_parts.append("Text contains factual claim indicators.")
    else:
        reasoning_parts.append("No strong factual claim indicators detected.")

    reasoning = " ".join(reasoning_parts)

    return InputProfile(
        input_type=profile_type,
        has_title=has_title,
        title_length=title_length,
        body_length=body_length,
        total_word_count=total_word_count,
        sentence_count=sentence_count,
        is_single_token=is_single_token,
        has_factual_claims=has_factual_claims,
        reasoning=reasoning,
    )
