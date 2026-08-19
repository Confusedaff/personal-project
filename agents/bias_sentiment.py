"""
Bias / Sentiment Agent — Emotional language and framing analysis.

Flags manipulative framing and loaded language as a supporting signal,
not a verdict on its own (since sensational-but-true and calm-but-false
both exist).
"""
from __future__ import annotations

import re
from typing import Any

from agents.base import BaseAgent, AgentResult, Label

# Loaded language lexicons (subsets for demonstration)
STRONG_BIAS_WORDS = {
    "shocking", "unbelievable", "exposed", "secret", "coverup", "conspiracy",
    "mainstream media", "sheeple", "wake up", "they don't want you to know",
    "exposed", "bombshell", "devastating", "destroyed", "slammed", "blasted",
    "rips", "annoys", "destroys", "crushes", "demolishes", "annihilates",
    "hoax", "fraud", "scam", "propaganda", "brainwash", "manipulate",
    "censored", "banned", "silenced", "suppressed", "truth", "real truth",
    "what they hiding", "share before deleted", "gone viral", "must see",
    "breaking", "urgent", "alert", "warning", "danger",
}

MODERATE_BIAS_WORDS = {
    "allegedly", "reportedly", "sources say", "claimed", "insisted",
    "demanded", "refused", "attacked", "defended", "blamed", "accused",
    "controversial", "questionable", "dubious", "debunked", "misleading",
    "inflammatory", "provocative", "divisive", "radical", "extreme",
}

EMOTIONAL_PATTERNS = [
    # ALL CAPS emphasis (common in clickbait/fake news)
    re.compile(r'\b[A-Z]{4,}\b'),
    # Excessive exclamation marks
    re.compile(r'!{2,}'),
    # Emotional intensifiers
    re.compile(r'\b(absolutely|totally|completely|utterly|100%)\b', re.I),
    # Fear-mongering patterns
    re.compile(r'\b(danger|threat|crisis|catastrophe|disaster|emergency)\b', re.I),
    # Appeal to authority without citation
    re.compile(r'\b(experts say|studies show|scientists confirm|doctors reveal)\b', re.I),
    # Urgency/call to action
    re.compile(r'\b(share now|before they|don\'t let them|act now|time running out)\b', re.I),
]

HEADCINE_PATTERNS = [
    re.compile(r'\b\d+\s+(?:dead|killed|injured|arrested)\b', re.I),
    re.compile(r'\b(EXPOSED|CAUGHT|DESTROYED|SLAMMED)\b'),
    re.compile(r'\bwhat\s+(?:they|the government|the media)\s+(?:don\'t|won\'t|can\'t)\b', re.I),
]


def _compute_bias_score(text: str) -> dict:
    """Compute a composite bias/sentiment score."""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    strong_hits = [w for w in STRONG_BIAS_WORDS if w in text_lower]
    moderate_hits = [w for w in MODERATE_BIAS_WORDS if w in text_lower]

    pattern_hits = []
    for p in EMOTIONAL_PATTERNS:
        matches = p.findall(text)
        pattern_hits.extend(matches)

    headline_hits = []
    for p in HEADCINE_PATTERNS:
        matches = p.findall(text)
        headline_hits.extend(matches)

    # Score components
    strong_ratio = len(strong_hits) / total_words
    moderate_ratio = len(moderate_hits) / total_words
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 3) / total_words

    # Composite score (0 = clean, 1 = heavily biased)
    bias_score = min(
        strong_ratio * 20 +
        moderate_ratio * 10 +
        caps_ratio * 5 +
        len(pattern_hits) * 0.05 +
        len(headline_hits) * 0.1,
        1.0
    )

    return {
        "bias_score": round(bias_score, 4),
        "strong_bias_words": strong_hits[:10],
        "moderate_bias_words": moderate_hits[:10],
        "emotional_patterns": len(pattern_hits),
        "headline_patterns": len(headline_hits),
        "caps_words": len([w for w in words if w.isupper() and len(w) > 3]),
        "total_words": total_words,
    }


class BiasSentimentAgent(BaseAgent):
    name = "bias_sentiment"

    def run(self, article: dict[str, Any]) -> AgentResult:
        title = article.get("title", "")
        text = article.get("text", "")
        full_text = f"{title} {text}"

        analysis = _compute_bias_score(full_text)
        bias_score = analysis["bias_score"]

        # Bias is a supporting signal, not a verdict
        # High bias doesn't mean fake — just means high sensationalism
        if bias_score > 0.6:
            label = Label.FAKE
            confidence = 0.3 + bias_score * 0.3  # moderate confidence
        elif bias_score < 0.15:
            label = Label.REAL
            confidence = 0.2 + (1 - bias_score) * 0.2
        else:
            label = Label.UNCERTAIN
            confidence = 0.15

        reasoning_parts = [
            f"Bias score: {bias_score:.3f}",
        ]
        if analysis["strong_bias_words"]:
            reasoning_parts.append(
                f"Strong bias words: {', '.join(analysis['strong_bias_words'][:5])}"
            )
        if analysis["headline_patterns"]:
            reasoning_parts.append(
                f"{analysis['headline_patterns']} clickbait headline patterns detected"
            )
        if analysis["caps_words"] > 3:
            reasoning_parts.append(
                f"{analysis['caps_words']} ALL-CAPS words (possible emphasis manipulation)"
            )

        return AgentResult(
            agent_name=self.name,
            label=label,
            confidence=confidence,
            reasoning=". ".join(reasoning_parts),
            evidence=[analysis],
            raw_output=analysis,
        )
