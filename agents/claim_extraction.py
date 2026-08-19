"""
Claim Extraction Agent — LLM-based factual claim extraction.

Uses an LLM (or rule-based fallback) to extract checkable factual assertions
from an article. This is what makes verification possible: instead of
classifying the whole blob, we pull out the 2-4 specific claims that can
be checked against external sources.
"""
from __future__ import annotations

import json
import re
import os
from typing import Any

from agents.base import BaseAgent, AgentResult, Label

# LLM integration — supports OpenAI-compatible APIs
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are a fact-checking claim extractor. Given a news article,
extract the 2-5 most important CHECKABLE factual assertions that could be verified
against external sources.

For each claim, provide:
- "claim": the factual assertion as a self-contained sentence
- "type": one of "statistic", "event", "attribution", "quote", "causal", "other"
- "checkability": "high" (specific, verifiable fact), "medium" (partially verifiable), "low" (opinion/hard to verify)

Focus on claims that:
1. State specific facts (numbers, dates, names, events)
2. Attribute actions or statements to specific people/organizations
3. Make causal or correlational assertions

Skip:
- Opinions and editorial commentary
- Vague or unfalsifiable statements
- Historical background that is widely accepted

Return ONLY a JSON array, no other text:
[{"claim": "...", "type": "...", "checkability": "..."}]
"""

RULE_BASED_PATTERNS = [
    # Statistical claims
    re.compile(r'\b\d+[\d,.]*\s*(?:percent|%|million|billion|trillion)\b', re.I),
    # Attribution claims
    re.compile(r'\b(said|stated|announced|claimed|reported|confirmed|denied)\b', re.I),
    # Specific event claims
    re.compile(r'\b(on\s+\w+\s+\d{1,2}|last\s+(?:week|month|year|tuesday|monday|wednesday|thursday|friday|saturday|sunday))\b', re.I),
    # Numbers
    re.compile(r'\b(?:over|nearly|about|more than|less than|approximately)\s+\d+\b', re.I),
]


def _extract_claims_rule_based(text: str) -> list[dict[str, str]]:
    """Fallback: extract sentences containing factual indicators."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    claims = []
    for sent in sentences:
        if len(sent) < 20 or len(sent) > 300:
            continue
        for pattern in RULE_BASED_PATTERNS:
            if pattern.search(sent):
                claims.append({
                    "claim": sent.strip(),
                    "type": "other",
                    "checkability": "medium",
                })
                break
        if len(claims) >= 5:
            break
    # If no pattern matched, take the first few substantive sentences
    if not claims:
        for sent in sentences:
            if len(sent) > 30:
                claims.append({
                    "claim": sent.strip(),
                    "type": "other",
                    "checkability": "medium",
                })
            if len(claims) >= 3:
                break
    return claims


def _extract_claims_llm(text: str) -> list[dict[str, str]]:
    """Use an OpenAI-compatible LLM to extract claims."""
    if not HAS_OPENAI:
        return _extract_claims_rule_based(text)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("CLAIM_EXTRACTION_MODEL", "gpt-4o-mini")

    if not api_key:
        return _extract_claims_rule_based(text)

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # Truncate to avoid token limits
    truncated = text[:6000] if len(text) > 6000 else text

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": truncated},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()
        # Extract JSON array from response
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            claims = json.loads(match.group())
            return claims[:5]  # cap
    except Exception:
        pass

    return _extract_claims_rule_based(text)


class ClaimExtractionAgent(BaseAgent):
    name = "claim_extraction"

    def run(self, article: dict[str, Any]) -> AgentResult:
        text = article.get("full_text") or article.get("text", "")
        claims = _extract_claims_llm(text)

        high_count = sum(1 for c in claims if c.get("checkability") == "high")

        # More checkable claims = more useful signal
        confidence = min(0.3 + high_count * 0.15 + len(claims) * 0.05, 0.95)

        return AgentResult(
            agent_name=self.name,
            label=Label.UNCERTAIN,  # extraction doesn't judge
            confidence=confidence,
            reasoning=f"Extracted {len(claims)} claims ({high_count} high-checkability)",
            evidence=claims,
            raw_output={"claims": claims, "method": "llm" if HAS_OPENAI else "rule_based"},
        )
