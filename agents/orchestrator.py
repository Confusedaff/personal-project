"""
Orchestrator / Aggregator — Weighs all agent signals into one verdict.

This is the actual "agentic" part: an LLM or rules engine that weighs
all agent outputs (don't just average them — e.g. a fact-check agent
finding a false claim should be able to override a classifier saying
"real") and produces one verdict with a transparent evidence trail.
"""
from __future__ import annotations

import json
from typing import Any
from dataclasses import dataclass

from agents.base import BaseAgent, AgentResult, Label


@dataclass
class Verdict:
    """Final aggregated verdict from all agents."""
    label: Label
    confidence: float
    reasoning: str
    evidence_trail: list[dict]
    agent_results: list[dict]
    needs_human_review: bool
    review_reason: str

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "evidence_trail": self.evidence_trail,
            "agent_results": self.agent_results,
            "needs_human_review": self.needs_human_review,
            "review_reason": self.review_reason,
        }


# Agent weight configuration — fact-check has highest weight because it
# actually verifies claims against external truth, not just style patterns.
AGENT_WEIGHTS = {
    "claim_extraction": 0.0,  # extraction only, no classification
    "ml_classifier": 0.15,    # style signal (useful but not ground truth)
    "fact_check": 0.40,       # THE accuracy lever — actual claim verification
    "source_credibility": 0.15,
    "media_forensics": 0.10,
    "bias_sentiment": 0.05,   # supporting signal only
    "ingestion": 0.0,         # no classification
}

# Override rules: fact-check findings can override other agents
OVERRIDE_THRESHOLD = 0.7  # if fact_check confidence > this and contradicts, override


class Orchestrator(BaseAgent):
    name = "orchestrator"

    def run(self, article: dict[str, Any]) -> AgentResult:
        agent_results = article.get("agent_results", [])
        if not agent_results:
            return AgentResult(
                agent_name=self.name,
                label=Label.UNCERTAIN,
                confidence=0.0,
                reasoning="No agent results to aggregate",
            )

        # Parse results
        results: list[AgentResult] = []
        for r in agent_results:
            if isinstance(r, AgentResult):
                results.append(r)
            elif isinstance(r, dict):
                results.append(AgentResult(
                    agent_name=r.get("agent", "unknown"),
                    label=Label(r.get("label", "uncertain")),
                    confidence=r.get("confidence", 0),
                    reasoning=r.get("reasoning", ""),
                    evidence=r.get("evidence", []),
                    raw_output=r.get("raw_output", {}),
                ))

        # === Weighted voting ===
        weighted_fake = 0.0
        weighted_real = 0.0
        total_weight = 0.0
        evidence_trail = []

        for r in results:
            weight = AGENT_WEIGHTS.get(r.agent_name, 0.0)
            if weight == 0:
                continue

            # Scale weight by agent confidence
            effective_weight = weight * r.confidence

            if r.label == Label.FAKE:
                weighted_fake += effective_weight
            elif r.label == Label.REAL:
                weighted_real += effective_weight

            total_weight += effective_weight

            evidence_trail.append({
                "agent": r.agent_name,
                "label": r.label.value,
                "confidence": r.confidence,
                "weight": weight,
                "effective_weight": round(effective_weight, 4),
                "reasoning": r.reasoning,
            })

        # === Override rules ===
        override = False
        override_reason = ""

        # Fact-check override: if fact-check agent is highly confident and
        # contradicts the ML classifier, it wins
        fact_check_result = next((r for r in results if r.agent_name == "fact_check"), None)
        ml_result = next((r for r in results if r.agent_name == "ml_classifier"), None)

        if fact_check_result and fact_check_result.confidence > OVERRIDE_THRESHOLD:
            if ml_result and ml_result.label != fact_check_result.label:
                override = True
                override_reason = (
                    f"Fact-check agent (conf={fact_check_result.confidence:.3f}) "
                    f"overrides ML classifier (conf={ml_result.confidence:.3f}) "
                    f"because external verification takes precedence over style analysis"
                )

        # Source credibility override: known bad domain with low credibility score
        source_result = next((r for r in results if r.agent_name == "source_credibility"), None)
        if source_result and source_result.confidence > 0.6 and source_result.label == Label.FAKE:
            if ml_result and ml_result.label == Label.REAL:
                override = True
                override_reason = (
                    f"Source credibility agent (conf={source_result.confidence:.3f}) "
                    f"overrides ML classifier: source is known low-credibility"
                )

        # === Compute final verdict ===
        if override:
            # Use the overriding agent's label
            overriding = fact_check_result or source_result
            final_label = overriding.label
            final_confidence = overriding.confidence
            evidence_trail.append({
                "agent": "orchestrator",
                "label": "override",
                "reasoning": override_reason,
            })
        elif total_weight > 0:
            fake_ratio = weighted_fake / total_weight
            real_ratio = weighted_real / total_weight

            if fake_ratio > real_ratio:
                final_label = Label.FAKE
                final_confidence = fake_ratio
            else:
                final_label = Label.REAL
                final_confidence = real_ratio
        else:
            final_label = Label.UNCERTAIN
            final_confidence = 0.0

        # === Human review decision ===
        needs_review = False
        review_reason = ""

        if final_confidence < 0.5:
            needs_review = True
            review_reason = f"Low confidence ({final_confidence:.3f})"
        elif override:
            needs_review = True
            review_reason = f"Agent conflict requiring override: {override_reason}"
        elif abs(weighted_fake - weighted_real) < 0.1 and total_weight > 0:
            needs_review = True
            review_reason = "Agents strongly disagree (close split)"

        # === Build reasoning string ===
        agent_summary = ", ".join(
            f"{r.agent_name}={r.label.value}({r.confidence:.2f})"
            for r in results if r.agent_name in AGENT_WEIGHTS and AGENT_WEIGHTS[r.agent_name] > 0
        )
        reasoning = (
            f"Orchestrated verdict: {final_label.value} (confidence={final_confidence:.3f}). "
            f"Agent signals: [{agent_summary}]. "
            f"Override applied: {override}. "
            f"Human review needed: {needs_review}"
        )

        return AgentResult(
            agent_name=self.name,
            label=final_label,
            confidence=final_confidence,
            reasoning=reasoning,
            evidence=evidence_trail,
            raw_output={
                "weighted_fake": round(weighted_fake, 4),
                "weighted_real": round(weighted_real, 4),
                "total_weight": round(total_weight, 4),
                "override": override,
                "override_reason": override_reason,
                "needs_human_review": needs_review,
                "review_reason": review_reason,
            },
        )
