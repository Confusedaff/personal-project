"""
Orchestrator / Aggregator — Weighs all agent signals into one verdict.

This is the actual "agentic" part: a rules engine that weighs all agent
outputs and produces one verdict with a transparent evidence trail.

Features:
  - Input-aware dynamic weighting: adjusts agent weights based on input
    richness (MINIMAL / SHORT / MEDIUM / ARTICLE).
  - Confidence-aware effective weighting: multiplies base weights by agent
    confidence and evidence quality.
  - Override rules: strong factual evidence can override weaker style-based
    signals; low-confidence results produce UNCERTAIN verdicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.base import AgentResult, BaseAgent, Label
from agents.input_profile import InputProfile, InputType, analyze_input


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
    input_profile: str = ""
    agent_weights: dict[str, float] = field(default_factory=dict)
    weighting_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "evidence_trail": self.evidence_trail,
            "agent_results": self.agent_results,
            "needs_human_review": self.needs_human_review,
            "review_reason": self.review_reason,
            "input_profile": self.input_profile,
            "agent_weights": {k: round(v, 4) for k, v in self.agent_weights.items()},
            "weighting_reason": self.weighting_reason,
        }


# ── Base weight profiles per input type ──────────────────────────────────
# These are starting points — the orchestrator also applies confidence and
# evidence-quality multipliers on top of these base weights.

BASE_WEIGHTS: dict[InputType, dict[str, float]] = {
    InputType.MINIMAL: {
        "ml_classifier": 0.10,
        "fact_check": 0.65,
        "source_credibility": 0.10,
        "media_forensics": 0.05,
        "bias_sentiment": 0.10,
    },
    InputType.SHORT: {
        "ml_classifier": 0.15,
        "fact_check": 0.55,
        "source_credibility": 0.12,
        "media_forensics": 0.08,
        "bias_sentiment": 0.10,
    },
    InputType.MEDIUM: {
        "ml_classifier": 0.25,
        "fact_check": 0.50,
        "source_credibility": 0.10,
        "media_forensics": 0.05,
        "bias_sentiment": 0.10,
    },
    InputType.ARTICLE: {
        "ml_classifier": 0.35,
        "fact_check": 0.40,
        "source_credibility": 0.10,
        "media_forensics": 0.05,
        "bias_sentiment": 0.10,
    },
}

# Non-classifying agents that should always get zero weight in voting
_SKIP_AGENTS = {"claim_extraction", "ingestion", "orchestrator"}

# Override rules: fact-check findings can override other agents
OVERRIDE_THRESHOLD = 0.7  # if fact_check confidence > this and contradicts, override


def _evidence_quality_factor(result: AgentResult) -> float:
    """Estimate evidence quality for an agent result.

    Returns a multiplier in [0.5, 1.0]:
      - Agents with evidence get a full factor.
      - Agents with no evidence but a label get a reduced factor.
      - Agents that returned UNCERTAIN get a reduced factor.
    """
    if result.label == Label.UNCERTAIN:
        return 0.5

    has_evidence = bool(result.evidence)
    if not has_evidence:
        # Still provide some signal but penalize slightly
        return 0.7

    return 1.0


def _fact_check_evidence_strength(fact_result: AgentResult) -> float:
    """Estimate how strong the fact-check evidence is.

    Looks at the number of sources, claims checked, and whether evidence
    was actually found (vs. direct LLM fallback only).
    """
    raw = fact_result.raw_output or {}
    sources = raw.get("sources", [])
    claims_checked = raw.get("claims_checked", 0)
    claims_with_evidence = raw.get("claims_with_evidence", 0)
    direct_llm = raw.get("direct_llm_verifications", 0)

    if claims_checked == 0:
        return 0.3

    coverage = claims_with_evidence / claims_checked if claims_checked else 0.0
    source_bonus = min(len(sources) / 5.0, 1.0) * 0.3  # up to 0.3 bonus for sources

    # High coverage + real sources = strong evidence
    # Low coverage + only direct LLM = weaker evidence
    strength = 0.4 + (coverage * 0.4) + source_bonus

    # Penalize if all claims were verified only by direct LLM (no external sources)
    if direct_llm >= claims_checked and len(sources) == 0:
        strength *= 0.6

    return min(max(strength, 0.2), 1.0)


def compute_dynamic_weights(
    profile: InputProfile,
    results: list[AgentResult],
) -> tuple[dict[str, float], str]:
    """Compute effective weights based on input profile and agent results.

    Returns (weights_dict, reasoning_string).
    """
    base = BASE_WEIGHTS[profile.input_type]
    effective_weights: dict[str, float] = {}
    reasoning_parts: list[str] = []

    for r in results:
        if r.agent_name in _SKIP_AGENTS:
            continue

        base_w = base.get(r.agent_name, 0.0)
        if base_w == 0:
            continue

        # Start with base weight
        w = base_w

        # Apply agent confidence scaling
        w *= r.confidence

        # Apply evidence quality factor
        eq = _evidence_quality_factor(r)
        w *= eq

        # Special handling for fact_check: apply evidence strength
        if r.agent_name == "fact_check":
            es = _fact_check_evidence_strength(r)
            w *= es
            reasoning_parts.append(
                f"fact_check: base={base_w:.2f}, conf={r.confidence:.2f}, "
                f"evidence_quality={eq:.2f}, evidence_strength={es:.2f}"
            )
        else:
            reasoning_parts.append(
                f"{r.agent_name}: base={base_w:.2f}, conf={r.confidence:.2f}, "
                f"evidence_quality={eq:.2f}"
            )

        effective_weights[r.agent_name] = w

    # Normalize so weights sum to 1.0
    total = sum(effective_weights.values())
    if total > 0:
        for k in effective_weights:
            effective_weights[k] /= total

    reasoning = "; ".join(reasoning_parts)
    return effective_weights, reasoning


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

        # ── Input profiling ──
        profile = article.get("input_profile")
        if profile is None or not isinstance(profile, InputProfile):
            # Build profile from enriched article data
            profile = analyze_input(
                title=article.get("title", ""),
                text=article.get("text", ""),
                url=article.get("url", ""),
            )

        # ── Dynamic weight calculation ──
        effective_weights, weight_reasoning = compute_dynamic_weights(profile, results)

        weighting_reason = (
            f"Input profile: {profile.input_type.value}. "
            f"{profile.reasoning} "
            f"Effective weights: {', '.join(f'{k}={v:.3f}' for k, v in effective_weights.items())}. "
            f"Computation: {weight_reasoning}"
        )

        # === Weighted voting ===
        weighted_fake = 0.0
        weighted_real = 0.0
        total_weight = 0.0
        evidence_trail = []

        for r in results:
            weight = effective_weights.get(r.agent_name, 0.0)
            if weight == 0:
                # Still record zero-weight agents for transparency
                if r.agent_name not in _SKIP_AGENTS:
                    evidence_trail.append({
                        "agent": r.agent_name,
                        "label": r.label.value,
                        "confidence": r.confidence,
                        "weight": 0.0,
                        "effective_weight": 0.0,
                        "reasoning": r.reasoning,
                    })
                continue

            effective_weight = weight

            if r.label == Label.FAKE:
                weighted_fake += effective_weight
            elif r.label == Label.REAL:
                weighted_real += effective_weight

            total_weight += effective_weight

            evidence_trail.append({
                "agent": r.agent_name,
                "label": r.label.value,
                "confidence": r.confidence,
                "weight": round(weight, 4),
                "effective_weight": round(effective_weight, 4),
                "reasoning": r.reasoning,
            })

        # === Override rules ===
        override = False
        override_reason = ""

        fact_check_result = next((r for r in results if r.agent_name == "fact_check"), None)
        ml_result = next((r for r in results if r.agent_name == "ml_classifier"), None)

        # Fact-check override: strong external evidence overrides style prediction
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

        # ── Input-type-aware confidence adjustment ──
        # For MINIMAL inputs with weak fact-check evidence, reduce confidence
        # to avoid forcing a verdict on insufficient data.
        if profile.input_type == InputType.MINIMAL and not override:
            fc_raw = fact_check_result.raw_output if fact_check_result else {}
            fc_sources = fc_raw.get("sources", [])
            fc_claims_checked = fc_raw.get("claims_checked", 0)

            # If fact-check had no real external sources, penalize more aggressively
            if len(fc_sources) == 0 and fc_claims_checked <= 1:
                final_confidence *= 0.4
                if final_confidence < 0.5:
                    final_label = Label.UNCERTAIN
                    final_confidence = 0.0
                    evidence_trail.append({
                        "agent": "orchestrator",
                        "label": "confidence_downgrade",
                        "reasoning": (
                            "Minimal input with no verifiable factual claims found. "
                            "Downgrading to UNCERTAIN to avoid false certainty."
                        ),
                    })

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
            for r in results if r.agent_name in effective_weights and effective_weights[r.agent_name] > 0
        )
        reasoning = (
            f"Orchestrated verdict: {final_label.value} (confidence={final_confidence:.3f}). "
            f"Input profile: {profile.input_type.value}. "
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
                "input_profile": profile.input_type.value,
                "agent_weights": {k: round(v, 4) for k, v in effective_weights.items()},
                "weighting_reason": weighting_reason,
                "input_profile_details": profile.to_dict(),
            },
        )
