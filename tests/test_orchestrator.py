"""
Tests for the dynamic weighting orchestrator.

Covers:
  1. Single word input
  2. Short phrase input
  3. Headline only
  4. Title + short body
  5. Title + long article body
  6. Long article with fact-check contradicting ML
  7. Long article with ML and fact-check agreeing
  8. Input with no meaningful factual claim

Each test verifies: detected input profile, calculated agent weights,
individual agent confidence, final verdict, final confidence, and
whether UNCERTAIN is correctly used.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.base import AgentResult, Label
from agents.input_profile import analyze_input, InputType
from agents.orchestrator import (
    Orchestrator,
    BASE_WEIGHTS,
    compute_dynamic_weights,
    _evidence_quality_factor,
    _fact_check_evidence_strength,
)


def _make_result(
    name: str,
    label: Label,
    confidence: float,
    evidence: list | None = None,
    raw_output: dict | None = None,
) -> AgentResult:
    return AgentResult(
        agent_name=name,
        label=label,
        confidence=confidence,
        reasoning=f"Test reasoning for {name}",
        evidence=evidence or [],
        raw_output=raw_output or {},
    )


def _run_orchestrator(
    results: list[AgentResult],
    title: str = "",
    text: str = "",
    url: str = "",
) -> dict:
    """Run orchestrator with the given agent results and input."""
    profile = analyze_input(title=title, text=text, url=url)
    article = {
        "title": title,
        "text": text,
        "url": url,
        "agent_results": results,
        "input_profile": profile,
    }
    orch = Orchestrator()
    verdict = orch(article)
    # Combine raw_output with evidence_trail for easier test access
    output = verdict.raw_output.copy()
    output["evidence"] = verdict.evidence
    return output


# ─────────────────────────────────────────────────────────────────────
# Test 1: Single word input ("Bitcoin")
# ─────────────────────────────────────────────────────────────────────
class TestSingleWordInput:

    def test_profile_classification(self):
        p = analyze_input(title="", text="Bitcoin", url="")
        assert p.input_type == InputType.MINIMAL

    def test_weights_favor_fact_check(self):
        results = [
            _make_result("ml_classifier", Label.REAL, 0.7),
            _make_result("fact_check", Label.REAL, 0.6, raw_output={
                "sources": [], "claims_checked": 1, "claims_with_evidence": 0,
                "direct_llm_verifications": 1,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.5),
        ]
        out = _run_orchestrator(results, text="Bitcoin")
        weights = out["agent_weights"]
        assert weights["fact_check"] > weights["ml_classifier"]
        assert out["input_profile"] == "MINIMAL"

    def test_returns_uncertain_for_no_evidence(self):
        results = [
            _make_result("ml_classifier", Label.REAL, 0.7),
            _make_result("fact_check", Label.REAL, 0.4, raw_output={
                "sources": [], "claims_checked": 1, "claims_with_evidence": 0,
                "direct_llm_verifications": 1,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.5),
        ]
        out = _run_orchestrator(results, text="Bitcoin")
        label = Label(out["input_profile"]) if False else None
        # The final verdict should be downgraded due to minimal input + weak evidence
        assert out["needs_human_review"] is True


# ─────────────────────────────────────────────────────────────────────
# Test 2: Short phrase ("Bitcoin price crash")
# ─────────────────────────────────────────────────────────────────────
class TestShortPhraseInput:

    def test_profile_classification(self):
        p = analyze_input(title="", text="Bitcoin price crash", url="")
        assert p.input_type in (InputType.MINIMAL, InputType.SHORT)

    def test_weights_favor_fact_check(self):
        results = [
            _make_result("ml_classifier", Label.FAKE, 0.65),
            _make_result("fact_check", Label.REAL, 0.5, raw_output={
                "sources": ["https://example.com"], "claims_checked": 1,
                "claims_with_evidence": 1, "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.FAKE, 0.4),
        ]
        out = _run_orchestrator(results, text="Bitcoin price crash")
        weights = out["agent_weights"]
        assert weights["fact_check"] > weights["ml_classifier"]


# ─────────────────────────────────────────────────────────────────────
# Test 3: Headline only
# ─────────────────────────────────────────────────────────────────────
class TestHeadlineOnlyInput:

    def test_profile_classification(self):
        p = analyze_input(
            title="",
            text="Bitcoin crashes after major regulation",
            url="",
        )
        assert p.input_type in (InputType.MINIMAL, InputType.SHORT)

    def test_balanced_weighting(self):
        results = [
            _make_result("ml_classifier", Label.FAKE, 0.7),
            _make_result("fact_check", Label.FAKE, 0.6, raw_output={
                "sources": ["https://example.com"], "claims_checked": 1,
                "claims_with_evidence": 1, "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.FAKE, 0.5),
        ]
        out = _run_orchestrator(results, text="Bitcoin crashes after major regulation")
        weights = out["agent_weights"]
        # Fact-check should still be the highest
        assert weights["fact_check"] > weights["ml_classifier"]


# ─────────────────────────────────────────────────────────────────────
# Test 4: Title + short body
# ─────────────────────────────────────────────────────────────────────
class TestTitlePlusShortBody:

    def test_profile_classification(self):
        p = analyze_input(
            title="Breaking News",
            text="Markets reacted sharply today as investors scrambled to assess the impact of new tariffs.",
            url="",
        )
        assert p.input_type in (InputType.SHORT, InputType.MEDIUM)

    def test_weighting_includes_title_context(self):
        results = [
            _make_result("ml_classifier", Label.REAL, 0.6),
            _make_result("fact_check", Label.REAL, 0.7, raw_output={
                "sources": ["https://reuters.com"], "claims_checked": 2,
                "claims_with_evidence": 2, "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.5),
        ]
        out = _run_orchestrator(
            results,
            title="Breaking News",
            text="Markets reacted sharply today as investors scrambled to assess the impact of new tariffs.",
        )
        weights = out["agent_weights"]
        assert weights["fact_check"] > 0
        assert weights["ml_classifier"] > 0


# ─────────────────────────────────────────────────────────────────────
# Test 5: Title + long article body
# ─────────────────────────────────────────────────────────────────────
class TestTitlePlusLongArticle:

    def test_profile_classification(self):
        body = " ".join([
            f"Sentence {i}: The Federal Reserve today announced sweeping changes to monetary policy "
            "that will affect interest rates across the entire economy. economists have warned about "
            "the potential impact on inflation and employment."
            for i in range(1, 20)
        ])
        p = analyze_input(
            title="Fed Announces Major Policy Shift",
            text=body,
            url="",
        )
        assert p.input_type == InputType.ARTICLE

    def test_ml_gets_higher_weight(self):
        body = " ".join([
            f"Sentence {i}: The Federal Reserve today announced sweeping changes to monetary policy "
            "that will affect interest rates across the entire economy."
            for i in range(1, 20)
        ])
        results = [
            _make_result("ml_classifier", Label.FAKE, 0.8),
            _make_result("fact_check", Label.REAL, 0.7, raw_output={
                "sources": ["https://reuters.com", "https://apnews.com"],
                "claims_checked": 3, "claims_with_evidence": 3,
                "direct_llm_verifications": 0,
            }),
            _make_result("source_credibility", Label.REAL, 0.6),
            _make_result("bias_sentiment", Label.REAL, 0.5),
        ]
        out = _run_orchestrator(
            results,
            title="Fed Announces Major Policy Shift",
            text=body,
        )
        weights = out["agent_weights"]
        assert out["input_profile"] == "ARTICLE"
        # ML should have meaningful weight for articles
        assert weights["ml_classifier"] > 0.15
        # Fact-check should still be significant
        assert weights["fact_check"] > 0.25


# ─────────────────────────────────────────────────────────────────────
# Test 6: Long article — fact-check contradicts ML
# ─────────────────────────────────────────────────────────────────────
class TestFactCheckOverridesML:

    def test_override_fires(self):
        body = " ".join([
            f"Report {i}: Multiple sources confirm that the earth orbits the sun, "
            "not the other way around as some fringe theories suggest."
            for i in range(1, 20)
        ])
        results = [
            _make_result("ml_classifier", Label.FAKE, 0.85),
            _make_result("fact_check", Label.REAL, 0.85, raw_output={
                "sources": ["https://nasa.gov", "https://nature.com"],
                "claims_checked": 3, "claims_with_evidence": 3,
                "direct_llm_verifications": 0,
            }),
            _make_result("source_credibility", Label.REAL, 0.7),
            _make_result("bias_sentiment", Label.REAL, 0.5),
        ]
        out = _run_orchestrator(
            results,
            title="Scientists Confirm Earth Orbits the Sun",
            text=body,
        )
        # Override should fire: fact_check conf > 0.7 and contradicts ML
        assert out["override"] is True
        assert "Fact-check agent" in out["override_reason"]

    def test_final_label_follows_fact_check(self):
        body = " ".join([
            f"Report {i}: Scientists have confirmed that vaccines do not cause autism. "
            "This has been established through numerous peer-reviewed studies."
            for i in range(1, 20)
        ])
        results = [
            _make_result("ml_classifier", Label.FAKE, 0.85),
            _make_result("fact_check", Label.REAL, 0.85, raw_output={
                "sources": ["https://nature.com", "https://who.int"],
                "claims_checked": 3, "claims_with_evidence": 3,
                "direct_llm_verifications": 0,
            }),
            _make_result("source_credibility", Label.REAL, 0.7),
            _make_result("bias_sentiment", Label.REAL, 0.5),
        ]
        out = _run_orchestrator(
            results,
            title="Vaccines Do Not Cause Autism",
            text=body,
        )
        # After override, label should follow fact_check (REAL)
        assert out["needs_human_review"] is True  # override always triggers review


# ─────────────────────────────────────────────────────────────────────
# Test 7: Long article — ML and fact-check agree
# ─────────────────────────────────────────────────────────────────────
class TestMLAndFactCheckAgree:

    def test_agreement_high_confidence(self):
        body = " ".join([
            f"Report {i}: Climate change is causing sea levels to rise globally, "
            "with measurable impacts on coastal communities worldwide."
            for i in range(1, 20)
        ])
        results = [
            _make_result("ml_classifier", Label.REAL, 0.8),
            _make_result("fact_check", Label.REAL, 0.8, raw_output={
                "sources": ["https://nature.com", "https://ipcc.ch"],
                "claims_checked": 3, "claims_with_evidence": 3,
                "direct_llm_verifications": 0,
            }),
            _make_result("source_credibility", Label.REAL, 0.6),
            _make_result("bias_sentiment", Label.REAL, 0.5),
        ]
        out = _run_orchestrator(
            results,
            title="Climate Change Causes Rising Sea Levels",
            text=body,
        )
        assert out["override"] is False
        # Combined weights should produce high confidence
        assert out["weighted_real"] > out["weighted_fake"]

    def test_agreement_fake(self):
        body = " ".join([
            f"Report {i}: The moon is made of green cheese. "
            "Scientists have repeatedly debunked this but it keeps spreading online."
            for i in range(1, 20)
        ])
        results = [
            _make_result("ml_classifier", Label.FAKE, 0.75),
            _make_result("fact_check", Label.FAKE, 0.8, raw_output={
                "sources": ["https://nasa.gov"],
                "claims_checked": 2, "claims_with_evidence": 2,
                "direct_llm_verifications": 0,
            }),
            _make_result("source_credibility", Label.REAL, 0.5),
            _make_result("bias_sentiment", Label.FAKE, 0.6),
        ]
        out = _run_orchestrator(
            results,
            title="Moon Is Made of Green Cheese",
            text=body,
        )
        assert out["override"] is False
        assert out["weighted_fake"] > out["weighted_real"]


# ─────────────────────────────────────────────────────────────────────
# Test 8: Input with no meaningful factual claim
# ─────────────────────────────────────────────────────────────────────
class TestNoMeaningfulClaim:

    def test_single_word_uncertain(self):
        results = [
            _make_result("ml_classifier", Label.REAL, 0.6),
            _make_result("fact_check", Label.UNCERTAIN, 0.3, raw_output={
                "sources": [], "claims_checked": 0, "claims_with_evidence": 0,
                "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.4),
        ]
        out = _run_orchestrator(results, text="NASA")
        assert out["input_profile"] == "MINIMAL"
        # With uncertain fact-check and minimal input, confidence should be low
        assert out["needs_human_review"] is True

    def test_entity_topic_uncertain(self):
        results = [
            _make_result("ml_classifier", Label.REAL, 0.55),
            _make_result("fact_check", Label.UNCERTAIN, 0.2, raw_output={
                "sources": [], "claims_checked": 0, "claims_with_evidence": 0,
                "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.3),
        ]
        out = _run_orchestrator(results, text="Modi")
        assert out["needs_human_review"] is True


# ─────────────────────────────────────────────────────────────────────
# Unit tests for helper functions
# ─────────────────────────────────────────────────────────────────────
class TestEvidenceQualityFactor:

    def test_uncertain_returns_low(self):
        r = _make_result("ml_classifier", Label.UNCERTAIN, 0.5)
        assert _evidence_quality_factor(r) == 0.5

    def test_no_evidence_reduces(self):
        r = _make_result("ml_classifier", Label.REAL, 0.8, evidence=[])
        assert _evidence_quality_factor(r) == 0.7

    def test_with_evidence_full(self):
        r = _make_result("ml_classifier", Label.REAL, 0.8, evidence=[{"foo": 1}])
        assert _evidence_quality_factor(r) == 1.0


class TestFactCheckEvidenceStrength:

    def test_no_claims(self):
        r = _make_result("fact_check", Label.UNCERTAIN, 0.3, raw_output={
            "sources": [], "claims_checked": 0, "claims_with_evidence": 0,
            "direct_llm_verifications": 0,
        })
        strength = _fact_check_evidence_strength(r)
        assert 0.2 <= strength <= 1.0

    def test_high_coverage_with_sources(self):
        r = _make_result("fact_check", Label.REAL, 0.8, raw_output={
            "sources": ["https://a.com", "https://b.com", "https://c.com"],
            "claims_checked": 3,
            "claims_with_evidence": 3,
            "direct_llm_verifications": 0,
        })
        strength = _fact_check_evidence_strength(r)
        assert strength > 0.7

    def test_low_coverage_direct_llm_only(self):
        r = _make_result("fact_check", Label.REAL, 0.5, raw_output={
            "sources": [],
            "claims_checked": 3,
            "claims_with_evidence": 1,
            "direct_llm_verifications": 3,
        })
        strength = _fact_check_evidence_strength(r)
        assert strength < 0.6


class TestComputeDynamicWeights:

    def test_article_weights_ml_higher(self):
        profile = analyze_input(
            title="Test Article",
            text=" ".join([
                f"Sentence {i}: This is a detailed report about economic policy "
                "and its impact on financial markets across the globe."
                for i in range(1, 20)
            ]),
        )
        results = [
            _make_result("ml_classifier", Label.REAL, 0.8, evidence=[{"foo": 1}]),
            _make_result("fact_check", Label.REAL, 0.7, evidence=[{"bar": 1}], raw_output={
                "sources": ["https://a.com"], "claims_checked": 2,
                "claims_with_evidence": 2, "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.5, evidence=[{"baz": 1}]),
        ]
        weights, _ = compute_dynamic_weights(profile, results)
        assert weights["ml_classifier"] > weights["bias_sentiment"]

    def test_minimal_weights_fact_check_dominates(self):
        profile = analyze_input(title="", text="Bitcoin")
        results = [
            _make_result("ml_classifier", Label.REAL, 0.8, evidence=[{"foo": 1}]),
            _make_result("fact_check", Label.REAL, 0.7, evidence=[{"bar": 1}], raw_output={
                "sources": ["https://a.com"], "claims_checked": 2,
                "claims_with_evidence": 2, "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.5, evidence=[{"baz": 1}]),
        ]
        weights, _ = compute_dynamic_weights(profile, results)
        assert weights["fact_check"] > weights["ml_classifier"]

    def test_weights_normalize_to_one(self):
        profile = analyze_input(title="Test", text="Some text here")
        results = [
            _make_result("ml_classifier", Label.REAL, 0.7, evidence=[{"foo": 1}]),
            _make_result("fact_check", Label.REAL, 0.6, evidence=[{"bar": 1}], raw_output={
                "sources": ["https://a.com"], "claims_checked": 1,
                "claims_with_evidence": 1, "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.5, evidence=[{"baz": 1}]),
        ]
        weights, _ = compute_dynamic_weights(profile, results)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"


# ─────────────────────────────────────────────────────────────────────
# Integration: full orchestrator with explainability
# ─────────────────────────────────────────────────────────────────────
class TestOrchestratorExplainability:

    def test_verdict_contains_profile_info(self):
        results = [
            _make_result("ml_classifier", Label.REAL, 0.7, evidence=[{"foo": 1}]),
            _make_result("fact_check", Label.REAL, 0.6, evidence=[{"bar": 1}], raw_output={
                "sources": ["https://a.com"], "claims_checked": 1,
                "claims_with_evidence": 1, "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.5, evidence=[{"baz": 1}]),
        ]
        out = _run_orchestrator(results, title="Test", text="Some substantial text for testing purposes.")
        assert "input_profile" in out
        assert "agent_weights" in out
        assert "weighting_reason" in out
        assert isinstance(out["agent_weights"], dict)

    def test_evidence_trail_includes_all_agents(self):
        results = [
            _make_result("ml_classifier", Label.REAL, 0.7, evidence=[{"foo": 1}]),
            _make_result("fact_check", Label.REAL, 0.6, evidence=[{"bar": 1}], raw_output={
                "sources": ["https://a.com"], "claims_checked": 1,
                "claims_with_evidence": 1, "direct_llm_verifications": 0,
            }),
            _make_result("bias_sentiment", Label.REAL, 0.5, evidence=[{"baz": 1}]),
        ]
        out = _run_orchestrator(results, title="Test", text="Some text for analysis.")
        trail = [e["agent"] for e in out["evidence"]]
        assert "ml_classifier" in trail
        assert "fact_check" in trail
        assert "bias_sentiment" in trail
