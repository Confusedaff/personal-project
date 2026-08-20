"""
Tests for the input profiling module.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.input_profile import analyze_input, InputType


class TestMinimalInput:
    """Single word or entity with no body."""

    def test_single_word(self):
        p = analyze_input(title="", text="Bitcoin", url="")
        assert p.input_type == InputType.MINIMAL
        assert p.is_single_token is True
        assert p.total_word_count == 1

    def test_single_word_with_title(self):
        p = analyze_input(title="Modi", text="", url="")
        assert p.input_type == InputType.MINIMAL
        assert p.has_title is True
        assert p.body_length == 0

    def test_two_word_entity(self):
        p = analyze_input(title="", text="COVID pandemic", url="")
        assert p.input_type == InputType.MINIMAL

    def test_single_token_federal_agency(self):
        p = analyze_input(title="", text="NASA", url="")
        assert p.input_type == InputType.MINIMAL

    def test_no_input(self):
        p = analyze_input(title="", text="", url="")
        assert p.input_type == InputType.MINIMAL


class TestShortInput:
    """Short phrases or brief headlines."""

    def test_short_phrase(self):
        p = analyze_input(title="", text="Bitcoin price crash today", url="")
        assert p.input_type == InputType.SHORT

    def test_headline_only(self):
        p = analyze_input(
            title="",
            text="Bitcoin crashes after major regulation news hits market",
            url="",
        )
        assert p.input_type == InputType.SHORT

    def test_brief_with_title(self):
        p = analyze_input(title="Breaking", text="Markets are down sharply", url="")
        assert p.input_type == InputType.SHORT


class TestMediumInput:
    """Some text but not a full article."""

    def test_partial_paragraph(self):
        body = (
            "The Federal Reserve announced today that it will raise interest rates "
            "by 25 basis points. The decision comes after months of debate about "
            "inflationary pressures. Markets reacted negatively to the news, "
            "with the S&P 500 falling 1.5% in after-hours trading."
        )
        p = analyze_input(title="Fed Raises Rates", text=body, url="")
        assert p.input_type == InputType.MEDIUM
        assert p.has_title is True
        assert p.has_factual_claims is True

    def test_medium_without_title(self):
        body = (
            "Scientists at MIT have developed a new type of battery that can "
            "charge in under 5 minutes. The breakthrough uses a novel lithium "
            "chemistry that significantly improves energy density. The research "
            "team says commercial production could begin within two years."
        )
        p = analyze_input(title="", text=body, url="")
        assert p.input_type == InputType.MEDIUM


class TestArticleInput:
    """Full article body or title + long body."""

    def test_long_article(self):
        body = (
            "In a dramatic shift in U.S. trade policy, the administration announced "
            "sweeping tariffs on imported steel and aluminum, citing national security "
            "concerns. The tariffs, which take effect immediately, impose a 25% duty on "
            "all steel imports and a 10% duty on aluminum.\n\n"
            "Industry leaders expressed mixed reactions. The American Steel Institute "
            "welcomed the move, saying it would protect domestic jobs and national "
            "security. However, the National Association of Manufacturers warned that "
            "higher costs would ultimately hurt consumers and downstream industries.\n\n"
            "Economists at the Peterson Institute estimated the tariffs could cost the "
            "U.S. economy up to $300 billion annually. 'This is a significant escalation "
            "of trade tensions,' said Peterson Institute senior fellow Gary Hufbauer. "
            "The tariffs also apply to imports from allies including the European Union, "
            "Canada, and Japan.\n\n"
            "The WTO expressed concern about the broader implications for global trade. "
            "Several countries signaled they would challenge the tariffs under WTO rules. "
            "The EU trade commissioner called the tariffs 'unjustified and illegal' under "
            "international trade agreements.\n\n"
            "Stock markets reacted sharply to the announcement. The Dow Jones Industrial "
            "Average fell 400 points in after-hours trading, while shares of major steel "
            "producers surged. Analysts expect the trade tensions to persist for months, "
            "with potential retaliatory measures from affected trading partners."
        )
        p = analyze_input(
            title="Trump Imposes Sweeping Tariffs on Steel and Aluminum",
            text=body,
            url="",
        )
        assert p.input_type == InputType.ARTICLE
        assert p.sentence_count >= 10
        assert p.total_word_count >= 100
        assert p.has_factual_claims is True

    def test_long_article_without_title(self):
        body = " ".join([
            f"Sentence number {i} describes some aspect of the ongoing debate about "
            "artificial intelligence regulation in the European Union. The EU has been "
            "working on comprehensive legislation to govern AI systems since 2021."
            for i in range(1, 25)
        ])
        p = analyze_input(title="", text=body, url="")
        assert p.input_type == InputType.ARTICLE


class TestFactualClaimDetection:
    """Test the heuristic for detecting factual claims."""

    def test_claim_detected(self):
        text = "The stock market crashed yesterday after the Federal Reserve announced new policies."
        p = analyze_input(text=text)
        assert p.has_factual_claims is True

    def test_no_claim_in_entity(self):
        p = analyze_input(text="Bitcoin")
        assert p.has_factual_claims is False

    def test_opinion_may_not_trigger(self):
        # Short opinion with fewer indicators
        text = "This movie is really great and I love it"
        p = analyze_input(text=text)
        assert p.has_factual_claims is False


class TestProfileSerialization:
    """Test that InputProfile serializes correctly."""

    def test_to_dict(self):
        p = analyze_input(title="Test", text="Some body text here", url="")
        d = p.to_dict()
        assert "input_type" in d
        assert "has_title" in d
        assert "total_word_count" in d
        assert "reasoning" in d
        assert isinstance(d["input_type"], str)
