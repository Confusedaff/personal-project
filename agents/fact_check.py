"""
Fact-Check Retrieval Agent — Web search + fact-check API cross-check.

This is the real accuracy lever: for each extracted claim, searches the
live web and known fact-check databases. This closes the gap between
"does this read fake" (style) and "is this claim actually true" (facts).
"""
from __future__ import annotations

import os
import re
from typing import Any

from agents.base import BaseAgent, AgentResult, Label

# Google Fact Check Tools API (free tier: 100 queries/day)
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def _search_fact_check_api(query: str) -> list[dict]:
    """Query Google Fact Check Tools API."""
    api_key = os.environ.get("FACTCHECK_API_KEY", "")
    if not api_key or not HAS_HTTPX:
        return []

    try:
        resp = httpx.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query, "key": api_key, "languageCode": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for claim in data.get("claims", [])[:3]:
            review = claim.get("claimReview", [{}])[0]
            results.append({
                "claim_text": claim.get("text", ""),
                "claimant": claim.get("claimant", ""),
                "verdict": review.get("textualRating", ""),
                "publisher": review.get("publisher", {}).get("name", ""),
                "url": review.get("url", ""),
                "review_date": review.get("datePublished", ""),
            })
        return results
    except Exception:
        return []


def _search_web(query: str) -> list[dict]:
    """Lightweight web search via DuckDuckGo HTML (no API key needed)."""
    if not HAS_HTTPX:
        return []

    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        results = []
        # Simple extraction from result blocks
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for result in soup.find_all("div", class_="result")[:5]:
            title_el = result.find("a", class_="result__a")
            snippet_el = result.find("a", class_="result__snippet")
            url_el = result.find("a", class_="result__url")
            if title_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    "url": url_el.get_text(strip=True) if url_el else "",
                })
        return results
    except Exception:
        return []


def _analyze_claim_verdict(claim_text: str, fact_results: list[dict],
                           web_results: list[dict]) -> dict:
    """Determine if evidence supports or contradicts the claim."""
    verdict_signal = "no_evidence"
    supporting = 0
    contradicting = 0
    sources = []

    # Check fact-check API results
    for fr in fact_results:
        rating = fr.get("verdict", "").lower()
        sources.append(fr.get("url", ""))
        if any(w in rating for w in ["false", "fake", "misleading", "pants on fire", "mostly false"]):
            contradicting += 2
        elif any(w in rating for w in ["true", "mostly true", "correct"]):
            supporting += 2
        elif any(w in rating for w in ["mixture", "half true", "partly"]):
            supporting += 1
            contradicting += 1

    # Check web results for corroborating/contradicting signals
    for wr in web_results:
        snippet = (wr.get("snippet", "") + " " + wr.get("title", "")).lower()
        # Simple keyword overlap
        claim_words = set(claim_text.lower().split())
        snippet_words = set(snippet.split())
        overlap = len(claim_words & snippet_words)
        if overlap > 3:
            sources.append(wr.get("url", ""))
            # Check if snippet contradicts
            negation_words = {"not", "false", "debunked", "incorrect", "wrong", "misleading"}
            if any(nw in snippet for nw in negation_words):
                contradicting += 1
            else:
                supporting += 1

    if contradicting > supporting:
        verdict_signal = "contradicted"
    elif supporting > contradicting:
        verdict_signal = "supported"
    elif supporting == contradicting and supporting > 0:
        verdict_signal = "mixed"
    else:
        verdict_signal = "no_evidence"

    return {
        "verdict_signal": verdict_signal,
        "supporting": supporting,
        "contradicting": contradicting,
        "sources": sources[:10],
    }


class FactCheckAgent(BaseAgent):
    name = "fact_check"

    def run(self, article: dict[str, Any]) -> AgentResult:
        claims = article.get("claims", [])
        if not claims:
            # Try to extract from the raw article
            claims = [{"claim": article.get("full_text", article.get("text", "")),
                       "type": "other", "checkability": "medium"}]

        claim_analyses = []
        total_contradicted = 0
        total_supported = 0
        all_sources = []

        for claim_obj in claims[:5]:  # limit to 5 claims
            claim_text = claim_obj.get("claim", "")
            if not claim_text or len(claim_text) < 10:
                continue

            # Search fact-check databases
            fact_results = _search_fact_check_api(claim_text)

            # Also do a general web search
            web_results = _search_web(claim_text)

            analysis = _analyze_claim_verdict(claim_text, fact_results, web_results)
            analysis["original_claim"] = claim_text
            claim_analyses.append(analysis)

            total_contradicted += analysis["contradicting"]
            total_supported += analysis["supporting"]
            all_sources.extend(analysis["sources"])

        # Overall verdict
        if total_contradicted > total_supported:
            label = Label.FAKE
            confidence = min(0.5 + total_contradicted * 0.1, 0.95)
        elif total_supported > total_contradicted:
            label = Label.REAL
            confidence = min(0.5 + total_supported * 0.1, 0.95)
        else:
            label = Label.UNCERTAIN
            confidence = 0.3

        # Count how many claims had evidence
        claims_with_evidence = sum(
            1 for c in claim_analyses if c["verdict_signal"] != "no_evidence"
        )

        return AgentResult(
            agent_name=self.name,
            label=label,
            confidence=confidence,
            reasoning=(
                f"Checked {len(claim_analyses)} claims: "
                f"{total_supported} supporting signals, {total_contradicted} contradicting. "
                f"{claims_with_evidence}/{len(claim_analyses)} claims had external evidence."
            ),
            evidence=claim_analyses,
            raw_output={
                "total_supporting": total_supported,
                "total_contradicting": total_contradicted,
                "claims_checked": len(claim_analyses),
                "claims_with_evidence": claims_with_evidence,
                "sources": list(set(all_sources))[:20],
            },
        )
