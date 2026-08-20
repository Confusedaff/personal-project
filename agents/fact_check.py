"""
Fact-Check Retrieval Agent — Web search + fact-check API cross-check + LLM fallback.

v2: LLM-based verdict analysis instead of regex pattern matching.
v3: Groq as primary LLM provider.
v4: Direct LLM verification when no evidence is found via search.
"""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

from agents.base import BaseAgent, AgentResult, Label

logger = logging.getLogger(__name__)

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ---------------------------------------------------------------------------
# LLM provider config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_ANTHROPIC_MODEL = os.environ.get("FACTCHECK_LLM_MODEL", "claude-sonnet-4-6")
_anthropic_client = (
    anthropic.Anthropic()
    if HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY")
    else None
)


def _groq_complete(prompt: str, max_tokens: int = 400) -> str | None:
    """Call Groq's OpenAI-compatible chat completions endpoint."""
    if not HAS_HTTPX or not GROQ_API_KEY:
        return None
    try:
        resp = httpx.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Groq completion failed")
        return None


def _anthropic_complete(prompt: str, max_tokens: int = 400) -> str | None:
    if _anthropic_client is None:
        return None
    try:
        resp = _anthropic_client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        logger.exception("Anthropic completion failed")
        return None


def _llm_complete(prompt: str, max_tokens: int = 400) -> str | None:
    """Groq first (fast/cheap), fall back to Anthropic if configured."""
    text = _groq_complete(prompt, max_tokens=max_tokens)
    if text is not None:
        return text
    return _anthropic_complete(prompt, max_tokens=max_tokens)


def _strip_code_fence(text: str) -> str:
    return text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


# ---------------------------------------------------------------------------
# Direct LLM verification (fallback when search yields nothing)
# ---------------------------------------------------------------------------

_DIRECT_VERIFICATION_PROMPT = """You are a fact-checking expert. Given the following factual claim, determine if it is TRUE or FALSE based on your knowledge.

Claim: {claim}

Important rules:
- Answer TRUE if the claim is factually accurate or mostly accurate.
- Answer FALSE if the claim is factually inaccurate or mostly inaccurate.
- Do NOT guess — if you genuinely cannot determine truthfulness, say UNCERTAIN.
- Base your answer on well-established facts, not opinions.

Respond with JSON only, no other text:
{{"verdict": "TRUE" or "FALSE" or "UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "brief one-sentence explanation"}}
"""


def _direct_llm_verify(claim_text: str) -> dict:
    """Send the claim directly to the LLM for a TRUE/FALSE verdict."""
    text = _llm_complete(
        _DIRECT_VERIFICATION_PROMPT.format(claim=claim_text),
        max_tokens=300,
    )
    if text is None:
        return {"verdict_signal": "no_evidence", "confidence": 0.0,
                "reasoning": "LLM unavailable for direct verification", "sources": [],
                "method": "direct_llm_unavailable"}

    try:
        parsed = json.loads(_strip_code_fence(text))
        raw_verdict = parsed.get("verdict", "UNCERTAIN").upper()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")

        if raw_verdict == "TRUE":
            signal = "supported"
        elif raw_verdict == "FALSE":
            signal = "contradicted"
        else:
            signal = "mixed"
            confidence = min(confidence, 0.4)

        return {
            "verdict_signal": signal,
            "confidence": confidence,
            "reasoning": f"[Direct LLM] {reasoning}",
            "sources": [],
            "method": "direct_llm",
        }
    except Exception:
        logger.exception("direct LLM verification parse failed for: %r", claim_text)
        return {"verdict_signal": "no_evidence", "confidence": 0.0,
                "reasoning": "direct LLM verification failed to parse", "sources": [],
                "method": "direct_llm_parse_error"}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

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
        logger.exception("fact-check API query failed: %r", query)
        return []


def _search_web(query: str) -> list[dict]:
    """Web search via DuckDuckGo API (JSON endpoint) with HTML fallback."""
    if not HAS_HTTPX:
        return []

    # Try DuckDuckGo Instant Answer API first (more reliable than HTML scraping)
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
        data = resp.json()
        results = []
        # Abstract (main answer)
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append({
                "title": data.get("Heading", query),
                "snippet": abstract,
                "url": data.get("AbstractURL", ""),
            })
        # Related topics
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                })
        if results:
            return results
    except Exception:
        logger.debug("DuckDuckGo JSON API failed for: %r", query)

    # Fallback: DuckDuckGo HTML search
    try:
        from bs4 import BeautifulSoup
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
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
        logger.exception("DuckDuckGo HTML search failed: %r", query)
        return []


def _search_brave(query: str) -> list[dict]:
    """Fallback: Brave Search API (if configured)."""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key or not HAS_HTTPX:
        return []
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 5},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("web", {}).get("results", [])[:5]:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("description", ""),
                "url": r.get("url", ""),
            })
        return results
    except Exception:
        logger.exception("Brave search failed: %r", query)
        return []


@lru_cache(maxsize=2048)
def _cached_web_search(query: str) -> tuple[dict, ...]:
    return tuple(_search_web(query))


def _generate_search_queries(claim_text: str) -> list[str]:
    """Ask the model for targeted, checkable search queries."""
    text = _llm_complete(
        "Generate 3 short, specific web search queries (3-8 words each) that would "
        "help verify or refute this factual claim. Focus on key nouns, dates, numbers, "
        "and organizations mentioned in the claim.\n"
        "Respond with a JSON array of strings only, no other text.\n\n"
        f"Claim: {claim_text}",
        max_tokens=200,
    )
    if text is None:
        return [claim_text]
    try:
        queries = json.loads(_strip_code_fence(text))
        if isinstance(queries, list) and queries:
            return [str(q) for q in queries[:3]]
    except Exception:
        logger.exception("query generation parse failed for claim: %r", claim_text)
    return [claim_text]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

_VERDICT_PROMPT = """You are checking a single factual claim against retrieved evidence.

Claim: {claim}

Evidence (fact-check database results and web search snippets, may be noisy or irrelevant):
{evidence}

Decide the verdict signal for this claim based ONLY on the evidence above:
- "supported": evidence confirms the claim
- "contradicted": evidence shows the claim is false or a different fact is true
- "mixed": evidence conflicts or partially supports/contradicts
- "no_evidence": evidence is irrelevant, too thin, or doesn't address the claim

Respond with JSON only, no other text:
{{"verdict_signal": "...", "confidence": 0.0-1.0, "reasoning": "one sentence", "supporting_urls": ["..."], "contradicting_urls": ["..."]}}
"""


def _format_evidence(fact_results: list[dict], web_results: list[dict]) -> str:
    lines = []
    for fr in fact_results:
        lines.append(
            f"[fact-check] {fr.get('publisher', '?')} rated the claim "
            f"\"{fr.get('claim_text', '')}\" as {fr.get('verdict', '?')} "
            f"({fr.get('url', '')})"
        )
    for wr in web_results:
        lines.append(f"[web] {wr.get('title', '')}: {wr.get('snippet', '')} ({wr.get('url', '')})")
    return "\n".join(lines) if lines else "(no results returned)"


def _analyze_claim_verdict(claim_text: str, fact_results: list[dict],
                            web_results: list[dict]) -> dict:
    if not fact_results and not web_results:
        return {"verdict_signal": "no_evidence", "confidence": 0.0,
                "reasoning": "no search results retrieved", "sources": []}

    evidence_text = _format_evidence(fact_results, web_results)
    text = _llm_complete(
        _VERDICT_PROMPT.format(claim=claim_text, evidence=evidence_text),
        max_tokens=400,
    )
    urls = [r.get("url", "") for r in fact_results + web_results if r.get("url")]

    if text is None:
        return {"verdict_signal": "mixed", "confidence": 0.2,
                "reasoning": "LLM unavailable; evidence retrieved but not analyzed",
                "sources": urls[:10]}

    try:
        parsed = json.loads(_strip_code_fence(text))
        sources = list(dict.fromkeys(
            parsed.get("supporting_urls", []) + parsed.get("contradicting_urls", [])
        ))
        return {
            "verdict_signal": parsed.get("verdict_signal", "no_evidence"),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reasoning": parsed.get("reasoning", ""),
            "sources": sources[:10],
        }
    except Exception:
        logger.exception("verdict analysis parse failed for claim: %r", claim_text)
        return {"verdict_signal": "no_evidence", "confidence": 0.0,
                "reasoning": "verdict analysis failed", "sources": urls[:10]}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FactCheckAgent(BaseAgent):
    name = "fact_check"

    def run(self, article: dict[str, Any]) -> AgentResult:
        claims = article.get("claims", [])
        if not claims:
            claims = [{"claim": article.get("full_text", article.get("text", "")),
                       "type": "other", "checkability": "medium"}]
        claims = [c for c in claims[:5] if len(c.get("claim", "")) >= 10]

        def _process(claim_obj: dict) -> dict:
            claim_text = claim_obj["claim"]
            queries = _generate_search_queries(claim_text)

            # ── Phase 1: Search for evidence ──
            fact_results, web_results = [], []
            seen = set()
            with ThreadPoolExecutor(max_workers=(len(queries) * 2) or 1) as pool:
                futures = {}
                for q in queries:
                    futures[pool.submit(_search_fact_check_api, q)] = "fact"
                    futures[pool.submit(_cached_web_search, q)] = "web"
                for fut in as_completed(futures):
                    kind = futures[fut]
                    try:
                        results = fut.result()
                    except Exception:
                        continue
                    for r in results:
                        r = dict(r)
                        url = r.get("url", "")
                        if url and url in seen:
                            continue
                        seen.add(url)
                        (fact_results if kind == "fact" else web_results).append(r)

            # ── Phase 2: Analyze with LLM if evidence found ──
            if fact_results or web_results:
                analysis = _analyze_claim_verdict(claim_text, fact_results, web_results)
            else:
                # ── Phase 3: No evidence found — fallback to direct LLM verification ──
                logger.info("No search results for claim, falling back to direct LLM: %r", claim_text[:80])
                analysis = _direct_llm_verify(claim_text)

            analysis["original_claim"] = claim_text
            analysis["search_queries"] = queries
            return analysis

        with ThreadPoolExecutor(max_workers=min(len(claims), 5) or 1) as pool:
            claim_analyses = list(pool.map(_process, claims))

        supported = sum(1 for c in claim_analyses if c["verdict_signal"] == "supported")
        contradicted = sum(1 for c in claim_analyses if c["verdict_signal"] == "contradicted")
        mixed = sum(1 for c in claim_analyses if c["verdict_signal"] == "mixed")
        with_evidence = sum(1 for c in claim_analyses if c["verdict_signal"] != "no_evidence")
        direct_llm_count = sum(1 for c in claim_analyses if c.get("method") == "direct_llm")

        avg_conf = (
            sum(c["confidence"] for c in claim_analyses) / len(claim_analyses)
            if claim_analyses else 0.0
        )

        if contradicted > supported:
            label = Label.FAKE
        elif supported > contradicted:
            label = Label.REAL
        else:
            label = Label.UNCERTAIN

        coverage = (with_evidence / len(claim_analyses)) if claim_analyses else 0.0
        confidence = round(min(0.3 + avg_conf * 0.5 + coverage * 0.2, 0.95), 2)

        all_sources = list(dict.fromkeys(
            url for c in claim_analyses for url in c.get("sources", [])
        ))

        methods = [c.get("method", "search") for c in claim_analyses]
        method_summary = f"{direct_llm_count} direct LLM" if direct_llm_count else "search"

        return AgentResult(
            agent_name=self.name,
            label=label,
            confidence=confidence,
            reasoning=(
                f"Checked {len(claim_analyses)} claims: {supported} supported, "
                f"{contradicted} contradicted, {mixed} mixed. "
                f"{with_evidence}/{len(claim_analyses)} claims had usable evidence. "
                f"Method: {method_summary}."
            ),
            evidence=claim_analyses,
            raw_output={
                "supported": supported,
                "contradicted": contradicted,
                "mixed": mixed,
                "claims_checked": len(claim_analyses),
                "claims_with_evidence": with_evidence,
                "direct_llm_verifications": direct_llm_count,
                "sources": all_sources[:20],
            },
        )
