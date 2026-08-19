"""
Fact-Check Retrieval Agent — Web search + fact-check API cross-check.

v2 changes from the original:
  - Claim verdicts (supported/contradicted/mixed/no_evidence) are now
    decided by an LLM given the claim + retrieved evidence, instead of
    regex entity extraction and hand-tuned negation/contradiction
    pattern matching.
  - Web search now prefers a stable JSON API (Tavily) with the old
    DuckDuckGo HTML scrape kept only as a last-resort fallback.
  - Search queries per claim are LLM-generated instead of regex-derived.
  - HTTP calls run in parallel via a thread pool instead of sequentially.
  - Identical queries are cached in-process.
  - Failures are logged instead of silently swallowed.

v3 changes:
  - Added Groq as the primary LLM provider (fast + cheap inference).
    Falls back to Anthropic if Groq is unavailable/unconfigured.

  *** SECURITY NOTE ***
  GROQ_API_KEY below is hardcoded per request. Anyone who gets this file
  (git history, a shared repo, a screenshot) gets your key. At minimum:
    - add this file's path to .gitignore before committing anything else
    - rotate the key immediately if it's ever pushed to a public repo
    - prefer os.environ.get("GROQ_API_KEY") over hardcoding when you can
  Replace the placeholder string below with your real key.
"""
from __future__ import annotations

import json
import logging
import os
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

# Hardcoded per request — see security note above. Replace this placeholder.
GROQ_API_KEY = "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"
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
    if not HAS_HTTPX or not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_YOUR_"):
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
            timeout=15,
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
    """Web search via Tavily (preferred) with a DuckDuckGo HTML fallback."""
    if not HAS_HTTPX:
        return []

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "url": r.get("url", ""),
                }
                for r in data.get("results", [])[:5]
            ]
        except Exception:
            logger.exception("Tavily query failed, falling back to DDG: %r", query)

    try:
        from bs4 import BeautifulSoup
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
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
        logger.exception("DuckDuckGo fallback search failed: %r", query)
        return []


@lru_cache(maxsize=2048)
def _cached_web_search(query: str) -> tuple[dict, ...]:
    return tuple(_search_web(query))


def _generate_search_queries(claim_text: str) -> list[str]:
    """Ask the model for a few targeted, checkable search queries."""
    text = _llm_complete(
        "Generate 3 short web search queries (3-8 words each) that would "
        "help verify or refute this claim. Respond with a JSON array of "
        f"strings only, no other text.\n\nClaim: {claim_text}",
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

            analysis = _analyze_claim_verdict(claim_text, fact_results, web_results)
            analysis["original_claim"] = claim_text
            analysis["search_queries"] = queries
            return analysis

        with ThreadPoolExecutor(max_workers=min(len(claims), 5) or 1) as pool:
            claim_analyses = list(pool.map(_process, claims))

        supported = sum(1 for c in claim_analyses if c["verdict_signal"] == "supported")
        contradicted = sum(1 for c in claim_analyses if c["verdict_signal"] == "contradicted")
        mixed = sum(1 for c in claim_analyses if c["verdict_signal"] == "mixed")
        with_evidence = sum(1 for c in claim_analyses if c["verdict_signal"] != "no_evidence")

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

        return AgentResult(
            agent_name=self.name,
            label=label,
            confidence=confidence,
            reasoning=(
                f"Checked {len(claim_analyses)} claims: {supported} supported, "
                f"{contradicted} contradicted, {mixed} mixed. "
                f"{with_evidence}/{len(claim_analyses)} claims had usable evidence."
            ),
            evidence=claim_analyses,
            raw_output={
                "supported": supported,
                "contradicted": contradicted,
                "mixed": mixed,
                "claims_checked": len(claim_analyses),
                "claims_with_evidence": with_evidence,
                "sources": all_sources[:20],
            },
        )