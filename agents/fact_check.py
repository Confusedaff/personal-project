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


def _extract_key_entities(text: str) -> dict:
    """Extract key entities and relationships from a claim."""
    text_lower = text.lower()
    entities = {
        "proper_nouns": set(),
        "roles": set(),
        "numbers": set(),
        "dates": set(),
        "relationships": [],
        "subjects": set(),
        "objects": set(),
    }

    # Extract proper nouns (capitalized words not at sentence start)
    words = text.split()
    for i, word in enumerate(words):
        clean = re.sub(r'[^\w]', '', word)
        if clean and clean[0].isupper() and (i > 0 or len(clean) > 1):
            entities["proper_nouns"].add(clean.lower())

    # Extract role/title patterns
    role_patterns = [
        r'president(?:\s+of\s+\w+)?',
        r'prime\s+minister(?:\s+of\s+\w+)?',
        r'ceo(?:\s+of\s+\w+)?',
        r'king|queen|emperor|ruler',
        r'capital(?:\s+of\s+\w+)?',
        r'largest|smallest|biggest|tallest|fastest',
    ]
    for pattern in role_patterns:
        matches = re.findall(pattern, text_lower)
        entities["roles"].update(matches)

    # Extract numbers with context
    num_matches = re.findall(r'(\d[\d,.]*)\s*(percent|%|million|billion|trillion|years?|days?|km|miles?)?', text_lower)
    for num, unit in num_matches:
        entities["numbers"].add(f"{num}{unit}".strip())

    # Extract "X is/are/was/were Y" patterns (primary relationship)
    is_patterns = [
        re.compile(r'(.+?)\s+(?:is|are|was|were)\s+(.+?)(?:\.|,|$)', re.I),
    ]
    for pattern in is_patterns:
        for match in pattern.finditer(text):
            subj = match.group(1).strip().lower()
            obj = match.group(2).strip().lower()
            # Clean up articles and pronouns
            subj = re.sub(r'^(the|a|an)\s+', '', subj)
            obj = re.sub(r'^(the|a|an|mr|mrs|ms|dr)\s+', '', obj)
            if len(subj) > 2 and len(obj) > 2:
                entities["subjects"].add(subj)
                entities["objects"].add(obj)
                entities["relationships"].append({
                    "subject": subj,
                    "object": obj,
                    "full": match.group(0).lower(),
                    "type": "is_relationship",
                })

    # Extract "X of Y" patterns
    rel_patterns = [
        re.compile(r'(\w+)\s+of\s+(\w+(?:\s+\w+)?)', re.I),
    ]
    for pattern in rel_patterns:
        for match in pattern.finditer(text):
            subj = match.group(1).lower()
            obj = match.group(2).lower()
            entities["subjects"].add(subj)
            entities["objects"].add(obj)
            entities["relationships"].append({
                "subject": subj,
                "object": obj,
                "full": match.group(0).lower(),
                "type": "of_relationship",
            })

    return entities


def _check_entity_contradiction(claim_entities: dict, snippet: str) -> str:
    """Check if snippet contradicts the claim's entity relationships."""
    snippet_lower = snippet.lower()

    # Check for explicit "X is NOT Y" or "X was NOT Y"
    negation_patterns = [
        re.compile(r'(.+?)\s+(?:is|was|are|were)\s+not\s+(.+?)(?:\.|,|$)', re.I),
        re.compile(r'(.+?)\s+(?:is|was|are|were)\s+never\s+(.+?)(?:\.|,|$)', re.I),
        re.compile(r'denied\s+that\s+(.+?)\s+(?:is|was)\s+(.+?)(?:\.|,|$)', re.I),
        re.compile(r'false\s+(?:that|claim|claiming)\s+(.+?)\s+(?:is|was)\s+(.+?)(?:\.|,|$)', re.I),
    ]
    for pattern in negation_patterns:
        match = pattern.search(snippet_lower)
        if match:
            neg_subject = match.group(1).strip()
            neg_object = match.group(2).strip()
            for rel in claim_entities.get("relationships", []):
                # Check if subject matches
                subj_match = (neg_subject in rel["subject"] or rel["subject"] in neg_subject or
                             any(p in neg_subject for p in rel["subject"].split() if len(p) > 3))
                obj_match = (neg_object in rel["object"] or rel["object"] in neg_object or
                            any(p in neg_object for p in rel["object"].split() if len(p) > 3))
                if subj_match and obj_match:
                    return "contradicts"

    # Check for "actually" or "in fact" corrections
    correction_patterns = [
        re.compile(r'(?:actually|in\s+fact|in\sreality|the\s+actual)\s.*?(.+?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|$)', re.I),
        re.compile(r'(?:the\s+real|the\s+actual|the\s+true)\s+(.+?)\s+(?:is|was)\s+(.+?)(?:\.|,|$)', re.I),
    ]
    for pattern in correction_patterns:
        match = pattern.search(snippet_lower)
        if match:
            corr_subject = match.group(1).strip()
            corr_object = match.group(2).strip()
            for rel in claim_entities.get("relationships", []):
                subj_match = (corr_subject in rel["subject"] or rel["subject"] in corr_subject or
                             any(p in corr_subject for p in rel["subject"].split() if len(p) > 3))
                obj_mismatch = (corr_object not in rel["object"] and rel["object"] not in corr_object and
                               not any(p in corr_object for p in rel["object"].split() if len(p) > 3))
                if subj_match and obj_mismatch:
                    return "contradicts"

    # Check for "X, not Y" patterns
    not_pattern = re.compile(r'(\w+(?:\s+\w+)?)\s*,\s*not\s+(\w+(?:\s+\w+)?)', re.I)
    for match in not_pattern.finditer(snippet_lower):
        affirmed = match.group(1)
        negated = match.group(2)
        for rel in claim_entities.get("relationships", []):
            if negated in rel["object"] or rel["object"] in negated:
                return "contradicts"

    # KEY FIX: Check for same subject with different object (the main contradiction pattern)
    # Example: claim says "president of India is Trump" but snippet says "president of America is Trump"
    for rel in claim_entities.get("relationships", []):
        # Only check is_relationship (full claims)
        if rel.get("type") != "is_relationship":
            continue

        subj_words = set(rel["subject"].split())
        obj_words = set(rel["object"].split())

        # Find all "X is Y" patterns in snippet (use greedy match for full object)
        for snippet_match in re.finditer(
            rf'(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|$)',
            snippet_lower
        ):
            snip_subj = snippet_match.group(1)
            snip_obj = snippet_match.group(2).strip()

            # Check if subject overlaps (e.g., both mention "modi")
            subj_overlap = len(subj_words & set(snip_subj.split())) > 0

            if subj_overlap:
                # Subject matches - now check if object is different
                # Check if snippet explicitly confirms the SAME relationship
                confirmed_same = any(
                    re.search(rf'{re.escape(snip_subj)}\s+(?:is|was|are|were)\s+{re.escape(obj)}', snippet_lower)
                    for obj in [rel["object"]]
                )
                if confirmed_same:
                    break  # Snippet confirms the claim - no contradiction

                # Check if snippet explicitly states a DIFFERENT relationship
                obj_no_overlap = len(obj_words & set(snip_obj.split())) == 0
                if obj_no_overlap:
                    # Different object for same subject - contradiction only if snippet explicitly states it
                    if re.search(rf'{re.escape(snip_subj)}\s+(?:is|was|are|were)\s+{re.escape(snip_obj)}', snippet_lower):
                        return "contradicts"

    # Also check if snippet mentions the proper nouns in a contradictory context
    proper_nouns = claim_entities.get("proper_nouns", set())
    if proper_nouns:
        # Check for patterns like "X, who is Y" or "X, the Y of Z"
        for noun in proper_nouns:
            if noun in snippet_lower:
                # Check if snippet says this noun has a different role
                context_match = re.search(
                    rf'{noun}[^.]*?(?:is|was|are|were)\s+(\w+(?:\s+\w+)?)',
                    snippet_lower
                )
                if context_match:
                    mentioned_role = context_match.group(1)
                    # Check if this role contradicts the claim
                    for rel in claim_entities.get("relationships", []):
                        if noun in rel["object"] or noun in rel["full"]:
                            if mentioned_role not in rel["subject"] and rel["subject"] not in mentioned_role:
                                # Different role for same entity - potential contradiction
                                pass  # Weakened: don't flag as contradiction without more evidence

    # CRITICAL: Check if snippet states a DIFFERENT person/entity holds the SAME role
    # Example: claim says "president of India is Trump" but snippet says "Droupadi Murmu is president of India"
    # Only check "is_relationship" type (full claims like "X is Y"), skip partial "of_relationship"
    for rel in claim_entities.get("relationships", []):
        # Skip "of_relationship" if we already have an "is_relationship" for the same role
        if rel.get("type") == "of_relationship":
            has_is_rel = any(
                r.get("type") == "is_relationship" and
                any(w in r.get("full", "") for w in rel.get("subject", "").split() if len(w) > 3)
                for r in claim_entities.get("relationships", [])
            )
            if has_is_rel:
                continue  # Skip this partial relationship

        # For "X is Y" relationships, the role is the object, the person is the subject
        if rel.get("type") == "is_relationship":
            role_phrase = rel["object"]  # e.g., "prime minister of india"
            claimed_holder = rel["subject"]  # e.g., "narendra modi"
        else:
            role_phrase = rel["full"]  # e.g., "president of india"
            claimed_holder = rel["object"]  # e.g., "mr donald trump"

        # Clean role_phrase (remove trailing period/punctuation)
        role_phrase = role_phrase.rstrip('.').strip()

        # Check if snippet mentions the same role
        if role_phrase in snippet_lower:
            # Find who the snippet says holds this role
            role_holder_match = re.search(
                rf'(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)\s+)?(?:current\s+)?{re.escape(role_phrase)}',
                snippet_lower
            )
            if not role_holder_match:
                # Try reverse: "X is the Y of Z"
                role_holder_match = re.search(
                    rf'(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)\s+)?(?:current\s+)?(?:president|prime minister|capital|king|queen)\s+of\s+\w+',
                    snippet_lower
                )
            if role_holder_match:
                actual_holder = role_holder_match.group(1).lower()
                # Check if the holder is different from what the claim states
                if actual_holder != claimed_holder:
                    # Different person holds the same role - this is a contradiction!
                    # But only if the claim's holder is not mentioned as a former holder
                    is_former = re.search(
                        rf'{re.escape(claimed_holder)}[^.]*?(?:former|used to be|previously|past)',
                        snippet_lower
                    )
                    is_former_reverse = re.search(
                        rf'(?:former|used to be|previously|past)[^.]*?{re.escape(claimed_holder)}',
                        snippet_lower
                    )
                    if not is_former and not is_former_reverse:
                        return "contradicts"

    # Also check: snippet mentions a different proper noun for the same subject
    # Only check "is_relationship" type
    for rel in claim_entities.get("relationships", []):
        # Skip "of_relationship" if we already have an "is_relationship"
        if rel.get("type") == "of_relationship":
            has_is_rel = any(
                r.get("type") == "is_relationship" and
                any(w in r.get("full", "") for w in rel.get("subject", "").split() if len(w) > 3)
                for r in claim_entities.get("relationships", [])
            )
            if has_is_rel:
                continue

        subj_words = set(rel["subject"].split())
        claimed_obj = rel["object"]

        # Find all "X is Y" patterns in snippet
        for match in re.finditer(r'(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|$)', snippet_lower):
            snip_subj = match.group(1)
            snip_obj = match.group(2).strip()

            # Check if subject matches (same role)
            subj_match = any(w in snip_subj for w in subj_words if len(w) > 3)

            if subj_match:
                # Check if object is a different proper noun
                snip_proper_nouns = set()
                for word in snip_obj.split():
                    clean = re.sub(r'[^\w]', '', word)
                    if clean and clean[0].isupper():
                        snip_proper_nouns.add(clean.lower())

                # If snippet mentions different proper nouns for the same role
                if snip_proper_nouns and not any(p in claimed_obj.lower() for p in snip_proper_nouns):
                    if any(p in proper_nouns for p in snip_proper_nouns):
                        return "contradicts"

    return "no_contradiction"


def _analyze_claim_verdict(claim_text: str, fact_results: list[dict],
                           web_results: list[dict]) -> dict:
    """Determine if evidence supports or contradicts the claim."""
    verdict_signal = "no_evidence"
    supporting = 0
    contradicting = 0
    sources = []

    claim_entities = _extract_key_entities(claim_text)

    # Check fact-check API results (high reliability)
    for fr in fact_results:
        rating = fr.get("verdict", "").lower()
        sources.append(fr.get("url", ""))
        if any(w in rating for w in ["false", "fake", "misleading", "pants on fire", "mostly false"]):
            contradicting += 3
        elif any(w in rating for w in ["true", "mostly true", "correct"]):
            supporting += 3
        elif any(w in rating for w in ["mixture", "half true", "partly"]):
            supporting += 1
            contradicting += 1

    # Check web results with entity-aware contradiction detection
    for wr in web_results:
        snippet = (wr.get("snippet", "") + " " + wr.get("title", "")).lower()
        url = wr.get("url", "")

        # Entity-aware contradiction check
        contradiction = _check_entity_contradiction(claim_entities, snippet)
        if contradiction == "contradicts":
            contradicting += 2
            sources.append(url)
            continue

        # Check for keyword overlap (lower weight than entity checks)
        claim_words = set(claim_text.lower().split())
        snippet_words = set(snippet.split())
        # Remove common stop words from comparison
        stop_words = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "on", "at", "to", "for", "and", "or", "but", "he", "she", "it", "they", "we"}
        claim_content = claim_words - stop_words
        snippet_content = snippet_words - stop_words
        overlap = len(claim_content & snippet_content)

        if overlap < 2:
            continue

        sources.append(url)

        # More sophisticated negation detection (use word boundaries to avoid false matches)
        strong_negations = [
            r'\bnot\b', r'\bnever\b', r'\bneither\b', r'\bnor\b',
            r'\bfalse\b', r'\bfake\b', r'\bdebunked\b', r'\bincorrect\b', r'\bwrong\b',
            r'\bmisleading\b', r'\bmyth\b', r'\bhoax\b', r'\blied\b',
            r'\bcorrection\b', r'\bcorrected\b', r'\bretracted\b',
        ]
        # Check for explicit contradiction phrases (must be near claim entities to count)
        contradiction_phrases = [
            "is not", "was not", "are not", "were not",
            "is actually", "was actually",
            "not the", "neither", "nor",
        ]

        has_negation = any(re.search(neg, snippet) for neg in strong_negations)
        has_contradiction_phrase = any(phrase in snippet for phrase in contradiction_phrases)

        # Only count contradiction phrases if they appear near claim entities
        if has_contradiction_phrase:
            # Check if the phrase is actually about the claim's subject/object
            near_claim_entity = False
            for rel in claim_entities.get("relationships", []):
                subj = rel["subject"]
                obj = rel["object"]
                # Check if subject or object words appear near the contradiction phrase
                for phrase in contradiction_phrases:
                    phrase_pos = snippet.find(phrase)
                    if phrase_pos >= 0:
                        context = snippet[max(0, phrase_pos-50):phrase_pos+len(phrase)+50]
                        if subj in context or obj in context or any(w in context for w in subj.split() if len(w) > 3):
                            near_claim_entity = True
                            break
                if near_claim_entity:
                    break
            if near_claim_entity:
                contradicting += 1
        elif has_negation:
            # Only count as contradiction if negation is related to claim entities
            for rel in claim_entities.get("relationships", []):
                if rel["subject"] in snippet and rel["object"] in snippet:
                    contradicting += 1
                    break
            else:
                pass  # Negation not related to claim - no signal
        else:
            # No negation found - only count as support if snippet EXACTLY confirms the relationship
            confirmed = False
            for rel in claim_entities.get("relationships", []):
                # Must match the FULL relationship: "X is Y" where X and Y match
                if re.search(rf'{re.escape(rel["subject"])}\s+(?:is|was|are|were)\s+{re.escape(rel["object"])}', snippet):
                    supporting += 1
                    confirmed = True
                    break
            # Do NOT give credit for mere topic mentions - they provide zero signal

    if contradicting > supporting:
        verdict_signal = "contradicted"
    elif supporting > contradicting and supporting >= 1:
        verdict_signal = "supported"
    elif supporting > 0 and contradicting > 0:
        verdict_signal = "mixed"
    else:
        verdict_signal = "no_evidence"

    return {
        "verdict_signal": verdict_signal,
        "supporting": supporting,
        "contradicting": contradicting,
        "sources": sources[:10],
    }


def _generate_search_queries(claim_text: str) -> list[str]:
    """Generate multiple targeted search queries from a claim."""
    queries = []
    entities = _extract_key_entities(claim_text)

    # Original claim (for fact-check databases)
    queries.append(claim_text)

    # Only use "is_relationship" for search queries (full claims like "X is Y")
    # Skip "of_relationship" which creates bad queries like "minister is india"
    for rel in entities.get("relationships", []):
        if rel.get("type") == "is_relationship":
            # "X is Y" -> search "X is Y" to verify
            queries.append(f"{rel['subject']} is {rel['object']}")
            # Also search just the subject to find its actual properties
            queries.append(rel["subject"])

    # Search for proper nouns together
    proper_nouns = list(entities.get("proper_nouns", []))
    if len(proper_nouns) >= 2:
        queries.append(" ".join(proper_nouns[:3]))

    # If we have a specific claim like "X is Y of Z", search for "actual Y of Z"
    role_match = re.search(r'(?:president|prime minister|capital|largest|smallest)\s+of\s+(\w+)', claim_text.lower())
    if role_match:
        role_context = role_match.group(0)
        queries.append(f"actual {role_context}")
        queries.append(f"current {role_context}")

    return queries[:4]  # Limit to 4 queries to avoid rate limiting


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

            # Generate targeted search queries
            search_queries = _generate_search_queries(claim_text)

            # Collect results from multiple queries
            all_fact_results = []
            all_web_results = []
            seen_urls = set()

            for query in search_queries:
                # Search fact-check databases
                fact_results = _search_fact_check_api(query)
                for fr in fact_results:
                    url = fr.get("url", "")
                    if url and url not in seen_urls:
                        all_fact_results.append(fr)
                        seen_urls.add(url)

                # Also do a general web search
                web_results = _search_web(query)
                for wr in web_results:
                    url = wr.get("url", "")
                    if url and url not in seen_urls:
                        all_web_results.append(wr)
                        seen_urls.add(url)

            analysis = _analyze_claim_verdict(claim_text, all_fact_results, all_web_results)
            analysis["original_claim"] = claim_text
            analysis["search_queries"] = search_queries
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
