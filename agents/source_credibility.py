"""
Source Credibility Agent — Domain reputation, registration age, known misinformation databases.

Catches cases where the writing is polished but the source is a known bad actor.
Checks domain age, known misinformation source lists, and publisher reputation.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse
from datetime import datetime

from agents.base import BaseAgent, AgentResult, Label

# WHOIS lookup (optional)
try:
    import whois
    HAS_WHOIS = True
except ImportError:
    HAS_WHOIS = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Known fabrication/misinformation domains — sites that have repeatedly
# published demonstrably false claims or are widely flagged by fact-checkers.
# This is NOT a political bias list: it targets outlets with documented patterns
# of fabricating or debunks content, not partisan-but-real publishers.
# In production, this would be a regularly-updated database (MBFC, NewsGuard,
# or a custom fact-check aggregation). The inclusion of any domain here is an
# editorial judgment based on fact-checking track records, not political lean.
KNOWN_LOW_CREDIBILITY = {
    # Outlets with documented patterns of fabricated or false claims
    "infowars.com", "naturalnews.com", "beforeitsnews.com", "yournewswire.com",
    "globalresearch.ca", "theantimedia.com", "collective-evolution.com",
    "southfrontpress.com", "mintpressnews.com", "alternatecurrentpolitics.com",
    "worldnewsdailyreport.com", "newspunch.com", "usanews.com",
    # Satire sites sometimes confused for real news
    "ifunny.co",
    # Sites with significant misinformation track records per MBFC/fact-checkers
    "thegatewaypundit.com", "freedomoutpost.com",
}

KNOWN_HIGH_CREDIBILITY = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "npr.org", "pbs.org",
    "wsj.com", "economist.com", "nature.com", "science.org",
    "who.int", "cdc.gov", "nih.gov", "un.org",
    "ft.com", "bloomberg.com", "cnbc.com", "nbcnews.com",
    "cnn.com", "abcnews.go.com", "cbsnews.com", "foxnews.com",
    "theatlantic.com", "newyorker.com", "propublica.org",
}


def _check_domain_age(domain: str) -> dict:
    """Look up domain registration age via WHOIS."""
    if not HAS_WHOIS or not domain:
        return {"age_days": None, "registrar": "", "error": "whois not available"}

    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            age_days = (datetime.now() - creation).days
            return {
                "age_days": age_days,
                "registrar": w.registrar or "",
                "org": w.org or "",
                "expiration": str(w.expiration_date) if w.expiration_date else "",
            }
    except Exception:
        pass

    return {"age_days": None, "registrar": "", "error": "lookup failed"}


def _check_urlhaus(domain: str) -> dict:
    """Check if domain is flagged in URLhaus (abuse.ch) malware DB."""
    if not HAS_HTTPX or not domain:
        return {"flagged": False, "urlhaus_info": ""}

    try:
        resp = httpx.post(
            "https://urlhaus-api.abuse.ch/v1/host/",
            data={"host": domain},
            timeout=5,
        )
        data = resp.json()
        if data.get("query_status") in ("no_results", "invalid_host"):
            return {"flagged": False, "urlhaus_info": "clean"}
        if "urls_online" in data and data["urls_online"] > 0:
            return {"flagged": True, "urlhaus_info": f"{data['urls_online']} online malware URLs"}
        return {"flagged": False, "urlhaus_info": "flagged but no active URLs"}
    except Exception:
        return {"flagged": False, "urlhaus_info": "check failed"}


class SourceCredibilityAgent(BaseAgent):
    name = "source_credibility"

    def run(self, article: dict[str, Any]) -> AgentResult:
        url = article.get("url", "")
        metadata = article.get("metadata", {})
        domain = metadata.get("domain", "") or (
            urlparse(url).netloc.lower() if url else ""
        )

        evidence = []
        score = 0.5  # start neutral

        # 1. Known credibility lists
        domain_lower = domain.lower()
        if domain_lower.startswith("www."):
            domain_lower = domain_lower[4:]  # proper prefix strip, not lstrip
        if domain_lower in KNOWN_LOW_CREDIBILITY:
            score -= 0.3
            evidence.append({"check": "known_list", "result": "LOW credibility domain",
                             "domain": domain_lower})
        elif domain_lower in KNOWN_HIGH_CREDIBILITY:
            score += 0.25
            evidence.append({"check": "known_list", "result": "HIGH credibility domain",
                             "domain": domain_lower})
        else:
            evidence.append({"check": "known_list", "result": "Not in known lists",
                             "domain": domain_lower})

        # 2. Domain age
        domain_info = _check_domain_age(domain_lower)
        age_days = domain_info.get("age_days")
        if age_days is not None:
            if age_days < 90:
                score -= 0.2
                evidence.append({"check": "domain_age", "result": f"Very new domain: {age_days} days"})
            elif age_days < 365:
                score -= 0.1
                evidence.append({"check": "domain_age", "result": f"Relatively new: {age_days} days"})
            elif age_days > 3650:
                score += 0.1
                evidence.append({"check": "domain_age", "result": f"Well-established: {age_days} days"})
            else:
                evidence.append({"check": "domain_age", "result": f"{age_days} days old"})
        else:
            evidence.append({"check": "domain_age", "result": "Unknown"})

        # 3. URLhaus check
        urlhaus = _check_urlhaus(domain_lower)
        if urlhaus.get("flagged"):
            score -= 0.3
            evidence.append({"check": "urlhaus", "result": "FLAGGED as malware host",
                             "info": urlhaus["urlhaus_info"]})
        else:
            evidence.append({"check": "urlhaus", "result": "Clean"})

        # 4. TLD heuristics
        tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
        suspicious_tlds = {"xyz", "top", "club", "info", "buzz", "gq", "ml", "cf", "tk"}
        if tld in suspicious_tlds:
            score -= 0.05
            evidence.append({"check": "tld", "result": f"Suspicious TLD: .{tld}"})

        # Clamp score
        score = max(0.0, min(1.0, score))

        if score < 0.35:
            label = Label.FAKE
        elif score > 0.65:
            label = Label.REAL
        else:
            label = Label.UNCERTAIN

        return AgentResult(
            agent_name=self.name,
            label=label,
            confidence=abs(score - 0.5) * 2,  # distance from uncertain
            reasoning=f"Source credibility score: {score:.2f} for {domain}. "
                      f"{len(evidence)} checks performed.",
            evidence=evidence,
            raw_output={"domain": domain, "score": score, "domain_info": domain_info},
        )
