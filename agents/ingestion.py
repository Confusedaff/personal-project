"""
Ingestion Agent — Article intake and metadata extraction.

Scrapes the article URL (if provided), extracts clean text, images,
and metadata (domain, author, publish date, etc.). Falls back to
user-provided title+text when no URL is given.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from agents.base import BaseAgent, AgentResult, Label

# Lightweight scraping with fallbacks
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Domain extraction from text
_DOMAIN_RE = re.compile(
    r'\b(?:https?://)?'
    r'((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,}))',
    re.IGNORECASE,
)

KNOWN_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "npr.org", "pbs.org",
    "wsj.com", "economist.com", "cnn.com", "foxnews.com", "abcnews.go.com",
    "cbsnews.com", "nbcnews.com", "usatoday.com", "latimes.com",
    "politico.com", "theatlantic.com", "newyorker.com", "propublica.org",
    "bloomberg.com", "cnbc.com", "ft.com", "marketwatch.com",
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "thegatewaypundit.com", "breitbart.com",
}


def _extract_domains_from_text(text: str) -> list[str]:
    """Find news domain mentions in article text."""
    if not text:
        return []
    found = []
    for match in _DOMAIN_RE.finditer(text):
        domain = match.group(1).lower()
        # Filter out common non-news domains
        if any(skip in domain for skip in ["example.com", "localhost", ".local"]):
            continue
        found.append(domain)
    # Also check for known domains by name (e.g. "reuters" -> "reuters.com")
    text_lower = text.lower()
    for known in KNOWN_NEWS_DOMAINS:
        name_part = known.split(".")[0]
        if name_part in text_lower and known not in found:
            found.append(known)
    return list(dict.fromkeys(found))  # dedupe preserving order


def _extract_images(soup) -> list[str]:
    """Pull all article-relevant image URLs from parsed HTML."""
    images = []
    if soup is None:
        return images
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        if src.startswith("data:"):
            continue
        # skip tiny icons / tracking pixels
        width = img.get("width", "")
        height = img.get("height", "")
        try:
            if int(width) < 50 or int(height) < 50:
                continue
        except (ValueError, TypeError):
            pass
        images.append(src)
    return images[:20]  # cap at 20


def _extract_metadata(soup, url: str) -> dict[str, Any]:
    """Pull OpenGraph / Twitter / standard meta tags + domain info."""
    meta = {}
    parsed = urlparse(url) if url else None
    meta["domain"] = parsed.netloc.lower() if parsed else ""
    meta["tld"] = parsed.netloc.split(".")[-1].lower() if parsed else ""

    if soup is None:
        return meta

    def _get(prop: str, attr: str = "property") -> str:
        tag = soup.find("meta", attrs={attr: prop})
        return tag["content"].strip() if tag and tag.get("content") else ""

    meta["og_title"] = _get("og:title")
    meta["og_description"] = _get("og:description")
    meta["og_image"] = _get("og:image")
    meta["author"] = (
        _get("author", "name")
        or _get("article:author")
        or _get("twitter:creator")
        or ""
    )
    meta["published_date"] = (
        _get("article:published_time")
        or _get("datePublished", "itemprop")
        or ""
    )
    meta["section"] = _get("article:section") or ""
    return meta


class IngestionAgent(BaseAgent):
    name = "ingestion"

    def run(self, article: dict[str, Any]) -> AgentResult:
        title = article.get("title", "")
        text = article.get("text", "")
        url = article.get("url", "")
        images = list(article.get("images", []))
        metadata = dict(article.get("metadata", {}))

        soup = None
        if url and HAS_HTTPX and HAS_BS4:
            try:
                resp = httpx.get(url, timeout=10, follow_redirects=True,
                                 headers={"User-Agent": "FakeNewsDetector/1.0"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # try to get better title
                if not title:
                    title_tag = soup.find("title")
                    title = title_tag.get_text(strip=True) if title_tag else ""

                # try to get article body text from common containers
                if not text:
                    body = (
                        soup.find("article")
                        or soup.find("div", class_=re.compile(r"article-body|post-content|entry-content|story-body", re.I))
                        or soup.find("main")
                    )
                    if body:
                        paragraphs = body.find_all("p")
                        text = "\n".join(p.get_text(strip=True) for p in paragraphs)

                if not images:
                    images = _extract_images(soup)

                metadata.update(_extract_metadata(soup, url))
            except Exception:
                pass  # graceful fallback

        # Combine title + text if not already combined
        full_text = f"{title}. {text}" if title and text else (text or title)

        # Extract domains mentioned in the text
        text_domains = _extract_domains_from_text(full_text)
        if not metadata.get("domain") and text_domains:
            metadata["domain"] = text_domains[0]
            metadata["mentioned_domains"] = text_domains

        # Store enriched article for downstream agents
        enriched = {
            "title": title,
            "text": text,
            "full_text": full_text,
            "url": url,
            "images": images,
            "metadata": metadata,
            "source_domain": metadata.get("domain", ""),
            "author": metadata.get("author", ""),
            "published_date": metadata.get("published_date", ""),
            "mentioned_domains": text_domains,
        }

        return AgentResult(
            agent_name=self.name,
            label=Label.UNCERTAIN,  # ingestion doesn't judge
            confidence=0.0,
            reasoning=f"Ingested article: {len(full_text)} chars, {len(images)} images, "
                      f"source={metadata.get('domain', 'unknown')}",
            evidence=[enriched],
            raw_output=enriched,
        )
