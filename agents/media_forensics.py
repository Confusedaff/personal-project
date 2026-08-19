"""
Media Forensics Agent — Reverse image search and EXIF checks.

Catches reused/miscaptioned photos, a very common fake-news technique
that a text-only pipeline cannot see at all.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from agents.base import BaseAgent, AgentResult, Label

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Image analysis (optional, for EXIF and perceptual hashing)
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

import io
import struct


def _download_image(url: str) -> bytes | None:
    """Download image bytes from URL."""
    if not HAS_HTTPX:
        return None
    try:
        resp = httpx.get(url, timeout=10,
                         headers={"User-Agent": "FakeNewsDetector/1.0"},
                         follow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) > 10 * 1024 * 1024:  # skip >10MB
            return None
        return resp.content
    except Exception:
        return None


def _check_exif(data: bytes) -> dict:
    """Extract EXIF data for metadata consistency checks."""
    if not HAS_PIL:
        return {"available": False}
    try:
        img = Image.open(io.BytesIO(data))
        exif_data = img._getexif()
        if not exif_data:
            return {"available": True, "has_exif": False}

        result = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if isinstance(value, (str, int, float)):
                result[str(tag)] = value
            elif isinstance(value, bytes):
                result[str(tag)] = f"<{len(value)} bytes>"
            elif isinstance(value, tuple) and len(value) == 2:
                result[str(tag)] = f"{value[0]}/{value[1]}"

        # Key checks
        camera = result.get("Make", "") + " " + result.get("Model", "")
        software = result.get("Software", "")
        gps = "GPSLatitude" in result or "GPSLatitude" in str(result)
        date_original = result.get("DateTimeOriginal", "")

        return {
            "available": True,
            "has_exif": True,
            "camera": camera.strip(),
            "software": software,
            "has_gps": gps,
            "date_original": date_original,
            "tags": list(result.keys())[:20],
        }
    except Exception:
        return {"available": True, "has_exif": False, "error": "parse failed"}


def _compute_perceptual_hash(data: bytes) -> str:
    """Compute a perceptual hash for duplicate detection."""
    if not HAS_PIL or not HAS_IMAGEHASH:
        return ""
    try:
        img = Image.open(io.BytesIO(data))
        phash = imagehash.phash(img)
        return str(phash)
    except Exception:
        return ""


def _reverse_image_search(url: str) -> list[dict]:
    """
    Attempt reverse image search via TinEye or Google Images.
    In production, use an API key for TinEye or Google Cloud Vision.
    """
    if not HAS_HTTPX:
        return []

    # Placeholder: in production, implement actual API calls
    # For now, return empty — this is the integration point
    return []


class MediaForensicsAgent(BaseAgent):
    name = "media_forensics"

    def run(self, article: dict[str, Any]) -> AgentResult:
        images = article.get("images", [])

        if not images:
            return AgentResult(
                agent_name=self.name,
                label=Label.UNCERTAIN,
                confidence=0.1,
                reasoning="No images found in article — media forensics not applicable",
                evidence=[],
            )

        image_analyses = []
        suspicious_count = 0

        for img_url in images[:10]:  # limit to 10 images
            data = _download_image(img_url)
            if not data:
                image_analyses.append({
                    "url": img_url,
                    "status": "download_failed",
                    "suspicious": False,
                })
                continue

            analysis = {"url": img_url, "status": "analyzed", "suspicious": False}

            # EXIF check
            exif = _check_exif(data)
            analysis["exif"] = exif

            # Check for signs of manipulation
            # - GPS data in news photo (should be stripped or consistent)
            # - Software suggesting editing
            if exif.get("has_gps"):
                analysis["suspicious"] = True
                analysis["reason"] = "GPS data present in image"
                suspicious_count += 1

            software = exif.get("software", "").lower()
            editing_tools = ["photoshop", "gimp", "lightroom", "snapseed", "afterlight"]
            if any(tool in software for tool in editing_tools):
                analysis["suspicious"] = True
                analysis["reason"] = f"Editing software detected: {exif.get('software')}"
                suspicious_count += 1

            # Perceptual hash for duplicate detection
            phash = _compute_perceptual_hash(data)
            if phash:
                analysis["phash"] = phash

            image_analyses.append(analysis)

        # Reverse image search
        reverse_results = _reverse_image_search(images[0] if images else "")

        total_images = len(image_analyses)
        if total_images == 0:
            confidence = 0.1
            label = Label.UNCERTAIN
        elif suspicious_count > 0:
            confidence = min(0.4 + suspicious_count * 0.15, 0.85)
            label = Label.FAKE
        else:
            confidence = 0.3
            label = Label.REAL

        return AgentResult(
            agent_name=self.name,
            label=label,
            confidence=confidence,
            reasoning=(
                f"Analyzed {total_images} images: {suspicious_count} flagged as suspicious. "
                f"EXIF available for {sum(1 for a in image_analyses if a.get('exif', {}).get('has_exif'))} images."
            ),
            evidence=image_analyses,
            raw_output={
                "images_checked": total_images,
                "suspicious_count": suspicious_count,
                "reverse_search_results": reverse_results,
            },
        )
