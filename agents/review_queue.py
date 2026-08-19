"""
Human Review Queue — Routes low-confidence/borderline cases to people.

Straight out of the limitations.md recommendation: treat predictions as
"a triage signal for human fact-checkers."
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
QUEUE_FILE = ROOT / "data" / "review_queue.json"


class ReviewQueue:
    """Manages articles flagged for human review."""

    def __init__(self):
        self.queue: list[dict] = []
        self._load()

    def _load(self):
        if QUEUE_FILE.exists():
            try:
                self.queue = json.loads(QUEUE_FILE.read_text())
            except Exception:
                self.queue = []

    def _save(self):
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_FILE.write_text(json.dumps(self.queue, indent=2, default=str))

    def add(self, article: dict, verdict: dict, review_reason: str) -> int:
        """Add an article to the review queue. Returns the queue entry ID."""
        entry_id = len(self.queue) + 1
        entry = {
            "id": entry_id,
            "article": {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source_domain": article.get("source_domain", ""),
            },
            "verdict": verdict,
            "review_reason": review_reason,
            "status": "pending",  # pending, in_review, resolved
            "human_verdict": None,
            "human_notes": "",
            "added_at": time.time(),
            "reviewed_at": None,
        }
        self.queue.append(entry)
        self._save()
        return entry_id

    def get_pending(self) -> list[dict]:
        """Return all pending review items."""
        return [e for e in self.queue if e["status"] == "pending"]

    def get_stats(self) -> dict:
        """Queue statistics."""
        statuses = {}
        for e in self.queue:
            s = e["status"]
            statuses[s] = statuses.get(s, 0) + 1
        return {
            "total": len(self.queue),
            "by_status": statuses,
        }


# Singleton
_queue = None


def get_review_queue() -> ReviewQueue:
    global _queue
    if _queue is None:
        _queue = ReviewQueue()
    return _queue
