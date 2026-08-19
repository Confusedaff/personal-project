"""
Feedback Loop — Logs corrected verdicts back into retraining data.

Addresses the "style mimicry will degrade performance over time"
limitation flagged in limitations.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_FILE = ROOT / "data" / "feedback_log.json"


class FeedbackLoop:
    """Manages human feedback on predictions for retraining."""

    def __init__(self):
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        if FEEDBACK_FILE.exists():
            try:
                self.entries = json.loads(FEEDBACK_FILE.read_text())
            except Exception:
                self.entries = []

    def _save(self):
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        FEEDBACK_FILE.write_text(json.dumps(self.entries, indent=2, default=str))

    def log_correction(self, article_text: str, original_verdict: str,
                       corrected_verdict: str, corrector: str = "human",
                       notes: str = "") -> dict:
        """Log a corrected verdict for retraining data."""
        entry = {
            "id": len(self.entries) + 1,
            "article_text": article_text[:2000],  # truncate for storage
            "original_verdict": original_verdict,
            "corrected_verdict": corrected_verdict,
            "corrector": corrector,
            "notes": notes,
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        self._save()
        return entry

    def get_training_data(self) -> list[dict]:
        """Export corrected entries as training data."""
        return [
            {
                "text": e["article_text"],
                "label": 1 if e["corrected_verdict"] == "real" else 0,
            }
            for e in self.entries
            if e["corrected_verdict"] in ("real", "fake")
        ]

    def get_stats(self) -> dict:
        """Feedback statistics."""
        corrections = 0
        for e in self.entries:
            if e["original_verdict"] != e["corrected_verdict"]:
                corrections += 1
        return {
            "total_entries": len(self.entries),
            "corrections": corrections,
            "accuracy_rate": 1 - (corrections / max(len(self.entries), 1)),
        }


# Singleton
_feedback = None


def get_feedback_loop() -> FeedbackLoop:
    global _feedback
    if _feedback is None:
        _feedback = FeedbackLoop()
    return _feedback
