"""
ML Classifier Agent — Wraps the existing TF-IDF + Linear SVM model.

This is the original MVP model, now operating as one signal among several
in the multi-agent system. It provides a baseline style-based classification
that the orchestrator weighs alongside fact-check, source credibility, etc.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agents.base import BaseAgent, AgentResult, Label

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Lazy-load the model on first use
_model = None
_vectorizer = None
_model_name = None


def _load_model():
    global _model, _vectorizer, _model_name
    if _model is not None:
        return

    import joblib
    models_dir = ROOT / "models"
    _vectorizer = joblib.load(models_dir / "vectorizer.joblib")
    _model = joblib.load(models_dir / "best_model.joblib")
    _model_name = (models_dir / "best_model_name.txt").read_text().strip()


class MLClassifierAgent(BaseAgent):
    name = "ml_classifier"

    def run(self, article: dict[str, Any]) -> AgentResult:
        _load_model()

        from preprocessing import clean_text

        title = article.get("title", "")
        text = article.get("text", "")
        content = f"{title}. {text}" if title else text
        cleaned = clean_text(content, remove_dateline=True)

        if not cleaned.strip():
            return AgentResult(
                agent_name=self.name,
                label=Label.UNCERTAIN,
                confidence=0.0,
                reasoning="No usable tokens after cleaning",
            )

        X = _vectorizer.transform([cleaned])
        proba = _model.predict_proba(X)[0]
        fake_p, real_p = float(proba[0]), float(proba[1])
        label = Label.REAL if real_p >= fake_p else Label.FAKE
        confidence = max(fake_p, real_p)

        return AgentResult(
            agent_name=self.name,
            label=label,
            confidence=confidence,
            reasoning=f"TF-IDF + {_model_name}: P(fake)={fake_p:.3f}, P(real)={real_p:.3f}. "
                      f"Classifies based on writing style and vocabulary patterns.",
            evidence=[{
                "model": _model_name,
                "fake_probability": round(fake_p, 4),
                "real_probability": round(real_p, 4),
            }],
            raw_output={"model": _model_name, "proba": [fake_p, real_p]},
        )
