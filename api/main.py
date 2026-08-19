"""
Serving API for the Fake News Detection MVP (Objective 4, item 4: "Deploy
the best-performing model behind an API that returns a prediction with a
confidence score.")

Run with:  uvicorn api.main:app --reload --port 8000   (from the project root)

Endpoints
---------
GET  /health                      liveness check
POST /predict                     {title?, text} -> label, confidence, category
GET  /stats/model-comparison       accuracy/F1 per benchmarked model
GET  /stats/confusion-matrix       confusion matrix for the production model
GET  /stats/category-breakdown     accuracy by article subject
GET  /stats/trend                  monthly fake vs. real volume in the training corpus
GET  /limitations                  the stated limitations note
"""
import json
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import clean_text  # noqa: E402

app = FastAPI(
    title="Fake News Detection API",
    description="Use Case #15 — GenAI & Security. Serves TF-IDF + classical "
                "classifier predictions with confidence scores, plus the "
                "analytics that back the dashboard.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
model = joblib.load(MODELS_DIR / "best_model.joblib")
best_model_name = (MODELS_DIR / "best_model_name.txt").read_text().strip()


class PredictRequest(BaseModel):
    title: str = Field("", description="Article headline (optional)")
    text: str = Field(..., description="Article body text", min_length=1)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    model_used: str


def _load_report(name: str):
    path = REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Report {name} not generated yet. Run src/train.py.")
    return json.loads(path.read_text())


@app.get("/health")
def health():
    return {"status": "ok", "model": best_model_name}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    content = f"{req.title}. {req.text}" if req.title else req.text
    cleaned = clean_text(content, remove_dateline=True)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Input text produced no usable tokens after cleaning.")
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]  # [P(fake), P(real)]
    fake_p, real_p = float(proba[0]), float(proba[1])
    label = "real" if real_p >= fake_p else "fake"
    confidence = max(fake_p, real_p)
    return PredictResponse(
        label=label, confidence=round(confidence, 4),
        fake_probability=round(fake_p, 4), real_probability=round(real_p, 4),
        model_used=best_model_name,
    )


@app.get("/stats/model-comparison")
def model_comparison():
    return _load_report("model_comparison.json")


@app.get("/stats/confusion-matrix")
def confusion_matrix():
    data = _load_report("confusion_matrices.json")
    return data.get(best_model_name, data)


@app.get("/stats/category-breakdown")
def category_breakdown():
    return _load_report("category_breakdown.json")


@app.get("/stats/trend")
def trend():
    return _load_report("trend_data.json")


@app.get("/limitations")
def limitations():
    path = ROOT / "docs" / "limitations.md"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Limitations note not generated yet.")
    return {"limitations_markdown": path.read_text()}
