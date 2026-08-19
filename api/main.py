"""
Serving API for the Fake News Detection Multi-Agent System.

Now supports both the original single-model /predict endpoint and the new
multi-agent /analyze endpoint with full evidence trails.

Run with:  uvicorn api.main:app --reload --port 8000   (from the project root)

Endpoints
---------
GET  /health                      liveness check + agent status
POST /predict                     legacy: {title?, text} -> label, confidence
POST /analyze                     NEW: multi-agent analysis with evidence trail
GET  /stats/model-comparison       accuracy/F1 per benchmarked model
GET  /stats/confusion-matrix       confusion matrix for the production model
GET  /stats/category-breakdown     accuracy by article subject
GET  /stats/trend                  monthly fake vs. real volume in the training corpus
GET  /limitations                  the stated limitations note
GET  /review-queue                 human review queue status
POST /review/{id}/resolve          resolve a review queue entry
POST /feedback                     log a corrected verdict
GET  /knowledge-base               knowledge base stats
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import clean_text  # noqa: E402

# Import agent system
sys.path.insert(0, str(ROOT))
from agents.base import AgentResult, Label
from agents.ingestion import IngestionAgent
from agents.claim_extraction import ClaimExtractionAgent
from agents.ml_classifier import MLClassifierAgent
from agents.fact_check import FactCheckAgent
from agents.source_credibility import SourceCredibilityAgent
from agents.media_forensics import MediaForensicsAgent
from agents.bias_sentiment import BiasSentimentAgent
from agents.orchestrator import Orchestrator
from agents.knowledge_base import get_knowledge_base
from agents.review_queue import get_review_queue
from agents.feedback import get_feedback_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fakenews.api")

app = FastAPI(
    title="Fake News Detection API — Multi-Agent System",
    description="Multi-agent misinformation detection with claim extraction, "
                "fact-checking, source credibility, media forensics, and "
                "bias analysis. Returns verdicts with transparent evidence trails.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Legacy model (loaded lazily for /predict)
vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
model = joblib.load(MODELS_DIR / "best_model.joblib")
best_model_name = (MODELS_DIR / "best_model_name.txt").read_text().strip()

# Agent instances
ingestion_agent = IngestionAgent()
claim_extraction_agent = ClaimExtractionAgent()
ml_classifier_agent = MLClassifierAgent()
fact_check_agent = FactCheckAgent()
source_credibility_agent = SourceCredibilityAgent()
media_forensics_agent = MediaForensicsAgent()
bias_sentiment_agent = BiasSentimentAgent()
orchestrator = Orchestrator()


# ── Pydantic models ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    title: str = Field("", description="Article headline (optional)")
    text: str = Field(..., description="Article body text", min_length=1)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    model_used: str


class AnalyzeRequest(BaseModel):
    title: str = Field("", description="Article headline")
    text: str = Field("", description="Article body text")
    url: str = Field("", description="Article URL (will be scraped)")
    images: list[str] = Field(default_factory=list, description="Embedded image URLs")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    agents: list[str] = Field(
        default_factory=lambda: ["all"],
        description="Which agents to run: 'all', or subset of "
                    "['ml_classifier','fact_check','source_credibility','media_forensics','bias_sentiment']"
    )


class AnalyzeResponse(BaseModel):
    label: str
    confidence: float
    reasoning: str
    agent_results: list[dict]
    evidence_trail: list[dict]
    needs_human_review: bool
    review_reason: str
    elapsed_ms: float


class ReviewResolveRequest(BaseModel):
    human_verdict: str = Field(..., description="'real' or 'fake'")
    notes: str = Field("", description="Reviewer notes")


class FeedbackRequest(BaseModel):
    article_text: str = Field(..., description="Article text")
    original_verdict: str = Field(..., description="Original prediction")
    corrected_verdict: str = Field(..., description="Corrected verdict: 'real' or 'fake'")
    notes: str = Field("")


# ── Helpers ───────────────────────────────────────────────────────────

def _load_report(name: str):
    path = REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Report {name} not generated yet. Run src/train.py.")
    return json.loads(path.read_text())


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    kb = get_knowledge_base()
    queue = get_review_queue()
    return {
        "status": "ok",
        "model": best_model_name,
        "version": "2.0.0-multi-agent",
        "agents": [
            "ingestion", "claim_extraction", "ml_classifier",
            "fact_check", "source_credibility", "media_forensics",
            "bias_sentiment", "orchestrator"
        ],
        "knowledge_base": kb.get_stats(),
        "review_queue": queue.get_stats(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Legacy single-model endpoint (backward compatible)."""
    content = f"{req.title}. {req.text}" if req.title else req.text
    cleaned = clean_text(content, remove_dateline=True)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Input text produced no usable tokens after cleaning.")
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    fake_p, real_p = float(proba[0]), float(proba[1])
    label = "real" if real_p >= fake_p else "fake"
    confidence = max(fake_p, real_p)
    return PredictResponse(
        label=label, confidence=round(confidence, 4),
        fake_probability=round(fake_p, 4), real_probability=round(real_p, 4),
        model_used=best_model_name,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Multi-agent analysis with full evidence trail."""
    t0 = time.time()

    if not req.text and not req.url:
        raise HTTPException(status_code=400, detail="Either 'text' or 'url' must be provided.")

    # 1. Ingestion
    article = {
        "title": req.title,
        "text": req.text,
        "url": req.url,
        "images": req.images,
        "metadata": req.metadata,
    }
    ingestion_result = ingestion_agent(article)
    enriched = ingestion_result.raw_output.copy()
    enriched["claims"] = []  # populated by claim extraction

    # 2. Claim extraction
    claim_result = claim_extraction_agent(enriched)
    claims = claim_result.evidence  # list of claim dicts
    enriched["claims"] = claims

    # 3. Run specialist agents
    requested_agents = req.agents
    run_all = "all" in requested_agents

    agent_results: list[AgentResult] = [
        ingestion_result,
        claim_result,
    ]

    if run_all or "ml_classifier" in requested_agents:
        agent_results.append(ml_classifier_agent(enriched))
    if run_all or "fact_check" in requested_agents:
        agent_results.append(fact_check_agent(enriched))
    if run_all or "source_credibility" in requested_agents:
        agent_results.append(source_credibility_agent(enriched))
    if run_all or "media_forensics" in requested_agents:
        agent_results.append(media_forensics_agent(enriched))
    if run_all or "bias_sentiment" in requested_agents:
        agent_results.append(bias_sentiment_agent(enriched))

    # 4. Orchestrator — returns AgentResult (shares interface with Verdict:
    #    .label, .confidence, .reasoning, .evidence, .raw_output, .to_dict())
    orchestrator_input = enriched.copy()
    orchestrator_input["agent_results"] = agent_results
    verdict = orchestrator(orchestrator_input)

    # 5. Human review queue
    review_queue = get_review_queue()
    if verdict.raw_output.get("needs_human_review", False):
        queue_id = review_queue.add(
            article=enriched,
            verdict=verdict.to_dict(),
            review_reason=verdict.raw_output.get("review_reason", ""),
        )
        logger.info(f"Added to review queue: entry #{queue_id}")

    # 6. Log to knowledge base
    kb = get_knowledge_base()
    for claim_obj in claims:
        claim_text = claim_obj.get("claim", "")
        if claim_text:
            sources = []
            for r in agent_results:
                if r.agent_name == "fact_check":
                    sources = r.raw_output.get("sources", [])
            kb.add_entry(
                claim=claim_text,
                verdict=verdict.label.value,
                sources=sources[:5],
                article_text=enriched.get("full_text", "")[:500],
            )

    elapsed = (time.time() - t0) * 1000

    return AnalyzeResponse(
        label=verdict.label.value,
        confidence=verdict.confidence,
        reasoning=verdict.reasoning,
        agent_results=[r.to_dict() for r in agent_results],
        evidence_trail=verdict.evidence,
        needs_human_review=verdict.raw_output.get("needs_human_review", False),
        review_reason=verdict.raw_output.get("review_reason", ""),
        elapsed_ms=round(elapsed, 1),
    )


@app.get("/review-queue")
def review_queue_status():
    queue = get_review_queue()
    return queue.get_stats()


@app.get("/review-queue/pending")
def review_queue_pending():
    queue = get_review_queue()
    return queue.get_pending()


@app.post("/review/{entry_id}/resolve")
def resolve_review(entry_id: int, req: ReviewResolveRequest):
    queue = get_review_queue()
    for entry in queue.queue:
        if entry["id"] == entry_id:
            entry["status"] = "resolved"
            entry["human_verdict"] = req.human_verdict
            entry["human_notes"] = req.notes
            import time as _time
            entry["reviewed_at"] = _time.time()
            queue._save()

            # Also log to feedback loop
            feedback = get_feedback_loop()
            article_text = entry.get("article", {}).get("title", "")
            original_verdict = entry.get("verdict", {}).get("label", "")
            feedback.log_correction(
                article_text=article_text,
                original_verdict=original_verdict,
                corrected_verdict=req.human_verdict,
                notes=req.notes,
            )
            return {"status": "resolved", "entry_id": entry_id}

    raise HTTPException(status_code=404, detail=f"Review entry {entry_id} not found")


@app.post("/feedback")
def log_feedback(req: FeedbackRequest):
    feedback = get_feedback_loop()
    entry = feedback.log_correction(
        article_text=req.article_text,
        original_verdict=req.original_verdict,
        corrected_verdict=req.corrected_verdict,
        notes=req.notes,
    )
    return {"status": "logged", "id": entry["id"]}


@app.get("/knowledge-base/stats")
def knowledge_base_stats():
    kb = get_knowledge_base()
    return kb.get_stats()


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
