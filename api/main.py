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
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

# Import auth system
from api.auth import get_current_user
from api import user_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fakenews.api")

# Pre-warm knowledge base (lazy model loading happens in background)
logger.info("Pre-loading knowledge base...")
try:
    kb = get_knowledge_base()
    kb._ensure_model()
    logger.info("Knowledge base ready.")
except Exception as e:
    logger.warning(f"Knowledge base pre-load failed (non-fatal): {e}")

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

# ── Auth router ──
from api.auth import router as auth_router
app.include_router(auth_router, prefix="/auth", tags=["auth"])

# ── Static file serving for dashboard ──
from fastapi.staticfiles import StaticFiles
DASHBOARD_DIR = ROOT / "dashboard"
app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR)), name="dashboard")


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Public login page — no auth required."""
    login_path = DASHBOARD_DIR / "login.html"
    if not login_path.exists():
        raise HTTPException(status_code=503, detail="Login page not found.")
    return HTMLResponse(content=login_path.read_text(encoding="utf-8"))

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

DASHBOARD_PATH = ROOT / "dashboard" / "index.html"


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the analyst dashboard. Client-side JS handles auth redirect."""
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=503, detail="Dashboard not built. Run src/build_dashboard.py.")
    return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_alt():
    """Alias for / — serves the analyst dashboard."""
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=503, detail="Dashboard not built. Run src/build_dashboard.py.")
    return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health(user: dict = Depends(get_current_user)):
    kb = get_knowledge_base()
    user_queue = user_db.get_review_queue_stats(user["id"])
    return {
        "status": "ok",
        "model": best_model_name,
        "version": "2.0.0-multi-agent",
        "user": user["username"],
        "agents": [
            "ingestion", "claim_extraction", "ml_classifier",
            "fact_check", "source_credibility", "media_forensics",
            "bias_sentiment", "orchestrator"
        ],
        "knowledge_base": kb.get_stats(),
        "review_queue": user_queue,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, user: dict = Depends(get_current_user)):
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
def analyze(req: AnalyzeRequest, user: dict = Depends(get_current_user)):
    """Multi-agent analysis with full evidence trail."""
    t0 = time.time()
    user_id = user["id"]

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
    # Only run source credibility if a URL or domain is available
    has_source = bool(enriched.get("url") or enriched.get("source_domain")
                      or enriched.get("metadata", {}).get("domain"))
    if (run_all or "source_credibility" in requested_agents) and has_source:
        agent_results.append(source_credibility_agent(enriched))
    # Skip media forensics for text-only input (no images to analyze)
    has_images = bool(enriched.get("images"))
    if (run_all or "media_forensics" in requested_agents) and has_images:
        agent_results.append(media_forensics_agent(enriched))
    if run_all or "bias_sentiment" in requested_agents:
        agent_results.append(bias_sentiment_agent(enriched))

    # 4. Orchestrator — returns AgentResult (shares interface with Verdict:
    #    .label, .confidence, .reasoning, .evidence, .raw_output, .to_dict())
    orchestrator_input = enriched.copy()
    orchestrator_input["agent_results"] = agent_results
    verdict = orchestrator(orchestrator_input)

    # 5. Human review queue (per-user)
    if verdict.raw_output.get("needs_human_review", False):
        queue_id = user_db.add_to_review_queue(
            user_id=user_id,
            article=enriched,
            verdict=verdict.to_dict(),
            reason=verdict.raw_output.get("review_reason", ""),
        )
        logger.info(f"Added to review queue for user {user_id}: entry #{queue_id}")

    # 6. Log to knowledge base (per-user)
    for claim_obj in claims:
        claim_text = claim_obj.get("claim", "")
        if claim_text:
            sources = []
            for r in agent_results:
                if r.agent_name == "fact_check":
                    sources = r.raw_output.get("sources", [])
            user_db.add_kb_entry(
                user_id=user_id,
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
def review_queue_status(user: dict = Depends(get_current_user)):
    return user_db.get_review_queue_stats(user["id"])


@app.get("/review-queue/pending")
def review_queue_pending(user: dict = Depends(get_current_user)):
    return user_db.get_pending_reviews(user["id"])


@app.post("/review/{entry_id}/resolve")
def resolve_review(entry_id: int, req: ReviewResolveRequest, user: dict = Depends(get_current_user)):
    user_id = user["id"]
    success = user_db.resolve_review(user_id, entry_id, req.human_verdict, req.notes)
    if not success:
        raise HTTPException(status_code=404, detail=f"Review entry {entry_id} not found")

    # Also log to feedback loop (per-user)
    queue = user_db.get_review_queue(user_id)
    for entry in queue:
        if entry["id"] == entry_id:
            user_db.log_feedback(
                user_id=user_id,
                article_text=entry.get("article", {}).get("title", ""),
                original_verdict=entry.get("verdict", {}).get("label", ""),
                corrected_verdict=req.human_verdict,
                notes=req.notes,
            )
            break
    return {"status": "resolved", "entry_id": entry_id}


@app.post("/feedback")
def log_feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    entry = user_db.log_feedback(
        user_id=user["id"],
        article_text=req.article_text,
        original_verdict=req.original_verdict,
        corrected_verdict=req.corrected_verdict,
        notes=req.notes,
    )
    return {"status": "logged", "id": entry["id"]}


@app.get("/knowledge-base/stats")
def knowledge_base_stats(user: dict = Depends(get_current_user)):
    return user_db.get_kb_stats(user["id"])


@app.get("/stats/model-comparison")
def model_comparison(user: dict = Depends(get_current_user)):
    return _load_report("model_comparison.json")


@app.get("/stats/confusion-matrix")
def confusion_matrix(user: dict = Depends(get_current_user)):
    data = _load_report("confusion_matrices.json")
    return data.get(best_model_name, data)


@app.get("/stats/category-breakdown")
def category_breakdown(user: dict = Depends(get_current_user)):
    return _load_report("category_breakdown.json")


@app.get("/stats/trend")
def trend(user: dict = Depends(get_current_user)):
    return _load_report("trend_data.json")


@app.get("/limitations")
def limitations(user: dict = Depends(get_current_user)):
    path = ROOT / "docs" / "limitations.md"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Limitations note not generated yet.")
    return {"limitations_markdown": path.read_text()}
