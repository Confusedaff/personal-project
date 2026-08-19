# AI-Powered Fake News Detection — Multi-Agent System

Use Case #15 (GenAI & Security portfolio). A layered multi-agent system that
goes beyond style-based classification to perform actual factual verification.

## Architecture

```
Article / URL input
        ↓
  Ingestion Agent ── scrape, clean, extract metadata
        ↓
  ┌─────────────────────────────────────────────┐
  │  Multi-Agent Analysis Layer                 │
  │  (orchestrator dispatches to specialists)   │
  │                                             │
  │  ┌─────────────────┐  ┌──────────────────┐ │
  │  │ Claim Extraction │  │ ML Classifier    │ │
  │  │ (LLM-based)     │  │ (TF-IDF/SVM)    │ │
  │  └────────┬────────┘  └────────┬─────────┘ │
  │           ↓                    ↓           │
  │  ┌─────────────────┐  ┌──────────────────┐ │
  │  │ Fact-Check      │  │ Source           │ │
  │  │ Retrieval       │  │ Credibility      │ │
  │  └────────┬────────┘  └────────┬─────────┘ │
  │           ↓                    ↓           │
  │  ┌─────────────────┐  ┌──────────────────┐ │
  │  │ Media           │  │ Bias /           │ │
  │  │ Forensics       │  │ Sentiment        │ │
  │  └────────┬────────┘  └────────┬─────────┘ │
  │           └────────┬───────────┘           │
  │                    ↓                       │
  │         Orchestrator / Aggregator          │
  │     (weighs signals into one verdict)      │
  └─────────────────────────────────────────────┘
        ↓
  ┌─────────────────────────────────────────────┐
  │  Knowledge Base / Vector DB (RAG)           │
  │  Live Web Search APIs                       │
  └─────────────────────────────────────────────┘
        ↓
  Verdict + Confidence + Evidence Trail
        ↓
  ┌─────────────────────────────────────────────┐
  │  Human Review Queue (low-confidence cases)  │
  │  Feedback / Retraining Loop                 │
  └─────────────────────────────────────────────┘
```

## What each agent does

| Agent | What it adds | Why it raises accuracy |
|---|---|---|
| **Ingestion** | Scrapes URL, extracts metadata, images | Richer input for all downstream agents |
| **Claim Extraction** | Pulls checkable factual assertions (LLM) | Enables verification instead of style-matching |
| **ML Classifier** | TF-IDF + Linear SVM (original MVP) | Style signal as one input among several |
| **Fact-Check Retrieval** | Web search + Google Fact Check API | Actual claim verification against external truth |
| **Source Credibility** | Domain age, reputation, known bad-actor lists | Catches polished writing from known bad sources |
| **Media Forensics** | EXIF checks, reverse image search | Catches reused/miscaptioned photos |
| **Bias/Sentiment** | Loaded language, framing analysis | Supporting signal for sensationalism |
| **Orchestrator** | Weighs all signals with override rules | Produces one verdict with transparent evidence |

## Setup

```bash
pip install -r requirements.txt
```

### Optional dependencies

```bash
# Lightweight optionals (recommended)
pip install Pillow ImageHash    # media forensics: EXIF + perceptual hashing
pip install python-whois        # source credibility: domain age lookups

# Heavy optional (only if you want vector-search RAG in the knowledge base)
pip install sentence-transformers faiss-cpu  # pulls in PyTorch (~200MB+)
# Without this, the knowledge base degrades to keyword search — still functional.

# API keys for live verification
export FACTCHECK_API_KEY="your-key"   # Google Fact Check Tools API (free: 100/day)
export OPENAI_API_KEY="your-key"      # For LLM-based claim extraction
```

## Quick start

```bash
# Start the API (loads the pre-trained model)
uvicorn api.main:app --reload --port 8000

# Multi-agent analysis
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Fed holds interest rates steady",
    "text": "The Federal Reserve said on Wednesday it would keep interest rates unchanged...",
    "url": "https://example.com/article"
  }'

# Legacy single-model endpoint (backward compatible)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Fed holds interest rates steady","text":"The Federal Reserve said..."}'

# Interactive API docs
open http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + agent status |
| `POST` | `/predict` | Legacy single-model prediction |
| `POST` | `/analyze` | **NEW** multi-agent analysis with evidence trail |
| `GET` | `/review-queue` | Human review queue status |
| `GET` | `/review-queue/pending` | List pending review items |
| `POST` | `/review/{id}/resolve` | Resolve a review with human verdict |
| `POST` | `/feedback` | Log a corrected verdict |
| `GET` | `/knowledge-base/stats` | Knowledge base statistics |
| `GET` | `/stats/model-comparison` | Model benchmark metrics |
| `GET` | `/stats/confusion-matrix` | Confusion matrix |
| `GET` | `/stats/category-breakdown` | Accuracy by subject |
| `GET` | `/stats/trend` | Monthly volume trends |
| `GET` | `/limitations` | Limitations & responsible-use note |

## Project structure

```
project/
├── data/                       True.csv / Fake.csv + knowledge base
├── src/
│   ├── preprocessing.py        shared cleaning/tokenization
│   ├── train.py                trains & evaluates classical models
│   ├── ablation_dateline.py    source-leakage experiment
│   ├── export_js_model.py      compact model for dashboard
│   └── build_dashboard.py      rebuilds dashboard HTML
├── agents/                     NEW: multi-agent system
│   ├── __init__.py
│   ├── base.py                 BaseAgent, AgentResult, Label
│   ├── ingestion.py            URL scraping, metadata extraction
│   ├── claim_extraction.py     LLM-based claim extraction
│   ├── ml_classifier.py        wraps existing TF-IDF + SVM
│   ├── fact_check.py           web search + fact-check APIs
│   ├── source_credibility.py   domain reputation, WHOIS
│   ├── media_forensics.py      EXIF, reverse image search
│   ├── bias_sentiment.py       loaded language detection
│   ├── orchestrator.py         weighted voting + override rules
│   ├── knowledge_base.py       vector DB for RAG
│   ├── review_queue.py         human review routing
│   └── feedback.py             retraining data feedback loop
├── models/                     trained model artifacts
├── reports/                    JSON metrics/artifacts
├── api/
│   └── main.py                 FastAPI (supports /predict + /analyze)
├── dashboard/                  analytics dashboard
└── docs/
    └── limitations.md          limitations & responsible-use note
```

## How the orchestrator works

The orchestrator doesn't just average agent outputs — it uses weighted voting
with override rules:

- **Fact-check agent** has the highest weight (40%) because it verifies claims
  against external truth, not just style patterns.
- **Override rule**: if the fact-check agent is highly confident (>0.7) and
  contradicts the ML classifier, it wins. This prevents a style-based
  classifier from overriding actual factual evidence.
- **Source credibility override**: known bad domains override the ML classifier.
- **Human review routing**: low-confidence or conflicting cases go to a review
  queue instead of auto-publishing a verdict.

## Full reproduce (retrain from scratch)

All commands must be run from the **project root** (where this README lives),
not from inside `src/`. The training scripts use relative paths for `data/`.

```bash
python3 src/train.py               # trains + evaluates all 4 models
python3 src/ablation_dateline.py    # reproduces the leakage-ablation numbers
python3 src/export_js_model.py      # regenerates the compact client-side model
python3 src/build_dashboard.py      # rebuilds dashboard with new numbers
```
