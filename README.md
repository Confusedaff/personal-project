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

## Features

- **Multi-agent architecture**: 8 specialized agents working in concert
- **Factual verification**: LLM-based claim extraction + live web search (not just style matching)
- **Transparent evidence trail**: Every verdict comes with reasoning from each agent
- **Override rules**: Fact-check agent can override the ML classifier when it finds factual errors
- **Human review routing**: Low-confidence and conflicting cases go to a review queue
- **Feedback loop**: Corrected verdicts feed back into retraining data
- **Analyst dashboard**: Interactive HTML dashboard with model benchmarks, confusion matrices, and category breakdowns
- **In-browser fallback**: Compact client-side model for offline use when the backend is unreachable

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

### Dependencies

**Required:**

| Package | Purpose |
|---|---|
| `pandas>=2.0` | Data loading and manipulation |
| `numpy>=1.24` | Numerical operations |
| `scikit-learn>=1.3` | TF-IDF vectorization, SVM, and other classifiers |
| `fastapi>=0.110` | Serving API |
| `uvicorn>=0.29` | ASGI server for FastAPI |
| `pydantic>=2.0` | Request/response validation |
| `httpx>=0.27.0` | HTTP client for web search and scraping |
| `beautifulsoup4>=4.12` | HTML parsing for ingestion and web search |
| `joblib>=1.3` | Model serialization |
| `openai>=1.0` | LLM-based claim extraction |
| `anthropic>=0.39.0` | LLM fallback for claim extraction and fact-checking |
| `python-whois>=0.9` | Domain age lookups for source credibility |

**Optional (recommended):**

```bash
# Media forensics: EXIF + perceptual hashing
pip install Pillow ImageHash

# Knowledge base vector search (heavy — pulls in PyTorch ~200MB+)
pip install sentence-transformers faiss-cpu
# Without this, the knowledge base degrades to keyword search — still functional.
```

### Environment Variables

```bash
# API keys for live verification (optional — system degrades gracefully without them)
export FACTCHECK_API_KEY="your-key"      # Google Fact Check Tools API (free: 100/day)
export OPENAI_API_KEY="your-key"         # For LLM-based claim extraction
export GROQ_API_KEY="your-key"           # Groq (fast/cheap LLM inference, primary for fact-check)
export ANTHROPIC_API_KEY="your-key"      # Anthropic (fallback LLM provider)
export TAVILY_API_KEY="your-key"         # Tavily search API (preferred over DuckDuckGo fallback)
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
| `GET` | `/` | Serves the analyst dashboard |
| `GET` | `/health` | Liveness check + agent status |
| `POST` | `/predict` | Legacy single-model prediction |
| `POST` | `/analyze` | **Multi-agent analysis with evidence trail** |
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
Fake-News-main/
├── data/                       True.csv / Fake.csv + knowledge base
│   ├── Fake.csv                Fake news articles (~23,481)
│   ├── True.csv                Real news articles (~21,417)
│   ├── knowledge_base.json     Vector DB entries for RAG
│   └── review_queue.json       Human review queue state
├── src/
│   ├── preprocessing.py        shared cleaning/tokenization
│   ├── train.py                trains & evaluates classical models
│   ├── ablation_dateline.py    source-leakage experiment
│   ├── export_js_model.py      compact model for dashboard
│   └── build_dashboard.py      rebuilds dashboard HTML
├── agents/                     multi-agent system
│   ├── __init__.py             package exports
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
│   ├── best_model.joblib       production model (Linear SVM)
│   ├── vectorizer.joblib       fitted TF-IDF vectorizer
│   └── best_model_name.txt     model name for logging
├── reports/                    JSON metrics/artifacts
│   ├── model_comparison.json   accuracy/F1 per model
│   ├── confusion_matrices.json confusion matrix per model
│   ├── category_breakdown.json accuracy by article subject
│   ├── trend_data.json         monthly fake/real volume
│   ├── dateline_leakage_ablation.json  with/without dateline comparison
│   └── js_model.json           compact client-side model
├── api/
│   └── main.py                 FastAPI (supports /predict + /analyze)
├── dashboard/                  analytics dashboard
│   ├── template.html           dashboard template with __EMBEDDED_DATA__
│   └── index.html              generated dashboard (static HTML)
├── docs/
│   └── limitations.md          limitations & responsible-use note
├── requirements.txt            Python dependencies
└── README.md                   this file
```

## How the orchestrator works

The orchestrator doesn't just average agent outputs — it uses weighted voting
with override rules:

| Agent | Weight | Rationale |
|---|---|---|
| `claim_extraction` | 0.0 | Extraction only, no classification |
| `ml_classifier` | 0.15 | Style signal (useful but not ground truth) |
| `fact_check` | 0.40 | **THE accuracy lever** — actual claim verification |
| `source_credibility` | 0.15 | Domain reputation catches known bad actors |
| `media_forensics` | 0.10 | Image manipulation detection |
| `bias_sentiment` | 0.05 | Supporting signal only |
| `ingestion` | 0.0 | No classification |

### Override rules

1. **Fact-check override**: If the fact-check agent is highly confident (>0.7) and contradicts the ML classifier, it wins. This prevents a style-based classifier from overriding actual factual evidence.
2. **Source credibility override**: Known bad domains with low credibility scores override the ML classifier.
3. **Human review routing**: Low-confidence or conflicting cases go to a review queue instead of auto-publishing a verdict.

### Decision logic

```
weighted_fake = sum(weight_i * confidence_i) for all agents labeling "fake"
weighted_real = sum(weight_i * confidence_i) for all agents labeling "real"

if fact_check.confidence > 0.7 and fact_check.label != ml_classifier.label:
    → fact_check wins (override)
elif source_credibility.confidence > 0.6 and source_credibility.label == "fake" and ml_classifier.label == "real":
    → source_credibility wins (override)
else:
    → fake_ratio = weighted_fake / total_weight
    → real_ratio = weighted_real / total_weight
    → label = argmax(fake_ratio, real_ratio)
    → confidence = max(fake_ratio, real_ratio)
```

### Human review triggers

- Confidence < 0.5
- Override applied (agent conflict)
- Weighted vote difference < 0.1 (agents strongly disagree)

## Model benchmarks

All models use identical TF-IDF features (30,000 max features, 1-2 ngrams) and the same 8,978-article held-out test set.

| Model | Accuracy | F1 (weighted) | F1 (macro) | Inference (test set) |
|---|---|---|---|---|
| **Linear SVM** | **0.9937** | **0.9937** | **0.9936** | 0.016s |
| Random Forest | 0.9893 | 0.9893 | 0.9893 | 1.433s |
| Logistic Regression | 0.9881 | 0.9881 | 0.9881 | 0.005s |
| Multinomial Naive Bayes | 0.9581 | 0.9581 | 0.9580 | 0.008s |

### Confusion matrix (Linear SVM, production model)

|  | Predicted fake | Predicted real |
|---|---|---|
| **Actual fake** | 4,665 (correctly flagged) | 30 (missed) |
| **Actual real** | 27 (wrongly flagged) | 4,256 (correctly passed) |

### Dateline leakage ablation

| Setting | Accuracy | F1 |
|---|---|---|
| Dateline left in | ~99.5% | ~99.5% |
| Dateline stripped (production) | 99.37% | 99.37% |

Leaving the Reuters wire dateline in the text inflates accuracy by letting the model key on formatting rather than content. The production pipeline strips it; the residual accuracy above is attributable to actual linguistic/stylistic signal.

## Dashboard

The analyst dashboard is a self-contained HTML file at `dashboard/index.html` with:

- **Live desk**: Paste article text or a URL for multi-agent analysis
- **Model benchmarks**: Bar chart comparing all 4 classifiers
- **Confusion matrix**: Where the production model makes errors
- **Category breakdown**: Accuracy by article subject tag
- **Trend chart**: Monthly fake vs. real volume in the training corpus

### Rebuilding the dashboard

After retraining, rebuild the dashboard to reflect the latest numbers:

```bash
python3 src/train.py               # trains + evaluates all 4 models
python3 src/ablation_dateline.py    # reproduces the leakage-ablation numbers
python3 src/export_js_model.py      # regenerates the compact client-side model
python3 src/build_dashboard.py      # rebuilds dashboard with new numbers
```

The dashboard is served automatically by the API at `GET /` (or open `dashboard/index.html` directly in a browser).

## Full reproduce (retrain from scratch)

All commands must be run from the **project root** (where this README lives),
not from inside `src/`. The training scripts use relative paths for `data/`.

```bash
python3 src/train.py               # trains + evaluates all 4 models
python3 src/ablation_dateline.py    # reproduces the leakage-ablation numbers
python3 src/export_js_model.py      # regenerates the compact client-side model
python3 src/build_dashboard.py      # rebuilds dashboard with new numbers
```

## Limitations

See `docs/limitations.md` for the full list. Key points:

1. **Narrow dataset**: The corpus draws real articles from Reuters wire copy and fake articles from a small set of flagged outlets. The ML classifier has not seen sports, entertainment, science, health, or local-news writing.
2. **Style mimicry**: A bad actor who deliberately imitates wire-service formatting will erode the ML classifier's accuracy — but fact-check and source credibility agents provide independent signals.
3. **Fact-check agent**: Requires internet access and API keys. The free tier is limited to 100 queries/day.
4. **Media forensics**: Reverse image search is currently a stub. Cannot detect deepfakes or AI-generated images.
5. **Source credibility list**: The `KNOWN_LOW_CREDIBILITY` list reflects editorial judgment based on fact-check records, not a neutral classification.
6. **Recommended use**: Treat predictions as a **triage signal for human fact-checkers**, not an automated takedown mechanism.

## Data

This project uses the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) by Clement Bisaillon.

- `True.csv`: Real news articles (~21,417)
- `Fake.csv`: Fake news articles (~23,481)
- Total after cleaning: ~44,889 articles

The dataset is not bundled in this repository due to file size (~110 MB) and licensing. Download it from Kaggle and place both files in the `data/` directory.

## License

This project is for educational and portfolio demonstration purposes. See the dataset license on Kaggle for data usage terms.
