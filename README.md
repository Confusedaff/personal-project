# AI-Powered Fake News Detection — MVP

Use Case #15 (GenAI & Security portfolio). Classifies news articles as
fake or real from title + body text, using TF-IDF features and classical
classifiers, served behind a REST API with a companion analytics dashboard.

Trained and evaluated on the real Kaggle **"Fake and Real News Dataset"**
(`True.csv` / `Fake.csv`, ~44,900 articles combined) — see `data/`.

## Results (held-out 20% test set, 8,978 articles)

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| **Linear SVM (production)** | **0.9937** | **0.9937** |
| Random Forest | 0.9893 | 0.9893 |
| Logistic Regression | 0.9881 | 0.9881 |
| Multinomial Naive Bayes | 0.9581 | 0.9581 |

All four models clear the MVP target (F1 ≥ 0.90). Full metrics, per-model
confusion matrices, category breakdown, and trend data are in `reports/`.

**Important:** these numbers are measured *after* stripping the Reuters wire
dateline that would otherwise let a model shortcut the task by keying on
source formatting instead of content. See `docs/limitations.md` and
`reports/dateline_leakage_ablation.json` for the before/after comparison —
this is the "stated limitations note" required by the MVP spec, backed by an
actual experiment rather than boilerplate.

## Project structure

```
project/
├── data/                       True.csv / Fake.csv (Kaggle dataset)
├── src/
│   ├── preprocessing.py        shared cleaning/tokenization (train + serve)
│   ├── train.py                trains & evaluates all 4 classical models
│   ├── ablation_dateline.py    quantifies the source-leakage artifact
│   ├── export_js_model.py      exports a compact model for the live dashboard demo
│   └── build_dashboard.py      rebuilds dashboard/index.html from current reports/
├── models/
│   ├── vectorizer.joblib       fitted TF-IDF vectorizer
│   ├── best_model.joblib       selected production model (Linear SVM)
│   └── best_model_name.txt
├── reports/                    JSON metrics/artifacts consumed by the API + dashboard
├── api/
│   └── main.py                 FastAPI serving app
├── dashboard/
│   ├── index.html               analytics dashboard + live in-browser demo (ready to open)
│   └── template.html            source template used by src/build_dashboard.py
└── docs/
    └── limitations.md          limitations & responsible-use note
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick start (uses the model already included in this download)

The trained model (`models/`) and its metrics (`reports/`) ship with this
project, so you don't need the dataset just to run the API or view the
dashboard.

```bash
# 1. Start the API (loads the pre-trained model)
uvicorn api.main:app --reload --port 8000

# 2. In another terminal, try a prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Fed holds interest rates steady","text":"The Federal Reserve said on Wednesday it would keep interest rates unchanged..."}'

# Or open the interactive API docs in a browser:
#   http://127.0.0.1:8000/docs
```

```bash
# 3. Open the dashboard
open dashboard/index.html        # macOS
xdg-open dashboard/index.html    # Linux
start dashboard/index.html       # Windows
```

### Live demo: backend + in-browser fallback

The "Wire intake" live demo in the dashboard now calls the real API
(`POST /predict`) for every prediction. If that call fails or times out
(server not running, CORS blocked, network down), it automatically falls
back to a compact 3,500-term logistic regression baked into the page
(`reports/js_model.json`, exported by `src/export_js_model.py`) so the demo
still responds instantly. A badge next to the verdict shows which one
answered — `live · <model name>` or `backup · in-browser model` — and the
hint line under the button reports current connection status.

By default the dashboard looks for the API at `http://127.0.0.1:8000`
(the `uvicorn` default). To point it elsewhere, either:

- open it as `dashboard/index.html?api=https://your-api.example.com`, or
- add `<script>window.FND_API_BASE = 'https://your-api.example.com';</script>`
  before the dashboard's own `<script>` tag (e.g. if you host `index.html`
  yourself and want to bake the URL in).

If you deploy the API somewhere other than `localhost`, update
`allow_origins` in `api/main.py`'s CORS middleware to something narrower
than `"*"` for production use.

## Full reproduce (retrain from scratch)

Only needed if you want to regenerate the model and metrics yourself.

1. Download the dataset from Kaggle — search "Fake and Real News Dataset"
   (by Clément Bisaillon) — and place `Fake.csv` and `True.csv` in `data/`
   (see `data/README.md`).
2. Run the pipeline in order:

```bash
cd project
python3 src/train.py               # trains + evaluates all 4 models, saves best one
python3 src/ablation_dateline.py    # reproduces the leakage-ablation numbers
python3 src/export_js_model.py      # regenerates the compact client-side model
python3 src/build_dashboard.py      # rebuilds dashboard/index.html with the new numbers
```

Each script prints its key metrics to the console as it runs. `train.py`
takes under a minute on a laptop CPU; the others are faster.

## What's not included (out of scope for this pass)

- **BERT fine-tuning (stretch goal):** scaffold-ready but not run here —
  fine-tuning transformer models needs GPU time well beyond this pass. Add a
  `src/train_bert.py` using `transformers.Trainer` on the same
  `clean_content`/`label` columns when compute is available, and compare
  against `reports/model_comparison.json` using the same test split.
- **Persistent hosting:** the API and dashboard are provided to run locally
  or deploy to your own infrastructure; nothing is hosted for you here.
