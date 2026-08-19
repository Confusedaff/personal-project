"""
Rebuilds dashboard/index.html by embedding the current contents of reports/
(plus the compact client-side model) into dashboard/template.html.

Run this after src/train.py / src/ablation_dateline.py / src/export_js_model.py
any time you retrain, so the dashboard reflects the latest numbers instead of
the snapshot it shipped with.
"""
import json
from pathlib import Path
from preprocessing import STOPWORDS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DASHBOARD = ROOT / "dashboard"


def main():
    data = {
        "modelComparison": json.loads((REPORTS / "model_comparison.json").read_text()),
        "confusion": json.loads((REPORTS / "confusion_matrices.json").read_text()),
        "category": json.loads((REPORTS / "category_breakdown.json").read_text()),
        "trend": json.loads((REPORTS / "trend_data.json").read_text()),
        "ablation": json.loads((REPORTS / "dateline_leakage_ablation.json").read_text()),
        "jsModel": json.loads((REPORTS / "js_model.json").read_text()),
        "stopwords": sorted(STOPWORDS),
    }

    template = (DASHBOARD / "template.html").read_text(encoding="utf-8")
    out = template.replace("__EMBEDDED_DATA__", json.dumps(data))
    (DASHBOARD / "index.html").write_text(out, encoding="utf-8")

    size_kb = len(out.encode("utf-8")) / 1024
    print(f"Wrote dashboard/index.html ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
