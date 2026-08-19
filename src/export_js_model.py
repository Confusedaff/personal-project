"""
Exports a compact TF-IDF + Logistic Regression model as JSON so the
dashboard can score user-typed text client-side, with no backend call.

This is a *demo* artifact distinct from the production model in
models/best_model.joblib (Linear SVM, 30k features): it caps vocabulary
size so the exported JSON stays small enough to ship inline in an HTML
artifact, trading a little accuracy for portability.
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import build_dataset, STOPWORDS

RANDOM_STATE = 42
VOCAB_SIZE = 3500


def main():
    df = build_dataset("data", remove_dateline=True)
    X_tr_text, X_te_text, y_tr, y_te = train_test_split(
        df["clean_content"], df["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vec = TfidfVectorizer(max_features=VOCAB_SIZE, ngram_range=(1, 1),
                           min_df=8, stop_words=list(STOPWORDS), sublinear_tf=True)
    X_tr = vec.fit_transform(X_tr_text)
    X_te = vec.transform(X_te_text)

    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="weighted")
    print(f"Compact demo model: vocab={len(vec.vocabulary_)}  acc={acc:.4f}  f1={f1:.4f}")

    vocab = vec.vocabulary_  # term -> column index
    idf = vec.idf_
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    # term -> [idf, coefficient]  (kept together to halve the number of JSON keys)
    terms = {}
    for term, idx in vocab.items():
        terms[term] = [round(float(idf[idx]), 4), round(float(coef[idx]), 5)]

    export = {
        "intercept": round(intercept, 5),
        "terms": terms,
        "meta": {
            "vocab_size": len(terms),
            "test_accuracy": round(acc, 4),
            "test_f1_weighted": round(f1, 4),
            "note": "Compact single-token demo model for in-browser scoring. "
                    "Production model (Linear SVM, 30k bigram features) lives in models/."
        }
    }
    with open("reports/js_model.json", "w") as f:
        json.dump(export, f)

    import os
    size_kb = os.path.getsize("reports/js_model.json") / 1024
    print(f"Wrote reports/js_model.json  ({size_kb:.0f} KB, {len(terms)} terms)")


if __name__ == "__main__":
    main()
