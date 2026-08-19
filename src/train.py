"""
Train and evaluate classical TF-IDF classifiers on the Fake and Real News
Dataset (Objectives 4.1, items 1-3 of the project spec).

Outputs:
  models/best_model.joblib          - the selected production model (pipeline)
  models/vectorizer.joblib          - fitted TF-IDF vectorizer (also inside pipeline)
  reports/model_comparison.json     - accuracy/F1/precision/recall per model
  reports/confusion_matrices.json   - confusion matrix per model
  reports/category_breakdown.json   - accuracy by article subject, best model
  reports/trend_data.json           - monthly fake/real article volume
  reports/dateline_leakage_ablation.json - with/without dateline-stripping comparison
"""
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
)
import joblib

from preprocessing import build_dataset, STOPWORDS

RANDOM_STATE = 42
DATA_DIR = "data"


def evaluate(name, model, X_test, y_test, results, matrices):
    t0 = time.time()
    preds = model.predict(X_test)
    infer_s = time.time() - t0
    acc = accuracy_score(y_test, preds)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, preds, average="weighted")
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds).tolist()
    results[name] = {
        "accuracy": round(acc, 4),
        "precision_weighted": round(p_w, 4),
        "recall_weighted": round(r_w, 4),
        "f1_weighted": round(f1_w, 4),
        "precision_macro": round(p_m, 4),
        "recall_macro": round(r_m, 4),
        "f1_macro": round(f1_m, 4),
        "inference_seconds_test_set": round(infer_s, 3),
    }
    matrices[name] = {"labels": ["fake(0)", "real(1)"], "matrix": cm}
    print(f"{name:22s} acc={acc:.4f}  f1_weighted={f1_w:.4f}  f1_macro={f1_m:.4f}")
    return preds


def main():
    print("Loading and cleaning dataset (dateline-stripped, production pipeline)...")
    df = build_dataset(DATA_DIR, remove_dateline=True)
    print(f"Rows after cleaning: {len(df)}  |  fake={sum(df.label==0)}  real={sum(df.label==1)}")

    X_train_text, X_test_text, y_train, y_test, idx_train, idx_test = train_test_split(
        df["clean_content"], df["label"], df.index,
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=30000, ngram_range=(1, 2), min_df=5,
        stop_words=list(STOPWORDS), sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print("TF-IDF shape:", X_train.shape)

    results, matrices, fitted = {}, {}, {}

    print("\nTraining classical classifiers...")
    lr = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    evaluate("Logistic Regression", lr, X_test, y_test, results, matrices)
    fitted["Logistic Regression"] = lr

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    evaluate("Multinomial Naive Bayes", nb, X_test, y_test, results, matrices)
    fitted["Multinomial Naive Bayes"] = nb

    svm_base = LinearSVC(random_state=RANDOM_STATE, max_iter=5000)
    svm = CalibratedClassifierCV(svm_base, cv=3)  # adds predict_proba for confidence scores
    svm.fit(X_train, y_train)
    evaluate("Linear SVM", svm, X_test, y_test, results, matrices)
    fitted["Linear SVM"] = svm

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_preds = evaluate("Random Forest", rf, X_test, y_test, results, matrices)
    fitted["Random Forest"] = rf

    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    best_model = fitted[best_name]
    print(f"\nBest model by weighted F1: {best_name}  (F1={results[best_name]['f1_weighted']})")

    # ---- persist production artifacts ----
    joblib.dump(vectorizer, "models/vectorizer.joblib")
    joblib.dump(best_model, "models/best_model.joblib")
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    with open("reports/model_comparison.json", "w") as f:
        json.dump({"results": results, "best_model": best_name, "n_train": X_train.shape[0],
                   "n_test": X_test.shape[0], "n_features": X_train.shape[1]}, f, indent=2)
    with open("reports/confusion_matrices.json", "w") as f:
        json.dump(matrices, f, indent=2)

    # ---- category breakdown on test set, best model ----
    best_preds = best_model.predict(X_test)
    test_df = df.loc[idx_test].copy()
    test_df["pred"] = best_preds
    test_df["correct"] = (test_df["pred"] == test_df["label"]).astype(int)
    cat = (test_df.groupby("subject")
           .agg(n=("label", "size"), accuracy=("correct", "mean"),
                fake_count=("label", lambda s: int((s == 0).sum())),
                real_count=("label", lambda s: int((s == 1).sum())))
           .reset_index().sort_values("n", ascending=False))
    cat["accuracy"] = cat["accuracy"].round(4)
    cat.to_json("reports/category_breakdown.json", orient="records", indent=2)

    # ---- trend view: monthly fake vs real volume over the whole corpus ----
    trend_df = df.dropna(subset=["date_parsed"]).copy()
    trend_df["month"] = trend_df["date_parsed"].dt.to_period("M").astype(str)
    trend = (trend_df.groupby(["month", "label"]).size().unstack(fill_value=0))
    trend.columns = ["fake", "real"] if list(trend.columns) == [0, 1] else trend.columns
    trend = trend.reset_index().sort_values("month")
    trend.to_json("reports/trend_data.json", orient="records", indent=2)

    print("\nSaved model artifacts to models/, reports to reports/")
    return best_name, results


if __name__ == "__main__":
    main()
