"""
Quantifies how much of the model's accuracy comes from the Reuters wire
dateline artifact rather than genuine linguistic signal. Trains the same
Logistic Regression + TF-IDF pipeline twice: once on raw-cleaned text
(dateline left in) and once on dateline-stripped text (the production
setting used in train.py), on identical splits, and reports the delta.
"""
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import build_dataset, STOPWORDS

RANDOM_STATE = 42


def run(remove_dateline):
    df = build_dataset("data", remove_dateline=remove_dateline)
    X_tr_text, X_te_text, y_tr, y_te = train_test_split(
        df["clean_content"], df["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label"]
    )
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=5,
                           stop_words=list(STOPWORDS), sublinear_tf=True)
    X_tr = vec.fit_transform(X_tr_text)
    X_te = vec.transform(X_te_text)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="weighted")

    # Which features carry the most weight toward "real"? If the dateline is
    # left in, "reuters" itself should dominate.
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_real_idx = coefs.argsort()[-8:][::-1]
    top_real_terms = [feature_names[i] for i in top_real_idx]

    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
            "top_terms_favoring_real": top_real_terms}


def main():
    print("Training WITHOUT dateline stripping (raw wire text left in)...")
    with_leak = run(remove_dateline=False)
    print(with_leak)

    print("\nTraining WITH dateline stripping (production setting)...")
    without_leak = run(remove_dateline=True)
    print(without_leak)

    out = {
        "with_dateline_leakage": with_leak,
        "without_dateline_leakage_production": without_leak,
        "accuracy_delta": round(with_leak["accuracy"] - without_leak["accuracy"], 4),
        "note": ("Leaving the Reuters wire dateline in the text inflates accuracy by "
                 "letting the model key on formatting rather than content. The production "
                 "pipeline strips it; the residual accuracy above is attributable to actual "
                 "linguistic/stylistic signal.")
    }
    with open("reports/dateline_leakage_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved reports/dateline_leakage_ablation.json")


if __name__ == "__main__":
    main()
