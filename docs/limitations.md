# Limitations & Responsible-Use Note

This note is a required deliverable of the MVP (Objective 4, item 6) and should ship
alongside the model in any deployment.

## 1. The dataset has a narrow, source-concentrated origin

The corpus (Kaggle "Fake and Real News Dataset") draws its real articles almost
entirely from Reuters wire copy, and its fake articles from a small set of
outlets flagged by fact-checking organizations. Coverage skews heavily to
**political and world-news topics** (politicsNews and worldnews account for
most of the "real" class; News/politics/left-news dominate the "fake" class).
A model trained on this corpus is a demonstration of feasibility, not a
general-purpose misinformation detector — it has not seen sports, entertainment,
science, health, or local-news writing, and should not be assumed to generalize
to those domains without retraining on in-domain data.

**Measured finding — the `subject` tag is a perfect class proxy.** We checked
whether any subject tag is shared between the two classes: it is not. All
2,195 `politicsNews` articles and all 2,088 `worldnews` articles are real; all
1,771 `News`, 1,353 `politics`, 906 `left-news`, 318 `Government News`, 174
`Middle-east`, and 173 `US_News` articles are fake (see
`reports/category_breakdown.json`). This means `subject` alone would trivially
"predict" the label in this dataset — it is a collection artifact (how the two
source sets were tagged), not a real-world property of misinformation, and the
model does not use `subject` as an input feature specifically to avoid
laundering this shortcut into production. Anyone extending this project should
be aware that per-category accuracy numbers are partly a reflection of this
imbalance, not purely of model skill.

## 2. Measured finding: source-formatting leakage inflates accuracy

We tested this directly rather than assuming it. Reuters wire copy in this
corpus almost always opens with a formatting artifact — a dateline such as
`WASHINGTON (Reuters) -` — that a classifier can key on instead of learning
anything about *content*. We trained an identical TF-IDF + Logistic Regression
pipeline twice, once leaving that artifact in and once stripping it out:

| Setting | Accuracy | Weighted F1 | Top terms favoring "real" |
|---|---|---|---|
| Dateline left in (raw text) | 0.9926 | 0.9926 | `reuters`, `said`, `washington reuters`, `washington` |
| Dateline stripped (production) | 0.9881 | 0.9881 | `said`, `president donald`, weekday names, `republican` |

Stripping the dateline (and any leftover "Reuters" mentions elsewhere in the
body) removes the single biggest shortcut and drops accuracy by roughly half
a point — the model is still clearly using genuine signal, but the raw-text
number should not be quoted as if it reflects real-world misinformation
detection ability. See `reports/dateline_leakage_ablation.json` and
`src/ablation_dateline.py` for the full experiment and to reproduce it.

Even after stripping, terms tied to Reuters' house style and beat (formal
attribution like "said", weekday-dated reporting, Washington political
coverage) remain influential. This is a **residual source-style bias**, not
a bug — it means part of what the model has learned is "this reads like wire
journalism" rather than "this is factually accurate," and the two are
correlated but not identical.

## 3. Class-imbalance and threshold considerations

The corpus is close to balanced (23,481 fake vs. 21,417 real) but real-world
traffic will not be. A false negative (fake content labeled real) is treated
as more costly than a false positive for a moderation use case, since it
lets misinformation pass unreviewed. The confidence score returned by
`/predict` is calibrated on this corpus's balance; if deployed on a stream
with a very different fake:real ratio, the decision threshold should be
re-tuned rather than left at 0.5, and precision/recall should be re-measured
on traffic that matches the deployment distribution.

## 4. Style mimicry will degrade performance over time

Because part of the signal is *stylistic*, a bad actor who deliberately
imitates wire-service formatting and tone (the exact failure mode described
in the original use-case problem statement) will erode the model's accuracy
faster than a purely random writer would. The model should be retrained
periodically on fresh, labeled examples rather than treated as a
one-time-trained, permanently accurate system.

## 5. Recommended use

- Treat predictions as a **triage signal for human fact-checkers**, not an
  automated takedown mechanism, particularly near the confidence threshold.
- Log prediction confidence alongside outcomes so drift can be monitored and
  the retraining cadence can be data-driven rather than fixed.
- Re-validate on a sample of the actual target domain (e.g. social-media
  posts, if that's the deployment surface) before trusting reported metrics —
  this corpus is long-form news article text, which is stylistically
  different from short-form or social content.
