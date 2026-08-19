# Limitations & Responsible-Use Note

This note covers the multi-agent system (v2.0) and the original MVP.

## 1. Multi-agent architecture: what changed

The original MVP classified articles using TF-IDF + Linear SVM on the full
article text — essentially a style-based classifier. The v2.0 multi-agent
system adds factual verification on top:

- **Claim extraction agent** pulls out checkable factual assertions (using an LLM
  or rule-based fallback) instead of classifying the whole blob at once.
- **Fact-check retrieval agent** searches the live web and Google Fact Check Tools
  API for each claim, providing *actual verification* rather than style judgment.
- **Source credibility agent** checks domain reputation, registration age, and
  known misinformation databases.
- **Media forensics agent** performs EXIF checks and (optionally) reverse image
  search on embedded images.
- **Bias/sentiment agent** flags manipulative framing as a supporting signal.
- **Orchestrator** weighs all agent signals with override rules (e.g., a
  fact-check finding a false claim overrides the ML classifier saying "real").

This addresses the root cause identified in limitation #2 below: the original
model could only judge "does this read fake," never "is this claim actually true."

## 2. Source-formatting leakage (partially addressed)

The ML classifier agent still uses TF-IDF + Linear SVM, which retains the
source-style bias described below. However, the orchestrator now treats it as
*one signal among several* (weight: 15%) rather than the sole verdict. The
fact-check agent (weight: 40%) provides actual factual verification.

## 3. The dataset has a narrow, source-concentrated origin

The corpus (Kaggle "Fake and Real News Dataset") draws its real articles almost
entirely from Reuters wire copy, and its fake articles from a small set of
outlets flagged by fact-checking organizations. Coverage skews heavily to
**political and world-news topics** (politicsNews and worldnews account for
most of the "real" class; News/politics/left-news dominate the "fake" class).

The ML classifier agent trained on this corpus is a demonstration of feasibility,
not a general-purpose misinformation detector — it has not seen sports,
entertainment, science, health, or local-news writing, and should not be assumed
to generalize to those domains without retraining on in-domain data.

**Measured finding — the `subject` tag is a perfect class proxy.** We checked
whether any subject tag is shared between the two classes: it is not. All
2,195 `politicsNews` articles and all 2,088 `worldnews` articles are real; all
1,771 `News`, 1,353 `politics`, 906 `left-news`, 318 `Government News`, 174
`Middle-east`, and 173 `US_News` articles are fake (see
`reports/category_breakdown.json`). This means `subject` alone would trivially
"predict" the label in this dataset — it is a collection artifact, not a
real-world property of misinformation.

## 4. Style mimicry will still affect the ML classifier

Because part of the ML classifier's signal is *stylistic*, a bad actor who
deliberately imitates wire-service formatting will erode its accuracy. However,
the multi-agent system mitigates this: even if the classifier is fooled, the
fact-check and source credibility agents provide independent signals that don't
depend on writing style.

## 5. Fact-check agent limitations

- Requires internet access and API keys (Google Fact Check Tools API)
- Free tier limited to 100 queries/day
- Coverage depends on what's been fact-checked — obscure claims may have no
  matches
- Web search fallback uses DuckDuckGo HTML scraping — this is fragile, may
  be rate-limited or blocked by DuckDuckGo, and should not be relied on in
  production. A proper search API (SerpAPI, Bing, etc.) is recommended.
- Results are cached in the knowledge base to reduce API calls over time

## 6. Media forensics agent limitations

- Requires optional dependencies (Pillow, ImageHash) for full functionality
- **Reverse image search is currently a stub** — the `_reverse_image_search()`
  function returns empty results. To enable it, integrate a real API:
  TinEye (paid), Google Cloud Vision (paid), or SauceNAO (free tier).
  This is a known incomplete feature, not a hidden limitation.
- Cannot detect deepfakes or AI-generated images
- EXIF data is often stripped by social media platforms, limiting what can
  be extracted from social-media-sourced images

## 7. Human review queue

The orchestrator routes low-confidence or conflicting cases to a human review
queue (POST /review/{id}/resolve). This is the recommended deployment pattern:
treat the system as a **triage signal for human fact-checkers**, not an
automated takedown mechanism.

## 8. Feedback loop and retraining

Corrected verdicts from human review are logged to the feedback loop and can
be exported as retraining data (get_feedback_loop().get_training_data()). This
addresses the degradation problem: as adversaries adapt, corrected examples
feed back into model improvements.

## 9. Source credibility list is an editorial judgment

The `KNOWN_LOW_CREDIBILITY` list in `source_credibility.py` is **not** a
neutral, objective classification. It includes sites that have been widely
flagged by fact-checking organizations (MBFC, NewsGuard) for repeated
fabrication or demonstrable misinformation, but the inclusion of any
particular outlet reflects the author's editorial judgment based on
available fact-check records. Reasonable people may disagree about where
to draw the line between "partisan but factual" and "misinformation."

The list is intentionally conservative — it includes only outlets with
documented patterns of fabricated claims, not partisan outlets that
generally report factual information with a slant. In production, this
list should be replaced with a regularly-updated external database
(MBFC, NewsGuard, or a custom aggregation) rather than a hardcoded set.

## 10. Heavy optional dependencies

`sentence-transformers` (used by the knowledge base for vector search) is
a genuinely heavy dependency — it pulls in PyTorch, which can be several
hundred megabytes. The code already degrades gracefully to keyword search
if it's not installed, so it is **not required** for the system to function.
If disk space or install time is a concern, skip it and rely on the keyword
fallback.

## 11. Class-imbalance and threshold considerations

The corpus is close to balanced (23,481 fake vs. 21,417 real) but real-world
traffic will not be. The orchestrator's weighted voting and override rules
provide some robustness, but the decision threshold should be re-tuned for
deployment distributions with very different fake:real ratios.

## 10. Recommended use

- Treat predictions as a **triage signal for human fact-checkers**, not an
  automated takedown mechanism, particularly near the confidence threshold.
- The `/analyze` endpoint provides a full evidence trail — use it for
  explainability and audit purposes.
- Log prediction confidence alongside outcomes so drift can be monitored and
- the retraining cadence can be data-driven rather than fixed.
- Re-validate on a sample of the actual target domain before trusting
  reported metrics.
