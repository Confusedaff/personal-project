"""
Data loading and text preprocessing for the Fake News Detection MVP.

Loads the Kaggle "Fake and Real News Dataset" (True.csv / Fake.csv),
merges the two files, parses dates, and provides a text-cleaning
function shared by both the training pipeline and the serving API so
that inference-time preprocessing always matches training-time
preprocessing.
"""
import re
import pandas as pd

# A compact, hand-maintained English stopword list. Kept local (rather than
# pulling nltk's corpus at runtime) so the pipeline has no extra network
# dependency at train or inference time.
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's
with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves
""".split())

# The vast majority of the "True" articles in this corpus open with a wire
# dateline, e.g. "WASHINGTON (Reuters) - ". Left in place, a classifier can
# hit ~99%+ accuracy just by keying on that formatting artifact rather than
# on any genuine signal of misinformation. We strip it during cleaning so
# the model is forced to learn from actual language, and we report the
# before/after difference explicitly in the limitations note.
# Handles both single-city ("WASHINGTON (Reuters) - ") and multi-city
# ("EVERETT, Washington/WASHINGTON (Reuters) - ") dateline formats.
DATELINE_RE = re.compile(r"^\s*[A-Z][A-Za-z0-9\.\,\'\-/\s]{0,90}\(Reuters\)\s*-\s*")
URL_RE = re.compile(r"http\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
WS_RE = re.compile(r"\s+")
REUTERS_TOKEN_RE = re.compile(r"\breuters\b")


def strip_dateline(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return DATELINE_RE.sub("", text)


def clean_text(text: str, remove_dateline: bool = True) -> str:
    """Lowercase, strip wire datelines/URLs/punctuation, collapse whitespace.

    When remove_dateline is True this also purges any leftover standalone
    "Reuters" mentions elsewhere in the body (e.g. "(Reuters Health)",
    "according to Reuters"). Left in place, that single token is enough for
    a classifier to key on source identity rather than content -- see
    src/ablation_dateline.py for the measured effect.
    """
    if not isinstance(text, str):
        return ""
    if remove_dateline:
        text = strip_dateline(text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    if remove_dateline:
        text = REUTERS_TOKEN_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def load_raw(data_dir: str = "data"):
    fake = pd.read_csv(f"{data_dir}/Fake.csv")
    true = pd.read_csv(f"{data_dir}/True.csv")
    fake["label"] = 0  # 0 = fake
    true["label"] = 1  # 1 = real
    df = pd.concat([fake, true], ignore_index=True)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["title", "text"]).reset_index(drop=True)
    return df


def build_dataset(data_dir: str = "data", remove_dateline: bool = True) -> pd.DataFrame:
    df = load_raw(data_dir)
    df["content"] = (df["title"].fillna("") + ". " + df["text"].fillna(""))
    df["clean_content"] = df["content"].apply(lambda t: clean_text(t, remove_dateline=remove_dateline))
    df = df[df["clean_content"].str.len() > 0].reset_index(drop=True)
    return df
