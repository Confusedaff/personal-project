"""
Knowledge Base / Vector DB — RAG over fact-checked corpus.

Provides retrieval-augmented generation support by maintaining a vector
database of fact-checked claims and their verdicts. Feeds into the
fact-check retrieval agent for richer context.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Vector DB integration (optional — falls back to JSON file storage)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# Fallback: simple JSON-based storage
KB_FILE = ROOT / "data" / "knowledge_base.json"


class KnowledgeBase:
    """Simple fact-check knowledge base with optional vector search."""

    def __init__(self):
        self.entries: list[dict] = []
        self.model = None
        self.index = None
        self._model_loaded = False
        self._load()

    def _ensure_model(self):
        """Lazy-load the sentence-transformers model only when needed."""
        if self._model_loaded:
            return
        self._model_loaded = True
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.model = None

    def _load(self):
        """Load existing knowledge base from disk."""
        if KB_FILE.exists():
            try:
                self.entries = json.loads(KB_FILE.read_text())
            except Exception:
                self.entries = []

    def _save(self):
        """Persist knowledge base to disk."""
        KB_FILE.parent.mkdir(parents=True, exist_ok=True)
        KB_FILE.write_text(json.dumps(self.entries, indent=2))

    def _build_index(self):
        """Build FAISS index for vector search."""
        self._ensure_model()
        if not HAS_FAISS or not HAS_NUMPY or not self.model or not self.entries:
            return

        texts = [e.get("claim", "") for e in self.entries]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def add_entry(self, claim: str, verdict: str, sources: list[str],
                  article_text: str = "", metadata: dict | None = None):
        """Add a fact-checked claim to the knowledge base."""
        entry = {
            "id": len(self.entries),
            "claim": claim,
            "verdict": verdict,
            "sources": sources,
            "article_text": article_text[:500],
            "metadata": metadata or {},
            "added_at": time.time(),
        }
        self.entries.append(entry)
        self._save()
        # Rebuild index
        self.index = None
        self._build_index()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for similar fact-checked claims."""
        if not self.entries:
            return []

        self._ensure_model()

        # Vector search if available
        if self.model and HAS_FAISS:
            if self.index is None:
                self._build_index()
            if self.index is not None:
                query_embedding = self.model.encode([query])
                query_embedding = np.array(query_embedding, dtype="float32")
                distances, indices = self.index.search(query_embedding, min(top_k, len(self.entries)))
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < len(self.entries):
                        entry = dict(self.entries[idx])
                        entry["similarity"] = float(1 / (1 + dist))
                        results.append(entry)
                return results

        # Fallback: keyword matching
        query_words = set(query.lower().split())
        scored = []
        for entry in self.entries:
            claim_words = set(entry.get("claim", "").lower().split())
            overlap = len(query_words & claim_words)
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, entry in scored[:top_k]:
            r = dict(entry)
            r["similarity"] = score / max(len(query_words), 1)
            results.append(r)
        return results

    def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        verdicts = {}
        for e in self.entries:
            v = e.get("verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1
        return {
            "total_entries": len(self.entries),
            "verdict_distribution": verdicts,
            "has_vector_index": self.index is not None,
        }


# Singleton instance
_kb = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
