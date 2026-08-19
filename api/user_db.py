"""
Per-user database management.

Each user gets an isolated SQLite database at data/user_data/{user_id}/user.db.
Shared authentication data lives in data/users.db.
"""
import sqlite3
import json
import os
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
USERS_DB = DATA_DIR / "users.db"
USER_DATA_DIR = DATA_DIR / "user_data"


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def init_users_db():
    """Create the shared users table if it doesn't exist."""
    _ensure_dirs()
    conn = sqlite3.connect(str(USERS_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.commit()
    conn.close()


def create_user(username: str, email: str, hashed_password: str) -> int:
    """Insert a new user and create their private data directory. Returns user_id."""
    _ensure_dirs()
    conn = sqlite3.connect(str(USERS_DB))
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
            (username, email, hashed_password),
        )
        user_id = cursor.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Username or email already exists")
    conn.close()

    # Create per-user data directory and empty stores
    user_dir = USER_DATA_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    _init_user_stores(user_dir)
    return user_id


def _init_user_stores(user_dir: Path):
    """Initialize empty per-user data files."""
    for fname in ["analysis_results.json", "review_queue.json", "feedback_log.json", "knowledge_base.json"]:
        fpath = user_dir / fname
        if not fpath.exists():
            fpath.write_text("[]")
    # User settings
    settings_path = user_dir / "settings.json"
    if not settings_path.exists():
        settings_path.write_text("{}")


def get_user_by_id(user_id: int) -> Optional[dict]:
    conn = sqlite3.connect(str(USERS_DB))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    conn = sqlite3.connect(str(USERS_DB))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[dict]:
    conn = sqlite3.connect(str(USERS_DB))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_db_path(user_id: int) -> Path:
    """Return the path to the user's private data directory."""
    return USER_DATA_DIR / str(user_id)


def _load_json(user_id: int, filename: str) -> list:
    fpath = get_user_db_path(user_id) / filename
    if fpath.exists():
        try:
            return json.loads(fpath.read_text())
        except (json.JSONDecodeError, Exception):
            return []
    return []


def _save_json(user_id: int, filename: str, data: list):
    fpath = get_user_db_path(user_id) / filename
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(data, indent=2, default=str))


# ── Per-user data operations ──────────────────────────────────────────

def save_analysis(user_id: int, result: dict) -> int:
    results = _load_json(user_id, "analysis_results.json")
    entry_id = len(results) + 1
    result["id"] = entry_id
    result["timestamp"] = time.time()
    results.append(result)
    _save_json(user_id, "analysis_results.json", results)
    return entry_id


def get_user_analyses(user_id: int) -> list:
    return _load_json(user_id, "analysis_results.json")


def get_user_analysis(user_id: int, entry_id: int) -> Optional[dict]:
    for entry in _load_json(user_id, "analysis_results.json"):
        if entry.get("id") == entry_id:
            return entry
    return None


# ── Review queue ──────────────────────────────────────────────────────

def add_to_review_queue(user_id: int, article: dict, verdict: dict, reason: str) -> int:
    queue = _load_json(user_id, "review_queue.json")
    entry_id = len(queue) + 1
    entry = {
        "id": entry_id,
        "article": article,
        "verdict": verdict,
        "review_reason": reason,
        "status": "pending",
        "human_verdict": None,
        "human_notes": "",
        "added_at": time.time(),
        "reviewed_at": None,
    }
    queue.append(entry)
    _save_json(user_id, "review_queue.json", queue)
    return entry_id


def get_review_queue(user_id: int) -> list:
    return _load_json(user_id, "review_queue.json")


def get_pending_reviews(user_id: int) -> list:
    return [e for e in _load_json(user_id, "review_queue.json") if e.get("status") == "pending"]


def resolve_review(user_id: int, entry_id: int, human_verdict: str, notes: str = "") -> bool:
    queue = _load_json(user_id, "review_queue.json")
    for entry in queue:
        if entry["id"] == entry_id:
            entry["status"] = "resolved"
            entry["human_verdict"] = human_verdict
            entry["human_notes"] = notes
            entry["reviewed_at"] = time.time()
            _save_json(user_id, "review_queue.json", queue)
            return True
    return False


def get_review_queue_stats(user_id: int) -> dict:
    queue = _load_json(user_id, "review_queue.json")
    statuses = {}
    for e in queue:
        s = e.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    return {"total": len(queue), "by_status": statuses}


# ── Feedback ──────────────────────────────────────────────────────────

def log_feedback(user_id: int, article_text: str, original_verdict: str,
                 corrected_verdict: str, notes: str = "") -> dict:
    feedback = _load_json(user_id, "feedback_log.json")
    entry_id = len(feedback) + 1
    entry = {
        "id": entry_id,
        "article_text": article_text[:2000],
        "original_verdict": original_verdict,
        "corrected_verdict": corrected_verdict,
        "notes": notes,
        "timestamp": time.time(),
    }
    feedback.append(entry)
    _save_json(user_id, "feedback_log.json", feedback)
    return entry


def get_user_feedback(user_id: int) -> list:
    return _load_json(user_id, "feedback_log.json")


def get_feedback_stats(user_id: int) -> dict:
    entries = _load_json(user_id, "feedback_log.json")
    corrections = sum(1 for e in entries if e.get("original_verdict") != e.get("corrected_verdict"))
    return {
        "total_entries": len(entries),
        "corrections": corrections,
        "accuracy_rate": 1 - (corrections / max(len(entries), 1)),
    }


# ── Knowledge base ────────────────────────────────────────────────────

def add_kb_entry(user_id: int, claim: str, verdict: str, sources: list, article_text: str = "") -> int:
    kb = _load_json(user_id, "knowledge_base.json")
    entry_id = len(kb) + 1
    entry = {
        "id": entry_id,
        "claim": claim,
        "verdict": verdict,
        "sources": sources[:5],
        "article_text": article_text[:500],
        "added_at": time.time(),
    }
    kb.append(entry)
    _save_json(user_id, "knowledge_base.json", kb)
    return entry_id


def search_kb(user_id: int, query: str, top_k: int = 5) -> list:
    kb = _load_json(user_id, "knowledge_base.json")
    if not kb:
        return []
    query_words = set(query.lower().split())
    scored = []
    for entry in kb:
        claim_words = set(entry.get("claim", "").lower().split())
        overlap = len(query_words & claim_words)
        if overlap > 0:
            scored.append((overlap, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def get_kb_stats(user_id: int) -> dict:
    kb = _load_json(user_id, "knowledge_base.json")
    verdicts = {}
    for e in kb:
        v = e.get("verdict", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1
    return {"total_entries": len(kb), "verdict_distribution": verdicts}


# ── User settings ─────────────────────────────────────────────────────

def get_user_settings(user_id: int) -> dict:
    fpath = get_user_db_path(user_id) / "settings.json"
    if fpath.exists():
        try:
            return json.loads(fpath.read_text())
        except Exception:
            return {}
    return {}


def save_user_settings(user_id: int, settings: dict):
    fpath = get_user_db_path(user_id) / "settings.json"
    fpath.write_text(json.dumps(settings, indent=2))
