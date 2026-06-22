"""
AL TASNIM Production RAG Server — pgvector + BM25 hybrid.

One-time setup (run once after first install):
  python scripts/01_clean_chunks.py       # clean + deduplicate all_chunks.json
  python scripts/02_push_to_pgvector.py   # embed with BGE-M3, push to PostgreSQL

Run server:
  conda activate v12
  cd /home/abhay/Desktop/NGXP/tasnimv.0
  uvicorn prod_rag:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /healthz                   — status + chunk count
  POST /search  {query, top_k}    — hybrid ranked chunks
  POST /ask     {query, top_k}    — structured 7-part LLM answer + sources
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psycopg2
import requests
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Load .env file if present
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.split("#")[0].strip()   # strip inline comments
            os.environ.setdefault(_k.strip(), _v)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL      = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

PG_URL   = os.getenv("PG_URL", "postgresql://abhay@/altasnim?host=/var/run/postgresql")
PG_TABLE = os.getenv("PG_TABLE", "rag_chunks")

# LLM provider — set LLM_PROVIDER=groq in .env for cloud, =ollama for local
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL     = os.getenv("LLM_MODEL", "llama3.1:8b")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"

# Ollama (used when LLM_PROVIDER=ollama)
OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL  = os.getenv("LLM_MODEL", "llama3.1:8b")

BM25_WEIGHT  = 0.4
DENSE_WEIGHT = 0.6
RRF_K        = 60

MIN_RELEVANCE_SCORE = 0.009   # below this → rewrite query and retry
CACHE_SIM_THRESHOLD = 0.97    # cosine sim for semantic cache hit — higher = fewer false matches
CACHE_TTL_SECONDS   = 900     # 15-minute answer cache TTL

# ---------------------------------------------------------------------------
# Security — Auth, RBAC, PII sanitization, clarification gate
# ---------------------------------------------------------------------------
_RBAC_CFG_PATH = Path(__file__).parent / "config" / "rbac_config.yaml"
_RBAC_CONFIG: dict = {}
if _RBAC_CFG_PATH.exists():
    with _RBAC_CFG_PATH.open() as _f:
        _RBAC_CONFIG = yaml.safe_load(_f) or {}

# API key store — populated at startup from AL_TASNIM_API_KEYS env var
# Format: "key1:role:display_name,key2:role2:name2"
_API_KEYS: dict[str, dict] = {}
_AUTH_ENABLED = bool(os.getenv("AL_TASNIM_API_KEYS", "").strip())

def _load_api_keys() -> None:
    raw = os.getenv("AL_TASNIM_API_KEYS", "").strip()
    for entry in raw.split(","):
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) >= 2 and parts[0]:
            _API_KEYS[parts[0]] = {
                "role": parts[1],
                "name": parts[2] if len(parts) > 2 else "user",
            }

def _authenticate(request: Request) -> dict:
    """Return {role, name} or raise 401/403. Dev mode: all allowed as admin."""
    if not _AUTH_ENABLED:
        return {"role": "ngxp_admin", "name": "dev"}
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    user = _API_KEYS.get(api_key)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return user

def _allowed_confidentiality(user: dict) -> list[str]:
    """Confidentiality levels this role may access. Default: public only."""
    role = user.get("role", "read_only")
    return _RBAC_CONFIG.get("roles", {}).get(role, {}).get(
        "confidentiality", ["public"]
    )

def _allowed_tables(user: dict) -> list[str]:
    """SQL tables this role may query. ['*'] means all."""
    role = user.get("role", "read_only")
    return _RBAC_CONFIG.get("roles", {}).get(role, {}).get("tables", ["*"])

def _check_table_access(user: dict, table: str) -> bool:
    allowed = _allowed_tables(user)
    return "*" in allowed or table in allowed

# ── PII & prompt injection sanitization ──────────────────────────────────────
_PII_EMAIL   = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
_PII_PHONE   = re.compile(r'\b(?:\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{4}\b')
_PII_EMP_ID  = re.compile(r'\b(?:EMP|EID|ID)[-/]?\d{4,10}\b', re.IGNORECASE)
_INJECTION   = [re.compile(p, re.IGNORECASE) for p in [
    r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions',
    r'disregard\s+(?:your\s+)?(?:instructions|training)',
    r'you\s+are\s+now\s+(?:a\s+)?(?:different|new)',
    r'act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+)?(?:different|evil|unrestricted)',
    r'forget\s+(?:everything|your\s+training|all\s+instructions)',
    r'reveal\s+(?:your\s+)?(?:system\s+)?prompt',
    r'print\s+(?:your\s+)?(?:system\s+)?prompt',
    r'show\s+(?:me\s+)?(?:your\s+)?system\s+(?:instructions|prompt)',
    r'bypass\s+(?:your\s+)?(?:safety|filter|restriction)',
    r'jailbreak',
    r'\bDAN\s+mode\b',
    r'pretend\s+you\s+(?:are\s+)?not\s+an?\s+(?:ai|assistant)',
]]

def _sanitize_input(query: str) -> tuple[str, list[str]]:
    """Block injection; redact PII. Returns (clean_query, pii_flags_list)."""
    for pat in _INJECTION:
        if pat.search(query):
            raise HTTPException(
                status_code=400,
                detail="Query blocked: prompt injection pattern detected.",
            )
    flags: list[str] = []
    q = query
    if _PII_EMAIL.search(q):
        q = _PII_EMAIL.sub("[EMAIL REDACTED]", q); flags.append("pii_email")
    if _PII_PHONE.search(q):
        q = _PII_PHONE.sub("[PHONE REDACTED]", q); flags.append("pii_phone")
    if _PII_EMP_ID.search(q):
        q = _PII_EMP_ID.sub("[EMP-ID REDACTED]", q); flags.append("pii_emp_id")
    if flags:
        print(f"[security] PII redacted from query: {flags}", flush=True)
    return q, flags

# ── Mandatory clarification gate ─────────────────────────────────────────────
_VAGUE_OPS = re.compile(
    r'\b(when\s+will|why\s+is|what.{0,20}status|what.{0,20}progress|'
    r'give\s+me\s+an?\s+update|any\s+updates?|how\s+is\s+(?:it|the)\s+(?:going|progress)|'
    r'is\s+it\s+delayed|when\s+(?:will|is)\s+it|behind\s+schedule|what\'?s?\s+happening)\b',
    re.IGNORECASE,
)
_SPECIFIC_ID = re.compile(
    r'\b([A-Z]{2,6}[-/]\d{3,6}|SWER\w+|nimr|lekhwair|marmul|qarn.?alam|'
    r'[A-Z]{2,6}\d{3,6}|activity\s*code\s*\w+|\d{4,}|all\s+wells|total)\b',
    re.IGNORECASE,
)
_KNOWLEDGE_Q = re.compile(
    r'\b(what\s+is|what\s+does|explain|define|describe|how\s+does|'
    r'tell\s+me\s+about|meaning\s+of|which\s+milestone|difference\s+between)\b',
    re.IGNORECASE,
)

def _check_vague_query(query: str) -> str | None:
    """Return a clarification prompt if the query is too vague, else None."""
    if len(query.strip()) < 8:
        return ("Please provide more detail. Include a Well ID (e.g. NIMR-1687), "
                "Rig number (e.g. SWER101), activity code, or cluster name.")
    if _KNOWLEDGE_Q.search(query):
        return None
    if _VAGUE_OPS.search(query) and not _SPECIFIC_ID.search(query):
        return (
            "Your question needs more detail to return accurate data. Please specify:\n"
            "  • Well ID (e.g. NIMR-1687) or Rig number (e.g. SWER101), and\n"
            "  • What exactly you want — progress %, status, completion date, or delay reason.\n"
            "Example: \"What is the current progress of well NIMR-1687?\""
        )
    return None

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    id: str
    text: str
    source: str
    score: float
    meta: dict


# ---------------------------------------------------------------------------
# PostgreSQL connection
# ---------------------------------------------------------------------------
_pg_conn: psycopg2.extensions.connection | None = None


def _get_pg_conn() -> psycopg2.extensions.connection:
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(PG_URL)
        _pg_conn.autocommit = True
        register_vector(_pg_conn)
    return _pg_conn


def load_chunks_from_pg() -> list[Chunk]:
    """Load id + source + content + metadata from PostgreSQL for BM25 index.
    Parent chunks (is_parent=TRUE) are excluded from BM25 — they are context-only.
    Embeddings stay in the DB — dense search goes via pgvector SQL."""
    conn = _get_pg_conn()
    print(f"[load] Fetching chunks from {PG_TABLE} …", end=" ", flush=True)
    with conn.cursor() as cur:
        # Exclude parent-only chunks from BM25 — they have no embedding
        cur.execute(f"""
            SELECT id, source, content, metadata, parent_id
            FROM {PG_TABLE}
            WHERE is_parent IS NOT TRUE
        """)
        rows = cur.fetchall()

    chunks = []
    for r in rows:
        meta = r[3] if isinstance(r[3], dict) else (json.loads(r[3]) if r[3] else {})
        chunks.append(Chunk(id=r[0], source=r[1], text=r[2], meta=meta))

    sources = len({c.source for c in chunks})
    print(f"{len(chunks):,} chunks from {sources} sources")
    return chunks


def fetch_parent_content(parent_id: str | None) -> str | None:
    """Parent-Document Retriever: fetch the larger parent context for a child chunk."""
    if not parent_id:
        return None
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT content FROM {PG_TABLE} WHERE id = %s AND is_parent = TRUE",
            (parent_id,),
        )
        row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Embedding — BGE-M3 on GPU (Groq handles LLM, so GPU is free for embeddings)
# ---------------------------------------------------------------------------
_st_model = None


def _get_model():
    global _st_model
    if _st_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embed] Loading {EMBED_MODEL} on {device.upper()} …")
        _st_model = SentenceTransformer(
            EMBED_MODEL,
            model_kwargs={"torch_dtype": torch.float16},
            device=device,
        )
        print("[embed] Model ready")
    return _st_model


def embed_query(text: str) -> np.ndarray:
    m = _get_model()
    return m.encode(
        [f"Represent this sentence for searching relevant passages: {text}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


# ---------------------------------------------------------------------------
# BM25 index (in-memory, built from text loaded at startup)
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    print("[index] Building BM25 index …", end=" ", flush=True)
    idx = BM25Okapi([_tokenize(c.text) for c in chunks])
    print("done")
    return idx


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def pg_dense_search(
    qvec: np.ndarray, k: int, allowed_conf: list[str] | None = None
) -> list[Hit]:
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        # RBAC pre-filter: access check happens INSIDE the DB query,
        # before any chunks are returned to Python.
        if allowed_conf:
            cur.execute(
                f"""
                SELECT id, source, content, metadata,
                       1 - (embedding <=> %s::vector) AS score,
                       parent_id
                FROM {PG_TABLE}
                WHERE is_parent IS NOT TRUE
                  AND (
                      metadata->>'confidentiality_level' IS NULL
                      OR metadata->>'confidentiality_level' = ANY(%s)
                  )
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (qvec.tolist(), allowed_conf, qvec.tolist(), k),
            )
        else:
            cur.execute(
                f"""
                SELECT id, source, content, metadata,
                       1 - (embedding <=> %s::vector) AS score,
                       parent_id
                FROM {PG_TABLE}
                WHERE is_parent IS NOT TRUE
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (qvec.tolist(), qvec.tolist(), k),
            )
        rows = cur.fetchall()

    hits = []
    for r in rows:
        meta = r[3] if isinstance(r[3], dict) else (json.loads(r[3]) if r[3] else {})
        parent_id = r[5]
        # Parent-Document Retriever: use parent context if available
        if parent_id:
            parent_text = fetch_parent_content(parent_id)
            if parent_text:
                meta["_used_parent"] = True
                hits.append(Hit(r[0], parent_text, r[1], float(r[4]), meta))
                continue
        hits.append(Hit(r[0], r[2], r[1], float(r[4]), meta))
    return hits


def bm25_search(query: str, bm25: BM25Okapi,
                chunks: list[Chunk], k: int) -> list[Hit]:
    scores = bm25.get_scores(_tokenize(query))
    idx = np.argsort(-scores)[:k]
    return [Hit(chunks[i].id, chunks[i].text, chunks[i].source,
                float(scores[i]), chunks[i].meta) for i in idx]


def hybrid_search(
    query: str,
    qvec: np.ndarray,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    top_k: int = 8,
    allowed_conf: list[str] | None = None,
) -> list[Hit]:
    """RRF fusion of pgvector dense search + BM25 keyword search."""
    cand     = max(top_k * 8, 80)
    den_hits = pg_dense_search(qvec, cand, allowed_conf=allowed_conf)
    bm_hits  = bm25_search(query, bm25, chunks, cand)

    scores: dict[str, float] = {}
    hit_map: dict[str, Hit] = {}
    for rank, h in enumerate(den_hits, 1):
        scores[h.id] = scores.get(h.id, 0.0) + DENSE_WEIGHT / (RRF_K + rank)
        hit_map[h.id] = h
    for rank, h in enumerate(bm_hits, 1):
        scores[h.id] = scores.get(h.id, 0.0) + BM25_WEIGHT / (RRF_K + rank)
        hit_map[h.id] = h

    total  = DENSE_WEIGHT + BM25_WEIGHT
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [Hit(hit_map[cid].id, hit_map[cid].text, hit_map[cid].source,
                sc / total, hit_map[cid].meta)
            for cid, sc in ranked]


# ---------------------------------------------------------------------------
# Contextual Compression
# Extracts only relevant sentences from each chunk using BGE-M3 similarity.
# No extra LLM call — fast, local, uses the same embedding model.
# Prevents context overflow: LLM gets clean, focused evidence only.
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_COMPRESS_THRESHOLD = 0.50   # cosine sim to query — below this = not relevant
_MIN_SENTENCES_KEEP = 2      # always keep at least 2 sentences per chunk


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text.strip())
    out = []
    for p in parts:
        for sub in p.split('\n'):
            s = sub.strip()
            if len(s) > 30:
                out.append(s)
    return out


def compress_hits(hits: list[Hit], qvec: np.ndarray) -> list[Hit]:
    """
    Contextual Compression: for each retrieved chunk, keep only sentences
    that are semantically close to the query. Adjacent sentences are included
    for readability. Chunks with zero relevant sentences are dropped entirely.
    """
    model = _get_model()
    compressed: list[Hit] = []

    for hit in hits:
        sentences = _split_sentences(hit.text)
        if len(sentences) <= _MIN_SENTENCES_KEEP:
            compressed.append(hit)
            continue

        # Embed sentences and score against query
        sent_vecs = model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sims = (sent_vecs @ qvec).tolist()

        # Find relevant sentence indices (above threshold)
        relevant = [i for i, s in enumerate(sims) if s >= _COMPRESS_THRESHOLD]

        # If nothing passes threshold, keep top-2 by score
        if not relevant:
            top2 = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:_MIN_SENTENCES_KEEP]
            relevant = top2

        # Expand: include one sentence before/after each relevant sentence for context
        expanded: set[int] = set()
        for idx in relevant:
            if idx > 0:
                expanded.add(idx - 1)
            expanded.add(idx)
            if idx < len(sentences) - 1:
                expanded.add(idx + 1)

        compressed_text = " ".join(sentences[i] for i in sorted(expanded))
        compressed.append(Hit(hit.id, compressed_text, hit.source, hit.score, hit.meta))

    return compressed


# ---------------------------------------------------------------------------
# Semantic answer cache (in-memory, TTL-based)
# ---------------------------------------------------------------------------
_cache_lock: threading.Lock = threading.Lock()
_answer_cache: list[dict] = []


def _cache_lookup(qvec: np.ndarray) -> dict | None:
    now = time.time()
    with _cache_lock:
        for entry in _answer_cache:
            if now - entry["ts"] > CACHE_TTL_SECONDS:
                continue
            if float(qvec @ entry["vec"]) >= CACHE_SIM_THRESHOLD:
                return entry
    return None


def _cache_store(qvec: np.ndarray, query: str,
                 answer: dict[str, str], sources: list[str]) -> None:
    now = time.time()
    with _cache_lock:
        _answer_cache[:] = [e for e in _answer_cache
                             if now - e["ts"] <= CACHE_TTL_SECONDS]
        _answer_cache.append({"vec": qvec, "query": query,
                               "answer": answer, "sources": sources, "ts": now})


# ---------------------------------------------------------------------------
# Unified LLM call — Groq (cloud) or Ollama (local)
# ---------------------------------------------------------------------------
def _call_llm(system: str, user: str, timeout: int = 180) -> str:
    """Call the configured LLM provider and return the response text."""
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            raise RuntimeError("GROQ_API_KEY not set in .env file")
        delays = [10, 30, 60]
        for attempt, delay in enumerate(delays + [None], 1):
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                if delay is None:
                    # All retries exhausted — return a structured error string
                    # so FastAPI returns 200 with a graceful message instead of 500
                    return '{"direct_answer": "Service temporarily busy due to rate limiting. Please retry in 30 seconds.", "confidence_level": "Low", "evidence": "", "sources": []}'
                import logging
                logging.getLogger("prod_rag").warning(
                    "Groq 429 — waiting %ds (attempt %d)", delay, attempt
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    else:
        # Ollama fallback
        full_prompt = f"{system}\n\n{user}"
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": 2048,
                    "temperature": 0,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        return raw


# ---------------------------------------------------------------------------
# Per-request trace — one structured line per /ask call to stdout
# ---------------------------------------------------------------------------
def _trace(route: str, latency_ms: float, **kv) -> None:
    parts = [f"[trace] route={route}", f"latency={latency_ms:.0f}ms"]
    parts += [f"{k}={v}" for k, v in kv.items()]
    print("  ".join(parts), flush=True)


# ---------------------------------------------------------------------------
# Safe SQL Compiler — all configuration loaded from config/sql_config.yaml
#
# Architecture (mandated):
#   1. LLM outputs structured JSON intent — never raw SQL
#   2. Backend Safe SQL Compiler maps intent → pre-written SQL template
#   3. Strict validation against config allowlists — SELECT only, row cap
#   4. Parameterised execution — user values never interpolated into SQL
#   5. Results returned as plain text context → LLM synthesises the answer
# ---------------------------------------------------------------------------

def _load_sql_config() -> dict:
    """Load sql_config.yaml at startup. Crash loudly if it is missing."""
    import yaml
    cfg_path = Path(__file__).parent / "config" / "sql_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing SQL config: {cfg_path}\n"
            "Create config/sql_config.yaml before starting the server."
        )
    with cfg_path.open() as f:
        return yaml.safe_load(f)

_SQL_CFG: dict = _load_sql_config()

# Everything below reads from the config — no values hardcoded in Python
_ALLOWED_TABLE:     str      = _SQL_CFG.get("default_table", "well_monitoring")
_SOURCE_FILE:       str      = _SQL_CFG.get("default_source_file", "")
_MAX_ROWS:          int      = int(_SQL_CFG["max_rows"])
_SQL_TRIGGER_WORDS: set[str] = set(_SQL_CFG["trigger_words"])
_INTENT_CFG:      dict     = _SQL_CFG.get("intents", {})

# Config-driven fast-path routing table — built once at startup from sql_config.yaml.
# Each entry: (keyword, intent_name, extract_param)
# Longer keywords are checked first so "buffer status" wins over bare "buffer".
# To add a new fast-path: add fast_path_keywords to the intent in sql_config.yaml.
# No Python changes needed.
_FAST_PATH: list[tuple[str, str, str]] = []
for _iname, _icfg in _INTENT_CFG.items():
    for _kw in _icfg.get("fast_path_keywords", []):
        _FAST_PATH.append((_kw.lower(), _iname, _icfg.get("fast_path_extract", "")))
_FAST_PATH.sort(key=lambda x: -len(x[0]))   # longest match wins

# Auto-add select_columns and allowed_filter_columns from every intent as trigger words.
# This means any column name referenced in sql_config.yaml intents can route to SQL.
for _icfg in _INTENT_CFG.values():
    for _col in _icfg.get("select_columns", []):
        _SQL_TRIGGER_WORDS.add(_col.replace("_", " ").lower())
    for _col in _icfg.get("allowed_filter_columns", []):
        _SQL_TRIGGER_WORDS.add(_col.replace("_", " ").lower())


def _load_live_schema() -> dict[str, list[tuple[str, str]]]:
    """Read column names + data types for every non-system table directly from the DB.
    Returns {table_name: [(col_name, data_type), ...]} — always current, never stale."""
    try:
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND column_name != '_row_id'
                  AND table_name NOT IN ('rag_chunks', 'spatial_ref_sys')
                ORDER BY table_name, ordinal_position
                """
            )
            schema: dict[str, list[tuple[str, str]]] = {}
            for table, col, dtype in cur.fetchall():
                schema.setdefault(table, []).append((col, dtype))
        return schema
    except Exception as e:
        print(f"[schema] Could not load live schema: {e}")
        return {}


# Load once at startup — shared by _build_intent_description and future callers.
_LIVE_SCHEMA: dict[str, list[tuple[str, str]]] = _load_live_schema()


def _build_intent_description() -> str:
    """Build the LLM intent-classifier prompt from config + live DB schema.
    Fully driven by sql_config.yaml intents + live DB schema. No hardcoded values."""
    lines = ["Available intents and their parameters:\n"]
    for intent_name, cfg in _INTENT_CFG.items():
        lines.append(intent_name)
        lines.append(f'  description: {cfg.get("description", "")}')
        # Column hints: prefer config select_columns, fall back to live DB schema
        if "select_columns" in cfg:
            lines.append(f'  columns: {", ".join(cfg["select_columns"])}')
        else:
            table = cfg.get("table", _ALLOWED_TABLE)
            live_cols = [c for c, _ in _LIVE_SCHEMA.get(table, [])][:20]
            if live_cols:
                lines.append(f'  columns (from db): {", ".join(live_cols)}')
        if "allowed_group_by" in cfg:
            lines.append(f'  group_by options: {", ".join(cfg["allowed_group_by"])}')
        if "allowed_status_columns" in cfg:
            lines.append(f'  status_column options: {" | ".join(cfg["allowed_status_columns"])}')
        if "allowed_filter_columns" in cfg:
            lines.append(f'  filter_columns: {", ".join(cfg["allowed_filter_columns"])}')
        lines.append("")
    return "\n".join(lines)


def _compile_sql(intent: dict) -> tuple[str, dict]:
    """
    Generic config-driven SQL compiler.
    All data knowledge (tables, columns, sql patterns) lives in sql_config.yaml.
    Python contains only the query-building logic — zero column names, zero table names.
    To add a new intent: add it to sql_config.yaml only. No Python changes needed.
    """
    name = (intent.get("intent") or "").strip()
    intent_cfg = _INTENT_CFG.get(name)
    if not intent_cfg:
        raise ValueError(f"Unknown intent '{name}'. Allowed: {list(_INTENT_CFG.keys())}")

    table    = intent_cfg.get("table", _ALLOWED_TABLE)
    limit    = min(int(intent.get("limit", intent_cfg.get("default_limit", 20))), _MAX_ROWS)
    sql_type = intent_cfg.get("sql_type", "list")

    # Per-intent validation: column must be in allowed_filter_columns and exist in live schema
    live_col_map: dict[str, str] = dict(_LIVE_SCHEMA.get(table, []))
    allowed_filters = set(intent_cfg.get("allowed_filter_columns", []))

    def validate_col(col: str) -> None:
        if not col:
            return
        if allowed_filters and col not in allowed_filters:
            raise ValueError(
                f"Column '{col}' not in allowed_filter_columns for intent '{name}'. "
                f"Allowed: {sorted(allowed_filters)} — see config/sql_config.yaml."
            )
        if live_col_map and col not in live_col_map:
            raise ValueError(f"Column '{col}' does not exist in table '{table}'.")

    def build_cols() -> str:
        cfg_cols = intent_cfg.get("select_columns")
        if cfg_cols:
            return ", ".join(cfg_cols)
        live = [c for c in live_col_map if not c.startswith("_")][:20]
        return ", ".join(live) if live else "*"

    # ── count ──────────────────────────────────────────────────────────────────
    if sql_type == "count":
        col = intent.get("filter_column", "")
        val = intent.get("filter_value", "")
        validate_col(col)
        if col and val:
            return (f"SELECT COUNT(*) AS count FROM {table} WHERE {col} ILIKE %(val)s",
                    {"val": f"%{val}%"})
        return f"SELECT COUNT(*) AS count FROM {table}", {}

    # ── list ───────────────────────────────────────────────────────────────────
    if sql_type == "list":
        col = intent.get("filter_column", "")
        val = intent.get("filter_value", "")
        validate_col(col)
        cols   = build_cols()
        order  = intent_cfg.get("order_by", "")
        order_clause = f" ORDER BY {order}" if order else ""
        if col and val:
            return (f"SELECT {cols} FROM {table} WHERE {col} ILIKE %(val)s{order_clause} LIMIT {limit}",
                    {"val": f"%{val}%"})
        return f"SELECT {cols} FROM {table}{order_clause} LIMIT {limit}", {}

    # ── group_by ───────────────────────────────────────────────────────────────
    if sql_type == "group_by":
        group       = intent.get("group_by", intent_cfg.get("default_group_by", ""))
        allowed_grp = set(intent_cfg.get("allowed_group_by", []))
        if group and allowed_grp and group not in allowed_grp:
            raise ValueError(f"group_by '{group}' not allowed: {sorted(allowed_grp)}")
        agg_col = intent_cfg.get("aggregate_column", "")
        if group and agg_col:
            return (f"SELECT {group}, COUNT(*) AS count, "
                    f"ROUND(AVG({agg_col})::numeric * 100, 1) AS avg_pct "
                    f"FROM {table} GROUP BY {group} ORDER BY avg_pct DESC LIMIT {limit}", {})
        if group:
            return (f"SELECT {group}, COUNT(*) AS count FROM {table} "
                    f"WHERE {group} IS NOT NULL GROUP BY {group} ORDER BY count DESC LIMIT {limit}", {})
        if agg_col:
            return (f"SELECT COUNT(*) AS total, "
                    f"ROUND(AVG({agg_col})::numeric * 100, 1) AS avg_pct, "
                    f"ROUND(MIN({agg_col})::numeric * 100, 1) AS min_pct, "
                    f"ROUND(MAX({agg_col})::numeric * 100, 1) AS max_pct "
                    f"FROM {table}", {})
        return f"SELECT COUNT(*) AS total FROM {table}", {}

    # ── count_by_column ────────────────────────────────────────────────────────
    if sql_type == "count_by_column":
        grp_col = intent_cfg.get("group_column", "")
        if not grp_col:
            raise ValueError(f"Intent '{name}' missing group_column in sql_config.yaml")
        return (f"SELECT {grp_col}, COUNT(*) AS count FROM {table} "
                f"WHERE {grp_col} IS NOT NULL GROUP BY {grp_col} ORDER BY count DESC", {})

    # ── ranked ─────────────────────────────────────────────────────────────────
    if sql_type == "ranked":
        rank_col = intent_cfg.get("rank_column", "")
        if not rank_col:
            raise ValueError(f"Intent '{name}' missing rank_column in sql_config.yaml")
        direction = intent.get("direction", "top").lower()
        order = "DESC" if direction == "top" else "ASC"
        cols  = build_cols()
        return (f"SELECT {cols} FROM {table} "
                f"WHERE {rank_col} IS NOT NULL ORDER BY {rank_col} {order} LIMIT {limit}", {})

    # ── status_breakdown ───────────────────────────────────────────────────────
    if sql_type == "status_breakdown":
        allowed_sc = set(intent_cfg.get("allowed_status_columns", []))
        col = intent.get("status_column", next(iter(allowed_sc), ""))
        if allowed_sc and col not in allowed_sc:
            raise ValueError(f"status_column '{col}' not allowed: {sorted(allowed_sc)}")
        if not col:
            raise ValueError(f"Intent '{name}' requires a status_column parameter.")
        return (f"SELECT {col} AS status, COUNT(*) AS count FROM {table} "
                f"WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY count DESC", {})

    # ── range ──────────────────────────────────────────────────────────────────
    if sql_type == "range":
        range_col = intent_cfg.get("range_column", "")
        if not range_col:
            raise ValueError(f"Intent '{name}' missing range_column in sql_config.yaml")
        raw_min = float(intent.get("min_pct", intent.get("min_value", 0)))
        raw_max = float(intent.get("max_pct", intent.get("max_value", 100)))
        min_val = raw_min / 100 if raw_min > 1 else raw_min
        max_val = raw_max / 100 if raw_max > 1 else raw_max
        return (f"SELECT COUNT(*) AS count, "
                f"ROUND(MIN({range_col})::numeric*100,1) AS min_pct, "
                f"ROUND(MAX({range_col})::numeric*100,1) AS max_pct "
                f"FROM {table} WHERE {range_col} BETWEEN %(min_val)s AND %(max_val)s",
                {"min_val": min_val, "max_val": max_val})

    # ── detail ─────────────────────────────────────────────────────────────────
    if sql_type == "detail":
        id_col   = intent_cfg.get("id_column", "")
        name_col = intent_cfg.get("name_column", "")
        id_val   = str(intent.get("pdo_well_id", intent.get("well_id",
                       intent.get("id_value", "")))).strip()
        name_val = str(intent.get("well_name", intent.get("name_value", ""))).strip()
        cols     = build_cols()
        if id_val and id_col:
            try:
                return (f"SELECT {cols} FROM {table} WHERE {id_col} = %(id)s LIMIT 5",
                        {"id": int(id_val)})
            except ValueError:
                return (f"SELECT {cols} FROM {table} WHERE {id_col}::text ILIKE %(id)s LIMIT 5",
                        {"id": f"%{id_val}%"})
        if name_val and name_col:
            return (f"SELECT {cols} FROM {table} WHERE {name_col} ILIKE %(name)s LIMIT 5",
                    {"name": f"%{name_val}%"})
        raise ValueError(f"Intent '{name}' requires a lookup value (id or name).")

    # ── multi_filter ───────────────────────────────────────────────────────────
    if sql_type == "multi_filter":
        cols       = build_cols()
        conditions: list[str] = []
        params:     dict      = {}
        for fc in intent_cfg.get("allowed_filter_columns", []):
            val = str(intent.get(fc, "")).strip()
            if not val:
                continue
            col_type = live_col_map.get(fc, "text")
            if val.isdigit():
                if any(t in col_type for t in ("int", "numeric", "double", "real")):
                    conditions.append(f"{fc} = %({fc})s")
                    params[fc] = int(val)
                else:
                    conditions.append(f"{fc}::text = %({fc})s")
                    params[fc] = val
            else:
                conditions.append(f"{fc} ILIKE %({fc})s")
                params[fc] = f"%{val}%"
        where        = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        order        = intent_cfg.get("order_by", "")
        order_clause = f" ORDER BY {order}" if order else ""
        return f"SELECT {cols} FROM {table}{where}{order_clause} LIMIT {limit}", params

    raise ValueError(
        f"Unknown sql_type '{sql_type}' for intent '{name}'. "
        f"Valid: count, list, group_by, count_by_column, ranked, "
        f"status_breakdown, range, detail, multi_filter"
    )


# Words that force RAG routing — loaded from sql_config.yaml exclude_words list.
# To add a new exclusion: edit config/sql_config.yaml, no Python changes needed.
_SQL_EXCLUDE_WORDS: set[str] = set(_SQL_CFG.get("exclude_words", []))

# All fast-path keywords across every intent — used as a secondary SQL trigger
_ALL_FAST_PATH_KWS: set[str] = set()
for _iname, _icfg in _INTENT_CFG.items():
    for _kw in _icfg.get("fast_path_keywords", []):
        _ALL_FAST_PATH_KWS.add(_kw.lower())

def _is_sql_query(question: str) -> bool:
    q = question.lower()
    # Workforce / project scope / meeting questions always go to RAG
    if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in _SQL_EXCLUDE_WORDS):
        return False
    # Primary trigger: global trigger_words list (well counts, rig, progress, etc.)
    if any(re.search(r'\b' + re.escape(kw) + r'\b', q) for kw in _SQL_TRIGGER_WORDS):
        return True
    # Secondary trigger: fast_path_keywords from any intent (milestone status, crew group, etc.)
    return any(kw in q for kw in _ALL_FAST_PATH_KWS)


def _check_sql_table_exists() -> bool:
    try:
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s",
                (_ALLOWED_TABLE,),
            )
            if cur.fetchone()[0] == 0:
                return False
            cur.execute(f"SELECT COUNT(*) FROM {_ALLOWED_TABLE}")
            return cur.fetchone()[0] > 0
    except Exception:
        return False


_ACTIVITY_CODE_RE = re.compile(
    r'\b([A-Z]-[A-Z]-[A-Z0-9]+-[A-Z0-9]+-\d+)\b', re.IGNORECASE
)

# Matches bare 5-6 digit PDO well IDs: "well id 37797", "pdo 37797", "well no 37797"
_WELL_ID_RE = re.compile(r'\b(?:well\s+id|pdo(?:\s+well)?|well\s+no)[^\d]*(\d{4,6})\b', re.IGNORECASE)

def _classify_intent(question: str) -> dict:
    """LLM translates question → structured JSON intent. Never writes SQL."""
    q_lower = question.lower()

    # ── Config-driven fast-path routing ────────────────────────────────────────
    # Keywords and their target intents come from sql_config.yaml (fast_path_keywords).
    # Longer keywords are checked first (sorted at startup) so "buffer status" wins
    # over bare "buffer". To add a new source table / intent: edit the YAML only.
    #
    # Special case: activity_code regex also triggers activity_lookup even when no
    # keyword matches (e.g. bare code "F-C-FDI-EXC-13" without "activity code" phrase).
    if _ACTIVITY_CODE_RE.search(question):
        intent: dict = {"intent": "activity_lookup"}
        code_m = _ACTIVITY_CODE_RE.search(question)
        if code_m:
            intent["activity_code"] = code_m.group(1).upper()
        print(f"[sql] Fast-path (activity_code regex): {intent}")
        return intent

    for keyword, intent_name, extract_param in _FAST_PATH:
        if keyword in q_lower:
            intent = {"intent": intent_name}
            # Extract parameters based on the declared extraction type
            if extract_param == "activity_code":
                code_m = _ACTIVITY_CODE_RE.search(question)
                if code_m:
                    intent["activity_code"] = code_m.group(1).upper()
                else:
                    disc_m = re.search(r'\b(civil|mechanical|electrical|e&i)\b', q_lower)
                    if disc_m:
                        intent["discipline"] = disc_m.group(1).title()
                    else:
                        kw_m = re.search(
                            r'(?:about|for|like|keyword|related to)\s+["\']?([A-Za-z ]+)["\']?',
                            q_lower)
                        if kw_m:
                            intent["keyword"] = kw_m.group(1).strip()
            elif extract_param == "pdo_well_id":
                well_id_m = _WELL_ID_RE.search(question)
                if well_id_m:
                    intent["pdo_well_id"] = well_id_m.group(1)
                rig_m = re.search(r'\b(swer\w+|swerig\w+)\b', q_lower)
                if rig_m:
                    intent["rig_no"] = rig_m.group(1).upper()
            print(f"[sql] Fast-path '{keyword}' → {intent}")
            return intent

    # Fast-path: bare well ID lookup — "well id 37797", "pdo well 37797"
    well_id_m = _WELL_ID_RE.search(question)
    if well_id_m:
        intent = {"intent": "well_detail", "pdo_well_id": well_id_m.group(1)}
        print(f"[sql] Fast-path well-ID: {intent}")
        return intent

    system = (
        "You are an intent classifier for an AL TASNIM well operations database. "
        "Translate the user question into a structured JSON intent. "
        "Return ONLY valid JSON — no explanation, no markdown, no code fences.\n\n"
        + _build_intent_description()
    )
    user = f"Question: {question}\n\nJSON intent:"
    try:
        raw = _call_llm(system, user, timeout=60)
        raw = re.sub(r'```[a-z]*', '', raw).strip().strip('`').strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            raise ValueError("No JSON object in LLM response.")
        result = json.loads(m.group())
        if not result.get("intent"):
            raise ValueError(f"LLM returned null/empty intent: {result}")
        return result
    except Exception as e:
        print(f"[sql] Intent classification failed: {e}")
        return {"intent": "well_count"}


def _intent_source_file(intent: dict) -> str:
    """Return the correct source file for this intent (activity_master vs well_monitoring)."""
    name       = (intent.get("intent") or "").strip()
    intent_cfg = _INTENT_CFG.get(name, {})
    # If the intent config declares its own table, look up that table's source in sql_config
    table = intent_cfg.get("table", _ALLOWED_TABLE)
    for t in _SQL_CFG.get("tables", []):
        if t.get("name") == table:
            return t.get("source_file", _SOURCE_FILE)
    return _SOURCE_FILE


def sql_answer(question: str) -> dict[str, str]:
    """
    Safe SQL flow (config-driven, zero hardcoded values):
      1. LLM → structured intent JSON
      2. Config-driven Safe SQL Compiler → parameterised SQL
      3. Parameterised execution → DB results
      4. DB results → LLM for answer synthesis
    """
    intent = _classify_intent(question)
    print(f"[sql] Intent: {intent}")
    source_file = _intent_source_file(intent)

    try:
        sql, params = _compile_sql(intent)
    except ValueError as e:
        return {
            "direct_answer": f"Query could not be compiled: {e}",
            "evidence": f"Intent: {intent}",
            "source_citation": source_file,
            "confidence_level": "Low",
            "assumptions": "N/A",
            "risk_limitation": str(e),
            "recommended_next_action": "Rephrase using a supported query type or update config/sql_config.yaml.",
        }

    print(f"[sql] SQL: {sql[:120]}")

    try:
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols  = [d[0] for d in cur.description] if cur.description else []
            rows  = cur.fetchmany(_MAX_ROWS)
            total = cur.rowcount
    except Exception as e:
        return {
            "direct_answer": f"Database query failed: {e}",
            "evidence": "N/A",
            "source_citation": source_file,
            "confidence_level": "Low",
            "assumptions": "N/A",
            "risk_limitation": "Check that the table exists: python scripts/09_universal_ingest.py",
            "recommended_next_action": "Re-run the ingest pipeline to load the required table.",
        }

    if not rows:
        return {
            "direct_answer": "No records found matching your criteria.",
            "evidence": f"Intent: {intent}",
            "source_citation": source_file,
            "confidence_level": "High",
            "assumptions": "None",
            "risk_limitation": "Filter may be too narrow or data may not cover this period.",
            "recommended_next_action": "Try a broader filter or check the data coverage period.",
        }

    header       = " | ".join(cols)
    result_lines = [header, "-" * len(header)]
    for r in rows:
        result_lines.append(" | ".join("NULL" if v is None else str(v) for v in r))
    result_text = "\n".join(result_lines)
    if total > _MAX_ROWS:
        result_text += f"\n(showing first {_MAX_ROWS} of {total} rows)"

    system = (
        "You are a data analyst for AL TASNIM Enterprises LLC, an oil and gas civil "
        "construction company in Oman. Report the query results using EXACTLY these "
        "7 section headers, each on its own line followed by a colon, in this order:\n\n"
        "DIRECT ANSWER:\n"
        "EVIDENCE:\n"
        "SOURCE / CITATION:\n"
        "CONFIDENCE LEVEL:\n"
        "ASSUMPTIONS:\n"
        "RISK / LIMITATION:\n"
        "RECOMMENDED NEXT ACTION:\n\n"
        "No markdown, no asterisks, no bullet points. Report only what the data shows. "
        "Do not rename, skip, or reorder any section header."
    )
    user = (
        f"Question: {question}\n\n"
        f"Database query results from {source_file}:\n{result_text}\n\n"
        "Report the results using the 7-section format above."
    )
    try:
        raw = _call_llm(system, user, timeout=120)
    except Exception:
        raw = ""

    # Detect safety-filter refusals from llama3 (e.g. "I cannot provide information...")
    _REFUSAL_PATTERNS = ("i cannot", "i can't", "i'm unable", "i am unable",
                         "not able to provide", "against my")
    is_refusal = raw and any(p in raw.lower()[:120] for p in _REFUSAL_PATTERNS)

    if raw and not is_refusal:
        parsed = _parse_structured_answer(raw)
    else:
        # Build a clean answer directly from the SQL result table
        intent_name = (intent.get("intent") or "").strip()
        table_name  = _INTENT_CFG.get(intent_name, {}).get("table", _ALLOWED_TABLE)
        parsed = {
            "direct_answer":           result_text,
            "evidence":                f"Direct SQL query against {table_name}",
            "source_citation":         source_file,
            "confidence_level":        "High",
            "assumptions":             "Data is current as of the last ingestion run.",
            "risk_limitation":         "Shows a maximum of 50 rows.",
            "recommended_next_action": "Drill down with a more specific filter if needed.",
        }

    if not parsed.get("source_citation"):
        parsed["source_citation"] = source_file
    return parsed


# ---------------------------------------------------------------------------
# Query rewriter (self-correcting: called when retrieval scores are poor)
# ---------------------------------------------------------------------------
def _rewrite_query(original: str) -> str:
    system = (
        "You are a search query optimizer for an oil & gas operations knowledge base. "
        "Return ONLY the rephrased query — no explanation, no quotes."
    )
    user = (
        "Rephrase into a keyword-rich search query focusing on technical terms, "
        f"KPI names, Well IDs, or activity codes.\n\nOriginal: {original}\nRephrased:"
    )
    try:
        rewritten = _call_llm(system, user, timeout=30)
        return rewritten if rewritten else original
    except Exception:
        return original


# ---------------------------------------------------------------------------
# LLM answer — 7-part AL TASNIM structured output
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an AI Operational Intelligence Assistant for AL TASNIM, supporting well delivery, drilling progress, and operational intelligence.

CRITICAL RULES — STRICTLY ENFORCED:
1. Answer ONLY from the exact values written in the context provided below. Never guess, estimate, calculate, or infer any value.
2. If the exact answer is NOT explicitly present in the context, write exactly this sentence under DIRECT ANSWER:
   "NOT FOUND IN CONTEXT — the retrieved documents do not contain a specific answer to this question."
   Then write "N/A" for EVERY remaining section. Do not attempt to infer, approximate, or reason beyond what is written.
3. SOURCE / CITATION: List ONLY the exact filenames that appear inside the [Source: ...] tags in the provided context.
   Never invent, modify, abbreviate, or combine source names. If a filename does not appear in a [Source: ...] tag, do not mention it.
4. Never fabricate employee names, numbers, KPI values, or document names — even if they seem plausible.
5. Use NO markdown — no asterisks, no bold, no dashes, no bullet points. Plain text only.
6. You MUST use EXACTLY these 7 headers in EXACTLY this order. Do not rename, abbreviate, reorder, or skip any header.

DIRECT ANSWER:
[2-3 sentences directly answering the question using ONLY values explicitly stated in the context]

EVIDENCE:
[Quote the exact data points from the context that support the answer. If not found, write N/A]

SOURCE / CITATION:
[ONLY exact filenames from the [Source: ...] tags — nothing else, no invented names]

CONFIDENCE LEVEL:
[High / Medium / Low — one sentence explaining how directly the context answers this question]

ASSUMPTIONS:
[Any assumptions made. Write "None" if not applicable]

RISK / LIMITATION:
[Data gaps, missing values, date staleness, or incomplete records found in the context]

RECOMMENDED NEXT ACTION:
[One specific, actionable next step for an operations manager]"""


# Ordered list of (field_name, regex_pattern).
# Patterns are anchored to line-start (^) and treat the colon as optional (:?)
# so they match both "DIRECT ANSWER:" and "DIRECT ANSWER" (no colon).
# Flag re.MULTILINE makes ^ match start of any line.
_SECTION_PATTERNS = [
    # Each pattern ends with \s*$  so it only matches a standalone header line
    # (prevents "Source of data: X" from matching SOURCE inside body text).
    # re.MULTILINE makes ^ / $ match at each line boundary.
    ("direct_answer",
     r"^(?:DIRECT\s+ANSWER|DIRECTIVE)\s*:?\s*$"),
    ("evidence",
     r"^(?:EVIDENCE|SUPPORTING\s+DATA)\s*:?\s*$"),
    ("source_citation",
     r"^(?:SOURCE\s*/\s*CITATION|SOURCE\s+/\s+CITATION|SOURCE\s+CITATION)\s*:?\s*$"),
    ("confidence_level",
     r"^(?:CONFIDENCE\s+LEVEL|CONFIDENCE)\s*:?\s*$"),
    ("assumptions",
     r"^ASSUMPTIONS?\s*:?\s*$"),
    ("risk_limitation",
     r"^(?:RISK\s*/\s*LIMITATION|RISK\s*/\s*LIMITATIONS?|RISK\s+LIMITATION|RISK\s+&\s+LIMITATION)\s*:?\s*$"),
    ("recommended_next_action",
     r"^(?:RECOMMENDED\s+NEXT\s+ACTION|NEXT\s+STEPS?|RECOMMENDATION)\s*:?\s*$"),
]


def _clean_text(text: str) -> str:
    # Strip qwen3 thinking blocks (various formats)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_start\|>think.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    # Strip residual non-ASCII/Unicode artifacts left by qwen3 thinking mode
    # (Korean jamo U+1100-U+11FF, Korean syllables U+AC00-U+D7A3, BOM, zero-width)
    text = re.sub(r'^[-- ᄀ-ᇿ가-힣﻿​-\u200F]+\s*', '', text)
    text = re.sub(r'\*\*([^*]*)\*\*', r'\1', text)   # strip **bold**
    text = re.sub(r'\*([^*]*)\*',     r'\1', text)    # strip *italic*
    text = re.sub(r'#+ ',             '',    text)    # strip ### markdown headers
    text = re.sub(r'[ \t]+',          ' ',   text)    # collapse spaces/tabs
    text = re.sub(r'\n{3,}',          '\n\n', text)   # max one blank line
    # Grammar: "There are 1 wells/rows/records" → singular
    text = re.sub(r'\bThere are 1 (well|row|record|rig|cluster|entry|result)s\b',
                  r'There is 1 \1', text)
    text = re.sub(r'\bthere are 1 (well|row|record|rig|cluster|entry|result)s\b',
                  r'there is 1 \1', text)
    return text.strip()


def _parse_structured_answer(raw: str) -> dict[str, str]:
    """Split LLM output into 7 named sections. Falls back gracefully if a
    section header is missing — puts everything in direct_answer."""
    raw = _clean_text(raw)

    # Find start positions of each section header that actually appears.
    # re.MULTILINE makes ^ match start of any line, so headers without colons
    # (e.g. "DIRECT ANSWER\n...") are found the same as "DIRECT ANSWER:\n...".
    found: list[tuple[int, str]] = []
    _FLAGS = re.IGNORECASE | re.MULTILINE
    for field, pattern in _SECTION_PATTERNS:
        m = re.search(pattern, raw, _FLAGS)
        if m:
            found.append((m.start(), field, m.end()))

    found.sort(key=lambda x: x[0])

    result = {f: "" for f, _ in _SECTION_PATTERNS}

    if not found:
        result["direct_answer"] = raw
        return result

    for i, (_, field, header_end) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(raw)
        result[field] = raw[header_end:end].strip()

    # Fallback: if direct_answer is empty but we have content before the first section,
    # use that preamble text as the direct answer
    if not result["direct_answer"] and found:
        preamble = raw[:found[0][0]].strip()
        if preamble:
            result["direct_answer"] = preamble

    # Fallback: if direct_answer still empty, use evidence or first non-empty field
    if not result["direct_answer"]:
        for field, _ in _SECTION_PATTERNS:
            if result.get(field):
                result["direct_answer"] = result[field]
                break

    return result


def _validate_numbers(answer: dict[str, str], hits: list[Hit]) -> dict[str, str]:
    """Post-generation guard: if direct_answer contains a standalone number,
    verify that exact number appears in at least one retrieved chunk.
    If not found anywhere in context → override to NOT FOUND IN CONTEXT."""
    direct = answer.get("direct_answer", "")
    if not direct or "NOT FOUND IN CONTEXT" in direct:
        return answer

    # Extract all standalone numbers from the direct answer
    numbers = re.findall(r'\b(\d[\d,\.]*\d|\d)\b', direct)
    if not numbers:
        return answer

    # Build full context text from all retrieved chunks
    context_text = " ".join(h.text for h in hits)

    # Check: for each number in the answer, does it (or a close variant) appear in context?
    unverified = []
    for num in numbers:
        # Normalize: remove commas for comparison (14,411 → 14411)
        normalized = num.replace(",", "")
        if normalized in context_text or num in context_text:
            return answer  # at least one number verified → pass
        unverified.append(num)

    if unverified:
        answer["direct_answer"] = (
            "NOT FOUND IN CONTEXT — the retrieved documents do not contain "
            "the specific value needed to answer this question."
        )
        answer["evidence"] = "N/A"
        answer["confidence_level"] = "Low — answer could not be verified against retrieved context."
    return answer


def llm_answer(query: str, hits: list[Hit]) -> dict[str, str]:
    good_hits = [h for h in hits if h.score >= MIN_RELEVANCE_SCORE] or hits[:3]

    # Build context with full text (larger window catches summary rows)
    context_parts = [f"[Source: {h.source}]\n{h.text[:2500]}" for h in good_hits]
    context = "\n\n---\n\n".join(context_parts)

    # Explicit source list prevents the LLM from citing invented filenames
    available_sources = "\n".join(
        f"  - {s}" for s in dict.fromkeys(h.source for h in good_hits)
    )

    user_msg = (
        f"ALLOWED SOURCES (cite ONLY from this list — no other filenames):\n"
        f"{available_sources}\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {query}"
    )
    raw = _call_llm(_SYSTEM_PROMPT, user_msg, timeout=180)
    parsed = _parse_structured_answer(raw)
    return _validate_numbers(parsed, good_hits)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AL TASNIM RAG", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_chunks: list[Chunk] = []
_bm25: BM25Okapi | None = None


@app.on_event("startup")
def startup():
    global _chunks, _bm25
    _load_api_keys()
    print("\n" + "=" * 70)
    print("  AL TASNIM RAG v2 — pgvector + BM25 hybrid")
    print(f"  Auth: {'enabled (' + str(len(_API_KEYS)) + ' keys)' if _AUTH_ENABLED else 'disabled (dev mode)'}")
    print("=" * 70)

    # Verify table exists and has data
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {PG_TABLE};")
        total = cur.fetchone()[0]

    if total == 0:
        raise RuntimeError(
            f"Table '{PG_TABLE}' is empty.\n"
            "Run: python scripts/01_clean_chunks.py && python scripts/02_push_to_pgvector.py"
        )
    print(f"[db] PostgreSQL: {total:,} chunks in {PG_TABLE}")

    _chunks = load_chunks_from_pg()
    _bm25   = build_bm25(_chunks)

    print(f"\n[ready] {len(_chunks):,} chunks | pgvector HNSW + BM25 | BGE-M3 + Groq LLM")
    print("=" * 70 + "\n")


# --- request/response models ------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 8


class SearchHit(BaseModel):
    id: str
    source: str
    score: float
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: float


class AskRequest(BaseModel):
    query: str
    top_k: int = 12
    bypass_cache: bool = False   # set True in A/B tests to skip semantic cache


class AskResponse(BaseModel):
    query: str
    query_used: str
    direct_answer: str
    evidence: str
    source_citation: str
    confidence_level: str
    assumptions: str
    risk_limitation: str
    recommended_next_action: str
    sources: list[str]
    retrieval_attempts: int
    from_cache: bool
    latency_ms: float
    needs_clarification: bool = False
    pii_flags: list[str] = []


# --- endpoints ---------------------------------------------------------------
@app.get("/healthz")
def healthz():
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {PG_TABLE};")
        db_count = cur.fetchone()[0]
    return {
        "status": "ok",
        "chunks_in_db": db_count,
        "chunks_in_bm25": len(_chunks),
        "embed_model": EMBED_MODEL,
        "llm_model": OLLAMA_LLM_MODEL,
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    t0 = time.time()
    qvec = embed_query(req.query)
    hits = hybrid_search(req.query, qvec, _bm25, _chunks, req.top_k)
    hits = compress_hits(hits, qvec)
    return SearchResponse(
        query=req.query,
        hits=[SearchHit(id=h.id, source=h.source, score=h.score,
                        text=h.text, metadata=h.meta) for h in hits],
        latency_ms=round((time.time() - t0) * 1000, 1),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    t0 = time.time()

    # ── 1. Auth — verify API key, get role ───────────────────────────────────
    user = _authenticate(request)
    allowed_conf = _allowed_confidentiality(user)

    # ── 2. Input sanitization — block injection, redact PII ──────────────────
    clean_query, pii_flags = _sanitize_input(req.query)

    # ── 3. Clarification gate — block vague operational queries ──────────────
    clarification = _check_vague_query(clean_query)
    if clarification:
        lat = round((time.time() - t0) * 1000, 1)
        _trace("clarification", lat, role=user["role"])
        return AskResponse(
            query=req.query, query_used=clean_query,
            direct_answer=clarification,
            evidence="", source_citation="", confidence_level="",
            assumptions="", risk_limitation="", recommended_next_action="",
            sources=[], retrieval_attempts=0, from_cache=False,
            latency_ms=lat, needs_clarification=True, pii_flags=pii_flags,
        )

    qvec = embed_query(clean_query)

    # Semantic cache check — skipped when bypass_cache=True (A/B testing)
    cached = None if req.bypass_cache else _cache_lookup(qvec)
    if cached:
        a = cached["answer"]
        lat = round((time.time() - t0) * 1000, 1)
        _trace("cache", lat, role=user["role"])
        return AskResponse(
            query=req.query,
            query_used=cached["query"],
            direct_answer=a.get("direct_answer", ""),
            evidence=a.get("evidence", ""),
            source_citation=a.get("source_citation", ""),
            confidence_level=a.get("confidence_level", ""),
            assumptions=a.get("assumptions", ""),
            risk_limitation=a.get("risk_limitation", ""),
            recommended_next_action=a.get("recommended_next_action", ""),
            sources=cached["sources"],
            retrieval_attempts=0,
            from_cache=True,
            latency_ms=lat,
            pii_flags=pii_flags,
        )

    # ── Route: structured operational query → Text-to-SQL ────────────────────
    if _is_sql_query(clean_query) and _check_sql_table_exists():
        # RBAC: check table access before running any SQL
        intent = _classify_intent(clean_query)
        target_table = _INTENT_CFG.get(intent.get("intent", ""), {}).get(
            "table", _ALLOWED_TABLE
        )
        if not _check_table_access(user, target_table):
            lat = round((time.time() - t0) * 1000, 1)
            _trace("sql_denied", lat, role=user["role"], table=target_table)
            return AskResponse(
                query=req.query, query_used=clean_query,
                direct_answer=(
                    f"Access denied. Your role ({user['role']}) does not have "
                    f"permission to query the {target_table} data."
                ),
                evidence="", source_citation="",
                confidence_level="Low — access control", assumptions="",
                risk_limitation="RBAC restriction enforced.",
                recommended_next_action="Contact your system administrator to request elevated access.",
                sources=[], retrieval_attempts=0, from_cache=False,
                latency_ms=lat, pii_flags=pii_flags,
            )

        print(f"[ask] Routing to Text-to-SQL: {clean_query!r}")
        parsed  = sql_answer(clean_query)
        # Source comes from the intent config via sql_answer → parsed["source_citation"]
        sources = [s.strip() for s in parsed.get("source_citation", _SOURCE_FILE).split(",") if s.strip()]

        direct = parsed.get("direct_answer", "").strip()
        confidence = parsed.get("confidence_level", "").lower()
        if direct and "high" in confidence:
            _cache_store(qvec, clean_query, parsed, sources)

        lat = round((time.time() - t0) * 1000, 1)
        _trace("sql", lat, role=user["role"], confidence=confidence or "?")
        return AskResponse(
            query=req.query,
            query_used=clean_query,
            direct_answer=parsed.get("direct_answer", ""),
            evidence=parsed.get("evidence", ""),
            source_citation=parsed.get("source_citation", ""),
            confidence_level=parsed.get("confidence_level", ""),
            assumptions=parsed.get("assumptions", ""),
            risk_limitation=parsed.get("risk_limitation", ""),
            recommended_next_action=parsed.get("recommended_next_action", ""),
            sources=sources,
            retrieval_attempts=0,
            from_cache=False,
            latency_ms=lat,
            pii_flags=pii_flags,
        )

    # ── Route: knowledge query → Hybrid RAG (with RBAC pre-filter) ───────────
    active_query = clean_query
    hits: list[Hit] = []
    attempts = 0
    for attempt in range(2):
        attempts += 1
        search_vec = embed_query(active_query) if attempt > 0 else qvec
        hits = hybrid_search(
            active_query, search_vec, _bm25, _chunks, req.top_k,
            allowed_conf=allowed_conf,
        )
        if hits and hits[0].score >= MIN_RELEVANCE_SCORE:
            break
        if attempt == 0:
            print(f"[ask] Low score ({hits[0].score if hits else 0:.4f}) — rewriting …")
            active_query = _rewrite_query(clean_query)
            print(f"[ask] Rewritten: {active_query!r}")

    # Contextual Compression: strip irrelevant sentences before sending to LLM
    hits = compress_hits(hits, search_vec)

    parsed  = llm_answer(active_query, hits)
    sources = list(dict.fromkeys(h.source for h in hits))
    direct = parsed.get("direct_answer", "").strip()
    confidence = parsed.get("confidence_level", "").lower()
    is_good = (
        direct
        and "NOT FOUND IN CONTEXT" not in direct
        and "not found" not in direct.lower()
        and "high" in confidence
    )
    if is_good:
        _cache_store(qvec, clean_query, parsed, sources)

    lat = round((time.time() - t0) * 1000, 1)
    _trace(
        "rag", lat,
        role=user["role"],
        attempts=attempts,
        chunks=len(hits),
        top_score=f"{hits[0].score:.4f}" if hits else "0",
        confidence=confidence or "?",
        rewritten=("yes" if active_query != clean_query else "no"),
    )
    return AskResponse(
        query=req.query,
        query_used=active_query,
        direct_answer=parsed.get("direct_answer", ""),
        evidence=parsed.get("evidence", ""),
        source_citation=parsed.get("source_citation", ""),
        confidence_level=parsed.get("confidence_level", ""),
        assumptions=parsed.get("assumptions", ""),
        risk_limitation=parsed.get("risk_limitation", ""),
        recommended_next_action=parsed.get("recommended_next_action", ""),
        sources=sources,
        retrieval_attempts=attempts,
        from_cache=False,
        latency_ms=lat,
        pii_flags=pii_flags,
    )
