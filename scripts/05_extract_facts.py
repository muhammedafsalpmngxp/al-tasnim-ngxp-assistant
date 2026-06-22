#!/usr/bin/env python3
"""
Step 5 — Fact Injection: embed and push facts from config/facts.yaml into PostgreSQL.

To add, edit, or remove a fact: edit  config/facts.yaml  only — no Python changes needed.

Run:
    conda activate v12
    python scripts/05_extract_facts.py
"""
import hashlib
import json
import os
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.split("#")[0].strip()
            os.environ.setdefault(_k.strip(), _v)

HERE        = Path(__file__).parent.parent
FACTS_FILE  = HERE / "config" / "facts.yaml"
PG_URL      = os.getenv("PG_URL",      "postgresql://abhay@/altasnim?host=/var/run/postgresql")
PG_TABLE    = os.getenv("PG_TABLE",    "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")


# ---------------------------------------------------------------------------
# Load facts from YAML — the only source of truth
# ---------------------------------------------------------------------------
def load_facts() -> list[dict]:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML not installed. Run: conda install -c conda-forge pyyaml"
        )
    if not FACTS_FILE.exists():
        raise FileNotFoundError(
            f"Facts config not found: {FACTS_FILE}\n"
            "Create config/facts.yaml with a 'facts:' list."
        )
    with FACTS_FILE.open() as f:
        data = yaml.safe_load(f)
    facts = data.get("facts", [])
    if not facts:
        raise ValueError("facts.yaml has no entries under 'facts:'")
    return facts


# ---------------------------------------------------------------------------
# Build chunk record from a fact dict
# ---------------------------------------------------------------------------
def make_chunk(fact: dict) -> dict:
    text = " ".join(fact["text"].split())   # normalise YAML block scalar whitespace
    h = hashlib.md5(text.encode()).hexdigest()
    return {
        "id":           f"fact_{h[:16]}",
        "content":      text,
        "content_hash": h,
        "metadata": {
            "source":        fact.get("source", ""),
            "sheet":         fact.get("sheet", ""),
            "document_type": "fact",
            "category":      fact.get("category", "general"),
            "is_fact":       True,
        },
    }


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> list:
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] Loading {EMBED_MODEL} on {device.upper()} …")
    model = SentenceTransformer(
        EMBED_MODEL,
        model_kwargs={"torch_dtype": torch.float16},
        device=device,
    )
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    print(f"[embed] Done — {len(vecs)} vectors")
    return vecs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Step 5 — Fact Injection")
    print(f"  Source: {FACTS_FILE.relative_to(HERE)}")
    print("=" * 60)

    facts  = load_facts()
    chunks = [make_chunk(f) for f in facts]

    print(f"\n  Facts loaded: {len(chunks)}")
    for c in chunks:
        print(f"    [{c['metadata']['category']:15s}] {c['content'][:70]} …")

    print("\n[embed] Embedding facts …")
    vecs = embed_texts([c["content"] for c in chunks])

    print("\n[db] Connecting …")
    conn = psycopg2.connect(PG_URL)
    conn.autocommit = False
    register_vector(conn)

    rows = [
        (
            c["id"],
            c["metadata"]["source"],
            c["content"],
            vec,
            json.dumps(c["metadata"]),
            c["content_hash"],
        )
        for c, vec in zip(chunks, vecs)
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""INSERT INTO {PG_TABLE}
                   (id, source, content, embedding, metadata, content_hash)
               VALUES %s
               ON CONFLICT (id) DO UPDATE
                 SET content    = EXCLUDED.content,
                     embedding  = EXCLUDED.embedding,
                     metadata   = EXCLUDED.metadata""",
            rows,
        )
    conn.commit()
    conn.close()

    print(f"\n  Injected {len(chunks)} fact chunks into DB.")
    print("  Restart the server to rebuild BM25 with updated facts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
