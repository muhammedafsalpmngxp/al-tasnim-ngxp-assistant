"""
Production-ready Natural-Language → SQL assistant for SQL Server.

Pipeline per question:
  Intent check → Table retrieval → Value grounding → Relationship context →
  Query planner → SQL generation → Self-critique → Validate → Execute → Synthesize

Exposes ask(question, session_id=None) → FinalResponse
and shutdown() so the orchestrator's db_tool adapter works unchanged.

All tunables are environment variables — nothing hardcoded.
Schema is discovered dynamically from the live DB at startup.
"""

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Optional
from urllib.parse import quote_plus

import pyodbc
from dotenv import load_dotenv
from sqlalchemy import (
    Column, MetaData, Table, create_engine,
    inspect as sa_inspect, text,
)
from sqlalchemy.types import (
    Boolean, Date, DateTime, Float, Integer, String, Time,
)

# Disable pyodbc's built-in pool — conflicts with SQLAlchemy's pool and causes
# TCP reset errors during large remote schema operations.
pyodbc.pooling = False

# Resolve .env: walk up from this file until we find a .env file,
# then load it with override=True so the real values always win.
def _load_env() -> None:
    search = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):  # max 6 levels up
        candidate = os.path.join(search, ".env")
        if os.path.isfile(candidate):
            load_dotenv(candidate, override=True)
            # Use a basic print here — logger is not set up yet
            print(f"[pipeline] Loaded .env from: {candidate}", flush=True)
            return
        search = os.path.dirname(search)
    load_dotenv(override=True)  # fallback: let python-dotenv search CWD
    print("[pipeline] WARNING: .env not found by walk-up; falling back to CWD search", flush=True)

_load_env()

# ============================================================
# CONFIG — all from environment
# ============================================================

_PLACEHOLDERS = {"your_sql_server_ip", "your_password", "your_google_api_key",
                 "your_key", "localhost", ""}

DB_SERVER           = os.getenv("DB_SERVER")
DB_NAME             = os.getenv("DB_NAME")
DB_USER             = os.getenv("DB_READONLY_USER")
DB_PASSWORD         = os.getenv("DB_READONLY_PASSWORD")
DB_DRIVER           = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_CONNECT_TIMEOUT  = int(os.getenv("DB_CONNECT_TIMEOUT", "30"))
DB_QUERY_TIMEOUT    = int(os.getenv("DB_QUERY_TIMEOUT", "120"))
DB_POOL_SIZE        = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW     = int(os.getenv("DB_MAX_OVERFLOW", "10"))

LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "groq").split("#")[0].strip().lower()
if LLM_PROVIDER == "local":
    LLM_PROVIDER = "ollama"

GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME     = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE    = float(os.getenv("GROQ_TEMPERATURE", "0.0"))
GROQ_MAX_TOKENS     = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
GROQ_CONTEXT_WINDOW = int(os.getenv("GROQ_CONTEXT_WINDOW", "32768"))

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME   = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
LLM_TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS      = int(os.getenv("LLM_MAX_TOKENS", "2048"))

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")

_base_dir         = os.path.dirname(os.path.abspath(__file__))
_tool_dir         = os.path.dirname(_base_dir)
TABLE_INDEX_DIR   = os.getenv("TABLE_INDEX_DIR",  os.path.join(_tool_dir, "table_index_storage"))
VALUE_INDEX_DIR   = os.getenv("VALUE_INDEX_DIR",  os.path.join(_tool_dir, "value_index_storage"))
LOG_DIR           = os.getenv("LOG_DIR",           os.path.join(_tool_dir, "logs"))
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")

# How many random sample values to collect per varchar column.
# Values are picked randomly (not from the top of the index) so the sample
# represents the full spread of data — critical for correct table selection.
# No column-count limit — every varchar/nvarchar column in every table is sampled.
TABLE_SAMPLE_LIMIT  = int(os.getenv("TABLE_SAMPLE_LIMIT",  "10"))
# Sampling method:
#   "newid"       — ORDER BY NEWID() — truly random, works on any table size (default)
#   "tablesample" — TABLESAMPLE SYSTEM — faster on huge tables, page-level approximate
TABLE_SAMPLE_METHOD = os.getenv("TABLE_SAMPLE_METHOD", "newid").lower().strip()

# Human-readable name for this database shown inside LLM prompts.
# Change via .env — no code edits needed when switching databases.
DB_DOMAIN = os.getenv("DB_DOMAIN", "operational")

ROW_CAP            = int(os.getenv("ROW_CAP", "100"))
RETRIEVER_TOP_K    = int(os.getenv("RETRIEVER_TOP_K", "5"))
VALUE_TOP_K        = int(os.getenv("VALUE_TOP_K", "2"))
# Weight for dense (vector) score in hybrid retrieval. BM25 weight = 1 - HYBRID_ALPHA.
# 0.5 = equal weight; increase toward 1.0 for more semantic, toward 0.0 for more keyword.
HYBRID_ALPHA       = float(os.getenv("HYBRID_ALPHA", "0.5"))
MAX_QUERY_LEN      = int(os.getenv("MAX_QUERY_LEN", "500"))
HISTORY_MAX        = int(os.getenv("HISTORY_MAX", "5"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
VALUE_SAMPLE_LIMIT = int(os.getenv("VALUE_SAMPLE_LIMIT", "50"))   # rows per column
ENABLE_SQL_CRITIQUE = os.getenv("ENABLE_SQL_CRITIQUE", "1") == "1"

VALUE_HINT_MAX  = int(os.getenv("VALUE_HINT_MAX",  "4"))

# VALUE_INDEX_ENABLED=0 disables value grounding entirely — speeds up first-run init
# significantly on large databases (avoids hundreds of per-column DB queries).
# VALUE_INDEX_MAX_COLS caps how many varchar columns are sampled when enabled.
VALUE_INDEX_ENABLED  = os.getenv("VALUE_INDEX_ENABLED",  "0") == "1"
VALUE_INDEX_MAX_COLS = int(os.getenv("VALUE_INDEX_MAX_COLS", "100"))

DB_EXCLUDE_TABLES: set = {t.strip() for t in os.getenv("DB_EXCLUDE_TABLES", "").split(",") if t.strip()}

# ============================================================
# LOGGING
# ============================================================

def _setup_logging() -> logging.Logger:
    """Configure logging for the pipeline.

    When running in-process (orchestrator already configured the root logger),
    only add the rotating file handler so pipeline logs go to assistant.log
    without interfering with the orchestrator's console/level settings.
    When running standalone (no root handlers yet), configure everything.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "assistant.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        # Standalone mode — configure root with console + file
        root.setLevel(level)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # Always add the rotating file handler if not already present
    existing_files = {
        getattr(h, "baseFilename", None)
        for h in root.handlers
        if isinstance(h, RotatingFileHandler)
    }
    if os.path.abspath(log_file) not in existing_files:
        fh = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger(__name__)

logger = _setup_logging()

# ============================================================
# READ-ONLY GUARD + ROW CAP
# ============================================================

_BLOCKED = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE|MERGE|GRANT|REVOKE|DENY|BULK)\b',
    re.IGNORECASE,
)

def _validate_select_only(sql: str) -> None:
    s = sql.strip()
    if not s.upper().startswith("SELECT"):
        raise ValueError(f"Blocked: query must start with SELECT. Got: {s[:120]}")
    m = _BLOCKED.search(s)
    if m:
        raise ValueError(f"Blocked: forbidden keyword '{m.group()}' in generated SQL.")

def _inject_row_cap(sql: str, cap: int) -> str:
    if re.search(r'\bSELECT\s+(?:DISTINCT\s+)?TOP\s+\d+\b', sql, re.IGNORECASE):
        return sql
    return re.sub(r'\bSELECT\b', f'SELECT TOP {cap}', sql, count=1, flags=re.IGNORECASE)

# ============================================================
# SAFE SQL DATABASE
# ============================================================

_SA_TYPE_MAP = {
    'int':              Integer,
    'nvarchar':         String,
    'decimal':          Float,
    'datetime':         DateTime,
    'date':             Date,
    'time':             Time,
    'bit':              Boolean,
    'uniqueidentifier': String,
}

def _build_metadata_from_schema(schema: dict) -> MetaData:
    """Build SQLAlchemy MetaData from already-discovered schema — zero DB calls."""
    metadata = MetaData()
    for table_name, cols in schema.items():
        columns = [
            Column(col_name, _SA_TYPE_MAP.get(col_type, String)())
            for col_name, col_type in cols
        ]
        Table(table_name, metadata, *columns)
    return metadata

# Imported lazily to avoid heavy deps at module load
def _get_sql_db_class():
    from llama_index.core import SQLDatabase

    class SafeSQLDatabase(SQLDatabase):
        """Validates + caps every query BEFORE it reaches the DB."""
        def __init__(self, *args, row_cap: int = ROW_CAP, metadata=None, **kwargs):
            if metadata is not None:
                # Patch reflect to no-op — prevents LlamaIndex from re-scanning
                # the full remote schema on every SafeSQLDatabase.__init__ call,
                # which causes TCP timeouts on large databases.
                metadata.reflect = lambda *a, **kw: None
                kwargs["metadata"] = metadata
            super().__init__(*args, **kwargs)
            self._row_cap = row_cap

        def run_sql(self, command: str):
            _validate_select_only(command)
            command = _inject_row_cap(command, self._row_cap)
            logger.info(f"[SQL] {command[:400]}")
            return super().run_sql(command)

    return SafeSQLDatabase

# ============================================================
# DYNAMIC SCHEMA DISCOVERY
# ============================================================

_TYPE_NORMALIZE = re.compile(r'\(.*\)$')

def _normalize_type(sa_type: str) -> str:
    s = _TYPE_NORMALIZE.sub('', str(sa_type)).lower().strip()
    if 'varchar' in s or 'char' in s or 'text' in s:
        return 'nvarchar'
    if 'int' in s:
        return 'int'
    if 'decimal' in s or 'numeric' in s or 'float' in s or 'real' in s:
        return 'decimal'
    if 'date' in s or 'time' in s:
        return 'datetime'
    if 'bit' in s:
        return 'bit'
    if 'unique' in s:
        return 'uniqueidentifier'
    return s

def _discover_schema(engine, tables: list) -> dict:
    inspector = sa_inspect(engine)
    all_db_tables = set(inspector.get_table_names())
    schema = {}
    for table in tables:
        if table not in all_db_tables or table in DB_EXCLUDE_TABLES:
            continue
        try:
            cols = inspector.get_columns(table)
            schema[table] = [(c["name"], _normalize_type(c["type"])) for c in cols]
        except Exception as e:
            logger.warning(f"Could not introspect {table}: {e}")
    return schema

def _schema_hash(schema: dict) -> str:
    return hashlib.md5(
        json.dumps({k: v for k, v in sorted(schema.items())}, sort_keys=True).encode()
    ).hexdigest()

def _load_hash(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path) as f:
        return f.read().strip()

def _save_hash(path: str, value: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(value)

# ============================================================
# RELATIONSHIP DISCOVERY
# ============================================================

_REL_SKIP_COLS = {"id", "type", "status", "name", "code", "account", "location",
                  "created_at", "updated_at", "project_id"}

def _discover_relationships(engine, schema: dict) -> dict:
    target_tables = list(schema.keys())
    rels: dict = {t: [] for t in target_tables}
    target_lower = {t.lower(): t for t in target_tables}

    fk_sql = text("""
        SELECT tp.name, cp.name, tr.name, cr.name
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
        JOIN sys.tables  tp ON fkc.parent_object_id      = tp.object_id
        JOIN sys.columns cp ON fkc.parent_object_id      = cp.object_id
                            AND fkc.parent_column_id     = cp.column_id
        JOIN sys.tables  tr ON fkc.referenced_object_id  = tr.object_id
        JOIN sys.columns cr ON fkc.referenced_object_id  = cr.object_id
                            AND fkc.referenced_column_id = cr.column_id
    """)
    try:
        with engine.connect() as conn:
            for pt, pc, rt, rc in conn.execute(fk_sql).fetchall():
                if pt.lower() in target_lower and rt.lower() in target_lower:
                    p, r = target_lower[pt.lower()], target_lower[rt.lower()]
                    rels[p].append((pc, r, rc))
                    rels[r].append((rc, p, pc))
    except Exception as e:
        logger.warning(f"FK discovery failed: {e}")

    col_index: dict = {}
    for table, cols in schema.items():
        for col, _ in cols:
            col_index.setdefault(col.lower(), []).append((table, col))

    seen: set = set()
    for col_lower, table_cols in col_index.items():
        if len(table_cols) < 2 or col_lower in _REL_SKIP_COLS:
            continue
        for i, (t1, c1) in enumerate(table_cols):
            for t2, c2 in table_cols[i + 1:]:
                key = tuple(sorted([(t1, c1), (t2, c2)]))
                if key in seen:
                    continue
                seen.add(key)
                if (c1, t2) not in {(r[0], r[1]) for r in rels[t1]}:
                    rels[t1].append((c1, t2, c2))
                if (c2, t1) not in {(r[0], r[1]) for r in rels[t2]}:
                    rels[t2].append((c2, t1, c1))

    logger.info(f"Relationships: {sum(len(v) for v in rels.values())} discovered.")
    return rels

# ============================================================
# VALUE GROUNDING INDEX
# ============================================================

_VALUE_SKIP = re.compile(
    r'(data|attributes|url|json|notes|description|email|password|token|text|hash|blob|xml)',
    re.I,
)
_VALUE_TABLE_HASH_FILE = os.path.join(VALUE_INDEX_DIR, "value_version.txt")

def _build_value_documents(engine, schema: dict) -> list:
    from llama_index.core import Document
    docs = []
    sampled = 0
    for table, cols in schema.items():
        if sampled >= VALUE_INDEX_MAX_COLS:
            break
        for col, dtype in cols:
            if sampled >= VALUE_INDEX_MAX_COLS:
                break
            if dtype not in ("nvarchar",):
                continue
            if _VALUE_SKIP.search(col):
                continue
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text(
                        f"SELECT TOP {VALUE_SAMPLE_LIMIT} [{col}] "
                        f"FROM [{table}] "
                        f"WHERE [{col}] IS NOT NULL AND LEN([{col}]) < 150 "
                        f"ORDER BY NEWID()"
                    )).fetchall()
                for (val,) in rows:
                    v = str(val).strip()
                    if v:
                        docs.append(Document(
                            text=f"{v} | table:{table} | col:{col}",
                            metadata={"table": table, "column": col, "value": v},
                        ))
                sampled += 1
            except Exception:
                pass
    logger.info("Value index: %d docs from %d columns sampled.", len(docs), sampled)
    return docs

def _build_or_load_value_index(engine, schema: dict, embed_model):
    if not VALUE_INDEX_ENABLED:
        logger.info("Value index disabled (VALUE_INDEX_ENABLED=0) — skipping value grounding.")
        return None

    from llama_index.core import (
        StorageContext, VectorStoreIndex, load_index_from_storage,
    )
    current_hash = _schema_hash(schema)
    stored_hash  = _load_hash(_VALUE_TABLE_HASH_FILE)
    force        = os.getenv("REBUILD_INDEX", "0") == "1"
    index_exists = os.path.isfile(os.path.join(VALUE_INDEX_DIR, "index_store.json"))

    if index_exists and not force and stored_hash == current_hash:
        logger.info("Loading persisted value index...")
        try:
            sc  = StorageContext.from_defaults(persist_dir=VALUE_INDEX_DIR)
            idx = load_index_from_storage(sc, embed_model=embed_model)
            logger.info("Value index loaded.")
            return idx
        except Exception as e:
            logger.warning("Value index load failed (%s), rebuilding...", e)

    logger.info("Building value index (sampling DB values)...")
    docs = _build_value_documents(engine, schema)
    if not docs:
        logger.warning("No value documents — value grounding disabled.")
        return None
    idx = VectorStoreIndex.from_documents(docs, embed_model=embed_model, show_progress=False)
    os.makedirs(VALUE_INDEX_DIR, exist_ok=True)
    idx.storage_context.persist(persist_dir=VALUE_INDEX_DIR)
    _save_hash(_VALUE_TABLE_HASH_FILE, current_hash)
    logger.info(f"Value index built and saved ({len(docs)} docs).")
    return idx

# ============================================================
# TABLE SCHEMA INDEX
# ============================================================

_TABLE_HASH_FILE = os.path.join(TABLE_INDEX_DIR, "schema_version.txt")

# Columns whose values add noise rather than signal — skip during sampling.
_SAMPLE_SKIP_COLS = re.compile(
    r'(url|json|blob|xml|hash|token|password|email|notes|description|text|remark|comment|address|path|guid)',
    re.I,
)

def _table_description(table: str, cols: list) -> str:
    """Convert a CamelCase/underscore table name + columns into a readable sentence.
    Used as the opening line of the embedding text so the retriever can match on
    plain-English paraphrases of what the table stores.
    """
    spaced = re.sub(r'([A-Z])', r' \1', table.replace('_', ' ')).strip()
    words  = re.sub(r'\s+', ' ', spaced).strip().lower()
    key_cols = ", ".join(c for c, _ in cols[:15])
    return f"Stores {words} records. Contains fields: {key_cols}."


def _sample_col_sql(col: str, table: str) -> str:
    """Return a single-column random-sample SELECT for use inside UNION ALL.

    TABLE_SAMPLE_METHOD controls the strategy:
      "newid"       — ORDER BY NEWID() inside a subquery — truly random,
                      works correctly on all table sizes.
      "tablesample" — TABLESAMPLE SYSTEM(n ROWS) — page-level random,
                      faster on multi-million-row tables but approximate.

    Both avoid the bias of TOP without ORDER BY which returns the first rows
    the storage engine encounters (index order or table-scan order).
    """
    c_esc   = col.replace("'", "''")          # escape single quotes in col name literal
    n       = TABLE_SAMPLE_LIMIT
    not_empty = f"LEN(LTRIM(RTRIM(CAST([{col}] AS NVARCHAR(500))))) > 0"

    if TABLE_SAMPLE_METHOD == "tablesample":
        # TABLESAMPLE SYSTEM operates at the page level — fast on large tables.
        # We over-sample by 10× then take TOP n so we get enough distinct rows.
        inner = (
            f"SELECT TOP {n} [{col}] "
            f"FROM [{table}] TABLESAMPLE SYSTEM ({n * 10} ROWS) "
            f"WHERE [{col}] IS NOT NULL AND {not_empty}"
        )
    else:
        # NEWID() per row — exact uniform random sample, works on any size.
        inner = (
            f"SELECT TOP {n} [{col}] "
            f"FROM [{table}] "
            f"WHERE [{col}] IS NOT NULL AND {not_empty} "
            f"ORDER BY NEWID()"
        )

    return (
        f"SELECT N'{c_esc}' AS c, CAST([{col}] AS NVARCHAR(500)) AS v "
        f"FROM ({inner}) _s"
    )


def _sample_table_values(engine, schema: dict) -> dict:
    """Randomly sample TABLE_SAMPLE_LIMIT values from every varchar/nvarchar
    column in every table.

    Random (not deterministic TOP) so the sample represents the full spread of
    data in each column — critical so the LLM schema selector can match user
    query terms against actual stored values regardless of index order.

    Uses one UNION ALL query per table → DB round-trips = number of tables.

    Returns: {table_name: {col_name: [val1, val2, ...]}}
    """
    samples: dict = {}
    total = len(schema)
    logger.info(
        "[TABLE-SAMPLE] Randomly sampling ALL varchar columns in %d tables "
        "(%d values/column, method=%s)...",
        total, TABLE_SAMPLE_LIMIT, TABLE_SAMPLE_METHOD,
    )

    for i, (table, cols) in enumerate(schema.items(), 1):
        text_cols = [
            col for col, typ in cols
            if typ in ('nvarchar', 'varchar')
            and not _SAMPLE_SKIP_COLS.search(col)
        ]
        if not text_cols:
            continue

        union_sql = " UNION ALL ".join(_sample_col_sql(col, table) for col in text_cols)

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(union_sql)).fetchall()
            for col_name, val in rows:
                v = str(val).strip()
                if v:
                    samples.setdefault(table, {}).setdefault(col_name, []).append(v)
        except Exception as exc:
            logger.debug("[TABLE-SAMPLE] Skipped %s: %s", table, exc)

        if i % 100 == 0:
            logger.info("[TABLE-SAMPLE]   %d / %d tables processed...", i, total)

    sampled_cols = sum(len(v) for v in samples.values())
    logger.info(
        "[TABLE-SAMPLE] Done: %d columns with values across %d tables.",
        sampled_cols, len(samples),
    )
    return samples


def _rich_table_context(table: str, schema: dict, rels: dict, sample_values: dict = None) -> str:
    """Build a rich embedding text for a table.

    Includes:
    - A human-readable description generated from the table name and column list
    - Every column with its data type
    - Real sample values from varchar columns (if sampling was run)
    - Foreign-key / shared-column relationships

    The richer the text, the better the cosine similarity when a user asks a
    question that matches actual data values or column semantics rather than just
    the table name.
    """
    cols = schema.get(table, [])

    description = _table_description(table, cols)

    col_lines = []
    for col_name, col_type in cols:
        line = f"  - {col_name} ({col_type})"
        if sample_values:
            vals = (sample_values.get(table) or {}).get(col_name, [])
            if vals:
                preview = " | ".join(str(v)[:50] for v in vals[:TABLE_SAMPLE_LIMIT])
                line += f" — e.g. {preview}"
        col_lines.append(line)
    col_str = "\n".join(col_lines) if col_lines else "  (none)"

    rel_lines = [
        f"  [{table}].[{col}] → [{other_table}].[{other_col}]"
        for col, other_table, other_col in rels.get(table, [])
    ]
    rel_str = "\n".join(rel_lines) if rel_lines else "  (none)"

    return (
        f"Table: [{table}]\n"
        f"Description: {description}\n"
        f"Columns:\n{col_str}\n"
        f"Relationships:\n{rel_str}"
    )


def _build_or_load_table_index(sql_db, schema: dict, rels: dict, sample_values: dict = None):
    from llama_index.core import VectorStoreIndex
    from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema

    current_hash = _schema_hash(schema)
    stored_hash  = _load_hash(_TABLE_HASH_FILE)
    force        = os.getenv("REBUILD_INDEX", "0") == "1"
    index_exists = os.path.isfile(os.path.join(TABLE_INDEX_DIR, "index_store.json"))

    table_node_mapping = SQLTableNodeMapping(sql_db)

    if index_exists and not force and stored_hash == current_hash:
        logger.info("Loading persisted table index (schema unchanged)...")
        idx = ObjectIndex.from_persist_dir(
            persist_dir=TABLE_INDEX_DIR,
            object_node_mapping=table_node_mapping,
        )
        logger.info("Table index loaded (%d tables).", len(schema))
        return idx

    reason = "forced" if force else ("schema changed" if stored_hash != current_hash else "first run")
    logger.info("Building table index (%s, %d tables)...", reason, len(schema))

    table_schema_objs = [
        SQLTableSchema(
            table_name=t,
            context_str=_rich_table_context(t, schema, rels, sample_values or {}),
        )
        for t in schema
    ]

    idx = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex)
    idx.persist(persist_dir=TABLE_INDEX_DIR)
    _save_hash(_TABLE_HASH_FILE, current_hash)
    logger.info("Table index built and saved (%d tables).", len(schema))
    return idx

# ============================================================
# TABLE RETRIEVAL
# ============================================================

def _tokenize(text: str) -> list:
    """Split text into lowercase tokens, splitting on non-alphanumeric chars
    and also on camelCase / PascalCase / snake_case boundaries so that
    'ActivityCode', 'activity_code', 'ACTIVITY CODE' all produce the same tokens."""
    # Insert space before uppercase letters that follow lowercase (camelCase split)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    # Split on any non-alphanumeric run
    parts = re.split(r'[^a-zA-Z0-9]+', s)
    return [p.lower() for p in parts if p]


def _build_bm25_index(schema: dict, rels: dict, sample_values: dict) -> tuple:
    """Build a BM25Okapi index over rich table context strings (same text used
    for dense embeddings) so keyword matching complements semantic similarity.

    Returns (bm25_index, doc_names) where doc_names[i] is the table name for
    BM25 corpus document i.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning(
            "[BM25] rank_bm25 not installed — hybrid retrieval disabled. "
            "Run: pip install rank-bm25"
        )
        return None, []

    doc_names = list(schema.keys())
    corpus = [
        _tokenize(_rich_table_context(t, schema, rels, sample_values))
        for t in doc_names
    ]
    bm25 = BM25Okapi(corpus)
    logger.info("[BM25] Index built over %d tables.", len(doc_names))
    return bm25, doc_names


def _retrieve_tables_hybrid(
    question: str,
    raw_retriever,        # VectorIndexRetriever — returns NodeWithScore with .score
    bm25_index,           # BM25Okapi or None (falls back to dense-only)
    bm25_doc_names: list,
    fallback: list,
) -> list:
    """Hybrid table retrieval combining dense vector similarity (FAISS/cosine)
    with BM25 sparse keyword matching.

    Both scores are min-max normalised to [0, 1] before combining:
        final = HYBRID_ALPHA * dense_norm + (1 - HYBRID_ALPHA) * bm25_norm

    Returns top RETRIEVER_TOP_K table names sorted by combined score.
    Falls back to dense-only if BM25 is unavailable.
    """
    dense_scores: dict = {}  # {table_name: raw_dense_score}

    # --- Dense retrieval ---
    try:
        raw_nodes = raw_retriever.retrieve(question)
        for node in raw_nodes:
            score = getattr(node, "score", 0.0) or 0.0
            # Extract table name from node text: first line starts with "Table: [Name]"
            text_val = getattr(node, "node", node)
            text_val = getattr(text_val, "text", "") or ""
            m = re.match(r'Table:\s*\[([^\]]+)\]', text_val.strip())
            if m:
                dense_scores[m.group(1)] = score
    except Exception as exc:
        logger.warning("[HYBRID] Dense retrieval error (%s) — using fallback.", exc)
        return fallback[:RETRIEVER_TOP_K]

    if not dense_scores:
        return fallback[:RETRIEVER_TOP_K]

    # --- BM25 sparse retrieval ---
    bm25_scores: dict = {}
    if bm25_index is not None and bm25_doc_names:
        try:
            query_tokens = _tokenize(question)
            raw_bm25 = bm25_index.get_scores(query_tokens)  # ndarray, one score per doc
            for i, score in enumerate(raw_bm25):
                bm25_scores[bm25_doc_names[i]] = float(score)
        except Exception as exc:
            logger.warning("[HYBRID] BM25 scoring error (%s) — using dense only.", exc)

    # --- Combine over the union of tables that got any score ---
    all_tables_scored = set(dense_scores) | set(bm25_scores)

    # Normalise dense
    d_vals = list(dense_scores.values())
    d_min, d_max = (min(d_vals), max(d_vals)) if d_vals else (0.0, 1.0)
    d_range = d_max - d_min or 1.0

    # Normalise BM25 (over ALL docs so relative importance is preserved)
    b_vals = list(bm25_scores.values()) if bm25_scores else [0.0]
    b_min, b_max = (min(b_vals), max(b_vals)) if b_vals else (0.0, 1.0)
    b_range = b_max - b_min or 1.0

    combined: dict = {}
    for t in all_tables_scored:
        d_norm = (dense_scores.get(t, d_min) - d_min) / d_range
        b_norm = (bm25_scores.get(t, b_min) - b_min) / b_range if bm25_scores else 0.0
        combined[t] = HYBRID_ALPHA * d_norm + (1.0 - HYBRID_ALPHA) * b_norm

    ranked = sorted(combined, key=lambda t: combined[t], reverse=True)
    result = ranked[:RETRIEVER_TOP_K]

    logger.debug(
        "[HYBRID] top scores: %s",
        {t: round(combined[t], 3) for t in result},
    )
    return result if result else fallback[:RETRIEVER_TOP_K]


# ============================================================
# LLM SCHEMA SELECTOR
# ============================================================

def _rich_schema_with_samples(tables: list, schema: dict, rels: dict, sample_cache: dict) -> str:
    """Build a detailed schema block shown to the LLM schema selector.
    Every column is listed with its type and real sample values from the DB.
    """
    blocks = []
    for table in tables:
        cols = schema.get(table, [])
        col_lines = []
        for col_name, col_type in cols:
            line = f"    [{col_name}] ({col_type})"
            vals = (sample_cache.get(table) or {}).get(col_name, [])
            if vals:
                preview = " | ".join(str(v)[:40] for v in vals[:TABLE_SAMPLE_LIMIT])
                line += f"  →  sample values: {preview}"
            col_lines.append(line)
        col_str = "\n".join(col_lines) if col_lines else "    (none)"

        rel_lines = [
            f"    [{table}].[{col}] → [{ot}].[{oc}]"
            for col, ot, oc in rels.get(table, [])
        ]
        rel_str = "\n".join(rel_lines) if rel_lines else "    (none)"

        blocks.append(
            f"TABLE: [{table}]  ({len(cols)} columns)\n"
            f"  Columns with real data samples:\n{col_str}\n"
            f"  Relationships:\n{rel_str}"
        )
    return "\n\n".join(blocks)


_SCHEMA_SELECTOR_PROMPT = """You are a database schema analyst.
Your only job: given a user question and a set of candidate tables, identify which
table(s) and column(s) contain the data needed to answer the question.

You are shown each candidate table with:
- Every column name and its data type
- REAL sample values actually stored in that column

=== USER QUESTION ===
{question}

=== CANDIDATE TABLES (with complete columns and real data samples) ===
{rich_context}

=== ANALYSIS INSTRUCTIONS ===
1. Read every column and its sample values carefully for each table.
2. STRONG EVIDENCE — if a sample value in a column directly matches or contains a term
   from the user question, that column in that table is almost certainly the right one.
   Example logic: if the question mentions a specific code or identifier, find the column
   whose sample values contain that exact code.
3. COLUMN NAME EVIDENCE — if a column name semantically matches a concept in the question
   (e.g. a column named after a status, type, category, or identifier that the user asked about),
   that is strong evidence for that column.
4. Select the MINIMUM number of tables needed. One table is almost always enough.
5. For each filter you identify, specify:
   - The EXACT column name as it appears in the schema
   - "=" if the value is a short exact code/ID (int or short alpha-numeric)
   - "LIKE" if the value is a text phrase or partial name
   - The search value exactly as the user stated

Return ONLY valid JSON — no markdown, no explanation:
{{
  "selected_tables": ["ExactTableName"],
  "reasoning": "one sentence: which column/value evidence led to this choice",
  "filter_hints": {{
    "ExactTableName": [
      {{"column": "ExactColumnName", "operator": "=", "value": "ExactValue"}}
    ]
  }}
}}"""


def _llm_select_schema(
    question: str,
    candidate_tables: list,
    schema: dict,
    rels: dict,
    sample_cache: dict,
    llm,
) -> tuple:
    """Use the LLM to analyze all candidate tables with their full schemas and
    real sample values, then explicitly choose which tables and columns to use.

    Returns (selected_tables, filter_hints):
    - selected_tables: validated list of table names from schema
    - filter_hints: {table: [{column, operator, value}]} for WHERE generation
    Falls back to all candidates on any failure.
    """
    if not candidate_tables:
        return candidate_tables, {}

    rich_context = _rich_schema_with_samples(candidate_tables, schema, rels, sample_cache)

    try:
        raw = _llm_complete(
            _SCHEMA_SELECTOR_PROMPT.format(question=question, rich_context=rich_context),
            llm,
        )
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            logger.warning("[SCHEMA-SELECT] No JSON returned — using all candidates")
            return candidate_tables, {}

        result    = json.loads(m.group())
        selected  = result.get("selected_tables", [])
        hints     = result.get("filter_hints", {})
        reasoning = result.get("reasoning", "")

        # Validate — keep only tables that genuinely exist in the schema
        valid = [t for t in selected if t in schema]
        if not valid:
            logger.warning("[SCHEMA-SELECT] No valid tables in result — using all candidates")
            return candidate_tables, {}

        logger.info("[SCHEMA-SELECT] selected=%s reasoning=%r", valid, reasoning[:200])
        return valid, hints

    except Exception as exc:
        logger.warning("[SCHEMA-SELECT] Failed (%s) — using all candidates", exc)
        return candidate_tables, {}

# ============================================================
# VALUE GROUNDING
# ============================================================

def _retrieve_value_hints(question: str, value_retriever) -> str:
    if value_retriever is None:
        return ""
    try:
        nodes = value_retriever.retrieve(question)
        if not nodes:
            return ""
        seen_vals: set = set()
        lines = []
        for n in nodes:
            if len(lines) >= VALUE_HINT_MAX:
                break
            meta = getattr(n.node, "metadata", {}) or {}
            val   = meta.get("value", "")
            table = meta.get("table", "")
            col   = meta.get("column", "")
            key   = f"{table}.{col}:{val}"
            if key in seen_vals or not val:
                continue
            seen_vals.add(key)
            lines.append(f'  "{val}" → [{table}].[{col}]')
        if not lines:
            return ""
        return "=== REAL DATA VALUES (use these in WHERE clauses) ===\n" + "\n".join(lines) + "\n"
    except Exception as e:
        logger.warning(f"Value grounding error: {e}")
        return ""

# ============================================================
# LLM WRAPPER — provider-agnostic with exponential-backoff retry
# ============================================================

def _is_transient(err: Exception) -> bool:
    if hasattr(err, "status_code") and err.status_code in (500, 502, 503, 429):
        return True
    return any(k in str(err).lower() for k in ("rate limit", "429", "quota", "timeout", "connection", "temporarily"))

def _llm_complete(prompt: str, groq_llm) -> str:
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return groq_llm.complete(prompt).text.strip()
        except ValueError:
            raise
        except Exception as e:
            if not _is_transient(e) or attempt == MAX_RETRY_ATTEMPTS:
                raise
            delay = 2.0 * (2 ** (attempt - 1))
            logger.warning(f"LLM transient error (attempt {attempt}), retry in {delay:.0f}s: {e}")
            time.sleep(delay)
    raise RuntimeError("LLM: all retry attempts exhausted")

# ============================================================
# INTENT CLASSIFIER
# ============================================================

_INTENT_PROMPT = """You are a classifier for a database assistant.

Decide if the user's input is a DATABASE QUERY — meaning they are asking for
records, counts, lists, details, codes, names, values, or any information that
would be stored in a relational database.

If the input is a greeting, farewell, thank-you, or casual conversation unrelated
to data, classify it as OTHER.
If it asks for any kind of data — even something unfamiliar — classify it as DATA.
When in doubt, choose DATA.

User input: "{question}"

Reply with exactly one word — DATA or OTHER:"""

_CHITCHAT_PROMPT = """You are a friendly database assistant.
The user said something that is not a data query. Respond naturally in 2-4 sentences.
- Greeting → introduce yourself briefly and say you can answer questions about the database.
- Farewell → say goodbye warmly.
- Thanks / compliment → acknowledge and invite further questions.
- Help / capability question → explain you can look up records, counts, details, codes, and lists from the database.
- Anything else → respond politely and invite them to ask a data question.

Never make up data. Never mention SQL.

User said: "{question}"

Response:"""

def _classify_intent(question: str, groq_llm) -> str:
    try:
        result = _llm_complete(_INTENT_PROMPT.format(question=question), groq_llm)
        return "OTHER" if result.strip().upper().startswith("OTHER") else "DATA"
    except Exception:
        return "DATA"

def _chitchat_reply(question: str, groq_llm) -> str:
    return _llm_complete(_CHITCHAT_PROMPT.format(question=question), groq_llm)

# ============================================================
# CONVERSATION HISTORY  (per-session, keyed by session_id)
# ============================================================

_session_histories: dict = {}   # {session_id: deque(maxlen=HISTORY_MAX)}
_session_histories_lock = threading.Lock()

def _get_session_history(session_id: str) -> deque:
    """Return (creating if needed) the history deque for a given session."""
    with _session_histories_lock:
        if session_id not in _session_histories:
            _session_histories[session_id] = deque(maxlen=HISTORY_MAX)
        return _session_histories[session_id]


def _build_contextual_question(question: str, session_id: str = "default") -> str:
    history = _get_session_history(session_id)
    if not history:
        return question
    lines = []
    for prev_q, prev_a in list(history)[-2:]:
        lines.append(f"[Previous question]: {prev_q}")
        lines.append(f"[Previous answer summary]: {str(prev_a)[:200]}")
    lines.append(f"[Current question]: {question}")
    return "\n".join(lines)

# ============================================================
# QUERY PLANNER
# ============================================================

_PLANNER_PROMPT = """You are a database query planner. Your job is to analyze the question and
the provided table schemas, then produce a precise query plan.

=== TABLE SCHEMAS + RELATIONSHIPS ===
{schema_context}

=== QUESTION ===
{question}

Return a JSON object with exactly these keys:
{{
  "intent": "one sentence describing what the user wants",
  "required_tables": ["TableA"],
  "join_paths": ["[TableA].[col] = [TableB].[col]"],
  "filters": [],
  "aggregations": [],
  "ordering": {{
    "column": "exact_column_name_from_schema",
    "direction": "DESC"
  }}
}}

=== RULES ===
TABLES:
- required_tables must contain ONLY table names visible in the schema above. Never invent names.
- Use the MINIMUM number of tables needed. One table is better than two.
- join_paths must use EXACT column names from the schema. If no join is needed, set to [].

FILTERS:
- List ONLY conditions the user explicitly mentioned. Never invent filter values.
- Text / string / name / code / category columns → LIKE operator: "[ColumnName] LIKE '%value%'"
- Numeric IDs or short exact codes → = operator: "[ColumnName] = 'value'"
- Never use = on free-text columns. Always use LIKE for text searches.

ORDERING:
- Set ordering ONLY when the user asks for "top N", "highest", "lowest", "most", "least", "ranked", "best".
- Pick the most relevant NUMERIC column from the schema for ordering (look for decimal, int, float types).
- Set ordering.column to the EXACT column name from the schema — do not guess or invent.
- If no suitable numeric column exists in the schema, set ordering to null.
- If the question does NOT ask for ranking, set ordering to null.

TABLE DISAMBIGUATION (critical when schema contains similarly-named tables):
- Read EVERY table's column list carefully before choosing.
- Choose the table whose columns most directly answer the question.
- Never rely on the table name alone — always examine which columns match what the question needs.
- If two tables are equally plausible, add BOTH to required_tables.

{filter_hints}
Return ONLY valid JSON. No extra text, no markdown."""

def _plan_query(question: str, schema_context: str, groq_llm, filter_hints: dict = None) -> tuple:
    hints_str = ""
    if filter_hints:
        lines = []
        for table, hints in filter_hints.items():
            for h in hints:
                col, op, val = h.get("column", ""), h.get("operator", "="), h.get("value", "")
                if col and val:
                    lines.append(f"  [{table}].[{col}] {op} '{val}'")
        if lines:
            hints_str = (
                "=== LLM-IDENTIFIED FILTER COLUMNS (use these exactly in WHERE) ===\n"
                + "\n".join(lines) + "\n\n"
            )
    raw = _llm_complete(
        _PLANNER_PROMPT.format(schema_context=schema_context, question=question, filter_hints=hints_str),
        groq_llm,
    )
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            plan_dict = json.loads(m.group())
            return json.dumps(plan_dict, indent=2), plan_dict
        except json.JSONDecodeError:
            pass
    return raw, {}

_RANKING_WORDS = re.compile(
    r'\b(top\s+\d+|best|highest|lowest|most|least|ranked|ranking|rank)\b', re.IGNORECASE
)

def _needs_ranking(question: str) -> bool:
    return bool(_RANKING_WORDS.search(question))

# ============================================================
# SQL GENERATION
# ============================================================

_SQL_GEN_PROMPT = """You are a senior T-SQL expert. Write exactly ONE correct T-SQL SELECT statement.

=== SIMPLICITY FIRST ===
Write the SIMPLEST query that correctly answers the question.
- Use ONE table unless a JOIN is strictly necessary to answer the question.
- Add WHERE only for conditions the user explicitly stated.
- Add ORDER BY only if the user asked for ranking or sorting.
- Never add clauses to "seem thorough" — correctness beats complexity.

=== MANDATORY RULES ===
1. ONLY SELECT. Never write INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, EXEC, MERGE.
2. ALWAYS bracket every identifier: [TableName].[ColumnName].
3. Use ONLY table and column names that appear in the SCHEMA section below. Never invent names.
4. Row limit: use TOP {row_cap} unless the user specified a number (then use that number).
5. Numeric text columns: use AVG(TRY_CAST([col] AS FLOAT)) for averages.

6. TEXT SEARCH — CRITICAL (study the schema column types before writing WHERE):
   a. For nvarchar / varchar / text columns: ALWAYS use LIKE '%value%'. Never use = for text.
   b. For int / numeric / exact-code columns: use = 'value'.
   c. SQL Server collation is case-insensitive — no LOWER() or UPPER() needed.
   d. Multi-word text search: [col] LIKE '%word1%' AND [col] LIKE '%word2%'.
   e. For name/person lookups: search all name-like columns with OR.
   f. Look at the schema column TYPE to decide: nvarchar/varchar → LIKE, int/decimal → =.

7. RANKING — "top N / highest / lowest / most / least / best / ranked":
   a. Use ORDER BY [ordering_column] DESC (or ASC for lowest/least).
   b. The ordering column comes from the query plan — use it EXACTLY as named in schema.
   c. Do NOT add WHERE unless the user mentioned a specific filter.
   d. If plan ordering is null and no numeric column is in the schema, reply with exactly:
      CLARIFY: What metric should I use to rank? (e.g. which numeric column?)

8. SIMILARLY-NAMED TABLES — examine columns, not just names:
   a. Look at each candidate table's column list in the schema.
   b. Pick the table whose columns directly match what the question needs.
   c. If genuinely uncertain, query both with UNION ALL:
      SELECT [col] FROM [TableA] WHERE ... UNION ALL SELECT [col] FROM [TableB] WHERE ...

9. Value hints (real DB values) — use these exactly in WHERE clauses if relevant.

10. Return ONLY raw T-SQL. No markdown fences, no explanation.

=== SCHEMA ===
{schema_context}

{value_hints}
=== QUERY PLAN ===
{plan}

=== QUESTION ===
{question}

SQL:"""

def _generate_sql(question: str, schema_context: str, value_hints: str, plan: str, groq_llm) -> str:
    prompt = _SQL_GEN_PROMPT.format(
        row_cap=ROW_CAP,
        schema_context=schema_context,
        value_hints=value_hints,
        plan=plan,
        question=question,
    )
    sql = _llm_complete(prompt, groq_llm)
    return re.sub(r'^```(?:sql)?\s*', '', sql, flags=re.IGNORECASE).rstrip('`').strip()

# ============================================================
# SQL SELF-CRITIQUE
# ============================================================

_CRITIQUE_PROMPT = """Review this T-SQL SELECT statement against the schema and fix only real problems.
Do NOT change correct SQL. Do NOT add clauses the user did not ask for.

Question: {question}
SQL: {sql}
Schema: {schema_context}

Check each item below in order:
1. INVENTED NAMES — every table and column in the SQL must exist in the Schema above.
   If any name is missing from the schema, replace it with the closest real name from the schema.
2. CARTESIAN PRODUCT — every JOIN must have an ON condition. Add missing ON if needed.
3. STRING vs EXACT MATCH — inspect each WHERE condition:
   - If the column type in the schema is nvarchar/varchar/text → must use LIKE '%value%', not = 'value'.
   - If the column type is int/decimal/numeric or the value is a short exact code → = is correct.
   Fix any nvarchar column that uses = 'text_value' by changing to LIKE '%text_value%'.
4. AGGREGATION INFLATION — if a 1-to-many JOIN exists before SUM/AVG/COUNT, check for over-counting.
5. MISSING ORDER BY — if the question asks "top N / best / highest / lowest / most / least"
   and SQL has no ORDER BY, add ORDER BY using the most relevant numeric column from the schema.
6. INVENTED WHERE — if SQL has WHERE conditions on values the user never mentioned, remove them.
7. UNNECESSARY JOIN — if one table provides all required columns, remove the extra JOIN.

If the SQL passes all checks, reply with exactly: PASS
If any check fails, reply with the corrected SQL only — raw SQL, no explanation, no markdown."""

def _critique_sql(sql: str, question: str, schema_context: str, groq_llm) -> str:
    if not ENABLE_SQL_CRITIQUE:
        return sql
    result = _llm_complete(
        _CRITIQUE_PROMPT.format(question=question, sql=sql, schema_context=schema_context),
        groq_llm,
    )
    if result.strip().upper() == "PASS":
        return sql
    corrected = re.sub(r'^```(?:sql)?\s*', '', result, flags=re.IGNORECASE).rstrip('`').strip()
    if corrected.upper().startswith("SELECT"):
        logger.info("SQL self-critique applied a correction.")
        return corrected
    return sql

# ============================================================
# ANSWER SYNTHESIS
# ============================================================

_SYNTHESIS_PROMPT = """You are a database assistant. A SQL query ran against the database.
Your job is to turn the raw query results into a clear, concise answer.

=== STRICT ANTI-HALLUCINATION RULES (follow exactly) ===
1. Answer ONLY from the data in the Results section below. Never add, infer, or invent anything.
2. Do NOT say "based on the data" or "it appears" — state facts directly.
3. Do NOT use "highest" or "lowest" unless the SQL contains ORDER BY that confirms ranking.
4. If Results are EMPTY:
   - Reply with: "No matching records were found for your query."
   - Do NOT explain why. Do NOT suggest what the user might have meant.
   - Do NOT say the data might exist somewhere else. Just report: not found.
5. If MULTIPLE rows are returned, list ALL of them — never summarize away individual rows.
6. Format multiple rows as a bullet list or a simple plain-text table.
7. Never show SQL, column names in raw format, or technical details.
8. Keep the tone concise and professional.

Question: {question}
SQL used: {sql}
Results: {results}

Answer:"""

def _synthesize_answer(question: str, sql: str, results: str, groq_llm) -> str:
    return _llm_complete(
        _SYNTHESIS_PROMPT.format(question=question, sql=sql, results=results),
        groq_llm,
    )

# ============================================================
# SCHEMA CONTEXT BUILDER
# ============================================================

def _schema_context_for_tables(tables: list, schema: dict, rels: dict) -> str:
    """Build the schema context shown to the LLM during planning and SQL generation.

    Shows ALL columns and ALL relationships for each table — no caps.
    Capping columns was the root cause of the LLM picking wrong columns or
    hallucinating column names that don't exist.
    """
    blocks = []
    for table in tables:
        cols = schema.get(table, [])
        col_lines = [f"  - {c} ({t})" for c, t in cols]
        col_str = "\n".join(col_lines) if col_lines else "  (none)"
        rel_lines = [
            f"  [{table}].[{col}] → [{ot}].[{oc}]"
            for col, ot, oc in rels.get(table, [])
        ]
        rel_str = "\n".join(rel_lines) if rel_lines else "  (none)"
        blocks.append(
            f"Table: [{table}] ({len(cols)} columns)\n"
            f"Columns:\n{col_str}\n"
            f"Relationships:\n{rel_str}"
        )
    return "\n\n".join(blocks)

# ============================================================
# INTERNAL ASK — returns (answer, sql_used, is_clarification)
# ============================================================

def _ask_internal(
    question: str,
    value_retriever,
    sql_db,
    schema: dict,
    rels: dict,
    groq_llm,
    all_tables: list,
    raw_retriever,         # VectorIndexRetriever — provides per-table dense scores
    bm25_index,            # BM25Okapi built at init
    bm25_doc_names: list,  # parallel table-name list for bm25_index
    session_id: str = "default",
) -> tuple:
    """Returns (answer, sql_used, is_clarification, raw_rows, col_keys, tables_used)."""

    history = _get_session_history(session_id)

    intent = _classify_intent(question, groq_llm)
    if intent == "OTHER":
        reply = _chitchat_reply(question, groq_llm)
        history.append((question, reply))
        return reply, "", False, [], [], []

    contextual_q = _build_contextual_question(question, session_id)

    # Step 1: hybrid retrieval — dense cosine + BM25 keyword scores combined
    candidate_tables = _retrieve_tables_hybrid(
        contextual_q,
        raw_retriever,
        bm25_index,
        bm25_doc_names,
        all_tables,
    )
    logger.info("Hybrid-retrieved candidates: %s", candidate_tables)

    # Step 2: LLM analyzes every candidate's full schema + real sample values
    #         and explicitly identifies which tables/columns to query
    tables, filter_hints = _llm_select_schema(
        contextual_q, candidate_tables, schema, rels, _sample_cache, groq_llm,
    )
    logger.info("LLM-selected tables: %s  filter_hints: %s", tables, filter_hints)

    schema_context = _schema_context_for_tables(tables, schema, rels)
    value_hints    = _retrieve_value_hints(contextual_q, value_retriever)

    plan_str, plan_dict = _plan_query(contextual_q, schema_context, groq_llm, filter_hints)
    logger.info("Plan: %s", plan_str[:200])

    if _needs_ranking(question):
        ordering = plan_dict.get("ordering")
        if not ordering or not ordering.get("column"):
            msg = (
                "I need a bit more context to answer that. "
                "What metric should I use to rank the results? "
                "For example: by a numeric column such as a count, amount, percentage, or duration?"
            )
            history.append((question, msg))
            return msg, "", True, [], [], tables

    sql = _generate_sql(contextual_q, schema_context, value_hints, plan_str, groq_llm)

    if sql.upper().startswith("CLARIFY:"):
        msg = sql[len("CLARIFY:"):].strip()
        history.append((question, msg))
        return msg, "", True, [], [], tables

    sql = _critique_sql(sql, contextual_q, schema_context, groq_llm)

    results_str, results_meta = sql_db.run_sql(sql)

    # Extract raw rows if LlamaIndex returns them in the metadata dict
    raw_rows: list = []
    col_keys: list = []
    if isinstance(results_meta, dict):
        raw_rows = results_meta.get("result", []) or []
        col_keys = results_meta.get("col_keys", []) or []

    # ------------------------------------------------------------------
    # Zero-row fallback: if primary query returned nothing AND the
    # retriever surfaced additional candidate tables that weren't used in
    # the SQL, re-run the full plan→generate→critique→execute pipeline
    # with those alternative tables.  This handles cases like
    # ActivityMasterMapping vs ActivityMasterMapping_New where the LLM
    # picked the wrong one from similar-named candidates.
    # ------------------------------------------------------------------
    if not raw_rows and len(tables) > 1:
        used_in_sql = {m.lower() for m in re.findall(r'\[([^\]]+)\]', sql)}
        alt_tables = [t for t in tables if t.lower() not in used_in_sql]
        if alt_tables:
            primary_used = [t for t in tables if t.lower() in used_in_sql]
            logger.info(
                "Zero rows from %s — retrying with alternative candidate tables: %s",
                primary_used, alt_tables,
            )
            try:
                alt_schema   = _schema_context_for_tables(alt_tables, schema, rels)
                alt_plan_str, _ = _plan_query(contextual_q, alt_schema, groq_llm)
                alt_sql      = _generate_sql(contextual_q, alt_schema, value_hints, alt_plan_str, groq_llm)
                alt_sql      = _critique_sql(alt_sql, contextual_q, alt_schema, groq_llm)
                alt_results_str, alt_results_meta = sql_db.run_sql(alt_sql)
                if isinstance(alt_results_meta, dict):
                    alt_rows = alt_results_meta.get("result", []) or []
                    if alt_rows:
                        logger.info(
                            "Alternative table retry returned %d rows — using these results (sql=%s)",
                            len(alt_rows), alt_sql[:120],
                        )
                        raw_rows    = alt_rows
                        col_keys    = alt_results_meta.get("col_keys", []) or []
                        sql         = alt_sql
                        tables      = alt_tables
                        results_str = alt_results_str
            except Exception as _retry_exc:
                logger.warning("Alternative table retry failed: %s", _retry_exc)

    answer = _synthesize_answer(question, sql, results_str, groq_llm)
    history.append((question, answer))
    return answer, sql, False, raw_rows, col_keys, tables

# ============================================================
# LAZY INITIALIZATION
# ============================================================

_init_lock  = threading.Lock()
_initialized = False

_engine          = None
_pipeline_llm    = None  # LlamaIndex LLM — provider chosen by LLM_PROVIDER env var
_schema_data: dict = {}
_rels_data: dict   = {}
_sql_db          = None
_raw_table_retriever = None   # VectorIndexRetriever — returns NodeWithScore (used by orchestrator routing + hybrid retrieval)
_value_retriever     = None
_all_tables: list    = []
_sample_cache: dict  = {}     # {table: {col: [val1, val2, ...]}} — populated at init, reused per query
_bm25_index          = None   # BM25Okapi index over rich table context strings
_bm25_doc_names: list = []    # parallel list of table names — index i ↔ _bm25_index corpus[i]


def _initialize() -> None:
    global _initialized, _engine, _pipeline_llm, _schema_data, _rels_data
    global _sql_db, _raw_table_retriever, _value_retriever, _all_tables, _sample_cache
    global _bm25_index, _bm25_doc_names

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.groq import Groq

        # --- validate DB credentials ---
        db_check = {
            "DB_SERVER": DB_SERVER, "DB_NAME": DB_NAME,
            "DB_READONLY_USER": DB_USER, "DB_READONLY_PASSWORD": DB_PASSWORD,
        }
        missing = [k for k, v in db_check.items() if not v]
        if missing:
            raise RuntimeError(
                f"Missing required env vars: {', '.join(missing)}. "
                f"Set them in your .env file."
            )
        placeholder = [k for k, v in db_check.items() if v and v.strip() in _PLACEHOLDERS]
        if placeholder:
            raise RuntimeError(
                f"Placeholder values detected for: {', '.join(placeholder)}. "
                f"Replace them with real values in your .env file."
            )

        # --- validate LLM credentials for chosen provider ---
        if LLM_PROVIDER == "groq":
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq. Set it in your .env file.")
            if GROQ_API_KEY.strip() in _PLACEHOLDERS:
                raise RuntimeError("GROQ_API_KEY is still a placeholder. Replace it with a real key.")
        elif LLM_PROVIDER == "gemini":
            if not GOOGLE_API_KEY:
                raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini. Set it in your .env file.")
            if GOOGLE_API_KEY.strip() in _PLACEHOLDERS:
                raise RuntimeError("GOOGLE_API_KEY is still a placeholder. Replace it with a real key.")
            if not GOOGLE_API_KEY.startswith("AIza"):
                logger.warning(
                    "GOOGLE_API_KEY does not look like a Google AI Studio key (expected prefix 'AIza'). "
                    "Get a valid key at https://aistudio.google.com/apikey"
                )

        logger.info(
            "DB config: server=%s db=%s user=%s | pipeline LLM provider=%s",
            DB_SERVER, DB_NAME, DB_USER, LLM_PROVIDER,
        )

        _t0 = time.time()
        if LLM_PROVIDER == "gemini":
            from llama_index.llms.gemini import Gemini
            # LlamaIndex Gemini requires "models/<name>" prefix
            _gemini_model = GEMINI_MODEL_NAME if GEMINI_MODEL_NAME.startswith("models/") else f"models/{GEMINI_MODEL_NAME}"
            logger.info("[INIT 1/7] Configuring Gemini LLM: model=%s", _gemini_model)
            _pipeline_llm = Gemini(
                model_name=_gemini_model,
                api_key=GOOGLE_API_KEY,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        else:
            # Default: groq (also covers any unrecognised provider — safe fallback)
            from llama_index.llms.groq import Groq
            logger.info("[INIT 1/7] Configuring Groq LLM: model=%s", GROQ_MODEL_NAME)
            _pipeline_llm = Groq(
                model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY,
                temperature=GROQ_TEMPERATURE, max_tokens=GROQ_MAX_TOKENS,
                context_window=GROQ_CONTEXT_WINDOW,
            )
        Settings.llm = _pipeline_llm
        logger.info("[INIT 1/7] done (%.1fs)", time.time() - _t0)

        _t = time.time()
        logger.info("[INIT 2/7] Loading embedding model: %s", EMBED_MODEL_NAME)
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device="cpu")
        Settings.embed_model = embed_model
        logger.info("[INIT 2/7] done (%.1fs)", time.time() - _t)

        _t = time.time()
        logger.info("[INIT 3/7] Connecting to %s/%s ...", DB_SERVER, DB_NAME)
        encoded_pw = quote_plus(DB_PASSWORD)
        _engine = create_engine(
            f"mssql+pyodbc://{DB_USER}:{encoded_pw}@{DB_SERVER}/{DB_NAME}"
            f"?driver={DB_DRIVER.replace(' ', '+')}&Connect+Timeout={DB_CONNECT_TIMEOUT}",
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"timeout": DB_QUERY_TIMEOUT},
        ).execution_options(timeout=DB_QUERY_TIMEOUT)

        with _engine.connect() as c:
            ver = c.execute(text("SELECT @@VERSION")).fetchone()[0]
            logger.info("DB connected: %s", ver[:100])
        logger.info("[INIT 3/7] done (%.1fs)", time.time() - _t)

        _t = time.time()
        logger.info("[INIT 4/7] Discovering all tables from live DB ...")
        inspector = sa_inspect(_engine)
        all_db_tables = inspector.get_table_names()
        if not all_db_tables:
            raise RuntimeError(
                f"No tables found in database '{DB_NAME}' on server '{DB_SERVER}'. "
                f"Check that the user '{DB_USER}' has SELECT permission."
            )
        target_tables = [t for t in all_db_tables if t not in DB_EXCLUDE_TABLES]
        logger.info("[INIT 4/7] Discovered %d tables (%.1fs)", len(target_tables), time.time() - _t)

        _t = time.time()
        logger.info("[INIT 5/7] Introspecting columns and types ...")
        _schema_data = _discover_schema(_engine, target_tables)
        found = list(_schema_data.keys())
        if not found:
            raise RuntimeError(
                f"No tables could be introspected from '{DB_NAME}'. "
                f"Verify DB_READONLY_USER has schema-read permissions."
            )
        logger.info(
            "[INIT 5/7] Schema: %d tables, %d total columns (%.1fs)",
            len(found), sum(len(v) for v in _schema_data.values()), time.time() - _t,
        )

        _t = time.time()
        logger.info("[INIT 5b] Discovering FK / column-name relationships ...")
        _rels_data  = _discover_relationships(_engine, _schema_data)
        _all_tables = found
        logger.info("[INIT 5b] done (%.1fs)", time.time() - _t)

        _t = time.time()
        logger.info("[INIT 6/7] Building SQLAlchemy metadata + SafeSQLDatabase ...")
        metadata = _build_metadata_from_schema(_schema_data)
        SafeSQLDatabase = _get_sql_db_class()
        _sql_db = SafeSQLDatabase(
            _engine, include_tables=found, row_cap=ROW_CAP, metadata=metadata,
        )
        logger.info("[INIT 6/7] done (%.1fs)", time.time() - _t)

        _t = time.time()
        logger.info("[INIT 7/7] Building / loading vector indexes ...")
        logger.info("[INIT 7/7] Sampling table values for embedding + BM25 ...")
        sample_values = _sample_table_values(_engine, _schema_data)
        _sample_cache.update(sample_values)
        logger.info("[INIT 7/7] Sampled %d tables", len(sample_values))

        table_index          = _build_or_load_table_index(_sql_db, _schema_data, _rels_data, sample_values)
        # VectorIndexRetriever returns NodeWithScore — used by orchestrator routing + hybrid retrieval.
        # top_k = all tables so every table gets a dense score for hybrid combination.
        _raw_table_retriever = table_index._index.as_retriever(similarity_top_k=len(_schema_data))

        # BM25 sparse index — built over the same rich context strings as the dense index.
        # Using all-table retrieval from raw_retriever means BM25 covers tables the
        # dense top-K might miss when the question is keyword-heavy (e.g. exact codes).
        _bm25_index, _bm25_doc_names = _build_bm25_index(_schema_data, _rels_data, sample_values)

        value_index      = _build_or_load_value_index(_engine, _schema_data, embed_model)
        _value_retriever = value_index.as_retriever(similarity_top_k=VALUE_TOP_K) if value_index else None
        logger.info("[INIT 7/7] done (%.1fs)", time.time() - _t)

        _initialized = True
        logger.info(
            "DB pipeline ready. Total init time: %.1fs",
            time.time() - _t0,
        )

# ============================================================
# PUBLIC API — matches the contract expected by db_tool.py
# ============================================================

def ask(question: str, session_id: Optional[str] = None):
    """Process a natural-language question and return a FinalResponse.

    Called by the orchestrator's query_database tool (in-process).
    """
    from .models import FinalResponse

    start = time.time()
    try:
        _initialize()
        sid = session_id or "default"
        answer, sql_used, is_clarification, raw_rows, col_keys, tables_used = _ask_internal(
            question,
            _value_retriever,
            _sql_db,
            _schema_data,
            _rels_data,
            _pipeline_llm,
            _all_tables,
            _raw_table_retriever,
            _bm25_index,
            _bm25_doc_names,
            session_id=sid,
        )
        elapsed = (time.time() - start) * 1000

        if is_clarification:
            return FinalResponse(
                success=False,
                clarification_needed=True,
                clarification_question=answer,
                suggestions=[],
                tables_used=tables_used or [],
                execution_time_ms=elapsed,
            )

        # Convert raw row tuples → list of dicts (serialisable).
        # Guard against rows shorter than col_keys to avoid IndexError.
        try:
            n = len(col_keys)
            rows_as_dicts = [
                {col_keys[i]: row[i] for i in range(min(n, len(row)))}
                for row in raw_rows
            ] if col_keys and raw_rows else []
        except Exception as exc:
            logger.warning("[ASK] Row serialization failed: %s", exc)
            rows_as_dicts = []

        return FinalResponse(
            success=True,
            answer=answer,
            sql=sql_used or None,
            row_count=len(rows_as_dicts),
            data=rows_as_dicts,
            tables_used=tables_used or [],
            execution_time_ms=elapsed,
        )

    except ValueError as e:
        logger.warning(f"Safety block: {e}")
        return FinalResponse(
            success=False,
            error=str(e),
            execution_time_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        logger.exception(f"Pipeline error for question: {question!r}")
        return FinalResponse(
            success=False,
            error=str(e),
            execution_time_ms=(time.time() - start) * 1000,
        )


def shutdown() -> None:
    """Release DB connections. Called from orchestrator shutdown."""
    global _engine, _initialized
    if _engine:
        _engine.dispose()
        logger.info("DB engine disposed.")
    _initialized = False
    logger.info("DB pipeline shut down.")
