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

ROW_CAP            = int(os.getenv("ROW_CAP", "100"))
RETRIEVER_TOP_K    = int(os.getenv("RETRIEVER_TOP_K", "5"))
VALUE_TOP_K        = int(os.getenv("VALUE_TOP_K", "2"))
MAX_QUERY_LEN      = int(os.getenv("MAX_QUERY_LEN", "500"))
HISTORY_MAX        = int(os.getenv("HISTORY_MAX", "5"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
VALUE_SAMPLE_LIMIT = int(os.getenv("VALUE_SAMPLE_LIMIT", "50"))   # rows per column
ENABLE_SQL_CRITIQUE = os.getenv("ENABLE_SQL_CRITIQUE", "1") == "1"

SCHEMA_MAX_COLS = int(os.getenv("SCHEMA_MAX_COLS", "20"))
SCHEMA_MAX_RELS = int(os.getenv("SCHEMA_MAX_RELS", "5"))
VALUE_HINT_MAX  = int(os.getenv("VALUE_HINT_MAX",  "4"))

# VALUE_INDEX_ENABLED=0 disables value grounding entirely — speeds up first-run init
# significantly on large databases (avoids hundreds of per-column DB queries).
# VALUE_INDEX_MAX_COLS caps how many varchar columns are sampled when enabled.
VALUE_INDEX_ENABLED  = os.getenv("VALUE_INDEX_ENABLED",  "0") == "1"
VALUE_INDEX_MAX_COLS = int(os.getenv("VALUE_INDEX_MAX_COLS", "100"))

_exclude_raw = os.getenv("DB_EXCLUDE_TABLES", "")
DB_EXCLUDE_TABLES: set = {t.strip() for t in _exclude_raw.split(",") if t.strip()}

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
                        f"SELECT DISTINCT TOP {VALUE_SAMPLE_LIMIT} [{col}] "
                        f"FROM [{table}] WHERE [{col}] IS NOT NULL AND LEN([{col}]) < 150"
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

def _rich_table_context(table: str, schema: dict, rels: dict) -> str:
    cols = schema.get(table, [])
    col_str = ", ".join(f"{c} ({t})" for c, t in cols)
    rel_lines = [
        f"  [{table}].[{col}] → [{other_table}].[{other_col}]"
        for col, other_table, other_col in rels.get(table, [])
    ]
    rel_str = "\n".join(rel_lines) if rel_lines else "  (none)"
    return f"Table: [{table}]\nColumns: {col_str}\nRelationships:\n{rel_str}"

def _build_or_load_table_index(sql_db, schema: dict, rels: dict):
    from llama_index.core import VectorStoreIndex
    from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema

    current_hash = _schema_hash(schema)
    stored_hash  = _load_hash(_TABLE_HASH_FILE)
    force        = os.getenv("REBUILD_INDEX", "0") == "1"
    index_exists = os.path.isfile(os.path.join(TABLE_INDEX_DIR, "index_store.json"))

    table_node_mapping = SQLTableNodeMapping(sql_db)
    table_schema_objs  = [
        SQLTableSchema(
            table_name=t,
            context_str=_rich_table_context(t, schema, rels),
        )
        for t in schema
    ]

    if index_exists and not force and stored_hash == current_hash:
        logger.info("Loading persisted table index...")
        idx = ObjectIndex.from_persist_dir(
            persist_dir=TABLE_INDEX_DIR,
            object_node_mapping=table_node_mapping,
        )
        logger.info("Table index loaded.")
        return idx

    reason = "forced" if force else ("schema changed" if stored_hash != current_hash else "first run")
    logger.info(f"Building table index ({reason})...")
    idx = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex)
    idx.persist(persist_dir=TABLE_INDEX_DIR)
    _save_hash(_TABLE_HASH_FILE, current_hash)
    logger.info("Table index built and saved.")
    return idx

# ============================================================
# TABLE RETRIEVAL
# ============================================================

def _retrieve_tables(question: str, retriever, fallback: list) -> list:
    try:
        results = retriever.retrieve(question)
        names = []
        for r in results:
            if hasattr(r, "table_name"):
                names.append(r.table_name)
            elif hasattr(r, "node"):
                meta = getattr(r.node, "metadata", {}) or {}
                if "table_name" in meta:
                    names.append(meta["table_name"])
                else:
                    m = re.match(r'Table:\s*\[([^\]]+)\]', getattr(r.node, "text", "") or "")
                    if m:
                        names.append(m.group(1))
        seen: set = set()
        unique = [n for n in names if not (n in seen or seen.add(n))]
        return unique if unique else fallback[:RETRIEVER_TOP_K]
    except Exception as e:
        logger.warning(f"Table retrieval error ({e}), using fallback.")
        return fallback[:RETRIEVER_TOP_K]

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

_INTENT_PROMPT = """You are a classifier for a drilling/well-operations database assistant.

Is the user's input a DATABASE QUERY (asking for data about wells, employees, crews,
equipment, tasks, revenue, drilling sequences, or any business data)?

User input: "{question}"

Reply with exactly one word — DATA or OTHER:"""

_CHITCHAT_PROMPT = """You are a friendly assistant for a drilling and well operations database.
The user said something that is NOT a database query.

Respond naturally in 2-4 sentences:
- Greeting → briefly introduce yourself and mention what you can answer.
- Farewell → say goodbye warmly.
- Thanks / compliment → acknowledge and invite further questions.
- Help / capability question → list your topics with 1-2 examples.
- Anything else → respond politely and redirect toward data questions.

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
# CONVERSATION HISTORY
# ============================================================

_history: deque = deque(maxlen=HISTORY_MAX)

def _build_contextual_question(question: str) -> str:
    if not _history:
        return question
    lines = []
    for prev_q, prev_a in list(_history)[-2:]:
        lines.append(f"[Previous question]: {prev_q}")
        lines.append(f"[Previous answer summary]: {str(prev_a)[:200]}")
    lines.append(f"[Current question]: {question}")
    return "\n".join(lines)

# ============================================================
# QUERY PLANNER
# ============================================================

_PLANNER_PROMPT = """You are a database query planner for a drilling and well operations database.

Given the question and table schemas, produce a concise query plan.

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
    "column": "column_name_from_schema",
    "direction": "DESC"
  }}
}}

Rules:
- required_tables must be actual table names from the schema above.
- join_paths must use EXACT column names from the schema. If no join needed, set to [].
- filters: list ONLY conditions the user explicitly mentioned — never invent values.
  * String/text/category/type/name columns → use LIKE operator: "[Type] LIKE '%Civil%'"
  * Numeric IDs or exact lookup codes → use =: "[Well_ID] = '31477'"
  * NEVER write "column = 'text_value'" — always LIKE for free-text.
- ordering rules (CRITICAL):
  * If the user asks for "top N", "best", "highest", "lowest", "most", "least", "ranked":
    - Pick the most relevant numeric column from the retrieved tables.
    - Set ordering.column to that column name (EXACT as in schema) and direction to DESC (or ASC for lowest/least).
    - Prefer columns with names like: progress, actual_progress, pms, manhours, duration, Normal_duration, over_all_progress_percentages, Amount, Revenue.
  * If no suitable numeric column exists, set ordering to null — do NOT guess.
  * If the question does NOT ask for ranking, set ordering to null.

=== TABLE DISAMBIGUATION (critical when schema contains similarly-named tables) ===
When multiple tables have similar names (e.g. ActivityMasterMapping and ActivityMasterMapping_New):
1. Compare their column schemas CAREFULLY — choose the table whose columns best match the query.
2. A "_New" or "_Updated" suffix usually means the most current dataset — prefer it when both seem equally relevant.
3. If the question asks about codes, types, or categories, look for columns named [Code], [Type], [Activity_Code], [Category] — pick the table that has those columns.
4. If BOTH tables are equally plausible, add BOTH to required_tables and note "use UNION ALL in the SQL to query both".
5. NEVER blindly pick the first / alphabetically-first table — always examine columns.

- Return ONLY valid JSON, no extra text."""

def _plan_query(question: str, schema_context: str, groq_llm) -> tuple:
    raw = _llm_complete(
        _PLANNER_PROMPT.format(schema_context=schema_context, question=question),
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

_SQL_GEN_PROMPT = """You are a senior T-SQL expert for a drilling and well operations database.
Write exactly ONE correct SELECT statement.

=== SIMPLICITY RULE (read this first) ===
Write the SIMPLEST query that answers the question.
- Use only ONE table unless the question clearly requires data from multiple tables.
- Add a WHERE clause ONLY if the user explicitly mentioned a filter.
- Add ORDER BY ONLY if the user asked for ranking/sorting.
- Add a JOIN ONLY if the answer cannot be found in a single table.
- Never add complexity to "seem thorough". Simple is correct.

=== STRICT RULES ===
1. ONLY SELECT. Never INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, EXEC, MERGE.
2. ALWAYS bracket names: [TableName].[ColumnName].
3. Use ONLY tables and columns from the SCHEMA below — never invent names.
4. TOP N: use the number from the question. If no number, use TOP {row_cap}.
5. For numeric averages on text columns: AVG(TRY_CAST([col] AS FLOAT)).
6. TEXT / CATEGORY / NAME SEARCH (CRITICAL — must follow exactly):
   a. ALWAYS use LIKE '%value%' for string, name, label, category, or type columns.
   b. NEVER write = 'text_value' for string columns — it will miss rows.
   c. Chain-of-thought examples:
      • "Civil activity code"  → WHERE [Type] LIKE '%Civil%'        ✓
      • "Civil activity code"  → WHERE [Type] = 'Civil'             ✗ (wrong)
      • "well 31477"           → WHERE [Well_ID] = '31477'          ✓ (numeric ID, = is correct)
      • "Nimr field"           → WHERE [Field] LIKE '%Nimr%'        ✓
   d. SQL Server CI collation handles case automatically — no LOWER() needed.
   e. For person lookups, search BOTH [Name] AND [Email] columns with OR if both exist.
   f. For multi-word searches: LIKE '%word1%' AND [col] LIKE '%word2%'.
7. RANKING — when "top N", "best", "highest", "lowest", "most", "least":
   a. Use ORDER BY with the column from the plan's ordering field.
   b. Do NOT add WHERE unless user mentioned a specific filter.
   c. If plan ordering is null and no numeric column is obvious, reply with exactly:
      CLARIFY: What metric should I use to rank? (e.g. progress, revenue, duration)
8. Value hints below are real DB values — use them as-is in WHERE if relevant.
9. SIMILARLY-NAMED TABLES (CRITICAL when schema has multiple candidates):
   a. If the plan lists two tables with similar names, pick the one whose columns
      best match the query (e.g. has [Activity_Code], [Type], [Code] for code lookups).
   b. Prefer "_New" / "_Updated" variants — they typically hold current data.
   c. If you are uncertain which table has the data, use UNION ALL to query both:
      SELECT [col] FROM [TableA] WHERE ... UNION ALL SELECT [col] FROM [TableB] WHERE ...
   d. Never pick a table solely because it appears first — examine column names.
10. Return ONLY raw SQL — no markdown, no explanation.

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

_CRITIQUE_PROMPT = """Review this T-SQL SELECT statement for errors. Fix only real problems.

Question: {question}
SQL: {sql}
Schema: {schema_context}

Check for:
1. Cartesian product — missing or wrong JOIN condition.
2. Column names that do not exist in the schema.
3. Aggregation inflation — 1-to-many join before SUM/AVG inflates results.
4. Ranking without ORDER BY — if the question asks "top N / best / highest / lowest / most / least"
   and the SQL has no ORDER BY, add one using the most relevant numeric column.
5. Invented WHERE filters — if SQL has WHERE conditions the user never mentioned, remove them.
6. Unnecessary JOIN — if one table suffices, simplify.
7. String equality instead of LIKE — if WHERE uses = 'text_value' on any name, label, category,
   type, or free-text column, change it to LIKE '%text_value%'.
   Exception: = is correct for numeric IDs (e.g. Well_ID = '31477').

If the SQL is correct, reply with exactly: PASS
If there is an error, reply with the corrected SQL only (raw SQL, no explanation)."""

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

_SYNTHESIS_PROMPT = """You are a helpful assistant answering questions about drilling and well operations data.
A SELECT query ran and returned results. Turn the raw data into a clear, friendly answer.

=== STRICT RULES ===
1. Answer ONLY from the query results below — never infer or add facts.
2. Never say "highest" or "lowest" unless ORDER BY in the SQL confirms ranking.
3. If results are EMPTY, say politely that nothing was found.
4. If MULTIPLE items match, list ALL of them.
5. Format multiple rows as a bullet list or simple table.
6. Keep tone warm, concise, and professional.
7. Never show SQL, raw column names, or technical error details.

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
    blocks = []
    for table in tables:
        cols = schema.get(table, [])
        shown_cols = cols[:SCHEMA_MAX_COLS]
        col_str = ", ".join(f"{c} ({t})" for c, t in shown_cols)
        if len(cols) > SCHEMA_MAX_COLS:
            col_str += f", ... ({len(cols) - SCHEMA_MAX_COLS} more)"
        rel_lines = [
            f"  [{table}].[{col}] → [{ot}].[{oc}]"
            for col, ot, oc in rels.get(table, [])[:SCHEMA_MAX_RELS]
        ]
        rel_str = "\n".join(rel_lines) if rel_lines else "  (none)"
        blocks.append(f"Table: [{table}]\nColumns: {col_str}\nRelationships:\n{rel_str}")
    return "\n\n".join(blocks)

# ============================================================
# INTERNAL ASK — returns (answer, sql_used, is_clarification)
# ============================================================

def _ask_internal(
    question: str,
    table_retriever,
    value_retriever,
    sql_db,
    schema: dict,
    rels: dict,
    groq_llm,
    all_tables: list,
) -> tuple:
    """Returns (answer, sql_used, is_clarification, raw_rows, col_keys, tables_used)."""

    intent = _classify_intent(question, groq_llm)
    if intent == "OTHER":
        reply = _chitchat_reply(question, groq_llm)
        _history.append((question, reply))
        return reply, "", False, [], [], []

    contextual_q = _build_contextual_question(question)

    tables = _retrieve_tables(contextual_q, table_retriever, all_tables)
    logger.info("Retrieved tables: %s", tables)

    schema_context = _schema_context_for_tables(tables, schema, rels)
    value_hints    = _retrieve_value_hints(contextual_q, value_retriever)

    plan_str, plan_dict = _plan_query(contextual_q, schema_context, groq_llm)
    logger.info("Plan: %s", plan_str[:200])

    if _needs_ranking(question):
        ordering = plan_dict.get("ordering")
        if not ordering or not ordering.get("column"):
            msg = (
                "I need a bit more context to answer that. "
                "What metric should I use to rank them? "
                "For example: by progress percentage, revenue, duration, manhours, or something else?"
            )
            _history.append((question, msg))
            return msg, "", True, [], [], tables

    sql = _generate_sql(contextual_q, schema_context, value_hints, plan_str, groq_llm)

    if sql.upper().startswith("CLARIFY:"):
        msg = sql[len("CLARIFY:"):].strip()
        _history.append((question, msg))
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
    _history.append((question, answer))
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
_table_retriever     = None
_raw_table_retriever = None   # VectorIndexRetriever — returns NodeWithScore (used by orchestrator routing)
_value_retriever     = None
_all_tables: list    = []


def _initialize() -> None:
    global _initialized, _engine, _pipeline_llm, _schema_data, _rels_data
    global _sql_db, _table_retriever, _raw_table_retriever, _value_retriever, _all_tables

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
        table_index          = _build_or_load_table_index(_sql_db, _schema_data, _rels_data)
        _table_retriever     = table_index.as_retriever(similarity_top_k=RETRIEVER_TOP_K)
        # Raw vector retriever used by the orchestrator's semantic routing node.
        # Returns NodeWithScore objects (with .score) — unlike ObjectRetriever which returns SQLTableSchema objects.
        _raw_table_retriever = table_index._index.as_retriever(similarity_top_k=RETRIEVER_TOP_K)
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
        answer, sql_used, is_clarification, raw_rows, col_keys, tables_used = _ask_internal(
            question,
            _table_retriever,
            _value_retriever,
            _sql_db,
            _schema_data,
            _rels_data,
            _pipeline_llm,
            _all_tables,
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

        # Convert raw row tuples → list of dicts (serialisable)
        try:
            rows_as_dicts = [
                {col_keys[i]: row[i] for i in range(len(col_keys))}
                for row in raw_rows
            ] if col_keys and raw_rows else []
        except Exception:
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
