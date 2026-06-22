"""
Production-ready Natural-Language → SQL assistant for SQL Server.

Pipeline per question:
  Intent check → Table retrieval → Value grounding → Relationship context →
  Query planner → SQL generation → Self-critique → Validate → Execute → Synthesize

All tunables are environment variables — see .env.example.
Schema is discovered dynamically from the live DB at startup (no hardcoded columns,
no hardcoded table list — every table in the database is included automatically).
"""

import os
import re
import sys
import time
import json
import hashlib
import logging
import pyodbc
from collections import deque
from logging.handlers import RotatingFileHandler
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect as sa_inspect, text, MetaData, Table, Column
from sqlalchemy.types import Integer, String, DateTime, Date, Time, Float, Boolean, LargeBinary
from llama_index.core import (
    SQLDatabase, VectorStoreIndex, StorageContext,
    load_index_from_storage, Settings, Document,
)
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Disable pyodbc's built-in connection pool — it conflicts with SQLAlchemy's pool
# and is the root cause of TCP reset errors ("connection forcibly closed") during
# long reflection operations on large databases.
pyodbc.pooling = False

# ============================================================
# CONFIG — all from environment
# ============================================================

DB_SERVER           = os.getenv("DB_SERVER")
DB_NAME             = os.getenv("DB_NAME")
DB_USER             = os.getenv("DB_READONLY_USER")
DB_PASSWORD         = os.getenv("DB_READONLY_PASSWORD")
DB_DRIVER           = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_CONNECT_TIMEOUT  = int(os.getenv("DB_CONNECT_TIMEOUT", "30"))
DB_QUERY_TIMEOUT    = int(os.getenv("DB_QUERY_TIMEOUT", "120"))
DB_POOL_SIZE        = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW     = int(os.getenv("DB_MAX_OVERFLOW", "10"))

GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME     = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS     = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
GROQ_CONTEXT_WINDOW = int(os.getenv("GROQ_CONTEXT_WINDOW", "32768"))

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")

_base_dir              = os.path.dirname(os.path.abspath(__file__))
TABLE_INDEX_DIR        = os.getenv("TABLE_INDEX_DIR",  os.path.join(_base_dir, "table_index_storage"))
VALUE_INDEX_DIR        = os.getenv("VALUE_INDEX_DIR",  os.path.join(_base_dir, "value_index_storage"))
LOG_DIR                = os.getenv("LOG_DIR",           os.path.join(_base_dir, "logs"))
LOG_LEVEL              = os.getenv("LOG_LEVEL", "INFO")
ROW_CAP                = int(os.getenv("ROW_CAP", "100"))
RETRIEVER_TOP_K        = int(os.getenv("RETRIEVER_TOP_K", "2"))
VALUE_TOP_K            = int(os.getenv("VALUE_TOP_K", "2"))
MAX_QUERY_LEN          = int(os.getenv("MAX_QUERY_LEN", "500"))
HISTORY_MAX            = int(os.getenv("HISTORY_MAX", "5"))
MAX_RETRY_ATTEMPTS     = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
VALUE_SAMPLE_LIMIT     = int(os.getenv("VALUE_SAMPLE_LIMIT", "300"))
ENABLE_SQL_CRITIQUE    = os.getenv("ENABLE_SQL_CRITIQUE", "1") == "1"

# Prompt size controls — keep Groq under its payload limit.
# Reduce these if you still hit 413 errors; increase for richer context.
SCHEMA_MAX_COLS = int(os.getenv("SCHEMA_MAX_COLS", "20"))  # columns shown per table
SCHEMA_MAX_RELS = int(os.getenv("SCHEMA_MAX_RELS", "5"))   # relationships shown per table
VALUE_HINT_MAX  = int(os.getenv("VALUE_HINT_MAX",  "4"))   # value hints shown in prompt

# Optional: comma-separated table names to EXCLUDE from discovery.
# e.g. DB_EXCLUDE_TABLES=sysdiagrams,__EFMigrationsHistory
_exclude_raw = os.getenv("DB_EXCLUDE_TABLES", "")
DB_EXCLUDE_TABLES: set[str] = {
    t.strip() for t in _exclude_raw.split(",") if t.strip()
}

# ============================================================
# LOGGING
# ============================================================

def _setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "assistant.log"),
        maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(ch)
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

class SafeSQLDatabase(SQLDatabase):
    """Validates + caps every query BEFORE it reaches the DB."""
    def __init__(self, *args, row_cap: int = ROW_CAP, metadata=None, **kwargs):
        if metadata is not None:
            # SQLDatabase.__init__ always calls self._metadata.reflect() even when
            # metadata is provided, triggering a full remote schema scan that times
            # out on large databases. Patching reflect to a no-op on our pre-built
            # metadata prevents that without touching any other code path.
            metadata.reflect = lambda *a, **kw: None
            kwargs["metadata"] = metadata
        super().__init__(*args, **kwargs)
        self._row_cap = row_cap

    def run_sql(self, command: str):
        _validate_select_only(command)
        command = _inject_row_cap(command, self._row_cap)
        logger.info(f"[SQL] {command[:400]}")
        return super().run_sql(command)

# ============================================================
# DYNAMIC SCHEMA DISCOVERY
# Fetches ALL tables/columns/types from the live DB — nothing hardcoded.
# ============================================================

_TYPE_NORMALIZE = re.compile(r'\(.*\)$')

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
    """
    Build a SQLAlchemy MetaData object entirely from the already-discovered schema dict.
    Zero database calls — fast, no timeouts, works for any number of tables.
    """
    metadata = MetaData()
    for table_name, cols in schema.items():
        columns = [
            Column(col_name, _SA_TYPE_MAP.get(col_type, String)())
            for col_name, col_type in cols
        ]
        Table(table_name, metadata, *columns)
    return metadata

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

def discover_schema(engine, tables: list[str]) -> dict[str, list[tuple[str, str]]]:
    inspector = sa_inspect(engine)
    all_db_tables = set(inspector.get_table_names())
    schema: dict[str, list[tuple[str, str]]] = {}
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

def discover_relationships(engine, schema: dict) -> dict[str, list[tuple]]:
    target_tables = list(schema.keys())
    rels: dict[str, list] = {t: [] for t in target_tables}
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

    col_index: dict[str, list[tuple[str, str]]] = {}
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

def _build_value_documents(engine, schema: dict) -> list[Document]:
    docs: list[Document] = []
    for table, cols in schema.items():
        for col, dtype in cols:
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
            except Exception:
                pass
    logger.info(f"Value index: {len(docs)} value documents collected.")
    return docs

def _build_or_load_value_index(engine, schema: dict, embed_model) -> VectorStoreIndex | None:
    current_hash = _schema_hash(schema)
    stored_hash = _load_hash(_VALUE_TABLE_HASH_FILE)
    force = os.getenv("REBUILD_INDEX", "0") == "1"
    index_exists = os.path.isfile(os.path.join(VALUE_INDEX_DIR, "index_store.json"))

    if index_exists and not force and stored_hash == current_hash:
        logger.info("Loading persisted value index...")
        try:
            sc = StorageContext.from_defaults(persist_dir=VALUE_INDEX_DIR)
            idx = load_index_from_storage(sc, embed_model=embed_model)
            logger.info("Value index loaded.")
            return idx
        except Exception as e:
            logger.warning(f"Value index load failed ({e}), rebuilding...")

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

def _build_or_load_table_index(
    sql_db: SafeSQLDatabase,
    schema: dict,
    rels: dict,
) -> ObjectIndex:
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

def _retrieve_tables(question: str, retriever, fallback: list[str]) -> list[str]:
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
    """Returns a formatted block of real DB values capped by VALUE_HINT_MAX
    to avoid inflating the Groq prompt beyond the payload limit."""
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
# LLM WRAPPER — Groq with exponential-backoff retry
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
            logger.warning(f"Groq transient error (attempt {attempt}), retry in {delay:.0f}s: {e}")
            time.sleep(delay)
    raise RuntimeError("Groq: all retry attempts exhausted")

# ============================================================
# INTENT CLASSIFIER — LLM-only, no hardcoded patterns.
# Falls back to DATA on any error so genuine queries always proceed.
# ============================================================

_INTENT_PROMPT = """You are a classifier for a drilling/well-operations database assistant.

Is the user's input a DATABASE QUERY (asking for data about wells, employees, crews,
equipment, tasks, revenue, drilling sequences, or any business data)?

User input: "{question}"

Reply with exactly one word — DATA or OTHER:"""

_CHITCHAT_PROMPT = """You are a friendly assistant for a drilling and well operations database.
The user said something that is NOT a database query.

Respond naturally in 2-4 sentences:
- Greeting → briefly introduce yourself and mention what you can answer (employees, crews,
  wells, tasks, equipment, revenue, drilling sequences, and all other operational data).
- Farewell → say goodbye warmly.
- Thanks / compliment → acknowledge and invite further questions.
- Help / capability question → list your topics with 1-2 examples.
- Anything else → respond politely and redirect toward data questions.

Never make up data. Never mention SQL.

User said: "{question}"

Response:"""

def _classify_intent(question: str, groq_llm) -> str:
    """Returns 'DATA' or 'OTHER'. Always falls back to 'DATA' on any error."""
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
# PIPELINE STEP 1 — QUERY PLANNER
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
- filters must list ONLY conditions explicitly mentioned by the user — never invent values.
- ordering rules (CRITICAL):
  * If the user asks for "top N", "best", "highest", "lowest", "most", "least", "ranked":
    - Pick the most relevant numeric column from the retrieved tables.
    - Set ordering.column to that column name (EXACT as in schema) and direction to DESC (or ASC for lowest/least).
    - Prefer columns with names like: progress, actual_progress, pms, manhours, duration, Normal_duration, over_all_progress_percentages, Amount, Revenue.
  * If no suitable numeric column exists, set ordering to null — do NOT guess.
  * If the question does NOT ask for ranking, set ordering to null.
- Return ONLY valid JSON, no extra text."""

def _plan_query(question: str, schema_context: str, groq_llm) -> tuple[str, dict]:
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
# PIPELINE STEP 2 — SQL GENERATION
# ============================================================

_SQL_GEN_PROMPT = """You are a senior T-SQL expert for a drilling and well operations database.
Write exactly ONE correct SELECT statement.

=== SIMPLICITY RULE (read this first) ===
Write the SIMPLEST query that answers the question.
- Use only ONE table unless the question clearly requires data from multiple tables.
- Add a WHERE clause ONLY if the user explicitly mentioned a filter (a name, a value, a date, a status).
- Add ORDER BY ONLY if the user asked for ranking/sorting (top N, highest, lowest, sorted by).
- Add a JOIN ONLY if the answer cannot be found in a single table.
- Never add complexity to "seem thorough". Simple is correct.

=== STRICT RULES ===
1. ONLY SELECT. Never INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, EXEC, MERGE.
2. ALWAYS bracket names: [TableName].[ColumnName].
3. Use ONLY tables and columns from the SCHEMA below — never invent names.
4. TOP N: use the number from the question (e.g. "top 4" → TOP 4). If no number, use TOP {row_cap}.
5. For numeric averages on text columns: AVG(TRY_CAST([col] AS FLOAT)).
6. TEXT / NAME SEARCH — when the user mentions a specific name or value:
   a. Use LIKE '%value%', never =, for names, labels, or free-text fields.
   b. SQL Server CI collation handles case — no LOWER() needed.
   c. For person lookups, search BOTH [Name] AND [Email] columns if they exist.
   d. For partial/misspelled names use the clearest root (e.g. 'rajsh' → '%raj%').
   e. Use = only for numeric IDs or explicit codes (Well_ID, task_code).
7. RANKING — when the question uses "top N", "best", "highest", "lowest", "most", "least":
   a. Use ORDER BY with the column from the plan's ordering field.
   b. Do NOT add any WHERE clause unless the user also mentioned a specific filter.
   c. If the plan ordering is null and no numeric column is obvious, reply with exactly:
      CLARIFY: What metric should I use to rank? (e.g. progress, revenue, duration)
8. Value hints below are real DB values — use them as-is in WHERE if relevant.
9. Return ONLY raw SQL — no markdown, no explanation.

=== SCHEMA ===
{schema_context}

{value_hints}
=== QUERY PLAN ===
{plan}

=== QUESTION ===
{question}

SQL:"""

def _generate_sql(question: str, schema_context: str, value_hints: str, plan: str,
                  groq_llm) -> str:
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
# PIPELINE STEP 3 — SQL SELF-CRITIQUE
# ============================================================

_CRITIQUE_PROMPT = """Review this T-SQL SELECT statement for errors. Fix only real problems.

Question: {question}
SQL: {sql}
Schema: {schema_context}

Check for these errors:
1. Cartesian product — missing or wrong JOIN condition.
2. Column names that do not exist in the schema.
3. Aggregation inflation — 1-to-many join before SUM/AVG inflates results.
4. Ranking without ORDER BY — if the question asks "top N / best / highest / lowest / most / least"
   and the SQL has no ORDER BY, add one using the most relevant numeric column.
5. Invented WHERE filters — if the SQL has WHERE conditions using values the user never mentioned
   (locations, names, codes, statuses), remove those conditions entirely.
6. Unnecessary JOIN — if the answer can come from one table but the SQL joins another,
   simplify to a single-table query.

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
# PIPELINE STEP 4 — ANSWER SYNTHESIS (anti-hallucination)
# ============================================================

_SYNTHESIS_PROMPT = """You are a helpful assistant answering questions about drilling and well operations data.
A SELECT query ran and returned results. Turn the raw data into a clear, friendly answer.

=== STRICT RULES ===
1. Answer ONLY from the query results below — never infer or add facts.
2. Never say "highest" or "lowest" unless ORDER BY in the SQL confirms ranking.
3. Never say "all records" unless COUNT(*) was used and proven.
4. If results are EMPTY:
   - Say politely that nothing was found.
   - If it was a name search, suggest the spelling might differ or ask for more context.
5. If MULTIPLE people/items match, list ALL of them (bullet list or table).
6. Format multiple rows as a bullet list or simple table — never as a wall of text.
7. Keep tone warm, concise, and professional.
8. Never show SQL, column names in raw form, or technical error details.

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
# RICH SCHEMA CONTEXT
# ============================================================

def _schema_context_for_tables(tables: list[str], schema: dict, rels: dict) -> str:
    """Build schema context capped by SCHEMA_MAX_COLS / SCHEMA_MAX_RELS env vars
    to keep Groq prompts under the payload size limit."""
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
# MAIN PIPELINE — ask()
# ============================================================

def ask(
    question: str,
    table_retriever,
    value_retriever,
    sql_db: SafeSQLDatabase,
    schema: dict,
    rels: dict,
    groq_llm,
    all_tables: list[str],
) -> tuple[str, str]:
    """Returns (answer, sql_used). Empty sql_used means no DB query ran."""

    # Step 0: intent gate (LLM-only — no hardcoded patterns)
    intent = _classify_intent(question, groq_llm)
    if intent == "OTHER":
        reply = _chitchat_reply(question, groq_llm)
        _history.append((question, reply))
        return reply, ""

    contextual_q = _build_contextual_question(question)

    # Step 1: retrieve relevant tables via vector similarity
    tables = _retrieve_tables(contextual_q, table_retriever, all_tables)
    logger.info(f"Retrieved tables: {tables}")

    # Step 2: build schema context for those tables
    schema_context = _schema_context_for_tables(tables, schema, rels)

    # Step 3: value grounding
    value_hints = _retrieve_value_hints(contextual_q, value_retriever)

    # Step 4: query planner
    plan_str, plan_dict = _plan_query(contextual_q, schema_context, groq_llm)
    logger.info(f"Plan: {plan_str[:200]}")

    # Step 4b: clarification gate
    if _needs_ranking(question):
        ordering = plan_dict.get("ordering")
        if not ordering or not ordering.get("column"):
            msg = (
                "I need a bit more context to answer that. "
                "What metric should I use to rank them? "
                "For example: by progress percentage, revenue, duration, manhours, or something else?"
            )
            _history.append((question, msg))
            return msg, ""

    # Step 5: generate SQL
    sql = _generate_sql(contextual_q, schema_context, value_hints, plan_str, groq_llm)

    if sql.upper().startswith("CLARIFY:"):
        msg = sql[len("CLARIFY:"):].strip()
        _history.append((question, msg))
        return msg, ""

    # Step 6: self-critique
    sql = _critique_sql(sql, contextual_q, schema_context, groq_llm)

    # Step 7+8: validate + execute
    results_str, _ = sql_db.run_sql(sql)

    # Step 9: synthesize
    answer = _synthesize_answer(question, sql, results_str, groq_llm)

    _history.append((question, answer))
    return answer, sql

# ============================================================
# STARTUP CHECKS
# ============================================================

def _check_credentials() -> None:
    missing = [k for k, v in {
        "DB_SERVER": DB_SERVER, "DB_NAME": DB_NAME,
        "DB_READONLY_USER": DB_USER, "DB_READONLY_PASSWORD": DB_PASSWORD,
        "GROQ_API_KEY": GROQ_API_KEY,
    }.items() if not v]
    if missing:
        logger.error(f"Missing required env vars: {', '.join(missing)}")
        sys.exit(1)

def _check_odbc() -> None:
    logger.info("Testing ODBC connection...")
    try:
        c = pyodbc.connect(
            f"DRIVER={{{DB_DRIVER}}};SERVER={DB_SERVER};DATABASE={DB_NAME};"
            f"UID={DB_USER};PWD={DB_PASSWORD};Connect Timeout={DB_CONNECT_TIMEOUT};",
            timeout=DB_CONNECT_TIMEOUT,
        )
        c.close()
        logger.info("ODBC connection OK.")
    except Exception as e:
        logger.error(f"ODBC connection failed: {e}")
        sys.exit(1)


def _validate_input(raw: str) -> str:
    q = raw.strip()
    if not q:
        raise ValueError("Empty input.")
    if len(q) > MAX_QUERY_LEN:
        raise ValueError(f"Input too long ({len(q)} chars, max {MAX_QUERY_LEN}).")
    return q

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    _check_credentials()
    _check_odbc()

    # LLM — Groq only
    logger.info(f"Configuring Groq: {GROQ_MODEL_NAME}")
    groq_llm = Groq(
        model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY,
        temperature=0, max_tokens=GROQ_MAX_TOKENS, context_window=GROQ_CONTEXT_WINDOW,
    )
    Settings.llm = groq_llm

    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device="cpu")
    Settings.embed_model = embed_model

    # DB engine
    logger.info(f"Connecting to {DB_SERVER}/{DB_NAME}...")
    encoded_pw = quote_plus(DB_PASSWORD)
    engine = create_engine(
        f"mssql+pyodbc://{DB_USER}:{encoded_pw}@{DB_SERVER}/{DB_NAME}"
        f"?driver={DB_DRIVER.replace(' ', '+')}&Connect+Timeout={DB_CONNECT_TIMEOUT}",
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"timeout": DB_QUERY_TIMEOUT},
    ).execution_options(timeout=DB_QUERY_TIMEOUT)

    with engine.connect() as c:
        ver = c.execute(text("SELECT @@VERSION")).fetchone()[0]
        logger.info(f"DB: {ver[:80]}...")

    # Discover ALL tables from the live database — no hardcoded list
    logger.info("Discovering all tables from live DB...")
    inspector = sa_inspect(engine)
    all_db_tables = inspector.get_table_names()
    if not all_db_tables:
        logger.error("No tables found in the database.")
        sys.exit(1)
    if DB_EXCLUDE_TABLES:
        logger.info(f"Excluding tables: {sorted(DB_EXCLUDE_TABLES)}")
    target_tables = [t for t in all_db_tables if t not in DB_EXCLUDE_TABLES]
    logger.info(f"Discovered {len(target_tables)} tables (excluded {len(all_db_tables) - len(target_tables)}).")

    # Dynamic schema — discover columns from live DB
    logger.info("Introspecting columns and types...")
    schema = discover_schema(engine, target_tables)
    found   = list(schema.keys())
    skipped = [t for t in target_tables if t not in schema]
    if skipped:
        logger.warning(f"Could not introspect {len(skipped)} table(s): {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    if not found:
        logger.error("No tables could be introspected.")
        sys.exit(1)
    logger.info(f"Schema loaded: {len(found)} tables, "
                f"{sum(len(v) for v in schema.values())} columns total.")

    # Relationship discovery
    rels = discover_relationships(engine, schema)

    # Build metadata entirely from the already-discovered schema — zero DB calls,
    # no timeouts, no SQLAlchemy 2.x autoload deprecation issues.
    logger.info("Building SQLAlchemy metadata from discovered schema...")
    metadata = _build_metadata_from_schema(schema)
    logger.info(f"Metadata ready: {len(schema)} tables.")

    # SafeSQLDatabase — pass pre-built metadata to skip internal re-reflection
    sql_db = SafeSQLDatabase(engine, include_tables=found, row_cap=ROW_CAP, metadata=metadata)

    # Table index (schema retrieval)
    table_index     = _build_or_load_table_index(sql_db, schema, rels)
    table_retriever = table_index.as_retriever(similarity_top_k=RETRIEVER_TOP_K)

    # Value grounding index
    value_index     = _build_or_load_value_index(engine, schema, embed_model)
    value_retriever = value_index.as_retriever(similarity_top_k=VALUE_TOP_K) if value_index else None

    logger.info("System ready.")

    total_cols = sum(len(v) for v in schema.values())
    total_rels = sum(len(v) for v in rels.values())
    print("\n" + "=" * 70)
    print("DB ASSISTANT — ready")
    print(f"  LLM          : {GROQ_MODEL_NAME} (Groq)")
    print(f"  Tables       : {len(found)}  |  Columns: {total_cols}  |  Relationships: {total_rels}")
    print(f"  Value index  : {'enabled' if value_retriever else 'disabled'}")
    print(f"  SQL critique : {'on' if ENABLE_SQL_CRITIQUE else 'off'}  |  Row cap: {ROW_CAP}")
    if DB_EXCLUDE_TABLES:
        print(f"  Excluded     : {', '.join(sorted(DB_EXCLUDE_TABLES))}")
    print("  Type 'exit' or 'q' to quit.")
    print("=" * 70)

    while True:
        try:
            raw = input("\nYour question: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if raw.strip().lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        try:
            question = _validate_input(raw)
        except ValueError as ve:
            print(f"  {ve}")
            continue

        print("  Thinking...")
        try:
            answer, sql_used = ask(
                question,
                table_retriever, value_retriever,
                sql_db, schema, rels,
                groq_llm, found,
            )
            print("\nAnswer:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            if sql_used:
                print(f"\nSQL used:\n{sql_used}")

        except ValueError as ve:
            logger.warning(f"Safety block: {ve}")
            print(f"\n  Blocked: {ve}")
        except Exception:
            logger.exception(f"Query failed: {question!r}")
            print("\n  Could not answer that question. Please rephrase or try again.")


if __name__ == "__main__":
    main()
