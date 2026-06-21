"""
Production-ready Natural-Language → SQL assistant for SQL Server.
Architecture: Retrieve → Discover Relationships → Plan → Generate SQL →
              Self-Critique → Validate → Execute → Synthesize Answer

All tunables are environment variables — see .env.example.
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
from urllib.request import urlopen
from urllib.error import URLError
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ============================================================
# CONFIG — all from environment
# ============================================================

DB_SERVER          = os.getenv("DB_SERVER")
DB_NAME            = os.getenv("DB_NAME")
DB_USER            = os.getenv("DB_READONLY_USER")
DB_PASSWORD        = os.getenv("DB_READONLY_PASSWORD")
DB_DRIVER          = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "30"))
DB_QUERY_TIMEOUT   = int(os.getenv("DB_QUERY_TIMEOUT", "60"))
DB_POOL_SIZE       = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW    = int(os.getenv("DB_MAX_OVERFLOW", "10"))

GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME     = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS     = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
GROQ_CONTEXT_WINDOW = int(os.getenv("GROQ_CONTEXT_WINDOW", "32768"))

OLLAMA_MODEL_NAME      = os.getenv("OLLAMA_MODEL_NAME", "qwen3:4b")
OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))
OLLAMA_NUM_CTX         = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")

_base_dir         = os.path.dirname(os.path.abspath(__file__))
INDEX_PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", os.path.join(_base_dir, "table_index_storage"))
LOG_DIR           = os.getenv("LOG_DIR", os.path.join(_base_dir, "logs"))
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")
ROW_CAP           = int(os.getenv("ROW_CAP", "100"))
RETRIEVER_TOP_K   = int(os.getenv("RETRIEVER_TOP_K", "5"))
MAX_QUERY_LEN     = int(os.getenv("MAX_QUERY_LEN", "500"))
HISTORY_MAX       = int(os.getenv("HISTORY_MAX", "5"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
ENABLE_SQL_CRITIQUE = os.getenv("ENABLE_SQL_CRITIQUE", "1") == "1"
ENABLE_SAMPLE_VALUES = os.getenv("ENABLE_SAMPLE_VALUES", "1") == "1"

TARGET_TABLES = [
    "2026_Well_Delivery_Scope_Well_Type",
    "ActivityTaskPlan",
    "WMR",
    "Employee",
    "crews",
    "CrewEmployee",
    "Equipment",
    "task_daily",
    "Revenue",
    "SAP_DRILLING_SEQUENCE",
]

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
# STATIC SCHEMA — exact columns (table/column names hardcoded per spec)
# ============================================================

STATIC_SCHEMA: dict[str, list[tuple[str, str]]] = {
    "2026_Well_Delivery_Scope_Well_Type": [
        ("Sr_No", "smallint"), ("RIG", "tinyint"), ("Well_ID", "int"),
        ("Latest_ROL", "nvarchar"), ("Latest_Rif_Off", "date"),
        ("Well_Category", "nvarchar"), ("Station_Code", "nvarchar"),
        ("Field", "nvarchar"), ("Well_Location_Name", "nvarchar"), ("Lift_type", "nvarchar"),
    ],
    "ActivityTaskPlan": [
        ("row_id", "bigint"), ("source_id", "nvarchar"), ("Data", "nvarchar"),
        ("ancestor", "nvarchar"), ("duration", "nvarchar"), ("progress", "nvarchar"),
        ("crew_uid", "nvarchar"), ("crew_type", "nvarchar"), ("qty", "nvarchar"),
        ("manhours", "nvarchar"), ("weightage", "nvarchar"), ("parent", "nvarchar"),
        ("start_date", "datetime2"), ("end_date", "datetime2"),
        ("target_start", "datetime2"), ("target_end", "datetime2"),
        ("actual_start", "datetime2"), ("actual_end", "datetime2"),
        ("qtyactual", "nvarchar"), ("qtyforacst", "nvarchar"),
        ("manhoursactual", "nvarchar"), ("manhourforacst", "nvarchar"),
        ("code", "nvarchar"), ("text", "nvarchar"), ("type", "nvarchar"),
        ("schedule_id", "nvarchar"), ("project_id", "nvarchar"),
        ("task_assignee", "nvarchar"), ("supervisor_email", "nvarchar"),
        ("attributes", "nvarchar"), ("remaining_duration", "nvarchar"),
        ("Resume_Suspend", "nvarchar"), ("data_nonprod", "nvarchar"),
        ("created_at", "datetime2"), ("updated_at", "datetime2"),
        ("Well_ID", "nvarchar"), ("Parent_WBS", "nvarchar"), ("Time_Stamp", "nvarchar"),
    ],
    "CrewEmployee": [
        ("id", "int"), ("Account", "nvarchar"), ("Crew", "int"),
        ("EmployeeType", "int"), ("Employee", "int"),
    ],
    "crews": [
        ("ID", "nvarchar"), ("Code", "nvarchar"), ("Account", "nvarchar"),
        ("Location", "nvarchar"), ("CrewType", "nvarchar"), ("Supervisor", "nvarchar"),
        ("Employees", "nvarchar"), ("Equipments", "nvarchar"),
    ],
    "Employee": [
        ("id", "int"), ("UId", "nvarchar"), ("Name", "nvarchar"), ("Email", "nvarchar"),
        ("Status", "nvarchar"), ("Supervisor", "int"), ("Account", "nvarchar"),
        ("EmployeeType", "int"), ("Company", "nvarchar"), ("Manager", "int"),
        ("Location", "nvarchar"),
    ],
    "Equipment": [
        ("ID", "int"), ("UId", "nvarchar"), ("LicensePlate", "nvarchar"),
        ("Description", "nvarchar"), ("Status", "nvarchar"), ("Account", "nvarchar"),
        ("EquipmentType", "int"), ("Location", "int"), ("Manager", "nvarchar"),
    ],
    "Revenue": [
        ("id", "bigint"), ("rigcode", "nvarchar"), ("well_id", "nvarchar"),
        ("code", "nvarchar"), ("pms", "decimal"), ("step_type", "nvarchar"),
        ("planned_progress", "nvarchar"), ("plan_percent", "nvarchar"),
        ("acutal_progress", "decimal"), ("act_percent", "nvarchar"),
        ("total_purpose_value", "decimal"), ("planned_purpose_value", "nvarchar"),
        ("actual_purpose_value", "decimal"), ("planned_progress_next_week", "nvarchar"),
        ("plan_percent_next_week", "nvarchar"), ("planned_purpose_value_next_week", "nvarchar"),
        ("Title", "nvarchar"), ("created_at", "datetime2"),
    ],
    "SAP_DRILLING_SEQUENCE": [
        ("Work_Center", "nvarchar"), ("Operation_Short", "nvarchar"), ("Activity", "nvarchar"),
        ("Opr_System_status", "nvarchar"), ("Earl_start_date", "date"),
        ("EarliestEndDate", "date"), ("Station_Code", "nvarchar"),
        ("Normal_duration", "float"), ("Norm_duratn_un", "nvarchar"),
        ("Well_Name", "nvarchar"), ("Field", "nvarchar"), ("Responsible_asset", "nvarchar"),
        ("Well_ID", "varchar"), ("Well_Location", "nvarchar"), ("Well_Function", "nvarchar"),
        ("Well_Category", "nvarchar"), ("PCAP_Category", "nvarchar"),
        ("Move_days", "tinyint"), ("PDO_Well_Type", "nvarchar"),
    ],
    "task_daily": [
        ("id", "bigint"), ("ActionOn", "date"), ("task_code", "nvarchar"),
        ("schedule_id", "bigint"), ("project_id", "uniqueidentifier"),
        ("required", "decimal"), ("planned", "decimal"), ("duration", "decimal"),
        ("remaining_duration", "decimal"), ("progress", "decimal"),
        ("ready", "bit"), ("completed", "bit"), ("plan", "bit"),
        ("committed_start", "date"), ("committed_end", "date"),
        ("target_start", "date"), ("target_end", "date"),
        ("actual_start", "date"), ("actual_end", "date"),
        ("startDate", "date"), ("endDate", "date"),
        ("crew_type", "nvarchar"), ("crew_code", "nvarchar"), ("planned_crew", "nvarchar"),
        ("well_id", "nvarchar"), ("task_uom", "nvarchar"),
        ("data_hours", "decimal"), ("data_qty", "decimal"),
        ("data_employees", "nvarchar"), ("task_assignee", "nvarchar"),
        ("supervisor_email", "nvarchar"), ("url", "nvarchar"),
        ("task_data", "nvarchar"), ("daily_data", "nvarchar"),
        ("created_at", "datetime2"), ("updated_at", "datetime2"),
        ("daily_ph_name", "nvarchar"), ("daily_equipment_ids", "nvarchar"),
        ("daily_employee_ids", "nvarchar"), ("daily_actual_quantity", "decimal"),
        ("daily_actual_hours", "decimal"), ("daily_completed", "bit"),
        ("time_stamp", "nvarchar"),
    ],
    "WMR": [
        ("sl_no", "nvarchar"), ("rig_no", "nvarchar"), ("well_location", "nvarchar"),
        ("well_name_after_spud", "nvarchar"), ("pdo_well_id", "nvarchar"),
        ("well_type", "nvarchar"), ("northing", "nvarchar"), ("easting", "nvarchar"),
        ("locationdd", "nvarchar"), ("flow_linedl", "nvarchar"),
        ("location_po_no", "nvarchar"), ("location_po_recvd_date", "nvarchar"),
        ("location_-_purpose_value", "nvarchar"),
        ("last_week_exp.rig_on_location_sap_data", "nvarchar"),
        ("latest_exp.rig_on_location_sap_data", "nvarchar"),
        ("exp.rig_off_location_sap_data", "nvarchar"),
        ("date_-_material_po_placed", "nvarchar"),
        ("date_-_material_available_at_site", "nvarchar"),
        ("scr_no", "nvarchar"), ("scr_date", "nvarchar"),
        ("moc_raised", "nvarchar"), ("moc_approved", "nvarchar"),
        ("buffer_status", "nvarchar"), ("actual_pegged_date", "nvarchar"),
        ("last_week_cum_progress", "nvarchar"), ("cum_progress_for_this_week", "nvarchar"),
        ("actual_start_date", "nvarchar"), ("actual_finish_date", "nvarchar"),
        ("flaf_issue_date", "nvarchar"), ("ramz_id", "nvarchar"),
        ("ramz_id_received_date_same_day_as_flaf_issue_date", "nvarchar"),
        ("date_of_site_survey_report_issuance", "nvarchar"),
        ("progress", "nvarchar"), ("over_all_progress_percentages", "nvarchar"),
        ("actual_rig_on_date", "nvarchar"), ("actual_rig_off_date", "nvarchar"),
        ("actual_eng._completion_date", "nvarchar"),
        ("actual_comm._start_date", "nvarchar"),
        ("actual_comm._finish_date_with_in_2_days_from_actual_engg._completion_date", "nvarchar"),
        ("engg_kpi_after_rig-off_days", "nvarchar"), ("data_error", "nvarchar"),
        ("reason_if_kpi_not_met", "nvarchar"),
        ("remark_status_area_of_attention_issues_", "nvarchar"),
        ("flow_line_const._status_in_progress_completed", "nvarchar"),
        ("flow_line_commi._status_in_progress_completed", "nvarchar"),
        ("project_id", "nvarchar"), ("Week_Number", "nvarchar"),
    ],
}

_SCHEMA_HASH = hashlib.md5(
    json.dumps({k: v for k, v in sorted(STATIC_SCHEMA.items())}, sort_keys=True).encode()
).hexdigest()

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
    def __init__(self, *args, row_cap: int = ROW_CAP, **kwargs):
        super().__init__(*args, **kwargs)
        self._row_cap = row_cap

    def run_sql(self, command: str):
        _validate_select_only(command)
        command = _inject_row_cap(command, self._row_cap)
        logger.info(f"[SQL] {command[:400]}")
        return super().run_sql(command)

# ============================================================
# RELATIONSHIP DISCOVERY
# Auto-discovers joins from sys.foreign_keys + column-name matching.
# Nothing hardcoded — runs at startup against the live DB.
# ============================================================

_REL_SKIP_COLS = {
    "id", "type", "status", "name", "code", "account",
    "location", "created_at", "updated_at", "project_id",
}

def _discover_relationships(engine, target_tables: list[str]) -> dict[str, list[tuple]]:
    """
    Returns {table: [(col, other_table, other_col), ...]}
    Sources:
      1. sys.foreign_keys  — explicit FK constraints
      2. Column name match — same column name in 2+ target tables (implicit FK)
    """
    rels: dict[str, list] = {t: [] for t in target_tables}
    target_lower = {t.lower(): t for t in target_tables}

    # 1. Explicit foreign keys
    fk_sql = text("""
        SELECT
            tp.name AS parent_table, cp.name AS parent_col,
            tr.name AS ref_table,   cr.name AS ref_col
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc
            ON fk.object_id = fkc.constraint_object_id
        JOIN sys.tables  tp ON fkc.parent_object_id     = tp.object_id
        JOIN sys.columns cp ON fkc.parent_object_id     = cp.object_id
                            AND fkc.parent_column_id    = cp.column_id
        JOIN sys.tables  tr ON fkc.referenced_object_id = tr.object_id
        JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id
                            AND fkc.referenced_column_id= cr.column_id
    """)
    try:
        with engine.connect() as conn:
            for pt, pc, rt, rc in conn.execute(fk_sql).fetchall():
                if pt.lower() in target_lower and rt.lower() in target_lower:
                    actual_pt = target_lower[pt.lower()]
                    actual_rt = target_lower[rt.lower()]
                    rels[actual_pt].append((pc, actual_rt, rc))
                    rels[actual_rt].append((rc, actual_pt, pc))
    except Exception as e:
        logger.warning(f"FK discovery query failed: {e}")

    # 2. Column-name matching across tables
    col_index: dict[str, list[tuple[str, str]]] = {}  # col_lower -> [(table, col)]
    for table, cols in STATIC_SCHEMA.items():
        if table not in target_tables:
            continue
        for col, _ in cols:
            col_index.setdefault(col.lower(), []).append((table, col))

    seen: set[tuple] = set()
    for col_lower, table_cols in col_index.items():
        if len(table_cols) < 2:
            continue
        if col_lower in _REL_SKIP_COLS:
            continue
        for i, (t1, c1) in enumerate(table_cols):
            for t2, c2 in table_cols[i + 1:]:
                key = tuple(sorted([(t1, c1), (t2, c2)]))
                if key in seen:
                    continue
                seen.add(key)
                # Only add if not already present from FK discovery
                existing_t1 = {(r[0], r[1]) for r in rels[t1]}
                if (c1, t2) not in existing_t1:
                    rels[t1].append((c1, t2, c2))
                existing_t2 = {(r[0], r[1]) for r in rels[t2]}
                if (c2, t1) not in existing_t2:
                    rels[t2].append((c2, t1, c1))

    count = sum(len(v) for v in rels.values())
    logger.info(f"Discovered {count} relationships across {len(target_tables)} tables.")
    return rels

# ============================================================
# SAMPLE VALUE COLLECTION
# Fetches top-3 distinct values for key columns at startup.
# Gives the LLM real examples so it knows what the data looks like.
# ============================================================

_SAMPLE_SKIP = re.compile(
    r'(data|attributes|url|json|notes|description|email|password|token|text|hash)', re.I
)
_SAMPLE_KEY = re.compile(
    r'(_id|_code|_type|_status|_category|_name|_field|well_id|rig|crew|field|location|status|category|type)$',
    re.I
)

def _get_sample_values(engine, target_tables: list[str]) -> dict[str, dict[str, list[str]]]:
    """Returns {table: {col: [val1, val2, val3]}} for key columns only."""
    if not ENABLE_SAMPLE_VALUES:
        return {}

    samples: dict[str, dict] = {}
    for table in target_tables:
        cols = STATIC_SCHEMA.get(table, [])
        table_samples: dict[str, list] = {}
        sampled = 0
        for col, dtype in cols:
            if sampled >= 6:
                break
            if _SAMPLE_SKIP.search(col):
                continue
            if not _SAMPLE_KEY.search(col):
                continue
            if dtype not in ("nvarchar", "varchar", "int", "bigint", "smallint", "tinyint", "bit"):
                continue
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text(
                        f"SELECT DISTINCT TOP 3 [{col}] FROM [{table}] WHERE [{col}] IS NOT NULL"
                    )).fetchall()
                vals = [str(r[0]) for r in rows if r[0] is not None]
                if vals:
                    table_samples[col] = vals
                    sampled += 1
            except Exception:
                pass
        if table_samples:
            samples[table] = table_samples
    logger.info(f"Collected sample values for {len(samples)} tables.")
    return samples

# ============================================================
# RICH SCHEMA CONTEXT BUILDER
# Combines: columns + discovered relationships + sample values.
# Used both for the retriever index AND injected into prompts.
# ============================================================

def _build_rich_context(
    table: str,
    rels: dict[str, list[tuple]],
    samples: dict[str, dict[str, list]],
) -> str:
    cols = STATIC_SCHEMA.get(table, [])
    col_str = ", ".join(f"{c} ({t})" for c, t in cols)

    rel_lines = []
    for col, other_table, other_col in rels.get(table, []):
        rel_lines.append(f"  [{table}].[{col}] → [{other_table}].[{other_col}]")
    rel_str = "\n".join(rel_lines) if rel_lines else "  (none discovered)"

    sample_lines = []
    for col, vals in (samples.get(table) or {}).items():
        sample_lines.append(f"  {col}: {', '.join(vals)}")
    sample_str = "\n".join(sample_lines) if sample_lines else "  (not sampled)"

    return (
        f"Table: [{table}]\n"
        f"Columns: {col_str}\n"
        f"Relationships:\n{rel_str}\n"
        f"Sample values:\n{sample_str}"
    )


def _build_context_block(
    tables: list[str],
    rels: dict[str, list[tuple]],
    samples: dict[str, dict[str, list]],
) -> str:
    """Full rich context for a list of tables, used in SQL-gen prompts."""
    return "\n\n".join(_build_rich_context(t, rels, samples) for t in tables)

# ============================================================
# LLM CALL WRAPPER — Groq primary, Ollama fallback, with retry
# ============================================================

def _is_rate_limit(err: Exception) -> bool:
    if hasattr(err, "status_code") and err.status_code in (429, 413):
        return True
    msg = str(err).lower()
    return any(k in msg for k in ("rate limit", "429", "413", "quota", "too large", "payload"))

def _is_transient(err: Exception) -> bool:
    if _is_rate_limit(err):
        return True
    if hasattr(err, "status_code") and err.status_code in (500, 502, 503):
        return True
    msg = str(err).lower()
    return any(k in msg for k in ("timeout", "connection", "temporarily"))

def _llm_complete(prompt: str, groq_llm, ollama_llm) -> str:
    """Complete a prompt. Retries transient errors; falls back to Ollama on rate limits."""
    last_err = None

    def _try(llm, label):
        nonlocal last_err
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                return llm.complete(prompt).text.strip()
            except ValueError:
                raise
            except Exception as e:
                last_err = e
                if _is_rate_limit(e):
                    raise  # handled by outer fallback
                if not _is_transient(e) or attempt == MAX_RETRY_ATTEMPTS:
                    raise
                delay = 2.0 * (2 ** (attempt - 1))
                logger.warning(f"{label} transient error (attempt {attempt}), retry in {delay:.0f}s: {e}")
                time.sleep(delay)

    try:
        return _try(groq_llm, "Groq")
    except Exception as e:
        if _is_rate_limit(e):
            logger.warning("Groq limit — switching to Ollama.")
            return _try(ollama_llm, "Ollama")
        raise

# ============================================================
# INDEX — hash-versioned persistence
# ============================================================

_HASH_FILE = os.path.join(INDEX_PERSIST_DIR, "schema_version.txt")

def _index_is_stale() -> bool:
    if not os.path.isfile(_HASH_FILE):
        return True
    with open(_HASH_FILE) as f:
        return f.read().strip() != _SCHEMA_HASH

def _save_index_hash() -> None:
    os.makedirs(INDEX_PERSIST_DIR, exist_ok=True)
    with open(_HASH_FILE, "w") as f:
        f.write(_SCHEMA_HASH)

def _build_or_load_index(
    sql_db: SafeSQLDatabase,
    table_names: list[str],
    rels: dict,
    samples: dict,
) -> ObjectIndex:
    table_node_mapping = SQLTableNodeMapping(sql_db)
    # Each table gets a rich context_str (columns + relationships + samples)
    table_schema_objs = [
        SQLTableSchema(
            table_name=t,
            context_str=_build_rich_context(t, rels, samples),
        )
        for t in table_names
    ]

    force_rebuild = os.getenv("REBUILD_INDEX", "0") == "1"
    index_exists = os.path.isfile(os.path.join(INDEX_PERSIST_DIR, "index_store.json"))
    stale = _index_is_stale()

    if index_exists and not force_rebuild and not stale:
        logger.info(f"Loading persisted index from {INDEX_PERSIST_DIR}.")
        idx = ObjectIndex.from_persist_dir(
            persist_dir=INDEX_PERSIST_DIR,
            object_node_mapping=table_node_mapping,
        )
        logger.info("Index loaded.")
        return idx

    reason = "REBUILD_INDEX=1" if force_rebuild else ("schema changed" if stale else "first run")
    logger.info(f"Building index ({reason})...")
    idx = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex)
    idx.persist(persist_dir=INDEX_PERSIST_DIR)
    _save_index_hash()
    logger.info("Index built and saved.")
    return idx

# ============================================================
# TABLE RETRIEVAL — extracts table names from the object index
# ============================================================

def _retrieve_tables(question: str, retriever, fallback_tables: list[str]) -> list[str]:
    """Return the top-K most relevant table names for a question."""
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
                    txt = getattr(r.node, "text", "") or ""
                    m = re.match(r'Table:\s*\[([^\]]+)\]', txt)
                    if m:
                        names.append(m.group(1))
        # deduplicate while preserving order
        seen: set[str] = set()
        unique = [n for n in names if not (n in seen or seen.add(n))]
        return unique if unique else fallback_tables[:RETRIEVER_TOP_K]
    except Exception as e:
        logger.warning(f"Table retrieval failed ({e}), using fallback tables.")
        return fallback_tables[:RETRIEVER_TOP_K]

# ============================================================
# PIPELINE STEP 1 — SQL GENERATION PROMPT
# ============================================================

_SQL_GEN_SYSTEM = """You are a senior T-SQL expert for a drilling and well operations database.
Your only job is to write one correct SELECT statement.

=== STRICT RULES ===
1. ONLY SELECT. Never INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, EXEC, MERGE.
2. ALWAYS use square brackets: [TableName].[ColumnName].
   WMR columns with dots/hyphens MUST be bracketed: [exp.rig_off_location_sap_data]
3. Use ONLY tables and columns from the SCHEMA below. Never invent names.
4. Add TOP {row_cap} after SELECT unless user specifies a number.
5. Numeric averages on text columns: AVG(TRY_CAST([col] AS FLOAT)).
6. TEXT / NAME SEARCHES — case-insensitive fuzzy rules:
   a. NEVER use = for names, text, or labels. Always use LIKE '%value%'.
   b. SQL Server CI collation handles case automatically — no LOWER() needed.
   c. Search BOTH [Name] AND [Email] for person lookups.
   d. For partial/misspelled names: use the first clear letters, e.g. 'rajsh' → LIKE '%raj%'.
   e. For code/ID fields that look like identifiers (Well_ID, task_code), = is acceptable.
7. Return ONLY the raw SQL — no markdown, no explanation.

=== SCHEMA WITH RELATIONSHIPS AND SAMPLE VALUES ===
{rich_context}

=== QUERY PLAN ===
{plan}

=== QUESTION ===
{question}

SQL:"""

def _build_sql_prompt(question: str, rich_context: str, plan: str) -> str:
    return _SQL_GEN_SYSTEM.format(
        row_cap=ROW_CAP,
        rich_context=rich_context,
        plan=plan,
        question=question,
    )

# ============================================================
# PIPELINE STEP 2 — QUERY PLANNER
# Produces a structured plan before SQL generation.
# Tells the SQL generator which tables to join and how.
# ============================================================

_PLANNER_SYSTEM = """You are a database query planner for a drilling and well operations database.

Given the user question and table schemas (with relationships), produce a concise query plan.

Available tables and their relationships:
{rich_context}

Question: {question}

Return a JSON object with exactly these keys:
{{
  "intent": "one sentence describing what the user wants",
  "required_tables": ["TableA", "TableB"],
  "join_paths": ["[TableA].[col] = [TableB].[col]"],
  "filters": ["any WHERE conditions as plain text"],
  "aggregations": ["any GROUP BY or aggregate needed, or empty list"]
}}

RULES:
- required_tables must be actual table names from the schema.
- join_paths must use EXACT column names from the schema.
- If no join is needed, set join_paths to [].
- Return ONLY valid JSON. No extra text."""

def _plan_query(question: str, rich_context: str, groq_llm, ollama_llm) -> str:
    prompt = _PLANNER_SYSTEM.format(rich_context=rich_context, question=question)
    raw = _llm_complete(prompt, groq_llm, ollama_llm)
    # Extract JSON even if LLM wraps it in markdown
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            plan = json.loads(match.group())
            return json.dumps(plan, indent=2)
        except json.JSONDecodeError:
            pass
    return raw  # return raw if parsing fails — SQL gen still works

# ============================================================
# PIPELINE STEP 3 — SQL SELF-CRITIQUE
# LLM reviews its own SQL before execution.
# ============================================================

_CRITIQUE_SYSTEM = """Review this T-SQL SELECT statement for correctness.

Question: {question}
Generated SQL: {sql}
Schema context: {rich_context}

Check ONLY for critical errors:
1. Wrong or missing join conditions (cartesian product risk)
2. Columns referenced that do not exist in the schema
3. Aggregation duplication (e.g. joining Revenue 1→many then summing — inflates totals)
4. Business logic clearly wrong (e.g. filtering active employees when question asks inactive)

If the SQL is correct: reply with exactly the word PASS
If there is a critical error: reply with the corrected SQL only (raw SQL, no explanation)."""

def _critique_sql(sql: str, question: str, rich_context: str, groq_llm, ollama_llm) -> str:
    if not ENABLE_SQL_CRITIQUE:
        return sql
    prompt = _CRITIQUE_SYSTEM.format(question=question, sql=sql, rich_context=rich_context)
    result = _llm_complete(prompt, groq_llm, ollama_llm)
    if result.strip().upper() == "PASS":
        return sql
    # LLM returned a corrected SQL
    corrected = result.strip()
    # Validate it's actually a SELECT before accepting
    if corrected.upper().startswith("SELECT"):
        logger.info("SQL self-critique returned a correction.")
        return corrected
    return sql  # if critique response is invalid, keep original

# ============================================================
# PIPELINE STEP 4 — ANSWER SYNTHESIS
# Strict anti-hallucination rules: answer only from returned rows.
# ============================================================

_SYNTHESIS_SYSTEM = """You are a helpful assistant answering questions about drilling and well operations data.
A SQL query was run and returned results. Your job: turn the raw data into a clear, friendly answer.

=== STRICT ANSWERING RULES ===
1. Use ONLY the data in the query results below. Never infer or add facts.
2. Never say "highest" or "lowest" unless ORDER BY in the SQL confirms it.
3. Never say "all records" unless COUNT(*) was used and proven.
4. Never assume data that is not in the results.
5. If results are EMPTY:
   - Say politely that nothing was found.
   - If the question was a name search, suggest the name might be spelled differently.
   - Example: "I couldn't find anyone named 'rajsh'. Try searching 'raj' or check the full name."
6. If MULTIPLE similar people/items are found, list ALL of them clearly.
   Example: "I found 3 people matching 'rajesh': ..."
7. Format multiple records as a bullet list or simple table.
8. Keep tone warm, concise, and professional — like a knowledgeable colleague.
9. Never show raw SQL or technical error details.

Question: {question}
SQL used: {sql}
Query results: {results}

Answer:"""

def _synthesize_answer(question: str, sql: str, results: str, groq_llm, ollama_llm) -> str:
    prompt = _SYNTHESIS_SYSTEM.format(question=question, sql=sql, results=results)
    return _llm_complete(prompt, groq_llm, ollama_llm)

# ============================================================
# INTENT CLASSIFIER
# Fast gate: skip the SQL pipeline for greetings, small-talk,
# meta questions, or anything clearly not a data query.
# ============================================================

_NON_DATA_REGEX = re.compile(
    r'^('
    r'hi+|hello+|hey+|howdy|greetings|good\s*(morning|afternoon|evening|day)|'
    r'how are you|how\'s it going|what\'s up|sup\b|yo\b|'
    r'thanks?|thank you|cheers|great|awesome|cool|ok+|okay|'
    r'bye+|goodbye|see you|exit|quit|stop|'
    r'who are you|what are you|what can you do|help me|help\b'
    r')[\s!?.]*$',
    re.IGNORECASE,
)

_CHITCHAT_PROMPT = """You are a friendly assistant for a drilling and well operations database.
The user said something that is NOT a database query.

Respond naturally and helpfully in 2-4 sentences.
- If it's a greeting, introduce yourself briefly and mention what kinds of questions you can answer
  (employees, crews, wells, tasks, equipment, revenue, drilling sequences).
- If it's a farewell, say goodbye warmly.
- If it's thanks or a compliment, acknowledge it and invite further questions.
- If they're asking what you can do or need help, explain your capabilities concisely with 1-2 examples.
- For anything else, respond politely and redirect toward data questions if appropriate.

Never make up data. Never run or mention SQL. Keep it conversational.

User said: "{question}"

Response:"""

_INTENT_CLASSIFY_PROMPT = """You are a classifier for a drilling/well-operations database assistant.

Decide if the user's input is a DATABASE QUERY (asking for data about wells, employees, crews,
equipment, tasks, revenue, or drilling operations) or NOT (greeting, small-talk, compliment,
complaint, meta-question about the system, or anything unrelated to the database).

User input: "{question}"

Reply with exactly one word:
- DATA   — if this is a request for database information
- OTHER  — if this is not a database query

Reply:"""

def _classify_intent(question: str, groq_llm, ollama_llm) -> str:
    """Returns 'DATA' or 'OTHER'. Falls back to 'DATA' on any error (safer)."""
    # Fast regex path — no LLM call needed
    if _NON_DATA_REGEX.match(question.strip()):
        return "OTHER"
    # For short inputs (< 8 words) ask the LLM
    if len(question.split()) < 8:
        try:
            prompt = _INTENT_CLASSIFY_PROMPT.format(question=question)
            result = _llm_complete(prompt, groq_llm, ollama_llm)
            return "OTHER" if result.strip().upper().startswith("OTHER") else "DATA"
        except Exception:
            return "DATA"
    return "DATA"

def _chitchat_reply(question: str, groq_llm, ollama_llm) -> str:
    prompt = _CHITCHAT_PROMPT.format(question=question)
    return _llm_complete(prompt, groq_llm, ollama_llm)

# ============================================================
# CONVERSATION HISTORY
# ============================================================

_history: deque = deque(maxlen=HISTORY_MAX)

def _build_contextual_question(question: str) -> str:
    """Prepend last 2 Q&A turns so follow-up questions resolve correctly."""
    if not _history:
        return question
    lines = []
    for prev_q, prev_a in list(_history)[-2:]:
        lines.append(f"[Previous question]: {prev_q}")
        lines.append(f"[Previous answer summary]: {str(prev_a)[:200]}")
    lines.append(f"[Current question]: {question}")
    return "\n".join(lines)

# ============================================================
# MAIN PIPELINE — ask()
# Orchestrates all steps; no LlamaIndex engine needed.
# ============================================================

def ask(
    question: str,
    retriever,
    sql_db: SafeSQLDatabase,
    rels: dict,
    samples: dict,
    groq_llm,
    ollama_llm,
    all_tables: list[str],
) -> tuple[str, str]:
    """
    Returns (answer, sql_used).
    Steps: intent check → retrieve → plan → generate SQL → critique → validate → execute → synthesize
    """
    # Step 0: intent gate — block non-data questions before any SQL work
    intent = _classify_intent(question, groq_llm, ollama_llm)
    if intent == "OTHER":
        reply = _chitchat_reply(question, groq_llm, ollama_llm)
        _history.append((question, reply))
        return reply, ""

    contextual_q = _build_contextual_question(question)

    # Step 1: retrieve relevant tables
    tables = _retrieve_tables(contextual_q, retriever, all_tables)
    logger.info(f"Retrieved tables: {tables}")

    # Step 2: build rich context for those tables
    rich_context = _build_context_block(tables, rels, samples)

    # Step 3: query planner
    plan = _plan_query(contextual_q, rich_context, groq_llm, ollama_llm)
    logger.info(f"Query plan: {plan[:300]}")

    # Step 4: generate SQL
    sql_prompt = _build_sql_prompt(contextual_q, rich_context, plan)
    sql = _llm_complete(sql_prompt, groq_llm, ollama_llm)
    # Strip markdown code fences if LLM adds them
    sql = re.sub(r'^```(?:sql)?\s*', '', sql, flags=re.IGNORECASE).rstrip('`').strip()

    # Step 5: self-critique (optional)
    sql = _critique_sql(sql, contextual_q, rich_context, groq_llm, ollama_llm)

    # Step 6: validate (read-only guard + row cap) — runs inside SafeSQLDatabase.run_sql
    # Step 7: execute
    results_str, _ = sql_db.run_sql(sql)

    # Step 8: synthesize answer
    answer = _synthesize_answer(question, sql, results_str, groq_llm, ollama_llm)

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
    logger.info("Credentials check passed.")

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

def _check_ollama() -> None:
    try:
        urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        logger.info(f"Ollama reachable at {OLLAMA_BASE_URL}.")
    except (URLError, OSError):
        logger.warning(f"Ollama not reachable at {OLLAMA_BASE_URL} — Groq-only mode.")

# ============================================================
# INPUT VALIDATION
# ============================================================

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
    _check_ollama()

    # LLMs
    logger.info(f"Configuring Groq: {GROQ_MODEL_NAME}")
    groq_llm = Groq(
        model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY,
        temperature=0, max_tokens=GROQ_MAX_TOKENS, context_window=GROQ_CONTEXT_WINDOW,
    )
    logger.info(f"Configuring Ollama: {OLLAMA_MODEL_NAME}")
    ollama_llm = Ollama(
        model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_REQUEST_TIMEOUT, temperature=0,
        context_window=OLLAMA_NUM_CTX, additional_kwargs={"num_ctx": OLLAMA_NUM_CTX},
    )
    Settings.llm = groq_llm

    # Embedding
    logger.info(f"Configuring embedding: {EMBED_MODEL_NAME}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device="cpu")

    # DB engine
    logger.info(f"Connecting to {DB_SERVER}/{DB_NAME}...")
    encoded_pw = quote_plus(DB_PASSWORD)
    engine = create_engine(
        f"mssql+pyodbc://{DB_USER}:{encoded_pw}@{DB_SERVER}/{DB_NAME}"
        f"?driver={DB_DRIVER.replace(' ', '+')}&Connect+Timeout={DB_CONNECT_TIMEOUT}",
        pool_size=DB_POOL_SIZE, max_overflow=DB_MAX_OVERFLOW, pool_pre_ping=True,
    ).execution_options(timeout=DB_QUERY_TIMEOUT)

    with engine.connect() as c:
        ver = c.execute(text("SELECT @@VERSION")).fetchone()[0]
        logger.info(f"DB connected: {ver[:80]}...")

    # Discover available tables
    all_table_names = set(inspect(engine).get_table_names())
    found = [t for t in TARGET_TABLES if t in all_table_names]
    missing = [t for t in TARGET_TABLES if t not in all_table_names]
    if missing:
        logger.warning(f"Tables not found in DB: {missing}")
    if not found:
        logger.error("None of the target tables exist.")
        sys.exit(1)
    logger.info(f"Using {len(found)}/{len(TARGET_TABLES)} tables.")

    # Auto-discover relationships from live DB
    rels = _discover_relationships(engine, found)

    # Collect sample values for key columns
    logger.info("Collecting sample values (key columns only)...")
    samples = _get_sample_values(engine, found)

    # SafeSQLDatabase + index (rich context = schema + rels + samples)
    sql_db = SafeSQLDatabase(engine, include_tables=found, row_cap=ROW_CAP)
    obj_index = _build_or_load_index(sql_db, found, rels, samples)
    retriever = obj_index.as_retriever(similarity_top_k=RETRIEVER_TOP_K)
    logger.info("System ready.")

    # Session
    print("\n" + "=" * 70)
    print("DB ASSISTANT — ready")
    print(f"  Primary LLM : {GROQ_MODEL_NAME}")
    print(f"  Fallback LLM: {OLLAMA_MODEL_NAME}")
    print(f"  Tables      : {len(found)}  |  Relationships discovered: {sum(len(v) for v in rels.values())}")
    print(f"  Row cap     : {ROW_CAP}  |  SQL critique: {'on' if ENABLE_SQL_CRITIQUE else 'off'}")
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
                question, retriever, sql_db, rels, samples, groq_llm, ollama_llm, found
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
            print("\n  Could not answer that question. Please rephrase or check the logs.")


if __name__ == "__main__":
    main()
