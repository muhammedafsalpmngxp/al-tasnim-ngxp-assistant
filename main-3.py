"""
Production-ready Natural-Language → SQL assistant for SQL Server.
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
from llama_index.core import SQLDatabase, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ============================================================
# CONFIG — all from environment, no hard-coded values
# ============================================================

# Database
DB_SERVER           = os.getenv("DB_SERVER")
DB_NAME             = os.getenv("DB_NAME")
DB_USER             = os.getenv("DB_READONLY_USER")
DB_PASSWORD         = os.getenv("DB_READONLY_PASSWORD")
DB_DRIVER           = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_CONNECT_TIMEOUT  = int(os.getenv("DB_CONNECT_TIMEOUT", "30"))
DB_QUERY_TIMEOUT    = int(os.getenv("DB_QUERY_TIMEOUT", "60"))
DB_POOL_SIZE        = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW     = int(os.getenv("DB_MAX_OVERFLOW", "10"))

# LLM — Groq (primary)
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME     = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS     = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
GROQ_CONTEXT_WINDOW = int(os.getenv("GROQ_CONTEXT_WINDOW", "32768"))

# LLM — Ollama (fallback)
OLLAMA_MODEL_NAME       = os.getenv("OLLAMA_MODEL_NAME", "qwen3:4b")
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_REQUEST_TIMEOUT  = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))
OLLAMA_NUM_CTX          = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# Embedding
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# App behaviour
_base_dir       = os.path.dirname(os.path.abspath(__file__))
INDEX_PERSIST_DIR   = os.getenv("INDEX_PERSIST_DIR", os.path.join(_base_dir, "table_index_storage"))
LOG_DIR             = os.getenv("LOG_DIR",   os.path.join(_base_dir, "logs"))
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
ROW_CAP             = int(os.getenv("ROW_CAP", "100"))          # hard ceiling injected into every SELECT
RETRIEVER_TOP_K     = int(os.getenv("RETRIEVER_TOP_K", "3"))    # tables passed to LLM per query
MAX_QUERY_LEN       = int(os.getenv("MAX_QUERY_LEN", "500"))    # max user input characters
HISTORY_MAX         = int(os.getenv("HISTORY_MAX", "5"))        # conversation turns kept
MAX_RETRY_ATTEMPTS  = int(os.getenv("MAX_RETRY_ATTEMPTS", "3")) # transient-error retries

# Table names (OK to hardcode per project spec)
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
# LOGGING — rotating file + console, no credentials in output
# ============================================================

def _setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "assistant.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return logging.getLogger(__name__)


logger = _setup_logging()

# ============================================================
# STATIC SCHEMA — exact columns from live DB
# Table/column names are intentionally hardcoded (per spec).
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

# Schema content hash — used to detect stale index on disk
_SCHEMA_HASH = hashlib.md5(
    json.dumps({k: v for k, v in sorted(STATIC_SCHEMA.items())}, sort_keys=True).encode()
).hexdigest()


def _schema_context_str(table: str) -> str:
    cols = STATIC_SCHEMA.get(table, [])
    if not cols:
        return f"[{table}]: (no schema)"
    return f"[{table}]: " + ", ".join(f"{c} ({t})" for c, t in cols)


def _full_schema_block() -> str:
    lines = []
    for table, cols in STATIC_SCHEMA.items():
        col_str = ", ".join(f"{c} ({t})" for c, t in cols)
        lines.append(f"  [{table}]: {col_str}")
    return "\n".join(lines)


# ============================================================
# PROMPT — few-shot + chain-of-thought
# ============================================================

_PROMPT_TMPL = (
    "You are a senior {dialect} expert. Think step-by-step before writing SQL.\n\n"
    "=== STRICT RULES ===\n"
    "1. ONLY write SELECT statements. NEVER use INSERT, UPDATE, DELETE, DROP, TRUNCATE,\n"
    "   ALTER, CREATE, EXEC, EXECUTE, MERGE, GRANT, REVOKE, or any DDL/DML command.\n"
    "   If the question asks to modify data, reply exactly: 'I can only read data, not modify it.'\n"
    "2. ALWAYS wrap table and column names in square brackets: [TableName].[ColumnName].\n"
    "   WMR columns contain dots/hyphens — they MUST be in brackets, e.g. [exp.rig_off_location_sap_data].\n"
    "3. Use ONLY the exact table and column names in the EXACT SCHEMA. Never invent or rename.\n"
    "4. Always add TOP {row_cap} after SELECT unless the user specifies a different number.\n"
    "5. For averages on text-stored numbers: AVG(TRY_CAST([col] AS FLOAT)).\n"
    "6. 'working'/'active' employee → [Status] = 'active'; 'inactive'/'left' → [Status] = 'inactive'.\n"
    "7. Natural language → columns: 'quantity'→qty, 'manhours'→manhours, 'progress'→progress, 'duration'→duration.\n"
    "8. Return ONLY the raw SQL query — no markdown, no explanation, no extra text.\n\n"
    "=== CASE-INSENSITIVE & FUZZY SEARCH RULES ===\n"
    "These rules apply for ALL name, text, and keyword searches:\n"
    "A. NEVER use = for text/name searches. Always use LIKE with wildcards.\n"
    "   Example: [Name] LIKE '%rajesh%'  — this matches 'Rajesh', 'RAJESH', 'rajesh kumar', etc.\n"
    "   SQL Server's default collation (CI) is already case-insensitive, so LIKE '%rajesh%'\n"
    "   matches 'Rajesh', 'RAJESH', 'rajesh' automatically — no LOWER() needed.\n"
    "B. When searching a person by name, search BOTH [Name] AND [Email]:\n"
    "   WHERE [Name] LIKE '%rajesh%' OR [Email] LIKE '%rajesh%'\n"
    "C. When searching for a well, crew, task, or location, always use LIKE '%value%' not = 'value'.\n"
    "D. If the user types a partial name (e.g. 'raj'), still use LIKE '%raj%' to return all partial matches.\n"
    "E. If the user may have misspelled, use LIKE '%<first few chars>%' — e.g. 'rajsh' → LIKE '%raj%'.\n"
    "F. For code/ID fields (Well_ID, task_code, rigcode) that look like identifiers, = is acceptable.\n\n"
    "=== EXACT SCHEMA ===\n"
    + _full_schema_block() + "\n\n"
    "=== RETRIEVER CONTEXT ===\n"
    "{schema}\n\n"
    "=== FEW-SHOT EXAMPLES ===\n\n"
    "Q: How many active employees are there?\n"
    "Think: [Employee] table, filter Status='active', COUNT rows.\n"
    "SQL: SELECT COUNT(*) AS active_count FROM [Employee] WHERE [Status] = 'active'\n\n"
    "Q: Show top 5 employees by name\n"
    "Think: [Employee] table, select key columns, ORDER BY Name.\n"
    "SQL: SELECT TOP 5 [id], [Name], [Email], [Status] FROM [Employee] ORDER BY [Name]\n\n"
    "Q: Tell me about rajesh\n"
    "Think: User typed lowercase 'rajesh'. Use LIKE '%rajesh%' on both Name and Email — CI collation handles case.\n"
    "SQL: SELECT TOP {row_cap} [id], [Name], [Email], [Status], [Company], [Location] "
    "FROM [Employee] WHERE [Name] LIKE '%rajesh%' OR [Email] LIKE '%rajesh%'\n\n"
    "Q: Find employee RAJESH BOKKA\n"
    "Think: Full name search — use LIKE '%RAJESH%' and '%BOKKA%' or combined '%RAJESH BOKKA%'.\n"
    "SQL: SELECT TOP {row_cap} [id], [Name], [Email], [Status], [Company], [Location] "
    "FROM [Employee] WHERE [Name] LIKE '%RAJESH%' AND [Name] LIKE '%BOKKA%'\n\n"
    "Q: Show crews at location nimr\n"
    "Think: [crews], use LIKE for Location — case-insensitive.\n"
    "SQL: SELECT TOP {row_cap} [ID], [Code], [CrewType], [Location], [Supervisor] "
    "FROM [crews] WHERE [Location] LIKE '%nimr%'\n\n"
    "Q: What is the daily progress for task code DRILL-001?\n"
    "Think: [task_daily], task_code is an identifier — = is fine here.\n"
    "SQL: SELECT TOP {row_cap} [id], [ActionOn], [task_code], [progress], [daily_actual_quantity], [daily_actual_hours] "
    "FROM [task_daily] WHERE [task_code] = 'DRILL-001' ORDER BY [ActionOn] DESC\n\n"
    "Q: List crews and their employees\n"
    "Think: [crews] JOIN [CrewEmployee] on crews.ID = CrewEmployee.Crew.\n"
    "SQL: SELECT TOP {row_cap} c.[ID], c.[Code], c.[CrewType], ce.[Employee] "
    "FROM [crews] c JOIN [CrewEmployee] ce ON ce.[Crew] = c.[ID]\n\n"
    "Q: Show revenue by well\n"
    "Think: [Revenue], GROUP BY well_id, SUM actual_purpose_value.\n"
    "SQL: SELECT TOP {row_cap} [well_id], SUM([actual_purpose_value]) AS total_revenue "
    "FROM [Revenue] GROUP BY [well_id] ORDER BY total_revenue DESC\n\n"
    "Q: Tell the details of daily tasks\n"
    "Think: [task_daily], select descriptive columns, most recent first.\n"
    "SQL: SELECT TOP {row_cap} [id], [ActionOn], [task_code], [well_id], [crew_code], [progress], "
    "[daily_actual_quantity], [daily_actual_hours], [daily_completed] FROM [task_daily] ORDER BY [ActionOn] DESC\n\n"
    "Q: Which wells are in field NIMR?\n"
    "Think: [2026_Well_Delivery_Scope_Well_Type] has Field column — use LIKE.\n"
    "SQL: SELECT TOP {row_cap} [Well_ID], [Well_Location_Name], [Well_Category], [Station_Code] "
    "FROM [2026_Well_Delivery_Scope_Well_Type] WHERE [Field] LIKE '%NIMR%'\n\n"
    "Q: Show SAP drilling sequence for well 33151\n"
    "Think: [SAP_DRILLING_SEQUENCE] has Well_ID — identifier, = is fine.\n"
    "SQL: SELECT TOP {row_cap} [Well_ID], [Well_Name], [Activity], [Earl_start_date], "
    "[EarliestEndDate], [Normal_duration], [Opr_System_status] "
    "FROM [SAP_DRILLING_SEQUENCE] WHERE [Well_ID] = '33151'\n\n"
    "Q: Show WMR rig-off date for well NIMR-001\n"
    "Think: [WMR] — use LIKE on well_name_after_spud.\n"
    "SQL: SELECT TOP {row_cap} [well_name_after_spud], [rig_no], "
    "[exp.rig_off_location_sap_data], [actual_rig_off_date], [progress] "
    "FROM [WMR] WHERE [well_name_after_spud] LIKE '%NIMR-001%'\n\n"
    "=== NOW ANSWER ===\n"
    "Q: {query_str}\n"
    "Think step-by-step, then write the SQL.\n"
    "SQL:"
)

CUSTOM_TEXT_TO_SQL_PROMPT = PromptTemplate(_PROMPT_TMPL.replace("{row_cap}", str(ROW_CAP)))

# Response synthesis prompt — makes the LLM answer like a helpful human chatbot
_RESPONSE_TMPL = (
    "You are a helpful assistant answering questions about drilling and well operations data.\n"
    "A SQL query was run against the database. Your job is to turn the raw results into a\n"
    "clear, friendly, human-readable answer.\n\n"
    "=== RESPONSE RULES ===\n"
    "1. Answer in plain conversational English — like a helpful colleague, not a robot.\n"
    "2. If results are found, summarise the key information clearly. Use a bullet list or\n"
    "   short table when there are multiple records.\n"
    "3. If the search was for a name (e.g. 'rajesh'), mention how many matching people were\n"
    "   found and list them with their key details (name, email, status, etc.).\n"
    "4. If NO results were found, say so politely and suggest that:\n"
    "   - The name or value might be spelled differently in the database.\n"
    "   - The user can try a shorter or different spelling.\n"
    "   Example: 'I couldn't find anyone named \"rajsh\" in the database. "
    "This might be a spelling variation — try searching for \"raj\" or check the full name.'\n"
    "5. If multiple similar records are found (e.g. both 'Rajesh Kumar' and 'Rajesh Bokka'),\n"
    "   list all of them and note: 'I found X people matching that name — here they are:'\n"
    "6. Never show raw SQL or technical error messages in your reply.\n"
    "7. If the question was about data modification, say: 'I can only read data, not modify it.'\n"
    "8. Keep the tone warm, concise, and professional.\n\n"
    "Original question: {query_str}\n"
    "SQL query run: {sql_query}\n"
    "Query results: {context_str}\n\n"
    "Answer:"
)

RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(_RESPONSE_TMPL)


# ============================================================
# READ-ONLY GUARD + ROW CAP (runs BEFORE SQL hits the DB)
# ============================================================

_BLOCKED = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE|MERGE|GRANT|REVOKE|DENY|BULK)\b',
    re.IGNORECASE,
)


def _validate_select_only(sql: str) -> None:
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError(f"Blocked: query must start with SELECT. Got: {stripped[:120]}")
    m = _BLOCKED.search(stripped)
    if m:
        raise ValueError(f"Blocked: forbidden keyword '{m.group()}' in generated SQL.")


def _inject_row_cap(sql: str, cap: int) -> str:
    """Add TOP N if no TOP clause is already present."""
    if re.search(r'\bSELECT\s+(?:DISTINCT\s+)?TOP\s+\d+\b', sql, re.IGNORECASE):
        return sql
    return re.sub(r'\bSELECT\b', f'SELECT TOP {cap}', sql, count=1, flags=re.IGNORECASE)


class SafeSQLDatabase(SQLDatabase):
    """SQLDatabase that validates and caps every query BEFORE execution."""

    def __init__(self, *args, row_cap: int = ROW_CAP, **kwargs):
        super().__init__(*args, **kwargs)
        self._row_cap = row_cap

    def run_sql(self, command: str):
        _validate_select_only(command)
        command = _inject_row_cap(command, self._row_cap)
        logger.info(f"[SQL] {command[:300]}")
        return super().run_sql(command)


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
        conn_str = (
            f"DRIVER={{{DB_DRIVER}}};"
            f"SERVER={DB_SERVER};DATABASE={DB_NAME};"
            f"UID={DB_USER};PWD={DB_PASSWORD};"
            f"Connect Timeout={DB_CONNECT_TIMEOUT};"
        )
        c = pyodbc.connect(conn_str, timeout=DB_CONNECT_TIMEOUT)
        c.close()
        logger.info("ODBC connection OK.")
    except Exception as e:
        logger.error(f"ODBC connection failed: {e}")
        sys.exit(1)


def _check_ollama() -> None:
    """Warn (do not exit) if Ollama is unreachable."""
    try:
        urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        logger.info(f"Ollama reachable at {OLLAMA_BASE_URL}.")
    except (URLError, OSError):
        logger.warning(
            f"Ollama not reachable at {OLLAMA_BASE_URL}. "
            "Fallback LLM unavailable — only Groq will be used."
        )


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


def _build_or_load_index(sql_database: SafeSQLDatabase, table_names: list[str]) -> ObjectIndex:
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=t, context_str=_schema_context_str(t))
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

    reason = "REBUILD_INDEX=1" if force_rebuild else ("schema changed" if stale else "no index found")
    logger.info(f"Building new index ({reason})...")
    idx = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex)
    idx.persist(persist_dir=INDEX_PERSIST_DIR)
    _save_index_hash()
    logger.info("Index built and saved.")
    return idx


# ============================================================
# RETRY — wraps any callable, backs off on transient errors
# ============================================================

def _is_transient(err: Exception) -> bool:
    if hasattr(err, "status_code") and err.status_code in (429, 413, 500, 502, 503):
        return True
    msg = str(err).lower()
    return any(k in msg for k in ("rate limit", "429", "413", "quota", "too large",
                                   "payload", "timeout", "connection", "temporarily"))


def _with_retry(fn, label: str = ""):
    last_err = None
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except ValueError:
            raise  # safety blocks — never retry
        except Exception as e:
            last_err = e
            if not _is_transient(e) or attempt == MAX_RETRY_ATTEMPTS:
                raise
            delay = 2.0 * (2 ** (attempt - 1))  # 2s, 4s, 8s
            logger.warning(f"{label} transient error (attempt {attempt}/{MAX_RETRY_ATTEMPTS}), "
                           f"retrying in {delay:.0f}s: {type(e).__name__}")
            time.sleep(delay)
    raise last_err


# ============================================================
# CONVERSATION HISTORY
# ============================================================

_history: deque = deque(maxlen=HISTORY_MAX)


def _build_contextual_query(question: str) -> str:
    """Prepend last 2 turns so the LLM can handle follow-up questions."""
    if not _history:
        return question
    lines = []
    for prev_q, prev_a in list(_history)[-2:]:
        lines.append(f"[Previous question]: {prev_q}")
        lines.append(f"[Previous answer]: {str(prev_a)[:200]}")
    lines.append(f"[Current question]: {question}")
    return "\n".join(lines)


# ============================================================
# QUERY PIPELINE
# ============================================================

def _make_engine(sql_db: SafeSQLDatabase, obj_index: ObjectIndex, llm) -> SQLTableRetrieverQueryEngine:
    return SQLTableRetrieverQueryEngine(
        sql_db,
        obj_index.as_retriever(similarity_top_k=RETRIEVER_TOP_K),
        text_to_sql_prompt=CUSTOM_TEXT_TO_SQL_PROMPT,
        response_synthesis_prompt=RESPONSE_SYNTHESIS_PROMPT,
        llm=llm,
    )


def _is_rate_limit(err: Exception) -> bool:
    if hasattr(err, "status_code") and err.status_code in (429, 413):
        return True
    msg = str(err).lower()
    return any(k in msg for k in ("rate limit", "429", "413", "quota", "too large", "payload"))


def _safe_query(
    question: str,
    groq_engine: SQLTableRetrieverQueryEngine,
    ollama_engine: SQLTableRetrieverQueryEngine,
) -> tuple:
    """
    Returns (response, sql_used, used_fallback).
    Tries Groq first; falls back to Ollama on rate-limit/payload errors.
    Retries transient errors with exponential back-off.
    """
    contextual_q = _build_contextual_query(question)

    def _run_groq():
        return groq_engine.query(contextual_q)

    try:
        resp = _with_retry(_run_groq, label="Groq")
        sql = (resp.metadata or {}).get("sql_query", "")
        return resp, sql, False

    except Exception as e:
        if _is_rate_limit(e):
            logger.warning(f"Groq limit hit ({type(e).__name__}) — switching to Ollama fallback.")

            def _run_ollama():
                return ollama_engine.query(contextual_q)

            resp = _with_retry(_run_ollama, label="Ollama")
            sql = (resp.metadata or {}).get("sql_query", "")
            return resp, sql, True
        raise


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_input(raw: str) -> str:
    q = raw.strip()
    if not q:
        raise ValueError("Empty input.")
    if len(q) > MAX_QUERY_LEN:
        raise ValueError(
            f"Input too long ({len(q)} chars). Please keep questions under {MAX_QUERY_LEN} characters."
        )
    return q


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # ---- startup checks ----
    _check_credentials()
    _check_odbc()
    _check_ollama()

    # ---- LLMs ----
    logger.info(f"Configuring Groq: {GROQ_MODEL_NAME}")
    groq_llm = Groq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=GROQ_MAX_TOKENS,
        context_window=GROQ_CONTEXT_WINDOW,
    )

    logger.info(f"Configuring Ollama: {OLLAMA_MODEL_NAME}")
    ollama_llm = Ollama(
        model=OLLAMA_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
        temperature=0,
        context_window=OLLAMA_NUM_CTX,
        additional_kwargs={"num_ctx": OLLAMA_NUM_CTX},
    )

    Settings.llm = groq_llm

    # ---- embedding ----
    logger.info(f"Configuring embedding: {EMBED_MODEL_NAME}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device="cpu")

    # ---- DB engine (query timeout via execution_options) ----
    logger.info(f"Connecting to {DB_SERVER}/{DB_NAME}...")
    encoded_pw = quote_plus(DB_PASSWORD)
    engine = create_engine(
        f"mssql+pyodbc://{DB_USER}:{encoded_pw}@{DB_SERVER}/{DB_NAME}"
        f"?driver={DB_DRIVER.replace(' ', '+')}&Connect+Timeout={DB_CONNECT_TIMEOUT}",
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_pre_ping=True,
    ).execution_options(timeout=DB_QUERY_TIMEOUT)

    with engine.connect() as c:
        ver = c.execute(text("SELECT @@VERSION")).fetchone()[0]
        logger.info(f"DB connected: {ver[:80]}...")

    # ---- discover tables ----
    insp = inspect(engine)
    all_tables = set(insp.get_table_names())
    found = [t for t in TARGET_TABLES if t in all_tables]
    missing = [t for t in TARGET_TABLES if t not in all_tables]
    if missing:
        logger.warning(f"Tables not found in DB: {missing}")
    if not found:
        logger.error("None of the target tables exist in the database.")
        sys.exit(1)
    logger.info(f"Using {len(found)}/{len(TARGET_TABLES)} tables: {', '.join(found)}")

    # ---- SafeSQLDatabase + index ----
    sql_db = SafeSQLDatabase(engine, include_tables=found, row_cap=ROW_CAP)
    obj_index = _build_or_load_index(sql_db, found)

    # ---- query engines (both built at startup) ----
    groq_engine   = _make_engine(sql_db, obj_index, groq_llm)
    ollama_engine = _make_engine(sql_db, obj_index, ollama_llm)
    logger.info("Query engines ready.")

    # ---- session ----
    print("\n" + "=" * 70)
    print("DB ASSISTANT — ready")
    print(f"  Primary LLM : {GROQ_MODEL_NAME}")
    print(f"  Fallback LLM: {OLLAMA_MODEL_NAME}")
    print(f"  Tables      : {len(found)}")
    print(f"  Row cap     : {ROW_CAP}  |  History: {HISTORY_MAX} turns")
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

        print("  Processing...")
        try:
            response, sql_used, used_fallback = _safe_query(question, groq_engine, ollama_engine)

            answer = str(response)
            _history.append((question, answer))

            tag = " [via Ollama fallback]" if used_fallback else ""
            print(f"\nAnswer{tag}:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            if sql_used:
                print(f"\nSQL used:\n{sql_used}")

        except ValueError as ve:
            # Safety block — show the reason, nothing sensitive
            logger.warning(f"Safety block: {ve}")
            print(f"\n  Blocked: {ve}")

        except Exception as e:
            # Log full traceback to file, show a clean message to user
            logger.exception(f"Query failed for: {question!r}")
            print(f"\n  Could not answer that question. "
                  f"Please rephrase or check the logs for details.")


if __name__ == "__main__":
    main()