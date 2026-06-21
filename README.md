# DB Tool — Natural Language → SQL Assistant

Ask questions about drilling and well operations data in plain English.
The tool generates SQL, runs it against the live SQL Server database, and replies like a helpful colleague.

---

## Features

- **Natural language queries** — ask in plain English, get human-readable answers
- **Multi-step pipeline** — table retrieval → query planner → SQL generation → self-critique → execution → synthesis
- **Auto relationship discovery** — joins discovered from `sys.foreign_keys` + column-name matching at startup
- **Sample-value injection** — LLM sees real data examples so it knows valid values for each column
- **Case-insensitive / fuzzy search** — "rajesh", "RAJESH", "Rajesh Kumar" all work
- **Read-only enforced** — SELECT only; INSERT/UPDATE/DELETE/DROP are blocked before execution
- **SQL self-critique** — LLM reviews its own SQL before it runs (disable with `ENABLE_SQL_CRITIQUE=0`)
- **Anti-hallucination synthesis** — answer only from rows returned; never infers or adds facts
- **Groq (primary) + Ollama (fallback)** — automatic fallback if Groq is rate-limited
- **Conversation history** — follow-up questions ("show more", "filter those by field NIMR") work
- **Audit log** — every query and SQL statement logged to `logs/assistant.log`

---

## Quick start

### 1. Prerequisites

- Python 3.12+
- ODBC Driver 17 for SQL Server ([download](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server))
- A Groq API key (free at [console.groq.com](https://console.groq.com))
- *(Optional)* Ollama running locally for offline fallback

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in DB_SERVER, DB_NAME, DB_READONLY_USER, DB_READONLY_PASSWORD, GROQ_API_KEY
```

### 4. Run

```bash
python main.py
```

On first run the embedding model is downloaded from Hugging Face (~130 MB) and the vector index is built. Subsequent runs load the index from disk instantly.

---

## Environment variables

See [`.env.example`](.env.example) for the full list with descriptions.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DB_SERVER` | ✅ | — | SQL Server IP or hostname |
| `DB_NAME` | ✅ | — | Database name |
| `DB_READONLY_USER` | ✅ | — | Read-only SQL login |
| `DB_READONLY_PASSWORD` | ✅ | — | Password |
| `GROQ_API_KEY` | ✅ | — | Groq API key |
| `GROQ_MODEL_NAME` | | `llama-3.3-70b-versatile` | Groq model |
| `OLLAMA_MODEL_NAME` | | `qwen3:4b` | Ollama fallback model |
| `OLLAMA_BASE_URL` | | `http://localhost:11434` | Ollama server URL |
| `EMBED_MODEL_NAME` | | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `ROW_CAP` | | `100` | Max rows returned per query |
| `RETRIEVER_TOP_K` | | `3` | Tables passed to LLM per query |
| `LOG_LEVEL` | | `INFO` | Logging level |
| `REBUILD_INDEX` | | `0` | Set to `1` to force index rebuild |
| `ENABLE_SQL_CRITIQUE` | | `1` | Set to `0` to skip SQL self-critique (faster) |
| `ENABLE_SAMPLE_VALUES` | | `1` | Set to `0` to skip sample-value collection at startup |

---

## Tables covered

| Table | Description |
|---|---|
| `2026_Well_Delivery_Scope_Well_Type` | Well delivery scope and types for 2026 |
| `ActivityTaskPlan` | Planned activities and tasks per well |
| `WMR` | Weekly management report — well-level progress |
| `Employee` | Employee directory with status and location |
| `crews` | Crew definitions and assignments |
| `CrewEmployee` | Crew-to-employee mapping |
| `Equipment` | Equipment inventory and status |
| `task_daily` | Daily task progress and actuals |
| `Revenue` | Revenue data by well and rig |
| `SAP_DRILLING_SEQUENCE` | SAP drilling sequence and operations |

---

## Example questions

```
How many active employees are there?
Show details about Rajesh
Which crews are at location NIMR?
Show revenue by well for this year
Tell me the daily progress for well 33151
Which wells are in field NIMR E?
Show SAP drilling sequence for well 33151
List crews and their employees
```

---

## Utility scripts

| Script | Purpose |
|---|---|
| `export_all_tables.py` | Export all 10 target tables to CSV files in `exports/` |

---

## Branch structure

| Branch | Purpose |
|---|---|
| `main` | Stable production code |
| `dev` | Integration branch |
| `feature/db-tool-new` | This tool (active development) |
