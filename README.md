# AL TASNIM DB Assistant

A production-ready Natural Language to SQL (NL2SQL) API for the **AppMasterDB_Local** Microsoft SQL Server database. Ask questions in plain English and get data-driven answers from the Al Tasnim oil & gas well delivery database.

---

## Features

- **Natural language to SQL** — ask questions in plain English, get answers from the database
- **Hybrid retrieval** — BM25 (keyword) + dense embeddings (semantic) with Reciprocal Rank Fusion for accurate table selection
- **Chain-of-thought SQL generation** — few-shot examples guide the LLM step by step
- **Safety enforced** — only `SELECT` statements are permitted; all DDL/DML is blocked
- **Multi-provider LLM** — switch between Groq, Google Gemini, or local Ollama via one env variable
- **Conversational intent handling** — greetings, farewells, thanks, and unrelated queries all handled gracefully
- **FastAPI** — REST API with Swagger docs, health check, CORS support

---

## Architecture

```
POST /ask
    │
    ├── Intent detection (GREETING / FAREWELL / THANKS / UNRELATED / SQL_QUERY)
    │
    └── SQL_QUERY path:
            │
            ├── Hybrid Table Retrieval
            │     ├── BM25 sparse search (rank_bm25)
            │     ├── Dense semantic search (HuggingFace BAAI/bge-small-en-v1.5)
            │     └── Reciprocal Rank Fusion → top-5 tables
            │
            ├── Schema context built from db-schema.yaml (selected tables only)
            │
            ├── SQL Generation (LLM — few-shot + chain-of-thought)
            │
            ├── Safety Validation (regex block: INSERT/UPDATE/DELETE/DROP/...)
            │
            ├── SQL Execution on SQL Server (max 500 rows, WITH NOLOCK)
            │     └── Auto-retry up to MAX_SQL_RETRIES times on failure
            │
            └── Response Formatting (LLM converts rows → readable answer)
```

---

## Project Structure

```
db-assistant/
├── db-assist.py        # FastAPI application entry point
├── pipeline.py         # Main NL2SQL orchestration pipeline
├── config.py           # All configuration via pydantic-settings
├── llm_factory.py      # LLM and embedding model factory (Groq/Gemini/Ollama)
├── schema_loader.py    # YAML schema loader + LlamaIndex SQLDatabase builder
├── retriever.py        # BM25, dense, and hybrid table retrievers
├── safety.py           # SQL safety validator (SELECT-only enforcement)
├── prompts.py          # All prompt templates and few-shot examples
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── data/
    └── db-schema.yaml  # Table and column descriptions for LLM context
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [ODBC Driver 17 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
- Access to the SQL Server at `20.98.112.250`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials and preferred LLM provider
```

### 4. Run the server

```bash
python db-assist.py
```

> **Note:** First startup takes 3–4 minutes to build the in-memory embedding index for all 28 tables. Subsequent requests are fast.

Once ready you will see:
```
Hybrid retriever ready (top_k=5)
Pipeline ready. Serving on 0.0.0.0:8000
Application startup complete.
```

---

## API

### `GET /health`

Returns server status and number of tables loaded.

```json
{
  "status": "ok",
  "llm_provider": "groq",
  "db_name": "AppMasterDB_Local",
  "tables_loaded": 28
}
```

### `POST /ask`

Submit a natural-language question.

**Request:**
```json
{
  "question": "Which wells have less than 50% progress?"
}
```

**Response:**
```json
{
  "answer": "There are 12 wells with less than 50% progress. The lowest is well 34422 at 18.3%, followed by ...",
  "sql": "SELECT pdo_well_id, well_location, ROUND(cum_progress_for_this_week * 100, 2) AS progress_pct FROM WellMonitoringReport_Latest WITH (NOLOCK) WHERE cum_progress_for_this_week < 0.5 ORDER BY cum_progress_for_this_week ASC",
  "data": [
    { "pdo_well_id": 34422, "well_location": "AL BURJ_26_MC_OP2", "progress_pct": 18.3 },
    ...
  ],
  "tables_used": ["WellMonitoringReport_Latest"],
  "error": null
}
```

**Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Example Questions

| Question | Type |
|---|---|
| How many active wells are there? | Count |
| Which wells have less than 50% progress? | Filter |
| Show me employees with their crew type | JOIN |
| What is the total contract value per project? | Aggregation |
| Show daily task records between Jan and Mar 2026 | Date range |
| Top 10 crews by productivity in Nimr | AND/OR + TOP |
| Which wells don't have an actual start date? | NULL check |
| Hello! | Greeting |
| Thank you | Thanks |

---

## LLM Provider Configuration

Set `LLM_PROVIDER` in `.env` to switch providers:

| Provider | Value | Required Key |
|---|---|---|
| Groq Cloud | `groq` | `GROQ_API_KEY` |
| Google Gemini | `gemini` | `GOOGLE_API_KEY` |
| Ollama (local) | `local` | None (Ollama must be running) |

---

## Database

- **Server:** `20.98.112.250`
- **Database:** `AppMasterDB_Local`
- **Domain:** Oil & gas well delivery and flowline construction — Al Tasnim LLC, Oman (Nimr, Marmul, Al Burj fields)
- **Tables:** 28 tables covering wells, tasks, crews, employees, revenue, WMR, SAP drilling sequences

Schema is documented in [`data/db-schema.yaml`](data/db-schema.yaml).

---

## Security

- Only `SELECT` statements are allowed. Any query containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `TRUNCATE`, `ALTER`, `CREATE`, `EXEC`, `EXECUTE`, `MERGE`, `GRANT`, `REVOKE`, `DENY`, or `BULK` is rejected before execution.
- All queries use `WITH (NOLOCK)` to avoid locking production tables.
- Results are capped at 500 rows.
