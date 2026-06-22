# AL TASNIM Well Delivery Intelligence Assistant
### RAG + Text-to-SQL Operational Intelligence System

A production-ready FastAPI server that answers natural-language questions about well delivery, drilling progress, and operational data. Routes each question to the right engine — SQL for live database queries, RAG for document knowledge — with no hardcoded values anywhere.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Folder Structure](#2-folder-structure)
3. [Step 1 — Set Up Python Environment](#3-step-1--set-up-python-environment)
4. [Step 2 — Set Up PostgreSQL](#4-step-2--set-up-postgresql)
5. [Step 3 — Configure Environment Variables](#5-step-3--configure-environment-variables)
6. [Step 4 — Set Up LLM (Groq or Ollama)](#6-step-4--set-up-llm-groq-or-ollama)
7. [Step 5 — Place Data Files](#7-step-5--place-data-files)
8. [Step 6 — Ingest Data](#8-step-6--ingest-data)
9. [Step 7 — Start the Server](#9-step-7--start-the-server)
10. [Step 8 — Run Tests](#10-step-8--run-tests)
11. [API Reference](#11-api-reference)
12. [Configuration Reference](#12-configuration-reference)
13. [How the System Works](#13-how-the-system-works)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. System Requirements

| Component | Required | Notes |
|---|---|---|
| Python | 3.11 or 3.12 | conda recommended |
| PostgreSQL | 14+ with pgvector extension | must have pgvector installed |
| GPU (optional) | NVIDIA CUDA | for faster embeddings; CPU works too |
| Groq API key | Recommended | free tier at console.groq.com — best answer quality |
| Ollama | Alternative to Groq | local LLM, slower, no API key needed |

---

## 2. Folder Structure

```
rag_deploy/
├── prod_rag.py                  Main server — run this
├── requirements.txt             Python dependencies
├── .env.example                 Copy to .env and fill in your values
├── test_rag.py                  Full test suite (37 tests)
├── config/
│   ├── sql_config.yaml          All data sources, SQL intents, trigger words
│   ├── rbac_config.yaml         Role-based access control
│   ├── alerts_config.yaml       SQL alert rules (evaluated every 5 min)
│   └── rsr_scr_config.yaml      RSR/SCR email watcher config
└── scripts/
    ├── 09_universal_ingest.py   Load all data files into PostgreSQL
    └── 07_auto_extract_excel.py Scan Excel files for column metadata
```

---

## 3. Step 1 — Set Up Python Environment

```bash
conda create -n v12 python=3.12
conda activate v12
pip install -r requirements.txt
```

If you do not have conda:
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Step 2 — Set Up PostgreSQL

### Install pgvector

Ubuntu / Debian:
```bash
sudo apt install postgresql-16-pgvector
```

macOS (Homebrew):
```bash
brew install pgvector
```

### Create the database

```bash
createdb altasnim
psql altasnim -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Verify:
```bash
psql altasnim -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
# Should print: vector
```

---

## 5. Step 3 — Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
# PostgreSQL connection
PG_URL=postgresql://YOUR_USER@/altasnim?host=/var/run/postgresql

# LLM — choose groq (recommended) or ollama
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here

# Models
LLM_MODEL=llama-3.1-8b-instant
GROQ_SYNTHESIS_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=BAAI/bge-m3

# Ollama (only needed if LLM_PROVIDER=ollama)
OLLAMA_URL=http://localhost:11434
```

### Getting a Groq API key (free)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up → API Keys → Create API Key
3. Paste the key as `GROQ_API_KEY` in your `.env`

### PostgreSQL connection string formats

| Setup | PG_URL |
|---|---|
| Unix socket (Linux default) | `postgresql://USER@/altasnim?host=/var/run/postgresql` |
| TCP localhost | `postgresql://USER:PASSWORD@localhost:5432/altasnim` |
| Remote server | `postgresql://USER:PASSWORD@HOST:5432/altasnim` |

---

## 6. Step 4 — Set Up LLM (Groq or Ollama)

### Option A — Groq (Recommended)

No installation needed. Just set `LLM_PROVIDER=groq` and add your `GROQ_API_KEY` in `.env`.

Models used:
- **Classification** (intent routing): `llama-3.1-8b-instant` — fast, cheap
- **Synthesis** (answer generation): `llama-3.3-70b-versatile` — GPT-4o class, accurate

### Option B — Ollama (Local, No API Key)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.1:8b

# Start Ollama (it runs as a daemon automatically after install)
ollama serve
```

Then in `.env`:
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_URL=http://localhost:11434
```

> Ollama is slower (30–60 seconds per query) compared to Groq (2–5 seconds).

---

## 7. Step 5 — Place Data Files

The data files are not included in this repository (they contain client data). Place them in a `data/` folder alongside this README:

```
data/
├── CSV/
│   ├── POC 1 WellMonitoringReport_Feb_Data_VERIFIED.csv
│   ├── POC 1 ProjectIDs_VERIFIED.csv
│   └── Daily_Plan.csv
└── EXCEL/
    ├── Task_Plan.xlsx
    ├── Crew Master_VERIFIED.xlsx
    ├── Well Master _ Nimr _ 2026_VERIFIED.xlsx
    ├── Activity Master_VERIFIED.xlsx
    ├── Job Progress Report-Nimr.xlsx
    ├── DailyPlan-Nimr.xlsx
    ├── PH_Productivity_Nimr_2026-05-10.xlsx
    ├── Well Delivery Core KPIs (1) (1).xlsx
    ├── Operational Charts - Construction - WD Dashboard (1).xlsx
    ├── WMR-Nimr.xlsx
    └── Mockup Dashboard with Drilldown 04032026_VERIFIED.xlsx
```

All expected file names are defined in `config/sql_config.yaml` under the `tables:` section — that is the single source of truth.

---

## 8. Step 6 — Ingest Data

This loads all data files into PostgreSQL and builds the vector embeddings.

```bash
conda activate v12
python scripts/09_universal_ingest.py
```

To reload everything from scratch (drops and rebuilds all tables):
```bash
python scripts/09_universal_ingest.py --reset
```

Expected output:
```
[ingest] well_monitoring  → 10632 rows
[ingest] activity_master  → 340 rows
[ingest] well_master      → 188 rows
...
[rag] Pushed 13606 chunks to pgvector
[rag] Built BM25 index: 11375 chunks
[done] Ingest complete
```

> This takes 5–15 minutes on first run due to embedding generation.

---

## 9. Step 7 — Start the Server

```bash
conda activate v12
uvicorn prod_rag:app --host 0.0.0.0 --port 8000
```

Wait for this line before sending any requests:
```
[ready] 13,606 chunks | pgvector HNSW + BM25 | BGE-M3 + Groq LLM
```

To run in background:
```bash
nohup uvicorn prod_rag:app --host 0.0.0.0 --port 8000 > server.log &
```

To stop the background server:
```bash
pkill -f "uvicorn prod_rag"
```

> If port 8000 is already in use (e.g. by VS Code extension), use `--port 8002`.

---

## 10. Step 8 — Run Tests

```bash
conda activate v12
python test_rag.py

# If server is on a different port:
python test_rag.py --url http://localhost:8002

# If server is on a remote machine:
python test_rag.py --url http://192.168.1.100:8000
```

### What the test suite covers (37 tests)

| Section | Tests | What it checks |
|---|---|---|
| Health | 1 | DB connected, chunks loaded, models ready |
| Total counts | 2 | Counts all wells with correct number |
| Filtered counts by rig | 3 | SWER101=682, SWER102=954, etc. |
| Filtered counts by status | 3 | Completed/In Progress/Not Started |
| Status breakdown | 3 | GROUP BY status columns |
| Ranked lists | 3 | Top/bottom wells by progress |
| Progress range list | 3 | Wells below/above a % threshold |
| Well list with filter | 2 | List wells on a rig or by status |
| Well detail lookup | 3 | By PDO well ID and by well name |
| Activity master | 3 | Civil/mechanical activities, activity codes |
| Per-rig summary | 2 | Average progress per rig |
| RAG document knowledge | 5 | M-90, FLAF, rig-on steps, buffer status |
| RAG routing (must not SQL) | 2 | Scope/procedure questions stay in RAG |
| Edge cases | 3 | Unknown IDs, zero results, vague queries |

Expected result on a working system:
```
Score: 37/37 (100%)
```

---

## 11. API Reference

### GET /healthz

Check that the server is running and the database is loaded.

```bash
curl http://localhost:8000/healthz
```

Response:
```json
{
  "status": "ok",
  "chunks_in_db": 13606,
  "chunks_in_bm25": 11375,
  "embed_model": "BAAI/bge-m3",
  "llm_model": "llama3.1:8b"
}
```

---

### POST /ask

Ask a natural-language question. The server automatically routes to SQL (for operational data) or RAG (for document knowledge).

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many wells does SWER101 have?"}'
```

Request body:

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Natural language question |
| `top_k` | int | no | Number of RAG chunks to retrieve (default 8) |

Response fields:

| Field | Description |
|---|---|
| `direct_answer` | 2–3 sentence answer |
| `evidence` | Exact data points or SQL result that support the answer |
| `source_citation` | Source file name(s) |
| `confidence_level` | High / Medium / Low with explanation |
| `assumptions` | Assumptions the answer depends on |
| `risk_limitation` | Known data gaps or staleness |
| `recommended_next_action` | Suggested next step for an operations manager |
| `sources` | List of source documents |
| `from_cache` | true if served from semantic cache |
| `latency_ms` | End-to-end response time in milliseconds |

Example response:
```json
{
  "direct_answer": "The RIG SWER101 has 682 wells.",
  "evidence": "SQL: SELECT COUNT(*) FROM well_monitoring WHERE rig_no = 'SWER101' → count = 682",
  "source_citation": "POC 1 WellMonitoringReport_Feb_Data_VERIFIED.csv",
  "confidence_level": "High — verified directly from database.",
  "assumptions": "Data is current as of the last ingestion run.",
  "risk_limitation": "Shows a maximum of 50 rows.",
  "recommended_next_action": "Drill down with a more specific filter if needed.",
  "sources": ["POC 1 WellMonitoringReport_Feb_Data_VERIFIED.csv"],
  "from_cache": false,
  "latency_ms": 1177
}
```

---

### POST /search

Returns raw ranked chunks — useful for debugging retrieval quality without LLM synthesis.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "buffer status nimr cluster", "top_k": 5}'
```

Request body:

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Search query |
| `top_k` | int | no | Number of results (default 8, max 20) |

---

## 12. Configuration Reference

All domain knowledge lives in config files. Python contains only logic.

### config/sql_config.yaml

The most important config file. Controls everything about Text-to-SQL.

| Section | What it controls |
|---|---|
| `default_table` | Fallback table when intent has no `table:` set |
| `max_rows` | Maximum rows returned by any SQL query |
| `patterns.rig_id` | Regex pattern to detect rig IDs (e.g. SWER101) in queries |
| `exclude_words` | Words that force RAG routing (e.g. "workforce", "meeting") |
| `trigger_words` | Words that force SQL routing (e.g. "how many", "buffer status") |
| `tables` | One entry per data source — name, file, sheet, columns |
| `intents` | One entry per SQL intent — sql_type, columns, filters, descriptions |

**SQL intent types:**

| `sql_type` | What it generates | Example question |
|---|---|---|
| `count` | `SELECT COUNT(*) ... WHERE col = val` | How many wells does SWER101 have? |
| `list` | `SELECT cols ... WHERE col ILIKE val` | List wells on SWER102 |
| `ranked` | `SELECT cols ... ORDER BY col DESC LIMIT n` | Top 5 wells by progress |
| `range` | `SELECT COUNT(*) WHERE col BETWEEN a AND b` | How many wells are 50–80% complete? |
| `range_list` | `SELECT cols WHERE col BETWEEN a AND b` | List wells below 50% progress |
| `group_by` | `SELECT col, COUNT(*), AVG(agg) GROUP BY col` | Average progress per rig |
| `count_by_column` | `SELECT col, COUNT(*) GROUP BY col` | Buffer status breakdown |
| `status_breakdown` | `SELECT status, COUNT(*) GROUP BY status` | Location prep status breakdown |
| `detail` | `SELECT cols WHERE id_col = val` | Show details for PDO well ID 34568 |
| `multi_filter` | `SELECT cols WHERE col1=v1 AND col2=v2` | Activity code C-001 for civil discipline |

**To add a new data source:**
1. Add an entry to `tables:` in `sql_config.yaml`
2. Add an entry to `intents:` in `sql_config.yaml`
3. Re-run `python scripts/09_universal_ingest.py`
4. No Python changes needed

### config/rbac_config.yaml

Controls which API keys can access which tables.

API keys are set in `.env`:
```env
AL_TASNIM_API_KEYS=secret-admin-key:ngxp_admin:Admin,ops-key-001:al_tasnim_ops:OpsUser
```

If `AL_TASNIM_API_KEYS` is not set, the server runs in **dev mode** (all requests treated as admin).

Roles: `ngxp_admin`, `pdo_admin`, `cluster_manager`, `planner`, `al_tasnim_ops`, `civil_executive`, `read_only`

### config/alerts_config.yaml

Defines SQL alert rules evaluated every 5 minutes. Sends email via SMTP. Configure SMTP credentials in `.env`.

### config/rsr_scr_config.yaml

Watches a folder for RSR/SCR PDF files and auto-emails stakeholders. Set recipient emails in `.env` via `RSR_PLANNER_EMAIL`, `RSR_CIVIL_EMAIL`, etc.

---

## 13. How the System Works

```
User question
     │
     ▼
Is it a SQL question?
(trigger_words match / fast-path regex)
     │
   ┌─┴─────────────────┐
  YES                  NO
   │                   │
   ▼                   ▼
Intent classifier   Hybrid RAG
(cheap LLM → JSON) (pgvector HNSW
   │                + BM25 → RRF)
   ▼                   │
Config-driven          ▼
SQL compiler       Contextual
(sql_config.yaml)  compression
   │                   │
   ▼                   ▼
PostgreSQL         Good hits → LLM synthesis
   │               Bad hits → rewrite query → retry
   ▼                   │
DB result text         ▼
   │               _validate_numbers()
   ▼               (anti-hallucination)
_validate_sql_numbers()    │
(count guard)              │
   └──────────┬────────────┘
              ▼
     Groq llama-3.3-70b (if key set)
     or Ollama llama3.1:8b (fallback)
              │
              ▼
     7-section structured answer
     (DIRECT ANSWER / EVIDENCE /
      SOURCE / CONFIDENCE /
      ASSUMPTIONS / RISK /
      RECOMMENDED NEXT ACTION)
              │
              ▼
     Semantic cache store (15 min TTL)
              │
              ▼
           Response
```

### Two-tier LLM

| Task | Model | Why |
|---|---|---|
| Intent classification (SQL routing) | `llama-3.1-8b-instant` via Groq | Cheap, fast, just needs to output JSON |
| Answer synthesis (RAG + SQL) | `llama-3.3-70b-versatile` via Groq | GPT-4o class — accurate, no hallucinations |
| Fallback (no Groq key) | `llama3.1:8b` via Ollama | Local, free, slower |

### Anti-hallucination guards

1. **SQL count guard** — if DB returns `count=682` but LLM writes a different number, the answer is replaced with a direct statement from the DB row
2. **RAG number guard** — every number in a RAG answer is verified against retrieved chunks; if not found, answer is overridden with NOT FOUND IN CONTEXT
3. **Source citation guard** — LLM is given an explicit list of allowed source filenames; it cannot invent source names
4. **Live schema discovery** — SQL compiler reads column names from `information_schema.columns` at startup, not from YAML; stale YAML columns are silently skipped

---

## 14. Troubleshooting

### Server won't start — "table rag_chunks is empty"

The vector store is empty. Run the ingest first:
```bash
python scripts/09_universal_ingest.py
```

### Server won't start — "address already in use"

Port 8000 is taken (often by VS Code extension). Use a different port:
```bash
uvicorn prod_rag:app --host 0.0.0.0 --port 8002
python test_rag.py --url http://localhost:8002
```

### Answers are slow (30–60 seconds)

You are using Ollama with a local model. Switch to Groq for 10–20x faster responses:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

### "GROQ_API_KEY not set" error

Add your Groq API key to `.env`. Get one free at [console.groq.com](https://console.groq.com).

### SQL query returns wrong table / no results

Check what intent was classified by looking at the server log output (`[sql] Intent: {...}`). If the intent is wrong, add or improve the `fast_path_keywords` or `description` for that intent in `config/sql_config.yaml`.

### "column does not exist" error in answer

A column listed in `select_columns` in `sql_config.yaml` does not exist in the database. Either:
- Remove the column from `sql_config.yaml`
- Or re-run `python scripts/09_universal_ingest.py` to reload the data

The server silently skips unknown columns at startup (live schema validation), but this check only applies to columns verified at server start time.

### pgvector extension not found

```bash
sudo apt install postgresql-16-pgvector   # Ubuntu
psql altasnim -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Embedding model downloads on first run

The first startup downloads the `BAAI/bge-m3` model (~2GB). This is normal. Subsequent starts load from cache.

---

## Quick Reference

```bash
# Start server
conda activate v12 && uvicorn prod_rag:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/healthz

# Ask a SQL question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many wells does SWER101 have?"}'

# Ask a knowledge question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the M-90 milestone?"}'

# Run test suite
python test_rag.py

# Reload data
python scripts/09_universal_ingest.py

# Reload data from scratch
python scripts/09_universal_ingest.py --reset
```
