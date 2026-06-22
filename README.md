# AL TASNIM Well Delivery Intelligence Assistant

Hybrid RAG + Text-to-SQL + Agentic AI system for operational intelligence across AL TASNIM well delivery, drilling progress, and workforce data.

---

## System Overview

Two complementary services work together:

| Service | File | Port | Role |
|---|---|---|---|
| **RAG/SQL server** | `prod_rag.py` | 8000 | Retrieval, Text-to-SQL, vector search, BM25, LLM synthesis |
| **Agentic AI server** | `agent/app.py` | 8001 | LangGraph orchestration, multi-tool routing, evidence validation, human-review escalation |

The agentic layer sits in front of `prod_rag.py`. It adds clarification gates, multi-source orchestration, analytics, structured recommendations, and observability — without duplicating retrieval logic.

---

## Part 1 — RAG / SQL Server (`prod_rag.py`)

### What it does

A single FastAPI server that answers natural-language questions by routing each query to the right engine:

| Question type | Engine | How |
|---|---|---|
| Operational counts, filters, status | **Text-to-SQL** | Intent classifier → parameterised SQL → PostgreSQL |
| Knowledge, procedures, KT content | **Hybrid RAG** | pgvector dense + BM25 keyword → RRF → LLM |
| Headcount, workforce questions | **Hybrid RAG** | Workforce facts embedded at ingest → pgvector + BM25 |

Every answer is returned in a **7-section structured format** (DIRECT ANSWER / EVIDENCE / SOURCE / CONFIDENCE LEVEL / ASSUMPTIONS / RISK / RECOMMENDED NEXT ACTION) with source citations.

### Architecture

```
prod_rag.py                  ← single FastAPI server — all logic lives here
  ├─ Hybrid retrieval        pgvector HNSW dense + BM25, RRF fusion (60/40 BM25:dense)
  ├─ Contextual compression  BGE-M3 sentence-similarity — strips irrelevant sentences
  │                          before sending context to the LLM. No extra LLM call.
  ├─ Self-correcting loop    If top-hit score < threshold: LLM rewrites query → retry
  ├─ Semantic answer cache   In-memory cosine-similarity cache (TTL 15 min)
  ├─ Text-to-SQL             Intent classifier → parameterised SQL → PostgreSQL tables
  ├─ Anti-hallucination      Every number in the answer is verified against retrieved chunks
  └─ Parent-doc retrieval    Matched child chunk → full parent row fetched for LLM context

config/
  sql_config.yaml            All data sources + ingest config + Text-to-SQL intents + trigger words

scripts/
  09_universal_ingest.py     Ingest all sources defined in sql_config.yaml into PostgreSQL
  07_auto_extract_excel.py   Auto-scan Excel files for column metadata
  10_test_all.py             Run full test suite against the running RAG server (48 tests)
  12_test_agent.py           Run full test suite against the agentic layer (52 tests)
```

### Endpoints

#### `GET /healthz`
```json
{"status": "ok", "chunks_in_db": 13599, "chunks_in_bm25": 13599}
```

#### `POST /ask`
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many wells are in ROL buffer status in Nimr?"}'
```
Response fields: `direct_answer`, `evidence`, `source_citation`, `confidence_level`, `assumptions`, `risk_limitation`, `recommended_next_action`, `sources`, `retrieval_attempts`, `from_cache`, `latency_ms`.

#### `POST /search`
Returns raw ranked chunks — useful for debugging retrieval quality.
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "buffer status nimr cluster", "top_k": 5}'
```

---

## Part 2 — Agentic AI Module (`agent/`)

### What it adds over prod_rag.py

| Feature | prod_rag.py | agent/ |
|---|---|---|
| Single-tool routing | ✅ | ✅ |
| Multi-tool orchestration (SQL + RAG + analytics) | ❌ | ✅ |
| Clarification gate (vague queries) | ❌ | ✅ |
| Analytics layer (delay risk, completion estimate) | ❌ | ✅ |
| Structured recommendation with risk/cost/approval | ❌ | ✅ |
| Human-review escalation | ❌ | ✅ |
| Semantic query cache (5-min TTL) | ✅ | ✅ |
| Observability log (PostgreSQL) | ❌ | ✅ |
| LangGraph state machine | ❌ | ✅ |

### Architecture

```
agent/
  app.py          FastAPI server (port 8001) — POST /agent/ask, GET /agent/health
  orchestrator.py LangGraph CompiledStateGraph — 5 nodes, conditional routing
  router.py       Tier-1 rule-based regex → Tier-2 cheap LLM fallback (no hardcoding)
  tools.py        rag_tool, sql_tool, analytics_tool, recommend_tool
  validator.py    Evidence validation, confidence scoring, human-review triggers
  cache.py        In-memory TTL cache — exact-match + semantic cosine similarity
  config.py       Loads config/agent_config.yaml with ${ENV_VAR} substitution
  models.py       Pydantic AgentRequest / AgentResponse, AgentState TypedDict

config/
  agent_config.yaml   ALL agent settings — routes, prompts, LLM tiers, cache,
                      human-review triggers, clarification prompts, embed model
                      Nothing is hardcoded in Python.
```

### LangGraph flow

```
START
  └─► clarification_gate ──(vague?)──► END  [returns clarification question]
           │
        (specific)
           │
           ▼
      execute_tools  ──────────────────────────────────────────────┐
       (parallel SQL+RAG for multi/recommend routes)               │
           │                                                        │
      (recommend?)                                            (other routes)
           │                                                        │
           ▼                                                        │
  generate_recommendation                                           │
           │                                                        │
           └──────────────────────► validate_and_format ◄──────────┘
                                           │
                                      log_and_return
                                           │
                                          END
```

### Routing logic (all in `config/agent_config.yaml`)

| Route | When triggered | Tools called |
|---|---|---|
| `rag` | "procedure / explain / SOP / what is the X procedure" | RAG only |
| `sql` | "how many / list / current status / show / count" | SQL only |
| `analytics` | "when will it finish / forecast / delay risk / completion date" | SQL + Analytics |
| `multi` | "why is it delayed / root cause / reason for delay" | SQL + RAG |
| `recommend` | "what should we do / recommend / how to recover / action plan" | SQL + RAG + Analytics + LLM |
| `clarification` | Vague query without a specific well/rig/activity ID | None — asks user |

### Configuration — nothing hardcoded in Python

Every configurable value lives in `config/agent_config.yaml`:

| Section | Controls |
|---|---|
| `server` | Host, port, title, version |
| `rag_service` | Upstream prod_rag.py URL, endpoints, timeout |
| `llm.classifier` | Small model for Tier-2 routing (reads `${LLM_MODEL}`) |
| `llm.generator` | Ollama model for answer synthesis (reads `${LLM_MODEL}`) |
| `llm.reasoning` | Groq model for recommendations (reads `${GROQ_API_KEY}`) |
| `router.*_patterns` | All regex routing rules (rag / sql / analytics / multi / recommend) |
| `router.rag_priority_patterns` | RAG signals that win over SQL keyword matches |
| `clarification.vague_patterns` | Patterns that trigger the clarification gate |
| `clarification.specific_id_pattern` | Pattern that overrides the gate (well ID present) |
| `clarification.prompts` | The clarification question text shown to users |
| `validation` | Min answer length, hallucination phrases, refuse message |
| `human_review.trigger_patterns` | Queries that escalate to manager review |
| `human_review.operations_review_message` | Message shown with recommendations |
| `embeddings.model` | Sentence-transformer model for semantic cache (reads `${EMBED_MODEL}`) |
| `prompts.recommend_system` | System prompt for the recommendation LLM |
| `prompts.recommend_template` | User prompt template for recommendations |
| `prompts.llm_router_system` | System prompt for Tier-2 LLM classifier |
| `cache` | TTL, semantic threshold, enabled flag |
| `logging` | PostgreSQL log table name, trace fields |

### Adding a new route

1. Add regex patterns to the appropriate section in `config/agent_config.yaml` — no Python changes needed for simple routing.
2. If the route needs a new tool, add an async function in `agent/tools.py` returning `{"tool": "name", "answer": "...", "error": None}`.
3. Wire the new tool in `node_execute_tools` inside `agent/orchestrator.py`.
4. Add test cases to `scripts/12_test_agent.py`.

### Endpoints

#### `GET /agent/health`
```json
{"status": "ok", "service": "AL TASNIM Agentic AI", "version": "1.0.0"}
```

#### `POST /agent/ask`
```bash
curl -X POST http://localhost:8001/agent/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why is well 30750 behind schedule?",
    "session_id": "user-abc",
    "user_role": "operations",
    "top_k": 8
  }'
```

Response fields:

| Field | Description |
|---|---|
| `direct_answer` | The synthesised answer |
| `evidence` | SQL queries executed or analytics metrics used |
| `sources` | Document or table sources cited |
| `confidence` | `High` / `Medium` / `Low` |
| `assumptions` | Explicit assumptions the answer depends on |
| `risk_limitation` | Known limitations or risks flagged |
| `recommended_next_action` | What the user should do next |
| `route` | Which route was taken (`sql`, `rag`, `multi`, etc.) |
| `tools_called` | List of tools invoked |
| `needs_human_review` | `true` if manager approval is required |
| `human_review_reason` | Why review is needed |
| `clarification_needed` | `true` if the query was too vague |
| `clarification_text` | The clarification question to ask the user |
| `from_cache` | `true` if served from cache |
| `latency_ms` | End-to-end response time |

#### `GET /agent/config`
Returns active config summary (add `?role=admin` for full config).

---

## Data sources

Configured in the `tables:` section of `config/sql_config.yaml`. Active tables:

| Table | Source file | Description |
|---|---|---|
| `well_monitoring` | WellMonitoringReport_Feb_Data_VERIFIED.csv | 128-column well status, rig, progress, dates |
| `wmr_nimr` | WMR-Nimr.xlsx | Nimr cluster WMR — 370 wells, buffer status, pegged date |
| `activity_master` | Activity Master_VERIFIED.xlsx | Productivity norms, qty/hr, crew group per activity code |
| `well_master` | Well Master _ Nimr _ 2026_VERIFIED.xlsx | Well IDs, locations, attributes |
| `crew_master` | Crew Master_VERIFIED.xlsx | Crew groups, formations, PG/EG codes |
| `operational_wells` | ACC Dump | 483 wells across 27 fields with rig assignment |
| `operational_tasks` | Task_Plan.xlsx | 44,235 milestone records per well with progress |
| `well_delivery_kpis` | Well Delivery Core KPIs.xlsx | 13 KPI metrics — Nimr vs Marmul counts |
| `daily_plan_nimr` | DailyPlan-Nimr.xlsx | Daily plan for Nimr field operations |
| `ph_productivity` | PH_Productivity_Nimr_2026-05-10.xlsx | Person-hour productivity (May 2026) |
| `construction_delivery` | WD Dashboard.xlsx | Well delivery management (87 columns) |
| `project_ids` | ProjectIDs_VERIFIED.csv | Project code → name lookup |
| `daily_plan` | Daily_Plan.csv | Planned vs actual activity timelines |

---

## Setup

```bash
# 1. Activate environment
conda activate v12

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create the database
createdb altasnim
psql altasnim -c 'CREATE EXTENSION IF NOT EXISTS vector;'

# 4. Copy env and configure
cp .env.example .env
# Required: DATABASE_URL, LLM_MODEL, OLLAMA_URL, EMBED_MODEL
# Optional: GROQ_API_KEY (enables Tier-3 reasoning for recommendations)

# 5. Ingest all data sources
python scripts/09_universal_ingest.py

# 6. Start the RAG/SQL server (port 8000)
uvicorn prod_rag:app --host 0.0.0.0 --port 8000

# 7. Start the Agentic AI server (port 8001) — in a separate terminal
python agent/app.py
```

Data files live in `data/` (gitignored — store in shared drive, copy locally before ingesting).

---

## Running tests

```bash
# RAG/SQL server tests (48 tests) — requires prod_rag.py on :8000
conda activate v12
python scripts/10_test_all.py

# Agentic AI unit tests (48 tests) — no services needed
python scripts/12_test_agent.py

# Agentic AI full tests including integration (52 tests) — requires prod_rag.py on :8000
python scripts/12_test_agent.py --integration
```

---

## Adding a new data source

1. Copy the file to `data/EXCEL/` or `data/CSV/`
2. Add a table entry (with `type`, `dir`, `source_file`, `sheet`, `header_row`) to the `tables:` section of `config/sql_config.yaml`
3. Run `python scripts/09_universal_ingest.py`
4. Add an intent to the `intents:` section of `config/sql_config.yaml` for Text-to-SQL support
5. Server auto-loads the live schema at startup — no code changes needed

---

## Blueprint status

| Feature | Status |
|---|---|
| Hybrid Search (BM25 + Dense + RRF) | Done |
| Semantic Chunking | Done |
| Parent-Document Retriever | Done |
| HNSW tuning (m=32, ef=200) | Done |
| Structured output + citations | Done |
| Semantic answer cache | Done |
| Contextual Compression | Done |
| Self-correcting loop (rewrite + retry) | Done |
| Text-to-SQL with intent classification | Done |
| Anti-hallucination number validation | Done |
| Request tracing (intent, chunks, latency) | Done |
| LangGraph agentic orchestration | Done |
| Policy-based + LLM hybrid router | Done |
| Clarification gate (vague query detection) | Done |
| Multi-tool parallel execution | Done |
| Analytics layer (delay risk, progress metrics) | Done |
| Structured recommendation engine | Done |
| Human-review escalation | Done |
| Observability log (PostgreSQL) | Done |
| Semantic cache (agent layer, 5-min TTL) | Done |
| Auth / role-based access control | Pending — wire JWT middleware before production |
| Redis cache (Phase 2) | Pending — swap cache.py only |
| ML-based delay forecasting (Phase 2) | Pending — analytics_tool uses heuristics now |
| Groq / Cerebras Tier-3 LLM | Pending — set GROQ_API_KEY in .env to enable |

---

## TODO before production

- **Auth**: `/ask`, `/search`, and `/agent/ask` endpoints are open. Wire JWT / API-key middleware before exposing outside localhost.
- **Rate limiting**: no throttle on the LLM call path — add `slowapi` or nginx rate-limit for production traffic.
- **GROQ_API_KEY**: set in `.env` to enable the Tier-3 reasoning model for recommendations. Without it, recommendations fall back to the local Ollama model.
- **Redis**: replace `agent/cache.py` in-memory cache with Redis for multi-instance deployments.
