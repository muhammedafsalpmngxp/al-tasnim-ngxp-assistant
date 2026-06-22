# AL TASNIM — Agentic AI Orchestrator

Production LangGraph orchestration platform for the Operational Intelligence Assistant.
A natural-language question is routed by an LLM (with native tool-calling) to one or more
tools, results feed back to the LLM, and a grounded cited answer is returned.

**DB tool** — imported in-process (no HTTP, no latency). Runs a full NL→SQL pipeline
(Groq LLM + HuggingFace embeddings, dynamic schema discovery, LlamaIndex vector indexes).

**RAG tool** — optional HTTP service on port 8002 for document retrieval.

---

## Project layout

```
db-tool-new/
├── .env.example          ← template (copy to .env and fill in)
├── .env                  ← real secrets (never commit)
├── requirements.txt
├── src/
│   ├── graph/            state.py · nodes.py · builder.py   (LangGraph)
│   ├── adapters/         registry.py · db_tool.py · rag_tool.py · _resilience.py
│   ├── config.py         all settings via pydantic-settings
│   ├── llm.py            LLM factory (groq / gemini / local)
│   ├── prompts.py        system prompt
│   ├── evidence.py       collect + validate tool evidence
│   ├── observability.py  structured logging + LangSmith tracing
│   ├── schemas.py        Pydantic request/response models
│   ├── server.py         FastAPI app
│   └── main.py           entrypoint
├── tools/
│   ├── db_assistant/     NL→SQL pipeline (Groq + HuggingFace + LlamaIndex)
│   │   ├── src/
│   │   │   ├── pipeline.py   full pipeline: intent → table retrieval → SQL → answer
│   │   │   └── models.py     FinalResponse schema
│   │   ├── table_index_storage/   persisted vector index (auto-built first run)
│   │   └── value_index_storage/
│   └── rag_assistant/    document RAG service (port 8002)
└── tests/
    ├── test_graph.py
    ├── test_registry.py
    ├── test_rag_contract.py
    └── eval_routing.py
```

---

## Quick start

```bash
# 1. Create and activate a conda/venv environment
conda activate mycuda        # or: python -m venv .venv && .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — set LLM_PROVIDER, API key, and DB credentials

# 4. Run the orchestrator (port 8001)
set PYTHONPATH=.
python -m src.main

# 5. Optional: run the RAG service (separate terminal, port 8002)
cd tools\rag_assistant && python run.py
```

---

## LLM providers

### Groq (recommended — fast, free tier)

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...           # https://console.groq.com/keys
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```

### Google Gemini

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...         # https://aistudio.google.com/app/apikey
GEMINI_MODEL_NAME=gemini-2.0-flash
```

### Ollama (local, no API key)

```env
LLM_PROVIDER=local
LLM_MODEL=qwen2.5:7b           # must support tool-calling
# Ollama must be running: ollama serve
```

---

## LangSmith tracing

LangSmith gives full trace visibility into every LangGraph node, LLM call, and tool invocation.

**Setup:**

1. Create a free account at <https://smith.langchain.com>
2. Go to **Settings → API Keys → Create API Key**
3. In your `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...   # your key (also shown as LANGCHAIN_API_KEY in LangSmith docs)
LANGSMITH_PROJECT=al-tasnim-orchestrator
```

The orchestrator reads these in `src/observability.py` and sets the standard
`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` env vars that
LangChain/LangGraph pick up automatically.

Once enabled, every `/chat` request appears as a full trace in the LangSmith UI:
agent steps, tool calls, LLM inputs/outputs, latency, token counts.

---

## Architecture

```
POST /chat
  │
  ▼
FastAPI server (src/server.py)
  │  builds HumanMessage, invokes LangGraph
  ▼
LangGraph (src/graph/)
  agent node ──tool_calls──► ToolNode ──results──► agent node ... ──► END
  (LLM decides which tools to call using native tool-calling)
  │
  ├── query_database (src/adapters/db_tool.py)
  │     Imports tools.db_assistant.src.pipeline IN-PROCESS
  │     Pipeline: intent → table vector search → schema context →
  │               query plan → SQL generation → self-critique → execute → synthesize
  │
  └── query_documents (src/adapters/rag_tool.py)
        HTTP call to RAG service on port 8002
```

**First-run init:** pipeline initialization (model load + schema discovery + index build)
runs in a **background thread** at server startup. Queries wait for it to finish
(up to `PIPELINE_INIT_WAIT_SEC`). After first run the table index is cached in
`tools/db_assistant/table_index_storage/` — subsequent restarts take ~30s.

---

## API

### `POST /chat`

```json
{ "question": "list wells in Nimr field", "session_id": "default", "user_id": "optional" }
```

Response:

```json
{
  "success": true,
  "answer": "There are 12 active wells in Nimr field...",
  "tools_used": [
    {
      "tool": "query_database",
      "status": "ok",
      "row_count": 12,
      "source": "operational_database",
      "tables_used": ["2026_Well_Delivery_Scope_Well_Type", "Nimr Well Delivery Tracker (5)"]
    }
  ],
  "iterations": 2,
  "session_id": "default",
  "error": null,
  "execution_time_ms": 4230.5
}
```

### `GET /health`

Returns provider, model, reachable tools, RAG status.

### `GET /tools`

Lists all registered tool names.

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `groq` / `gemini` / `local` |
| `GROQ_API_KEY` | — | Required when `LLM_PROVIDER=groq` |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model |
| `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash` | Gemini model |
| `LLM_MODEL` | `qwen3:8b` | Ollama model name |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per LLM response |
| `DB_SERVER` | — | SQL Server host/IP |
| `DB_NAME` | — | Database name |
| `DB_READONLY_USER` | — | Read-only SQL user |
| `DB_READONLY_PASSWORD` | — | Password |
| `DB_TOOL_TIMEOUT` | `120` | Seconds before a DB query times out |
| `PIPELINE_INIT_WAIT_SEC` | `1800` | Max seconds to wait for first-run pipeline init |
| `VALUE_INDEX_ENABLED` | `0` | `1` = build value-grounding index (slow on large DBs) |
| `RAG_TOOL_URL` | `http://localhost:8002` | RAG service URL |
| `MAX_ITERATIONS` | `5` | Max agent tool-calling iterations |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `LOG_FILE` | — | Optional path to write logs to a file |
| `LANGSMITH_TRACING` | `false` | `true` to enable LangSmith |
| `LANGSMITH_API_KEY` | — | Your LangSmith API key |
| `LANGSMITH_PROJECT` | `al-tasnim-orchestrator` | LangSmith project name |

---

## Add a tool

```python
# src/adapters/my_tool.py
from langchain_core.tools import tool
from .registry import register_tool

@register_tool
@tool
async def my_tool(arg: str) -> str:
    """One-line description — the LLM reads this to decide when to call the tool."""
    ...
```

Import it in `src/adapters/__init__.py`. The tool appears in `/tools` and is
automatically available to the agent.

---

## Health check

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "how many active wells are in Nimr field?"}'
```

## Tests

```bash
set PYTHONPATH=.
pytest -q                          # offline unit tests
python tests/eval_routing.py       # routing accuracy (needs a live model)
```
