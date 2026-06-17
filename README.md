# AL TASNIM — LangGraph Orchestrator

Top-level orchestration layer for the Operational Intelligence Assistant. A user
question goes to an LLM that decides which tool(s) to call (native tool-calling),
the tools run, their results feed back to the LLM, and a final cited answer is
returned. Tools live under `tools/`. The **DB tool is imported in-process** (same
code, no HTTP); the **RAG tool is a separate HTTP service**.

## Layout
```
al-tasnim-ngxp-assistant/                (repo root = orchestrator)
├── .env.example   .gitignore   .dockerignore   requirements.txt   pytest.ini
├── Dockerfile.orchestrator   Dockerfile.rag   docker-compose.yml   README.md
├── src/
│   ├── graph/      state.py · nodes.py · builder.py         (the LangGraph)
│   ├── adapters/   registry.py · db_tool.py · rag_tool.py · _resilience.py
│   ├── config.py · llm.py · prompts.py · evidence.py · observability.py
│   ├── schemas.py · server.py · main.py
├── tools/
│   ├── db_assistant/    your DB tool (unchanged, imported in-process)
│   └── rag_assistant/   RAG skeleton (port 8002)
└── tests/   test_graph.py · test_registry.py · test_rag_contract.py · eval_routing.py
```

## Configuration — ONE root `.env`
There is a **single `.env` at the repo root** (next to `requirements.txt`). It
configures the orchestrator AND supplies the DB tool's settings. The DB tool's
credentials live here too — you no longer keep a separate `tools/db_assistant/.env`.

```bash
cp .env.example .env      # then fill DB_SERVER / DB_* and pick the model
```
Use `LLM_PROVIDER=local` (Ollama, works for both the orchestrator and the DB tool)
or `LLM_PROVIDER=gemini` (+ `GOOGLE_API_KEY`).

## Flow
```
START -> agent (LLM + tools) --tool_calls--> tools (run) --results--> agent ... -> END
                              \--no tool_calls-------------------------------------> END
```
`ToolNode` appends each tool result as a `ToolMessage` (the feedback). `tools_condition`
routes to `tools` while tool calls exist, else `END`. `MAX_ITERATIONS` + `RECURSION_LIMIT`
guarantee termination. Routing rules (CoT + few-shot, incl. the both-tools case) are in
`src/prompts.py`. DB queries are SELECT-only (deterministic builder + read-only DB user).

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate     # mac/linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                # edit it
```

## Run
```bash
# optional RAG service (port 8002), separate terminal
cd tools\rag_assistant && python run.py

# orchestrator (port 8001), from repo root
set PYTHONPATH=. && python -m src.main
```
Docker: `docker compose up --build` (builds rag + orchestrator).

## Check what's working
```bash
pytest -q                                           # offline: graph feedback loop, tools, RAG contract
curl http://localhost:8001/health                   # provider, db_env_present, rag_reachable, tools
curl -X POST http://localhost:8001/ask -H "Content-Type: application/json" \
     -d "{\"question\": \"status of rig 104\"}"
set PYTHONPATH=. && python tests\eval_routing.py     # routing accuracy (needs a live model)
```

## Logging
Every step logs with a correlation id (set `LOG_FILE=...` in `.env` to also write a file):
```
orchestrator.api    ASK question=...
orchestrator.agent  ROUTING -> ['query_database']
orchestrator.tool.db CALL / RESULT ok rows=...
orchestrator.api    DONE iterations=2 evidence=1 grounded=True took=...ms
```

## Add a tool
```python
# src/adapters/my_tool.py
from langchain_core.tools import tool
from .registry import register_tool

@register_tool
@tool
async def my_tool(arg: str) -> str:
    """Clear description — the LLM uses this to decide when to call it."""
    ...
```
Import it in `src/adapters/__init__.py`. Done.

## Roadmap
Clarification node → retrieval grader / self-correction → risk + human approval
(`interrupt()`) → caching (doc-only) → output-validation node.
