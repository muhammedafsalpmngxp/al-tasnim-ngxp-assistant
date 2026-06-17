# rag_assistant (Document RAG tool) — skeleton

Runs on **port 8002**. Implement `search()` in `src/server.py` with your hybrid
retrieval (vector + BM25 + optional rerank). The orchestrator calls this over
HTTP; when ready it needs **no orchestrator changes** — just point `RAG_TOOL_URL`
at this service.

## Contract
```
POST /search  { "query": "casing procedure", "top_k": 5 }
  -> { "success": true,
       "passages": [ {"text": "...", "source": "SOP-12 p4", "score": 0.83} ],
       "sources": ["SOP-12"] }
GET  /health -> { "status": "healthy" }
```

## Run
```
pip install -r requirements.txt
python run.py
```
