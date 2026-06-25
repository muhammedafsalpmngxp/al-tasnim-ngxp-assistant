"""
db-assist.py — FastAPI entry point for the AL TASNIM Intelligent DB Assistant v2.

Full request flow:
  1. Context injection   — enrich question with conversation history
  2. Intent detection    — handle greetings/farewells before routing
  3. Clarification check — ask user if question is too vague
  4. Decomposition       — split multi-part questions into sub-questions
  5. Routing (per sub-Q) — SIMPLE → pipeline, COMPLEX → FunctionAgent
  6. Execution           — run pipeline or agent (per-request isolated tracker)
  7. Synthesis           — combine sub-answers if decomposed
  8. Validation          — flag uncertain answers before returning
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from pipeline import NL2SQLPipeline
from prompts import random_greeting, random_farewell, random_thanks, UNRELATED_RESPONSE
from agent import (
    route_question,
    ask_complex_agent,
    check_clarification,
    inject_context,
    build_context_string,
    decompose_question,
    validate_answer,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_pipeline: NL2SQLPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Starting AL TASNIM DB Assistant v2 ...")
    _pipeline = NL2SQLPipeline(settings)
    logger.info("Pipeline ready. Serving on %s:%d", settings.APP_HOST, settings.APP_PORT)
    yield
    logger.info("Shutting down AL TASNIM DB Assistant.")
    _pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AL TASNIM DB Assistant",
    description=(
        "Intelligent natural-language interface for AppMasterDB_Local. "
        "Features: query clarification, conversation memory, multi-part decomposition, "
        "LLM-based routing, FunctionAgent for complex analysis."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    """One turn in the conversation history (user question or assistant answer)."""
    role: str = Field(default="user", description="'user' or 'assistant'")
    question: Optional[str] = Field(default=None, description="User question text")
    answer: Optional[str] = Field(default=None, description="Assistant answer text")
    content: Optional[str] = Field(default=None, description="Generic content field")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list,
        description=(
            "Previous conversation turns for context. Pass the last few Q&A pairs "
            "so the assistant can resolve references like 'that well' or 'those tasks'."
        ),
    )


class AskResponse(BaseModel):
    answer: str
    needs_clarification: bool = Field(
        default=False,
        description="True when the assistant is asking the user a clarifying question.",
    )
    sql: Optional[str] = None
    data: Optional[list[dict[str, Any]]] = None
    tables_used: Optional[list[str]] = None
    error: Optional[str] = None
    agent_type: Optional[str] = Field(
        default=None,
        description="Which path handled this request: conversational | clarification | simple | complex",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health() -> dict[str, Any]:
    """Liveness/readiness probe."""
    if _pipeline is None or not _pipeline.ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    try:
        tables_loaded = len(_pipeline._schema_loader.get_table_names())
    except Exception:
        tables_loaded = -1
    return {
        "status": "ok",
        "version": "2.0.0",
        "llm_provider": settings.LLM_PROVIDER,
        "db_name": settings.DB_NAME,
        "tables_loaded": tables_loaded,
    }


@app.post("/ask", response_model=AskResponse, tags=["query"])
async def ask(request: AskRequest) -> AskResponse:
    """
    Submit a natural-language question.

    Supports optional `conversation_history` to provide context from prior turns.
    Returns `needs_clarification: true` when the assistant asks the user a follow-up
    question — in that case, submit the user's reply as a new request with the same
    conversation history extended with this clarifying exchange.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    history = [t.model_dump() for t in request.conversation_history]

    logger.info("[REQ:%s] Question: %s", request_id, request.question[:120])

    try:
        # ── Step 1: Context injection ─────────────────────────────────────────
        # Enrich question with context from conversation history so references
        # like "that well" or "those tasks" are resolved to actual entities.
        if history:
            question = inject_context(request.question, history, _pipeline._llm)
        else:
            question = request.question

        # ── Step 2: Intent detection ──────────────────────────────────────────
        # Handle conversational intents without touching the router or agent —
        # saves 1–2 LLM calls on every greeting/farewell/thanks.
        intent = _pipeline._detect_intent(question)
        logger.info("[REQ:%s] Intent: %s", request_id, intent)

        if intent == "GREETING":
            return AskResponse(answer=random_greeting(), agent_type="conversational")
        if intent == "FAREWELL":
            return AskResponse(answer=random_farewell(), agent_type="conversational")
        if intent == "THANKS":
            return AskResponse(answer=random_thanks(), agent_type="conversational")
        if intent == "UNRELATED":
            return AskResponse(answer=UNRELATED_RESPONSE, agent_type="conversational")

        # ── Step 3: Clarification check ───────────────────────────────────────
        # If the question is too vague, return a clarifying question to the user
        # instead of attempting a likely-wrong database query.
        context_str = build_context_string(history)
        clarification = check_clarification(question, _pipeline._llm, context_str)
        if clarification["needs_clarification"]:
            logger.info("[REQ:%s] Returning clarification request", request_id)
            return AskResponse(
                answer=clarification["clarifying_question"],
                needs_clarification=True,
                agent_type="clarification",
            )

        # ── Step 4: Question decomposition ────────────────────────────────────
        # Split multi-part questions into focused sub-questions so each generates
        # accurate SQL instead of one confused mega-query.
        decomposed = decompose_question(question, _pipeline._llm)
        sub_questions = decomposed["sub_questions"]

        # ── Step 5 & 6: Route + Execute each sub-question ────────────────────
        sub_results: list[dict] = []
        last_agent_type = "simple"

        for sub_q in sub_questions:
            routing = route_question(sub_q, _pipeline._llm)
            agent_type = routing["agent"]
            logger.info(
                "[REQ:%s] Sub-Q [%s]: %s",
                request_id,
                agent_type.upper(),
                sub_q[:80],
            )

            if agent_type == "simple":
                result = await _pipeline.ask(sub_q)
                # Escalation signal: pipeline couldn't generate valid SQL — fallback to agent
                if result.get("answer") == "__ESCALATE_TO_AGENT__":
                    logger.info(
                        "[REQ:%s] Pipeline escalated to agent for: %s", request_id, sub_q[:60]
                    )
                    agent_type = "complex"
                    last_agent_type = "complex"
                else:
                    result["_agent_type"] = "simple"
            if agent_type == "complex":
                agent_result = await ask_complex_agent(
                    llm=_pipeline._llm,
                    schema_loader=_pipeline._schema_loader,
                    engine=_pipeline._engine,
                    question=sub_q,
                    request_id=request_id,
                    row_cap=200,
                )
                result = {
                    "answer": agent_result["response"],
                    "sql": agent_result.get("sql"),
                    "data": agent_result.get("data", []),
                    "tables_used": agent_result.get("tables_used", []),
                    "_agent_type": "complex",
                }

            sub_results.append({"question": sub_q, "result": result})

        # ── Step 7: Synthesis ─────────────────────────────────────────────────
        if len(sub_results) == 1:
            final_result = sub_results[0]["result"]
            used_agent = final_result.get("_agent_type", "simple")
        else:
            final_result = await _synthesize_answers(question, sub_results, _pipeline._llm)
            used_agent = last_agent_type

        # ── Step 8: Answer validation ─────────────────────────────────────────
        # Check if answer properly addresses the question. Flag uncertainty
        # rather than silently returning a hallucinated or off-topic answer.
        row_count = len(final_result.get("data") or [])
        validated = validate_answer(
            question, final_result["answer"], row_count, _pipeline._llm
        )

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            "[REQ:%s] Completed in %.0fms | agent=%s | rows=%d | valid=%s",
            request_id,
            elapsed,
            used_agent,
            row_count,
            validated["is_valid"],
        )

        return AskResponse(
            answer=validated["final_answer"],
            needs_clarification=False,
            sql=final_result.get("sql"),
            data=final_result.get("data") or [],
            tables_used=final_result.get("tables_used") or [],
            error=None,
            agent_type=used_agent,
        )

    except Exception as exc:
        elapsed = (time.time() - start_time) * 1000
        logger.exception("[REQ:%s] Unhandled error after %.0fms: %s", request_id, elapsed, exc)
        return AskResponse(
            answer="An unexpected error occurred while processing your request. Please try again.",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Synthesis helper — combines multiple sub-answers into one coherent response
# ---------------------------------------------------------------------------


async def _synthesize_answers(
    original_question: str, sub_results: list[dict], llm
) -> dict:
    """Combine multiple sub-question answers into one final response."""
    parts = []
    all_sql: list[str] = []
    all_data: list[dict] = []
    all_tables: list[str] = []

    for item in sub_results:
        q = item["question"]
        r = item["result"]
        parts.append(f"Sub-question: {q}\nAnswer: {r.get('answer', '(no answer)')}")
        if r.get("sql"):
            all_sql.append(r["sql"])
        if r.get("data"):
            all_data.extend(r["data"])
        if r.get("tables_used"):
            for t in r["tables_used"]:
                if t not in all_tables:
                    all_tables.append(t)

    combined = "\n\n---\n".join(parts)
    synthesis_prompt = (
        f"You are a database assistant. Several queries were run to answer the user's question.\n"
        f"Combine the following individual answers into ONE clear, well-structured response.\n\n"
        f"Original question: {original_question}\n\n"
        f"Individual answers:\n{combined}\n\n"
        f"Instructions:\n"
        f"- Write a single coherent answer covering all aspects of the original question\n"
        f"- Use numbered points or sections if it improves clarity\n"
        f"- Be concise and professional\n"
        f"- Do not repeat the individual sub-questions or say 'sub-question 1 says...'\n"
        f"- If any sub-answer said 'no data found', include that fact in the synthesis"
    )

    try:
        response = llm.complete(synthesis_prompt)
        synthesized = response.text.strip()
    except Exception as e:
        logger.warning("[SYNTHESIS] LLM call failed (%s); concatenating answers", e)
        synthesized = "\n\n".join(
            f"**{item['question']}**\n{item['result'].get('answer', '')}"
            for item in sub_results
        )

    return {
        "answer": synthesized,
        "sql": " ;\n".join(all_sql) if all_sql else None,
        "data": all_data,
        "tables_used": all_tables,
    }


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "db-assist:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )
