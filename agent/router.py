"""
agent/router.py — LLM-based query complexity classifier.
Routes questions to SIMPLE pipeline or COMPLEX FunctionAgent.
"""
import logging
import time

logger = logging.getLogger(__name__)

_ROUTER_PROMPT = """\
You are a query complexity classifier for a database assistant.

Classify the user's question as SIMPLE or COMPLEX.

SIMPLE = One SELECT query can fully answer it:
- Filtering rows (WHERE conditions)
- Counting, summing, averaging a column
- Sorting and listing records
- Joining two or more tables to get related data
- Finding min/max/average values
- Status checks, flag lookups, NULL checks

Examples of SIMPLE:
- "Which wells are below 50% progress?"
- "How many active wells are there?"
- "Show tasks for well 33151"
- "Which wells have rig-off but no commissioning finish date?"
- "Show top 10 wells by productivity"
- "Which tasks are suspended?"
- "List wells missing MOC approval"
- "Show drilling sequence for well 33151"
- "What was done on task X on date D?"
- "Which wells are waiting on PO?"
- "Show employee list with crew assignments"
- "What is the progress for well 33151?"
- "Show wells with rig-off after January 2026"

COMPLEX = Requires multiple SEPARATE queries combined with reasoning, OR true predictions/forecasts:
- "Which tasks are likely to finish late?" (needs all tasks + multi-factor delay analysis)
- "Predict when well 33151 will complete" (needs velocity calculation + projection)
- "Which wells are at risk of missing the engineering KPI?" (needs multi-factor analysis)
- "Break down progress by discipline and identify what is lagging" (needs aggregation + comparison)
- "Total planned vs actual manhours with overrun analysis" (needs CAST + aggregation + variance)
- "Compare progress velocity across clusters" (needs multi-table aggregation + comparison)
- "Which wells need procurement action urgently?" (needs PO status + date reasoning across multiple wells)

RULE: When in doubt → SIMPLE. Only choose COMPLEX when you are certain multiple separate
queries are needed OR the question explicitly requires predictions, forecasts, or
multi-factor analysis across many records simultaneously.

Question: {question}

Reply with exactly one word: SIMPLE or COMPLEX\
"""


def route_question(question: str, llm) -> dict:
    """
    Classify question complexity.
    Returns: {"agent": "simple"|"complex", "reasoning": str, "latency_ms": float}
    """
    start_time = time.time()
    try:
        response = llm.complete(_ROUTER_PROMPT.format(question=question))
        text = response.text.strip().upper()

        if "SIMPLE" in text:
            result = {"agent": "simple", "reasoning": "LLM classified as SIMPLE"}
        else:
            result = {"agent": "complex", "reasoning": "LLM classified as COMPLEX"}

        result["latency_ms"] = (time.time() - start_time) * 1000
        logger.info(
            "[ROUTER] '%s...' → %s (%.0fms)",
            question[:60],
            result["agent"].upper(),
            result["latency_ms"],
        )
        return result

    except Exception as e:
        logger.warning("[ROUTER] Failed (%s); defaulting to SIMPLE", e)
        return {
            "agent": "simple",
            "reasoning": f"Fallback to simple after error: {e}",
            "latency_ms": (time.time() - start_time) * 1000,
        }
