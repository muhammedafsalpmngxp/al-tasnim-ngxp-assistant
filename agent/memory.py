"""
agent/memory.py — Conversation context injection.
Rewrites the user's question to be self-contained using prior conversation history.
No server-side storage — client passes history, server enriches the question.
"""
import logging

logger = logging.getLogger(__name__)

_CONTEXT_INJECTION_PROMPT = """\
You are a conversation context processor for a database assistant.

Given a conversation history and a new user question, rewrite the question to be
completely self-contained by incorporating any implied context from prior turns.

Rules:
- If the question already contains all needed context, return it UNCHANGED
- Replace pronouns and references ("it", "that well", "those tasks", "them") with the actual entities from history
- Add well names, dates, or filters that are clearly implied from prior context
- Keep the rewritten question natural and concise — do NOT add padding or explanation
- Do NOT add information that was not in either the question or history
- If history is empty or irrelevant to the question, return the original question unchanged
- Return ONLY the rewritten question — no explanation, no prefix, nothing else

─── Conversation history ──────────────────────────────────────────────────────
{history}

─── New question ──────────────────────────────────────────────────────────────
{question}

─── Rewritten question (only the question, nothing else) ──────────────────────
\
"""


def inject_context(question: str, conversation_history: list, llm) -> str:
    """
    Enrich the question with context from conversation history.
    Returns the self-contained rewritten question.
    """
    if not conversation_history:
        return question

    try:
        # Build history text from last 3 exchanges (6 turns max)
        recent = conversation_history[-6:]
        history_lines = []
        for turn in recent:
            role = turn.get("role", "user")
            content = (
                turn.get("question")
                or turn.get("answer")
                or turn.get("content", "")
            )
            if content:
                history_lines.append(f"{role.capitalize()}: {content}")

        if not history_lines:
            return question

        history_text = "\n".join(history_lines)
        prompt = _CONTEXT_INJECTION_PROMPT.format(
            history=history_text,
            question=question,
        )

        response = llm.complete(prompt)
        enriched = response.text.strip()

        # Sanity check — if LLM returned empty or garbage, use original
        if enriched and len(enriched) >= len(question) * 0.5:
            if enriched != question:
                logger.info(
                    "[MEMORY] Context injected: '%s' → '%s'",
                    question[:60],
                    enriched[:60],
                )
            return enriched

        return question

    except Exception as e:
        logger.warning("[MEMORY] Context injection failed (%s); using original question", e)
        return question


def build_context_string(conversation_history: list) -> str:
    """
    Build a plain text summary of recent conversation turns.
    Used by the clarifier to understand prior context.
    """
    if not conversation_history:
        return ""

    parts = []
    for turn in conversation_history[-4:]:
        q = turn.get("question") or turn.get("content", "")
        a = turn.get("answer", "")
        if q:
            parts.append(f"User asked: {q}")
        if a:
            # Truncate long answers
            short_a = a[:300] + "..." if len(a) > 300 else a
            parts.append(f"Assistant answered: {short_a}")

    return "\n".join(parts)
