"""
agent/decomposer.py — Splits complex multi-part questions into focused sub-questions.
Each sub-question is self-contained and answerable independently.
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

_DECOMPOSITION_PROMPT = """\
You are a question analyzer for a database assistant.

Determine if the user's question is asking multiple DISTINCT things that would each need
a separate database query, or just one focused thing (even if complex).

─── Needs decomposition ────────────────────────────────────────────────────────
Split ONLY when the question explicitly asks for multiple separate pieces of information:
  ✓ "Show progress for well 33151 AND which tasks are behind" → 2 separate queries
  ✓ "Break down progress by discipline, identify what is lagging, AND estimate completion" → 3 queries
  ✓ "Show total manhours AND compare with planned AND flag overruns" → 3 queries

─── Does NOT need decomposition ────────────────────────────────────────────────
Keep as single question when it is one coherent analytical task:
  ✗ "Which tasks are likely to finish late?" → single analysis
  ✗ "Show progress for well 33151" → single lookup
  ✗ "Which wells are at risk of missing KPI?" → single analysis
  ✗ "Show top 10 wells by productivity" → single query
  ✗ "Which tasks are behind their target?" → single query with filter

─── Rules for sub-questions ────────────────────────────────────────────────────
  - Maximum 4 sub-questions
  - Each sub-question must be fully self-contained (include well names, dates, filters)
  - Sub-questions must be independently answerable
  - Do not over-split — prefer fewer, focused sub-questions

Respond with ONLY valid JSON (no markdown, no explanation):

If no decomposition needed:
{{"needs_decomposition": false, "sub_questions": ["{original}"]}}

If decomposition needed:
{{"needs_decomposition": true, "sub_questions": ["first focused question", "second focused question"]}}

─── Examples ───────────────────────────────────────────────────────────────────
Q: "Show progress for well 33151 and which tasks are behind"
→ {{"needs_decomposition": true, "sub_questions": ["What is the current overall progress for well 33151?", "Which tasks for well 33151 are behind their target end dates?"]}}

Q: "Which wells are below 50% progress?"
→ {{"needs_decomposition": false, "sub_questions": ["Which wells are below 50% progress?"]}}

Q: "Break down well 33151 progress by discipline, identify what is lagging, and estimate when it will complete"
→ {{"needs_decomposition": true, "sub_questions": ["What is the current progress percentage by discipline for well 33151?", "Which discipline for well 33151 is most behind its planned progress?", "Based on current progress rate for well 33151, what is the estimated completion date?"]}}

Q: "Which tasks are likely to finish late?"
→ {{"needs_decomposition": false, "sub_questions": ["Which tasks are likely to finish late?"]}}

Q: "Show total planned vs actual manhours for well 33151 and flag overruns"
→ {{"needs_decomposition": true, "sub_questions": ["What are the total planned manhours for well 33151?", "What are the total actual manhours spent on well 33151, and which tasks have manhour overruns?"]}}

─── User question ───────────────────────────────────────────────────────────────
{question}

JSON response:\
"""


def decompose_question(question: str, llm) -> dict:
    """
    Split multi-part question into focused sub-questions.

    Returns:
        {"needs_decomposition": bool, "sub_questions": list[str]}
    """
    try:
        prompt = _DECOMPOSITION_PROMPT.format(
            question=question,
            original=question.replace('"', '\\"'),
        )
        response = llm.complete(prompt)
        text = response.text.strip()

        # Strip markdown code fences if LLM wrapped in ```json ... ```
        if "```" in text:
            text = re.sub(r"```(?:json)?\s*", "", text)
            text = re.sub(r"```\s*$", "", text).strip()

        result = json.loads(text)
        sub_questions = result.get("sub_questions", [question])
        needs_decomp = result.get("needs_decomposition", False)

        # Validate: sub_questions must be a non-empty list of strings
        if not isinstance(sub_questions, list) or not sub_questions:
            sub_questions = [question]
            needs_decomp = False

        # Cap at 4 sub-questions
        sub_questions = sub_questions[:4]

        logger.info(
            "[DECOMPOSER] %s → %d sub-question(s)%s",
            "Decomposed" if needs_decomp else "Single question",
            len(sub_questions),
            f": {sub_questions}" if needs_decomp else "",
        )
        return {"needs_decomposition": needs_decomp, "sub_questions": sub_questions}

    except Exception as e:
        logger.warning("[DECOMPOSER] Failed (%s); treating as single question", e)
        return {"needs_decomposition": False, "sub_questions": [question]}
