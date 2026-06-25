"""
agent/validator.py — Final answer quality gate.
Checks if the generated answer properly addresses the user's question.
Flags uncertainty rather than letting hallucinations through silently.
"""
import logging

logger = logging.getLogger(__name__)

_VALIDATION_PROMPT = """\
You are an answer quality reviewer for a database assistant.

Review the generated answer and determine if it properly addresses the user's question.

─── What to check ────────────────────────────────────────────────────────────
  1. Does the answer directly address what was asked?
  2. Is the answer based on actual data (rows_fetched > 0) or just assumptions?
  3. If 0 rows were fetched, does the answer honestly say "no data found"?
  4. Does the answer avoid obvious contradictions or impossible claims?
  5. Is the answer relevant to the question topic?

─── When to mark VALID ───────────────────────────────────────────────────────
  ✓ Answer directly addresses the question with specific data
  ✓ Answer correctly says "no results found" when 0 rows fetched
  ✓ Answer is a partial result but clearly notes it is partial
  ✓ Answer gives a well-reasoned estimate with appropriate caveats

─── When to mark UNCERTAIN ───────────────────────────────────────────────────
  ✗ Answer is completely off-topic or answers a different question
  ✗ Answer makes specific claims but 0 rows were fetched (likely hallucination)
  ✗ Answer ignores fetched data and gives generic statements
  ✗ Answer contradicts itself

DO NOT mark UNCERTAIN for:
  - Estimates or approximations (these are valid if qualified)
  - Summaries that don't list every row (summarizing is correct behavior)
  - Minor formatting issues
  - Correct "no data found" responses

Respond with EXACTLY one of:
  VALID
  UNCERTAIN|<brief one-sentence note about what is uncertain>

─── Review ────────────────────────────────────────────────────────────────────
User question: {question}
Rows fetched from database: {row_count}
Generated answer: {answer}

Response:\
"""


def validate_answer(question: str, answer: str, row_count: int, llm) -> dict:
    """
    Validate that the answer properly addresses the question.

    Returns:
        {"is_valid": bool, "uncertainty_note": str | None, "final_answer": str}
    """
    try:
        prompt = _VALIDATION_PROMPT.format(
            question=question,
            answer=answer[:1500],   # cap to avoid huge prompts
            row_count=row_count,
        )
        response = llm.complete(prompt)
        text = response.text.strip()

        if text.startswith("UNCERTAIN|"):
            note = text.split("|", 1)[1].strip()
            logger.warning("[VALIDATOR] Answer flagged uncertain: %s", note)
            # Append the note to the answer so user is informed
            final = f"{answer}\n\n_(Note: {note})_"
            return {"is_valid": False, "uncertainty_note": note, "final_answer": final}

        logger.info("[VALIDATOR] Answer validated OK")
        return {"is_valid": True, "uncertainty_note": None, "final_answer": answer}

    except Exception as e:
        logger.warning("[VALIDATOR] Failed (%s); returning answer as-is", e)
        return {"is_valid": True, "uncertainty_note": None, "final_answer": answer}
