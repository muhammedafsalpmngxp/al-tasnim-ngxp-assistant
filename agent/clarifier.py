"""
agent/clarifier.py — Pre-processing step that checks if a question has enough
information to query the database accurately. If not, returns a specific
clarifying question for the user instead of attempting a wrong query.
"""
import logging

logger = logging.getLogger(__name__)

_CLARIFICATION_PROMPT = """\
You are a pre-processor for a database assistant.

PRIMARY RULE: When in doubt → respond COMPLETE. Always attempt the query first.

Your ONLY job: catch questions that are a single empty phrase with zero context.
For everything else — respond COMPLETE and let the database assistant handle it.

IMPORTANT: Be VERY lenient. 95% of questions should be COMPLETE.

The database is about oil & gas well delivery for AL TASNIM LLC in Oman. It contains:
wells, tasks, crews, employees, rigs, drilling sequences, approvals, progress, revenue,
productivity, KPIs, commissioning, flowlines, OHL, engineering, procurement.

─── ALWAYS mark COMPLETE (do not ask) ──────────────────────────────────────
  ✓ Any question that names a well, rig, crew, project, person, or ID code
     e.g. "details of SWER149", "rig details of SWER149", "well 33151 progress",
          "tasks for project X", "crew ABC details", "status of well 37318"
  ✓ Any question about a category: "active wells", "suspended tasks", "delayed milestones"
  ✓ Any question with "all", "list", "show", "how many", "total", "which"
  ✓ Any question about a concept: "drilling sequence", "rig details", "progress",
     "KPI", "manhours", "revenue", "productivity", "commissioning", "approval"
  ✓ Questions that mention a known domain term even without a specific ID
  ✓ Follow-up questions where the conversation history gives context
  ✓ Vague questions where the system can make a reasonable attempt
  ✓ Questions with typos, abbreviations, or informal language — attempt anyway

─── ONLY mark INCOMPLETE (ask user) ────────────────────────────────────────
ONLY when the question is a single word or tiny fragment with zero context:
  ✗ "Show me" (alone, nothing else)
  ✗ "Data" (alone, nothing else)
  ✗ "Compare" (alone, nothing else — no subject at all)
  ✗ "Tell me" (alone, nothing else)

Do NOT ask about:
  - What metric they want (attempt with all available columns)
  - Which specific ID (if not given, query all)
  - What time period (if not given, query all/latest)
  - What format they want

─── Examples ────────────────────────────────────────────────────────────────
"i want know about the rig details of SWER149"  → COMPLETE  (has ID + topic)
"rig details of SWER149"                         → COMPLETE  (has ID + topic)
"give the full details of well 37318"            → COMPLETE  (has ID + topic)
"show me the progress"                           → COMPLETE  (query all wells' progress)
"which tasks are behind?"                        → COMPLETE  (query all behind tasks)
"details of SWER149"                             → COMPLETE  (has an ID — attempt it)
"What about the KPI?"                            → COMPLETE  (query KPI data)
"how many active wells"                          → COMPLETE
"drilling sequence"                              → COMPLETE  (query drilling sequence table)
"productivity"                                   → COMPLETE  (query productivity data)
"Compare"                                        → INCOMPLETE|What would you like to compare? (e.g., well progress, crew productivity, planned vs actual)
"Show me"                                        → INCOMPLETE|What would you like to see? (e.g., well progress, task status, crew assignments)

─── Previous conversation context ───────────────────────────────────────────
{context}

─── User question ────────────────────────────────────────────────────────────
{question}

Respond with exactly one of:
  COMPLETE
  INCOMPLETE|<one brief, specific clarifying question>

Response:\
"""


def check_clarification(question: str, llm, context: str = "") -> dict:
    """
    Determine if the question needs clarification before answering.

    Returns:
        {"needs_clarification": bool, "clarifying_question": str | None}
    """
    try:
        prompt = _CLARIFICATION_PROMPT.format(
            question=question,
            context=context.strip() if context else "None",
        )
        response = llm.complete(prompt)
        text = response.text.strip()

        if text.startswith("INCOMPLETE|"):
            clarifying_q = text.split("|", 1)[1].strip()
            logger.info("[CLARIFIER] Needs clarification: %s", clarifying_q)
            return {"needs_clarification": True, "clarifying_question": clarifying_q}

        logger.info("[CLARIFIER] Question is complete")
        return {"needs_clarification": False, "clarifying_question": None}

    except Exception as e:
        logger.warning("[CLARIFIER] Failed (%s); treating question as complete", e)
        return {"needs_clarification": False, "clarifying_question": None}
