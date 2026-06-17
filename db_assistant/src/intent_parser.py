import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple

from pydantic import ValidationError

from .config import settings
from .models import Intent, IntentResponse
from .value_resolver import ValueResolver
from .llm_client import get_llm_client

logger = logging.getLogger(__name__)

# Maximum LLM retries when response is empty or unparseable
_MAX_LLM_RETRIES = 2


class IntentParser:
    def __init__(self, value_resolver: ValueResolver, schema_introspector=None):
        self.resolver = value_resolver
        self.schema = schema_introspector   # injected so prompt shows real columns
        self.llm = get_llm_client()
        self.provider = settings.LLM_PROVIDER

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self, user_query: str, clarification_context: str = None) -> IntentResponse:
        logger.debug("[STEP 1] Received query: %r | clarification_context: %r",
                     user_query, clarification_context)

        if not self.llm.check_available():
            logger.debug("[STEP 1] LLM not available, aborting")
            return IntentResponse(
                success=False,
                error=f"LLM provider '{self.provider}' is not available. "
                      "Please check your configuration.",
                clarification_needed=False,
            )

        full_query = (
            f"{clarification_context}\n\nUser follow-up: {user_query}"
            if clarification_context
            else user_query
        )

        schema_context = self._build_schema_context()
        prompt = self._build_prompt(full_query, schema_context)
        logger.debug("[STEP 2] Built prompt (%d chars)", len(prompt))

        # Retry loop — handles empty / malformed LLM responses
        for attempt in range(1, _MAX_LLM_RETRIES + 1):
            try:
                logger.debug("[STEP 3] Sending to LLM (%s), attempt %d/%d",
                             self.provider, attempt, _MAX_LLM_RETRIES)

                response = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": settings.TEMPERATURE,
                        "num_predict": settings.MAX_TOKENS,
                    },
                )

                raw = response["message"]["content"]
                logger.debug("[STEP 4] Raw LLM response (attempt %d):\n%s", attempt, raw)

                if not raw or not raw.strip():
                    logger.warning("[STEP 4] Empty LLM response on attempt %d", attempt)
                    if attempt < _MAX_LLM_RETRIES:
                        continue
                    return IntentResponse(
                        success=False,
                        clarification_needed=True,
                        clarification_question="I couldn't process your question. Could you rephrase it?",
                    )

                intent_json = self._extract_json(raw)
                logger.debug("[STEP 5] Extracted JSON: %s", intent_json)

                if not intent_json:
                    logger.warning("[STEP 5] Could not extract JSON on attempt %d", attempt)
                    if attempt < _MAX_LLM_RETRIES:
                        continue
                    return IntentResponse(
                        success=False,
                        clarification_needed=True,
                        clarification_question="I didn't understand that. Could you rephrase?",
                    )

                if intent_json.get("clarification_needed"):
                    logger.debug("[STEP 5] LLM flagged clarification_needed=true")
                    return IntentResponse(
                        success=False,
                        clarification_needed=True,
                        clarification_question=intent_json.get(
                            "clarification_question",
                            "Could you provide more details?",
                        ),
                    )

                table = intent_json.get("table", "")
                logger.debug("[STEP 6] Validating Intent — table=%r", table)

                if not table or table not in settings.ALLOWED_TABLES:
                    logger.warning("[STEP 6] Invalid table %r on attempt %d", table, attempt)
                    if attempt < _MAX_LLM_RETRIES:
                        continue
                    return IntentResponse(
                        success=False,
                        clarification_needed=True,
                        clarification_question=(
                            "I couldn't identify which data you are asking about. "
                            "Could you be more specific? For example: wells, rigs, tasks, employees, equipment."
                        ),
                    )

                intent = Intent(**intent_json)

                logger.debug("[STEP 7] Resolving filter values...")
                resolved_intent, suggestions = self._resolve_filters(intent)

                if suggestions:
                    logger.debug("[STEP 7] Ambiguous filter values: %s", suggestions[:5])
                    return IntentResponse(
                        success=False,
                        clarification_needed=True,
                        clarification_question=(
                            "Did you mean one of these? "
                            + ", ".join(str(s) for s in suggestions[:5])
                        ),
                        suggestions=suggestions,
                    )

                logger.debug(
                    "[STEP 8] Intent OK — table=%r columns=%s filters=%d",
                    resolved_intent.table,
                    resolved_intent.columns,
                    len(resolved_intent.filters),
                )
                return IntentResponse(success=True, intent=resolved_intent)

            except ValidationError as e:
                logger.warning("[STEP 6] Pydantic validation error (attempt %d): %s", attempt, e)
                if attempt < _MAX_LLM_RETRIES:
                    continue
                return IntentResponse(
                    success=False,
                    clarification_needed=True,
                    clarification_question="I didn't fully understand. Could you rephrase?",
                )
            except Exception as e:
                logger.error("[STEP 3-8] Intent parsing failed (attempt %d): %s", attempt, e)
                err = str(e).lower()
                if any(k in err for k in ("not available", "connection", "timed out", "refused")):
                    return IntentResponse(
                        success=False,
                        error=(
                            f"LLM provider '{self.provider}' is not available or timed out. "
                            "Check configuration and try again."
                        ),
                        clarification_needed=False,
                    )
                if attempt < _MAX_LLM_RETRIES:
                    continue
                return IntentResponse(
                    success=False,
                    clarification_needed=True,
                    clarification_question="I didn't understand that. Could you rephrase?",
                )

        # Should not reach here
        return IntentResponse(
            success=False,
            clarification_needed=True,
            clarification_question="I couldn't process your question. Please try again.",
        )

    # ------------------------------------------------------------------
    # Schema context — dynamically built from real DB columns
    # ------------------------------------------------------------------

    def _build_schema_context(self) -> str:
        """Return a TABLES section using real column names from the DB where available."""
        lines = ["AVAILABLE TABLES AND COLUMNS"]
        lines.append("=" * 60)

        for table in settings.ALLOWED_TABLES:
            if self.schema:
                try:
                    schema = self.schema.get_table_schema(table)
                    cols = [
                        f"{c} ({schema['columns'][c]['type']})"
                        for c in schema["column_list"]
                    ]
                    lines.append(f"\nTable: {table}")
                    lines.append("Columns: " + ", ".join(cols))
                    continue
                except Exception as e:
                    logger.debug("Could not load schema for %s: %s", table, e)

            # Fallback — table listed without column detail
            lines.append(f"\nTable: {table}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt — few-shot + chain-of-thought, zero hardcoded answers
    # ------------------------------------------------------------------

    def _build_prompt(self, user_query: str, schema_context: str) -> str:
        allowed_tables_str = "\n".join(f"  - {t}" for t in settings.ALLOWED_TABLES)
        operators_str = "=, !=, >, <, >=, <=, in, like, between, is_null, is_not_null"
        agg_funcs_str = "count, sum, avg, max, min"

        return f"""You are a database intent parser. Your ONLY job is to read a user question and output a JSON intent object that describes what data to fetch. You do NOT write SQL. You do NOT execute queries.

================================================================================
THINKING PROCESS (chain of thought — work through this silently before outputting)
================================================================================

Step 1 — Identify topic: What is the user asking about? (wells, rigs, tasks, employees, equipment, progress, revenue, drilling…)
Step 2 — Map to table: Which table in the schema best covers that topic?
Step 3 — Identify filter: Is there a specific value being filtered on? (a number, a name, a status…)
Step 4 — Map filter to column: Which column in that table holds that value?
Step 5 — Select columns: What columns would answer the question? Default to the most relevant ones, not all columns.
Step 6 — Output JSON: Produce the JSON intent object.

================================================================================
RULES
================================================================================

1. "table" MUST be one of the allowed table names listed below — exact spelling.
2. "columns" lists only columns that exist in the chosen table (see schema).
3. "filters" values are raw user values — do NOT look up or guess database IDs.
4. Operators allowed: {operators_str}
5. Aggregate functions allowed: {agg_funcs_str}
6. If the question is a greeting or small talk (Hi, Hello, Thanks…) set clarification_needed=true.
7. If the question is ambiguous and cannot be mapped to any table, set clarification_needed=true with a helpful question — do NOT guess.
8. Never fabricate column names. Use only columns listed in the schema below.

================================================================================
ALLOWED TABLES
================================================================================

{allowed_tables_str}

================================================================================
{schema_context}
================================================================================

================================================================================
BUSINESS TERM → COLUMN MAPPINGS (use these when users speak in business language)
================================================================================

"rig", "drilling rig", "rig number"     → column RIG  (table 2026_Well_Delivery_Scope_Well_Type)
                                          OR Work_Center (table SAP_DRILLING_SEQUENCE)
"well", "well id", "well number"        → column Well_ID
"field", "area", "location"             → column Field
"status", "current status", "rig off"  → columns Latest_ROL, Latest_Rif_Off, Well_Category
"lift type", "artificial lift"          → column Lift_type
"progress", "overall progress", "%"    → column over_all_progress_percentages (table WMR)
"week", "weekly"                        → column Week_Number (table WMR)
"task", "activity", "work item"        → columns code, text, progress (table ActivityTaskPlan)
"employee", "person", "worker", "staff"→ columns Name, Status, Location (table Employee)
"crew", "team"                          → columns Code, Supervisor (table crews)
"equipment", "vehicle", "truck"        → columns LicensePlate, Description, Status (table Equipment)
"daily task", "daily work"             → columns task_code, progress, ActionOn (table task_daily)
"revenue", "value", "cost"             → columns total_purpose_value, Title (table Revenue)
"drilling sequence", "SAP"             → columns Well_ID, Work_Center, Earl_start_date (table SAP_DRILLING_SEQUENCE)

================================================================================
OUTPUT FORMAT — return ONLY this JSON, no explanation, no markdown fences
================================================================================

{{
  "table": "<exact table name from allowed list>",
  "columns": ["<col1>", "<col2>"],
  "filters": [{{"col": "<column>", "op": "<operator>", "value": <value>}}],
  "aggregate": null,
  "order_by": [{{"col": "<column>", "direction": "ASC"}}],
  "limit": 100,
  "natural_language": "<original question>",
  "clarification_needed": false,
  "clarification_question": null
}}

For aggregation:
  "aggregate": {{"func": "count", "group_by": ["<col>"]}}

================================================================================
FEW-SHOT EXAMPLES
================================================================================

User: "What is the status of rig 104?"
Thinking: topic=rig status → table=2026_Well_Delivery_Scope_Well_Type, filter RIG=104, columns about status.
Output:
{{
  "table": "2026_Well_Delivery_Scope_Well_Type",
  "columns": ["Well_ID", "RIG", "Latest_ROL", "Latest_Rif_Off", "Well_Category", "Field"],
  "filters": [{{"col": "RIG", "op": "=", "value": 104}}],
  "aggregate": null,
  "order_by": [{{"col": "Well_ID", "direction": "ASC"}}],
  "limit": 100,
  "natural_language": "What is the status of rig 104?",
  "clarification_needed": false,
  "clarification_question": null
}}

---

User: "Show me all employees in Muscat"
Thinking: topic=employees, filter Location=Muscat → table=Employee.
Output:
{{
  "table": "Employee",
  "columns": ["Name", "Status", "Location"],
  "filters": [{{"col": "Location", "op": "=", "value": "Muscat"}}],
  "aggregate": null,
  "order_by": [{{"col": "Name", "direction": "ASC"}}],
  "limit": 100,
  "natural_language": "Show me all employees in Muscat",
  "clarification_needed": false,
  "clarification_question": null
}}

---

User: "How many wells are in each field?"
Thinking: topic=wells count grouped by field → table=2026_Well_Delivery_Scope_Well_Type, aggregate count grouped by Field.
Output:
{{
  "table": "2026_Well_Delivery_Scope_Well_Type",
  "columns": ["Field"],
  "filters": [],
  "aggregate": {{"func": "count", "group_by": ["Field"]}},
  "order_by": [{{"col": "Field", "direction": "ASC"}}],
  "limit": 100,
  "natural_language": "How many wells are in each field?",
  "clarification_needed": false,
  "clarification_question": null
}}

---

User: "What equipment is available in Nizwa?"
Thinking: topic=equipment, filter Location=Nizwa → table=Equipment.
Output:
{{
  "table": "Equipment",
  "columns": ["LicensePlate", "Description", "Status", "Location"],
  "filters": [{{"col": "Location", "op": "=", "value": "Nizwa"}}],
  "aggregate": null,
  "order_by": [{{"col": "Description", "direction": "ASC"}}],
  "limit": 100,
  "natural_language": "What equipment is available in Nizwa?",
  "clarification_needed": false,
  "clarification_question": null
}}

---

User: "Hi"
Thinking: greeting, not a data question → clarification_needed.
Output:
{{
  "table": "",
  "columns": [],
  "filters": [],
  "aggregate": null,
  "order_by": [],
  "limit": 100,
  "natural_language": "Hi",
  "clarification_needed": true,
  "clarification_question": "Hello! I can help you query data about wells, rigs, tasks, employees, equipment, and more. What would you like to know?"
}}

================================================================================
NOW PARSE THIS QUERY — follow the thinking steps, then output ONLY the JSON
================================================================================

User: "{user_query}"

JSON:"""

    # ------------------------------------------------------------------
    # JSON extraction — robust brace-matching parser
    # ------------------------------------------------------------------

    def _extract_json(self, raw: str) -> Dict:
        raw = raw.strip()

        # Direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Find outermost {...} block via brace counting
        start = raw.find("{")
        if start == -1:
            logger.debug("No '{' found in LLM output")
            return {}

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start: i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError as e:
                            logger.debug("JSON decode failed on extracted block: %s", e)
                            return {}
        logger.debug("Brace-matching did not find a complete JSON object")
        return {}

    # ------------------------------------------------------------------
    # Filter value resolution
    # ------------------------------------------------------------------

    def _resolve_filters(self, intent: Intent) -> Tuple[Intent, Optional[List]]:
        suggestions = []
        for f in intent.filters:
            if f.op in ["is_null", "is_not_null"]:
                continue
            resolved, status, sugg = self.resolver.resolve_value(
                intent.table, f.col, f.value
            )
            if resolved is None:
                if sugg:
                    suggestions.extend(sugg)
                f.resolution_status = "not_found"
            else:
                f.resolved_value = resolved
                f.resolution_status = status
        if suggestions:
            return intent, suggestions[:5]
        return intent, None
