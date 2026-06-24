"""
pipeline.py — NL2SQL pipeline orchestrating intent detection, table retrieval,
SQL generation, safety validation, execution, and response formatting.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import Settings
import llm_factory
from schema_loader import SchemaLoader
from retriever import BM25TableRetriever, DenseTableRetriever, HybridTableRetriever
from safety import validate_sql
from prompts import (
    INTENT_PROMPT,
    SQL_GENERATION_PROMPT,
    RESPONSE_FORMATTING_PROMPT,
    SQL_ERROR_RETRY_PROMPT,
    UNRELATED_RESPONSE,
    random_greeting,
    random_farewell,
    random_thanks,
)

logger = logging.getLogger(__name__)

# Maximum rows returned to the caller
_MAX_ROWS = 500


class NL2SQLPipeline:
    """End-to-end natural-language-to-SQL pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        logger.info("Initialising NL2SQL pipeline …")

        # 1. Database engine
        self._engine: Engine = self._create_engine()
        logger.info("SQLAlchemy engine created (db=%s)", settings.DB_NAME)

        # 2. LLM
        self._llm = llm_factory.get_llm(settings)
        logger.info("LLM loaded: provider=%s", settings.LLM_PROVIDER)

        # 3. Embedding model
        self._embed_model = llm_factory.get_embed_model(settings)
        logger.info("Embedding model loaded: %s", settings.EMBEDDING_MODEL)

        # 4. Schema loader
        self._schema_loader = SchemaLoader(settings.SCHEMA_YAML_PATH, self._engine)
        table_names = self._schema_loader.get_table_names()
        logger.info("Schema loaded: %d tables", len(table_names))

        # 5. Build hybrid retriever
        table_contexts = self._schema_loader.get_all_table_contexts()

        bm25 = BM25TableRetriever(table_contexts)
        dense = DenseTableRetriever(
            table_contexts=table_contexts,
            embed_model=self._embed_model,
            cache_dir=settings.INDEX_CACHE_DIR,
            schema_yaml_path=settings.SCHEMA_YAML_PATH,
            cache_version=settings.INDEX_CACHE_VERSION,
        )
        self._retriever = HybridTableRetriever(bm25, dense, settings.TABLE_TOP_K)
        logger.info("Hybrid retriever ready (top_k=%d)", settings.TABLE_TOP_K)

        self.ready = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(self, question: str) -> dict[str, Any]:
        """
        Process a natural-language question and return a structured response.

        Returns:
            {
                "answer":      str,
                "sql":         str | None,
                "data":        list[dict] | None,
                "tables_used": list[str] | None,
            }
        """
        logger.info("Received question: %s", question)

        # 1. Classify intent
        intent = self._detect_intent(question)
        logger.info("Detected intent: %s", intent)

        # 2. Route conversational intents
        if intent == "GREETING":
            return _ok(random_greeting())
        if intent == "FAREWELL":
            return _ok(random_farewell())
        if intent == "THANKS":
            return _ok(random_thanks())
        if intent == "UNRELATED":
            return _ok(UNRELATED_RESPONSE)

        # 3. SQL_QUERY path
        tables_used = self._retrieve_tables(question)
        logger.info("Retrieved tables: %s", tables_used)

        schema_context = self._build_schema_context(tables_used)

        # Generate SQL (with retry on execution failure)
        sql: str | None = None
        rows: list[dict] | None = None
        last_error: str = ""

        for attempt in range(1 + self._settings.MAX_SQL_RETRIES):
            if attempt == 0:
                sql = self._generate_sql(question, schema_context)
            else:
                logger.warning("SQL retry %d after error: %s", attempt, last_error)
                sql = self._retry_sql(question, sql or "", last_error, schema_context)

            logger.info("Generated SQL (attempt %d): %s", attempt + 1, sql)

            # Safety check
            is_safe, reason = validate_sql(sql)
            if not is_safe:
                return _err(
                    f"The generated query was blocked for safety reasons: {reason}. "
                    "Only SELECT statements are permitted."
                )

            # Execute
            try:
                rows = self._execute_sql(sql)
                break  # success
            except Exception as exc:
                last_error = str(exc)
                logger.warning("SQL execution failed (attempt %d): %s", attempt + 1, last_error)
                if attempt >= self._settings.MAX_SQL_RETRIES:
                    return _err(
                        f"The query could not be executed after "
                        f"{self._settings.MAX_SQL_RETRIES + 1} attempts. "
                        f"Last error: {last_error}"
                    )

        # Format response
        answer = self._format_response(question, sql or "", rows or [])

        return {
            "answer": answer,
            "sql": sql,
            "data": rows,
            "tables_used": tables_used,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_engine(self) -> Engine:
        """Build the mssql+pyodbc SQLAlchemy engine."""
        s = self._settings
        password_encoded = urllib.parse.quote_plus(s.DB_PASSWORD)
        driver_encoded = urllib.parse.quote_plus(s.DB_DRIVER)
        conn_str = (
            f"mssql+pyodbc://{s.DB_USER}:{password_encoded}@{s.DB_SERVER}/"
            f"{s.DB_NAME}?driver={driver_encoded}&TrustServerCertificate=yes"
        )
        return create_engine(conn_str, pool_pre_ping=True, echo=False)

    def _detect_intent(self, question: str) -> str:
        """Call the LLM to classify the user's intent."""
        prompt = INTENT_PROMPT.format(question=question)
        try:
            response = self._llm.complete(prompt)
            intent = response.text.strip().upper()
            valid = {"GREETING", "FAREWELL", "THANKS", "SQL_QUERY", "UNRELATED"}
            return intent if intent in valid else "SQL_QUERY"
        except Exception as exc:
            logger.warning("Intent detection failed (%s); defaulting to SQL_QUERY", exc)
            return "SQL_QUERY"

    def _retrieve_tables(self, question: str) -> list[str]:
        """Return relevant table names via hybrid retrieval."""
        return self._retriever.retrieve(question)

    def _build_schema_context(self, table_names: list[str]) -> str:
        """Build a combined schema context string for the given tables."""
        parts = []
        for name in table_names:
            parts.append(self._schema_loader.get_table_context(name))
        return "\n\n".join(parts)

    def _generate_sql(self, question: str, schema_context: str) -> str:
        """Generate SQL from the question and schema context."""
        prompt = SQL_GENERATION_PROMPT.format(
            schema_context=schema_context,
            question=question,
        )
        response = self._llm.complete(prompt)
        return _parse_sql(response.text)

    def _retry_sql(
        self, question: str, failed_sql: str, error_message: str, schema_context: str
    ) -> str:
        """Ask the LLM to fix a previously failed SQL query."""
        prompt = SQL_ERROR_RETRY_PROMPT.format(
            question=question,
            failed_sql=failed_sql,
            error_message=error_message,
            schema_context=schema_context,
        )
        response = self._llm.complete(prompt)
        return _parse_sql(response.text)

    def _execute_sql(self, sql: str) -> list[dict]:
        """Execute the SQL and return up to _MAX_ROWS rows as a list of dicts."""
        with self._engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = []
            for row in result:
                rows.append(dict(zip(columns, row)))
                if len(rows) >= _MAX_ROWS:
                    break
        return rows

    def _format_response(self, question: str, sql: str, rows: list[dict]) -> str:
        """Convert raw rows into a human-readable answer via the LLM."""
        # Serialise rows compactly (truncate if very large)
        rows_preview = rows[:50]  # send at most 50 rows to the formatter LLM
        rows_text = "\n".join(str(r) for r in rows_preview)
        if len(rows) > 50:
            rows_text += f"\n… ({len(rows) - 50} more rows not shown)"

        prompt = RESPONSE_FORMATTING_PROMPT.format(
            question=question,
            sql=sql,
            row_count=len(rows),
            rows=rows_text if rows else "(no rows returned)",
        )
        try:
            response = self._llm.complete(prompt)
            return response.text.strip()
        except Exception as exc:
            logger.warning("Response formatting failed (%s); returning raw data", exc)
            if not rows:
                return "The query returned no results."
            return f"Query returned {len(rows)} row(s):\n" + "\n".join(
                str(r) for r in rows[:20]
            )


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _parse_sql(llm_output: str) -> str:
    """Extract the SQL query from LLM output that may contain THINK + SQL sections."""
    # Try to extract content after "SQL:" label
    sql_match = re.search(r"SQL:\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        sql = llm_output.strip()

    # Strip markdown code fences if present
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)

    return sql.strip()


def _ok(answer: str) -> dict[str, Any]:
    return {"answer": answer, "sql": None, "data": None, "tables_used": None}


def _err(message: str) -> dict[str, Any]:
    return {"answer": message, "sql": None, "data": None, "tables_used": None}
