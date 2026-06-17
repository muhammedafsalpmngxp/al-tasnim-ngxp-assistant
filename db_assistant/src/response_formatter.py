from typing import List
from .config import settings
from .llm_client import get_llm_client
import logging

logger = logging.getLogger(__name__)

class ResponseFormatter:
    def __init__(self):
        self.llm = get_llm_client()
        self.provider = settings.LLM_PROVIDER

    def format(self, question: str, data: List[dict], columns: List[str], row_count: int, table: str) -> str:
        if not data or row_count == 0:
            return "No matching records found in the database for your query."

        return self._format_with_llm(question, data, columns, row_count, table)

    def _format_with_llm(self, question: str, data: List[dict], columns: List[str], row_count: int, table: str) -> str:
        records = []
        for i, row in enumerate(data[:10]):
            record_str = " | ".join([f"{col}: {row.get(col, 'NULL')}" for col in columns[:6]])
            records.append(f"Record {i+1}: {record_str}")

        records_text = "\n".join(records)

        prompt = f"""Question: {question}

Found {row_count} records in {table}.
First {min(10, row_count)} records:
{records_text}

Please provide a clear, accurate answer. Include evidence from the data.
Answer:"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful data assistant. Be concise and accurate."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.3, "num_predict": 400},
            )

            answer = response["message"]["content"].strip()

            if row_count > 10:
                answer += f"\n\n*Showing first 10 of {row_count} records from {table}.*"
            else:
                answer += f"\n\n*Found {row_count} records in {table}.*"

            return answer

        except Exception as e:
            logger.warning(f"LLM formatting failed: {e}")
            return self._fallback_format(data, columns, row_count, table)

    def _fallback_format(self, data: List[dict], columns: List[str], row_count: int, table: str) -> str:
        if row_count == 1:
            parts = [f"{col}: {row.get(col, 'NULL')}" for col in columns for row in data]
            return " | ".join(parts[:10])

        result = f"Found {row_count} records in {table}:\n\n"
        for i, row in enumerate(data[:5], 1):
            parts = [f"{col}: {row.get(col, 'NULL')}" for col in columns[:5]]
            result += f"{i}. " + " | ".join(parts) + "\n"

        if row_count > 5:
            result += f"\n... and {row_count - 5} more records."
        return result
