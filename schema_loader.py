"""
schema_loader.py — Loads the YAML schema and builds LlamaIndex schema objects.
"""
from __future__ import annotations

import yaml
from typing import Any

from llama_index.core import SQLDatabase
from llama_index.core.objects import SQLTableSchema


class SchemaLoader:
    """Loads table metadata from a YAML file and provides helpers for LlamaIndex."""

    def __init__(self, yaml_path: str, db_engine: Any) -> None:
        self._yaml_path = yaml_path
        self._db_engine = db_engine
        self._yaml_data: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------

    def load_yaml(self) -> dict[str, Any]:
        """Load and cache the YAML file. Returns the full tables dict."""
        if self._yaml_data is None:
            with open(self._yaml_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            # Support both {tables: {...}} and flat {table_name: {...}} layouts
            self._yaml_data = raw.get("tables", raw)
        return self._yaml_data

    def get_table_names(self) -> list[str]:
        """Return all table names defined in the YAML."""
        return list(self.load_yaml().keys())

    def get_table_context(self, table_name: str) -> str:
        """Return a formatted string describing a table for LLM context."""
        tables = self.load_yaml()
        if table_name not in tables:
            return f"Table: {table_name}\n(No schema information available)"

        meta = tables[table_name]
        description = meta.get("description", "No description available.")
        columns: dict[str, Any] = meta.get("columns", {})

        lines: list[str] = [
            f"Table: {table_name}",
            f"Description: {description}",
            "Columns:",
        ]
        for col_name, col_info in columns.items():
            if isinstance(col_info, dict):
                col_desc = col_info.get("description", col_info.get("desc", ""))
            else:
                col_desc = str(col_info)
            lines.append(f"  {col_name}: {col_desc}")

        return "\n".join(lines)

    def get_all_table_contexts(self) -> dict[str, str]:
        """Return {table_name: context_string} for every table in the YAML."""
        return {name: self.get_table_context(name) for name in self.get_table_names()}

    # ------------------------------------------------------------------
    # LlamaIndex objects
    # ------------------------------------------------------------------

    def build_sql_database(self) -> SQLDatabase:
        """Wrap the SQLAlchemy engine in a LlamaIndex SQLDatabase (only YAML tables)."""
        return SQLDatabase(self._db_engine, include_tables=self.get_table_names())

    def build_table_schemas(self) -> list[SQLTableSchema]:
        """Build a SQLTableSchema for each table defined in the YAML."""
        schemas: list[SQLTableSchema] = []
        for table_name in self.get_table_names():
            context_str = self.get_table_context(table_name)
            schemas.append(
                SQLTableSchema(table_name=table_name, context_str=context_str)
            )
        return schemas
