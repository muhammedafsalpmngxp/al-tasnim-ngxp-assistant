#!/usr/bin/env python3
"""
Step 11 — Export SQL Schema.

Reads every table that was loaded by the universal ingest pipeline and writes
a complete schema YAML to config/schema.yaml.

The YAML is structured as:
  tables:
    well_monitoring:
      source_file: ...
      row_count: 10632
      columns:
        - name: rig_no
          type: text
          sample: SWER101
        ...
    activity_master:
      ...

This file is used by:
  - Developers to understand what columns are available per table
  - The LLM intent classifier (future: auto-include in SQL prompt)
  - Debugging: confirm that ingest loaded the right headers

Run:
    conda activate v12
    python scripts/11_export_schema.py
"""
import os
from pathlib import Path

import psycopg2
import yaml

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.split("#")[0].strip()
            os.environ.setdefault(_k.strip(), _v)

PG_URL    = os.getenv("PG_URL", "postgresql://abhay@/altasnim?host=/var/run/postgresql")
HERE      = Path(__file__).parent.parent
OUT_FILE  = HERE / "config" / "schema.yaml"
SQL_CFG   = HERE / "config" / "sql_config.yaml"

# Internal tables managed by the RAG system — skip these
SKIP_TABLES = {"rag_chunks", "spatial_ref_sys"}

SAMPLE_ROWS = 3   # how many sample values to show per column


def get_conn():
    conn = psycopg2.connect(PG_URL)
    conn.autocommit = True
    return conn


def fetch_tables(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
        return [r[0] for r in cur.fetchall() if r[0] not in SKIP_TABLES]


def fetch_columns(conn, table: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = %s AND column_name != '_row_id' "
            "ORDER BY ordinal_position",
            (table,),
        )
        cols = cur.fetchall()

    result = []
    for col_name, col_type in cols:
        # Fetch a few non-null samples
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT {col_name} FROM {table} "
                    f"WHERE {col_name} IS NOT NULL AND CAST({col_name} AS TEXT) != '' "
                    f"LIMIT {SAMPLE_ROWS}"
                )
                samples = [str(r[0]) for r in cur.fetchall()]
        except Exception:
            samples = []

        result.append({
            "name":    col_name,
            "type":    col_type,
            "samples": samples,
        })
    return result


def fetch_row_count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]


def load_source_map() -> dict[str, str]:
    """Map table_name → source_file from sql_config.yaml."""
    if not SQL_CFG.exists():
        return {}
    with SQL_CFG.open() as f:
        cfg = yaml.safe_load(f)
    result = {}
    for t in cfg.get("tables", []):
        result[t["name"]] = t.get("source_file", "")
    return result


def main() -> None:
    print("=" * 60)
    print("  Step 11 — Export SQL Schema")
    print(f"  Output : {OUT_FILE.relative_to(HERE)}")
    print("=" * 60)

    conn       = get_conn()
    tables     = fetch_tables(conn)
    source_map = load_source_map()

    print(f"\n  Found {len(tables)} tables in PostgreSQL:\n")

    schema: dict = {"tables": {}}

    for table in tables:
        print(f"  Scanning {table} …", end="", flush=True)
        cols      = fetch_columns(conn, table)
        row_count = fetch_row_count(conn, table)

        schema["tables"][table] = {
            "source_file": source_map.get(table, ""),
            "row_count":   row_count,
            "column_count": len(cols),
            "columns":     cols,
        }
        print(f"  {row_count:,} rows × {len(cols)} cols")

    conn.close()

    with OUT_FILE.open("w") as f:
        yaml.dump(schema, f, allow_unicode=True, sort_keys=False,
                  default_flow_style=False, width=120)

    print(f"\n  Schema written to {OUT_FILE.relative_to(HERE)}")
    print(f"  Total tables : {len(tables)}")

    # Print a quick summary table
    print(f"\n  {'Table':<30} {'Rows':>8}  {'Cols':>5}  Source file")
    print(f"  {'-'*30} {'-'*8}  {'-'*5}  {'-'*40}")
    for name, info in schema["tables"].items():
        src = (info["source_file"] or "")[:40]
        print(f"  {name:<30} {info['row_count']:>8,}  {info['column_count']:>5}  {src}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
