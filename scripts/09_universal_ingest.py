#!/usr/bin/env python3
"""
Step 9 — Universal Data Ingestion Pipeline.

Reads the tables section of config/sql_config.yaml and processes all sources:
  - type: sql       → loads CSV or Excel sheet into a PostgreSQL table
  - type: workforce → extracts live headcount facts from the workforce Excel
  - (no type)       → legacy table, skipped by this script

To add a new file:
  1. Copy it to data/CSV/ or data/EXCEL/
  2. Add an entry (with type, dir, sheet, header_row) to the tables section
     of config/sql_config.yaml  — no Python editing needed
  3. Re-run this script
  4. Restart the server

Run:
    conda activate v12
    python scripts/09_universal_ingest.py
    python scripts/09_universal_ingest.py --reset   # drop and reload all tables
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import yaml
from psycopg2.extras import execute_values

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

HERE       = Path(__file__).parent.parent
DATA_DIR   = HERE / "data"
SQL_CFG    = HERE / "config" / "sql_config.yaml"
PG_URL     = os.getenv("PG_URL", "postgresql://abhay@/altasnim?host=/var/run/postgresql")
PG_TABLE   = os.getenv("PG_TABLE", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RESET      = "--reset" in sys.argv
BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_col(name: str) -> str:
    """Turn any column header into a valid SQL identifier."""
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    if not name or name[0].isdigit():
        name = 'col_' + name
    return name[:63]


def dedup_cols(cols: list[str]) -> list[str]:
    """Ensure no two column names are identical."""
    seen: dict[str, int] = {}
    result = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            result.append(c)
    return result


def pg_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    return "TEXT"


def get_conn() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(PG_URL)
    conn.autocommit = False
    return conn


# ---------------------------------------------------------------------------
# Read a file (CSV or Excel) into a DataFrame
# ---------------------------------------------------------------------------
def read_source(src: dict) -> pd.DataFrame | None:
    fpath = DATA_DIR / src.get("dir", "CSV") / src.get("source_file", "")
    if not fpath.exists():
        print(f"    [warn] File not found — skipping: {fpath}")
        return None

    header_row = int(src.get("header_row", 0))

    try:
        if fpath.suffix.lower() == ".csv":
            df = pd.read_csv(fpath, header=header_row, low_memory=False)
        else:
            sheet = src.get("sheet")
            df = pd.read_excel(fpath, sheet_name=sheet, header=header_row)
    except Exception as e:
        print(f"    [error] Could not read {fpath.name}: {e}")
        return None

    if df.empty:
        print(f"    [warn] {fpath.name} is empty after reading — skipped.")
        return None

    # Drop completely empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Clean column names
    df.columns = dedup_cols([clean_col(c) for c in df.columns])

    # Drop columns whose cleaned name is 'unnamed_...' AND all values are null
    unnamed = [c for c in df.columns if c.startswith("unnamed")]
    df = df.drop(columns=[c for c in unnamed if df[c].isna().all()])

    return df


# ---------------------------------------------------------------------------
# Load a DataFrame into PostgreSQL
# ---------------------------------------------------------------------------
def load_to_sql(conn, df: pd.DataFrame, table_name: str) -> int:
    with conn.cursor() as cur:
        if RESET:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            conn.commit()

        col_defs = ", ".join(f"{col} {pg_type(df[col])}" for col in df.columns)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                _row_id SERIAL PRIMARY KEY,
                {col_defs}
            )
        """)
        conn.commit()

        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        existing = cur.fetchone()[0]
        if existing > 0 and not RESET:
            print(f"    Table '{table_name}' already has {existing:,} rows — skipping. (--reset to reload)")
            return 0

    # Replace NaN with None → PostgreSQL NULL
    df = df.where(pd.notna(df), None)

    col_list   = list(df.columns)
    insert_sql = f"INSERT INTO {table_name} ({', '.join(col_list)}) VALUES %s"

    total = 0
    with conn.cursor() as cur:
        for start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[start: start + BATCH_SIZE]
            rows  = [tuple(r) for r in batch.itertuples(index=False, name=None)]
            execute_values(cur, insert_sql, rows, page_size=BATCH_SIZE)
            total += len(rows)
        conn.commit()

    # Indexes on common key columns if they exist
    with conn.cursor() as cur:
        for col in ["rig_no", "well_type", "week_number", "id", "well_id"]:
            if col in df.columns:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table_name}_{col}_idx "
                    f"ON {table_name} ({col})"
                )
        conn.commit()

    return total


# ---------------------------------------------------------------------------
# Workforce fact extraction (reuses logic from script 07)
# ---------------------------------------------------------------------------
def extract_workforce_facts(src: dict) -> list[dict]:
    import hashlib
    fpath = DATA_DIR / src.get("dir", "EXCEL") / src.get("source_file", "")
    if not fpath.exists():
        print(f"    [warn] Workforce file not found: {fpath}")
        return []

    print(f"    Reading workforce data from '{src.get('source_file', '')}' …")
    try:
        df = pd.read_excel(fpath, sheet_name="Base Data", header=0)
    except Exception as e:
        print(f"    [error] {e}")
        return []

    CHART_COL = "Chart List"
    TARGET    = "Nationality Wise Employee Count"
    NAT_COL   = "Nationality"
    CAT_COL   = "POS Category"
    ATNM_COL  = "ATNM / Hired"

    if CHART_COL not in df.columns:
        print(f"    [warn] 'Chart List' column not found in Base Data sheet.")
        return []

    subset = df[df[CHART_COL] == TARGET].copy()
    if subset.empty:
        print(f"    [warn] No rows for '{TARGET}' in Base Data.")
        return []

    nat_counts = subset[NAT_COL].value_counts()
    expats     = int(nat_counts.get("Expats", 0))
    nationals  = int(subset[NAT_COL].str.lower().ne("expats").sum())
    cat_counts = subset[CAT_COL].value_counts() if CAT_COL in subset else {}
    labour     = int(cat_counts.get("Labour", 0)) if hasattr(cat_counts, 'get') else 0
    staff      = int(cat_counts.get("Staff",  0)) if hasattr(cat_counts, 'get') else 0
    total      = len(subset)

    print(f"    Extracted — Total:{total:,}  Expats:{expats:,}  Nationals:{nationals:,}  Labour:{labour:,}  Staff:{staff:,}")

    m = {"total": total, "expats": expats, "nationals": nationals,
         "labour": labour, "staff": staff, "source": src["file"]}

    raw_facts = [
        f"AL TASNIM total number of employees: {total:,}. Expat employees {expats:,} and National employees {nationals:,}. Grand Total workforce headcount is {total:,} people.",
        f"AL TASNIM expat employee count: {expats:,}. Number of expatriate workers employed by AL TASNIM is {expats:,}.",
        f"AL TASNIM national employee count: {nationals:,}. Number of national / local employees: {nationals:,}.",
        f"AL TASNIM labour headcount: {labour:,}. Staff headcount: {staff:,}. Combined labour and staff grand total: {total:,}.",
        f"AL TASNIM workforce summary: Total employees {total:,}. Labour {labour:,}. Staff {staff:,}. Expats {expats:,}. Nationals {nationals:,}.",
    ]

    chunks = []
    for text in raw_facts:
        h = hashlib.md5(text.encode()).hexdigest()
        chunks.append({
            "id":           f"fact_{h[:16]}",
            "content":      text,
            "content_hash": h,
            "metadata": {
                "source":        src["file"],
                "document_type": "fact",
                "category":      "workforce",
                "is_fact":       True,
                "auto_extracted": True,
            },
        })
    return chunks


def push_facts(conn, chunks: list[dict], vecs) -> None:
    from pgvector.psycopg2 import register_vector
    register_vector(conn)

    # Purge previous auto-extracted workforce facts
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {PG_TABLE} "
            f"WHERE metadata->>'category' = 'workforce' "
            f"AND metadata->>'auto_extracted' = 'true'"
        )
    conn.commit()

    rows = [
        (c["id"], c["metadata"]["source"], c["content"], vec,
         json.dumps(c["metadata"]), c["content_hash"])
        for c, vec in zip(chunks, vecs)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""INSERT INTO {PG_TABLE}
                   (id, source, content, embedding, metadata, content_hash)
               VALUES %s
               ON CONFLICT (id) DO UPDATE
                 SET content   = EXCLUDED.content,
                     embedding = EXCLUDED.embedding,
                     metadata  = EXCLUDED.metadata""",
            rows,
        )
    conn.commit()


def embed_texts(texts: list[str]):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Embedding {len(texts)} facts on {device.upper()} …")
    model = SentenceTransformer(
        EMBED_MODEL,
        model_kwargs={"torch_dtype": torch.float16},
        device=device,
    )
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Update sql_config.yaml with newly loaded tables
# ---------------------------------------------------------------------------
def update_sql_config(loaded_tables: list[dict]) -> None:
    with SQL_CFG.open() as f:
        cfg = yaml.safe_load(f)

    existing_tables = {t["name"]: t for t in cfg.get("tables", [])}
    for t in loaded_tables:
        name = t["name"]
        if name in existing_tables:
            existing_tables[name].update(t)  # merge: preserve dir/type/sheet/header_row
        else:
            existing_tables[name] = t

    cfg["tables"] = list(existing_tables.values())
    with SQL_CFG.open("w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\n  sql_config.yaml updated — {len(cfg['tables'])} tables registered.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  Step 9 — Universal Data Ingestion Pipeline")
    print(f"  Config : {SQL_CFG.relative_to(HERE)}")
    print(f"  Mode   : {'RESET (drop + reload)' if RESET else 'resume (skip existing)'}")
    print("=" * 70)

    if not SQL_CFG.exists():
        print(f"ERROR: {SQL_CFG} not found.")
        sys.exit(1)

    with SQL_CFG.open() as f:
        cfg = yaml.safe_load(f)

    # Only process tables that have a type (sql or workforce).
    # Entries without type are legacy tables loaded outside this script.
    sources = [t for t in cfg.get("tables", []) if t.get("type") in ("sql", "workforce")]
    print(f"\n  Sources defined : {len(sources)}")

    conn = get_conn()

    # Collect results for summary
    loaded_tables  = []
    failed         = []
    workforce_done = False

    for i, src in enumerate(sources, 1):
        stype = src.get("type", "").lower()
        fname = src.get("source_file", src.get("name", ""))
        print(f"\n[{i:02d}/{len(sources)}] {fname}  [{stype}]")

        # ── WORKFORCE fact extraction ──────────────────────────────────────────
        if stype == "workforce":
            chunks = extract_workforce_facts(src)
            if chunks:
                vecs = embed_texts([c["content"] for c in chunks])
                push_facts(conn, chunks, vecs)
                print(f"    Injected {len(chunks)} workforce fact chunks into RAG DB.")
                workforce_done = True
            else:
                failed.append(fname)
            continue

        # ── SQL load ───────────────────────────────────────────────────────────
        if stype == "sql":
            table_name = src.get("name", "")
            if not table_name:
                print(f"    [error] 'table' not specified in config — skipping.")
                failed.append(fname)
                continue

            df = read_source(src)
            if df is None:
                failed.append(fname)
                continue

            print(f"    Rows: {len(df):,}  |  Cols: {len(df.columns)}")
            print(f"    Sample cols: {list(df.columns[:6])}")

            try:
                inserted = load_to_sql(conn, df, table_name)
                if inserted > 0:
                    print(f"    Loaded {inserted:,} rows into '{table_name}'.")
                loaded_tables.append({
                    "name":        table_name,
                    "source_file": src.get("source_file", ""),
                    "description": src.get("description", ""),
                    "row_count":   int(len(df)),
                    "columns":     list(df.columns[:20]),
                })
            except Exception as e:
                conn.rollback()
                print(f"    [error] DB load failed: {e}")
                failed.append(fname)
            continue

        print(f"    [warn] Unknown type '{stype}' — skipped.")

    conn.close()

    # Update sql_config.yaml with all newly loaded tables
    if loaded_tables:
        update_sql_config(loaded_tables)

    # Summary
    sql_count = len(loaded_tables)
    print(f"\n{'='*70}")
    print(f"  INGESTION COMPLETE")
    print(f"{'='*70}")
    print(f"  SQL tables loaded   : {sql_count}")
    print(f"  Workforce facts     : {'yes' if workforce_done else 'no'}")
    print(f"  Failed              : {len(failed)}")
    if failed:
        print(f"\n  Failed files:")
        for f in failed:
            print(f"    - {f}")
    print(f"\n  Next: restart the server")
    print(f"  uvicorn prod_rag:app --host 0.0.0.0 --port 8000")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
