#!/usr/bin/env python3
"""
Step 7 — Live Excel Fact Extractor.

Reads workforce metrics directly from the Excel source file — no hardcoded numbers.
Run this daily (cron) or whenever the Excel file changes.

What it does:
  1. Auto-detects the latest *.xlsx in data/EXCEL/
  2. Reads the Base Data sheet — the raw employee table
  3. Computes counts by Nationality, POS Category, ATNM/Hired
  4. Generates fact text dynamically from those computed values
  5. Upserts into PostgreSQL — old facts are replaced with fresh numbers

Run:
    conda activate v12
    python scripts/07_auto_extract_excel.py

Schedule (Linux cron — daily at 6 AM):
    0 6 * * * /home/abhay/anaconda3/envs/v12/bin/python \
        /home/abhay/Desktop/NGXP/tasnimv.0/scripts/07_auto_extract_excel.py
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
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

import yaml

HERE        = Path(__file__).parent.parent
DATA_DIR    = HERE / "data" / "EXCEL"
PG_URL      = os.getenv("PG_URL",      "postgresql://abhay@/altasnim?host=/var/run/postgresql")
PG_TABLE    = os.getenv("PG_TABLE",    "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# Load workforce extraction config from sql_config.yaml so no column names or
# category labels are hardcoded in Python.
def _load_workforce_cfg() -> dict:
    sql_cfg_path = HERE / "config" / "sql_config.yaml"
    with sql_cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    for t in cfg.get("tables", []):
        if t.get("type") == "workforce":
            return t
    return {}

_WF_CFG = _load_workforce_cfg()
FACT_CATEGORY = _WF_CFG.get("name", "workforce")


# ---------------------------------------------------------------------------
# Locate the workforce Excel — reads WORKFORCE_EXCEL from .env
# ---------------------------------------------------------------------------
def find_excel() -> Path:
    filename = os.getenv("WORKFORCE_EXCEL", "").strip()
    if filename:
        path = DATA_DIR / filename
        if path.exists():
            print(f"  Source Excel : {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
            return path
        raise FileNotFoundError(
            f"WORKFORCE_EXCEL='{filename}' set in .env but file not found at:\n  {path}\n"
            "Update WORKFORCE_EXCEL in .env to the correct filename."
        )
    # Fallback: find the xlsx that has the configured sheet name
    sheet_name = _WF_CFG.get("sheet", "Base Data")
    print(f"  WORKFORCE_EXCEL not set — scanning for file with '{sheet_name}' sheet …")
    import openpyxl
    for xlsx in sorted(DATA_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_size, reverse=True):
        try:
            wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
            if sheet_name in wb.sheetnames:
                wb.close()
                print(f"  Found : {xlsx.name}  ({xlsx.stat().st_size / 1024:.0f} KB)")
                return xlsx
            wb.close()
        except Exception:
            continue
    raise FileNotFoundError(
        f"No Excel file with a '{sheet_name}' sheet found in data/EXCEL/.\n"
        "Set WORKFORCE_EXCEL=<filename> in .env to specify it explicitly."
    )


# ---------------------------------------------------------------------------
# Extract workforce metrics from Base Data (flat employee table)
# ---------------------------------------------------------------------------
def extract_workforce(excel_path: Path) -> dict:
    """
    Reads the workforce sheet and computes headcount breakdowns.
    All column names and category labels come from sql_config.yaml (workforce table entry).
    """
    sheet      = _WF_CFG.get("sheet",              "Base Data")
    chart_col  = _WF_CFG.get("dedup_chart_col",    "Chart List")
    chart_val  = _WF_CFG.get("dedup_chart_value",  "Nationality Wise Employee Count")
    nat_col    = _WF_CFG.get("nationality_col",    "Nationality")
    expat_lbl  = _WF_CFG.get("expat_label",        "Expats")
    cat_col    = _WF_CFG.get("category_col",       "POS Category")
    emp_col    = _WF_CFG.get("employment_col",     "ATNM / Hired")
    id_col     = _WF_CFG.get("id_col",             "Personnel Number")
    labour_lbl = _WF_CFG.get("labour_label",       "Labour")
    staff_lbl  = _WF_CFG.get("staff_label",        "Staff")
    atnm_lbl   = _WF_CFG.get("atnm_label",         "ATNM")
    hired_lbl  = _WF_CFG.get("hired_label",        "Hired")

    print(f"  Reading '{sheet}' sheet …")
    df = pd.read_excel(excel_path, sheet_name=sheet, header=0)

    # The sheet repeats each employee once per chart type.
    # Filter to one chart type so every employee appears exactly once.
    missing = [c for c in [chart_col, nat_col, cat_col, emp_col, id_col]
               if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in '{sheet}': {missing}")

    subset = df[df[chart_col] == chart_val].copy()
    print(f"  Rows for '{chart_val}': {len(subset):,}")

    # Nationality breakdown — bucket everything non-expat as national
    nat_counts = subset[nat_col].value_counts()
    expats     = int(nat_counts.get(expat_lbl, 0))
    nationals  = int(subset[nat_col].str.lower().ne(expat_lbl.lower()).sum())

    # Category breakdown
    cat_counts = subset[cat_col].value_counts()
    labour = int(cat_counts.get(labour_lbl, 0))
    staff  = int(cat_counts.get(staff_lbl,  0))

    # Employment type breakdown
    hire_counts = subset[emp_col].value_counts()
    atnm  = int(hire_counts.get(atnm_lbl,  0))
    hired = int(hire_counts.get(hired_lbl, 0))

    total = len(subset)

    metrics = {
        "total":     total,
        "expats":    expats,
        "nationals": nationals,
        "labour":    labour,
        "staff":     staff,
        "atnm":      atnm,
        "hired":     hired,
        "source":    excel_path.name,
    }

    print(f"\n  Extracted metrics:")
    print(f"    Total employees : {total:,}")
    print(f"    Expats          : {expats:,}")
    print(f"    Nationals       : {nationals:,}")
    print(f"    Labour          : {labour:,}")
    print(f"    Staff           : {staff:,}")
    print(f"    ATNM            : {atnm:,}")
    print(f"    Hired           : {hired:,}")
    return metrics


# ---------------------------------------------------------------------------
# Generate fact chunks from extracted metrics
# ---------------------------------------------------------------------------
def build_facts(m: dict) -> list[dict]:
    """
    Builds human-readable fact strings from extracted metrics.
    Text is generated here — numbers come entirely from 'm' (the live data).
    """
    src = m["source"]

    raw_facts = [
        (
            f"AL TASNIM total number of employees: {m['total']:,}. "
            f"Breakdown: Expat employees {m['expats']:,} and National employees {m['nationals']:,}. "
            f"Grand Total workforce headcount is {m['total']:,} people.",
            "Nationality breakdown"
        ),
        (
            f"AL TASNIM expat employee count: {m['expats']:,}. "
            f"Number of expatriate workers employed by AL TASNIM is {m['expats']:,}. "
            f"ATNM employees total: {m['expats']:,}.",
            "Expat count"
        ),
        (
            f"AL TASNIM national employee count: {m['nationals']:,}. "
            f"Number of national / local employees: {m['nationals']:,}. "
            f"Hired national workforce: {m['nationals']:,}.",
            "National count"
        ),
        (
            f"AL TASNIM labour headcount: {m['labour']:,}. "
            f"Total labour workforce: {m['labour']:,} workers. "
            f"Staff headcount: {m['staff']:,}. "
            f"Combined labour and staff grand total: {m['total']:,}.",
            "Labour vs Staff"
        ),
        (
            f"AL TASNIM staff headcount: {m['staff']:,}. "
            f"Number of staff employees at AL TASNIM: {m['staff']:,}. "
            f"Staff workforce total: {m['staff']:,} people. "
            f"AL TASNIM has {m['staff']:,} staff members.",
            "Staff count"
        ),
        (
            f"AL TASNIM workforce summary: "
            f"Total employees {m['total']:,}. "
            f"Labour {m['labour']:,}. "
            f"Staff {m['staff']:,}. "
            f"Expats {m['expats']:,}. "
            f"Nationals {m['nationals']:,}. "
            f"Grand total headcount {m['total']:,}.",
            "Workforce summary"
        ),
    ]

    chunks = []
    for text, sheet in raw_facts:
        h = hashlib.md5(text.encode()).hexdigest()
        chunks.append({
            "id":           f"fact_{h[:16]}",
            "content":      text,
            "content_hash": h,
            "metadata": {
                "source":        src,
                "sheet":         sheet,
                "document_type": "fact",
                "category":      FACT_CATEGORY,
                "is_fact":       True,
                "auto_extracted": True,
            },
        })
    return chunks


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> list:
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[embed] Loading {EMBED_MODEL} on {device.upper()} …")
    model = SentenceTransformer(
        EMBED_MODEL,
        model_kwargs={"torch_dtype": torch.float16},
        device=device,
    )
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    print(f"[embed] Done — {len(vecs)} vectors")
    return vecs


# ---------------------------------------------------------------------------
# Remove stale workforce facts from a previous run before inserting fresh ones
# ---------------------------------------------------------------------------
def purge_old_workforce_facts(conn) -> int:
    """Delete workforce facts whose content is stale (different text, same category).
    ON CONFLICT DO UPDATE handles same-ID updates; but if text changed → new ID,
    so we also purge all auto_extracted workforce facts before reinserting."""
    with conn.cursor() as cur:
        cur.execute(
            f"""DELETE FROM {PG_TABLE}
                WHERE metadata->>'category' = %s
                  AND metadata->>'auto_extracted' = 'true'""",
            (FACT_CATEGORY,),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Step 7 — Live Excel Fact Extractor")
    print("=" * 60)

    excel_path = find_excel()
    metrics    = extract_workforce(excel_path)
    chunks     = build_facts(metrics)

    print(f"\n  Facts generated: {len(chunks)}")

    vecs = embed_texts([c["content"] for c in chunks])

    print("\n[db] Connecting …")
    conn = psycopg2.connect(PG_URL)
    conn.autocommit = False
    register_vector(conn)

    deleted = purge_old_workforce_facts(conn)
    print(f"  Purged {deleted} stale workforce facts from previous run.")

    rows = [
        (
            c["id"],
            c["metadata"]["source"],
            c["content"],
            vec,
            json.dumps(c["metadata"]),
            c["content_hash"],
        )
        for c, vec in zip(chunks, vecs)
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""INSERT INTO {PG_TABLE}
                   (id, source, content, embedding, metadata, content_hash)
               VALUES %s
               ON CONFLICT (id) DO UPDATE
                 SET content    = EXCLUDED.content,
                     embedding  = EXCLUDED.embedding,
                     metadata   = EXCLUDED.metadata""",
            rows,
        )
    conn.commit()
    conn.close()

    print(f"\n  Injected {len(chunks)} live workforce facts.")
    print("  Restart the server to rebuild BM25 with fresh numbers.")
    print("=" * 60)


if __name__ == "__main__":
    main()
