"""
Evaluation harness for natural-language query regression testing.

Add question/expected_table pairs to EVAL_CASES and run:
    python tests/eval_harness.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.connection_pool import ConnectionPool
from src.value_resolver import ValueResolver
from src.intent_parser import IntentParser

EVAL_CASES = [
    {"question": "What is the status of rig 104?", "expected_table": "2026_Well_Delivery_Scope_Well_Type"},
    {"question": "What is the overall progress of Well 33151?", "expected_table": "WMR"},
]

def run_eval():
    conn_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={settings.DB_SERVER};"
        f"DATABASE={settings.DB_NAME};"
        f"UID={settings.DB_READONLY_USER};"
        f"PWD={settings.DB_READONLY_PASSWORD};"
        f"ApplicationIntent=ReadOnly;"
    )
    pool = ConnectionPool(conn_string, max_size=3)
    resolver = ValueResolver(pool, settings.FUZZY_THRESHOLD, settings.AMBIGUITY_MARGIN)
    parser = IntentParser(resolver)

    passed = 0
    for case in EVAL_CASES:
        result = parser.parse(case["question"])
        if result.success and result.intent and result.intent.table == case["expected_table"]:
            passed += 1
            print(f"PASS: {case['question']}")
        else:
            table = result.intent.table if result.intent else None
            print(f"FAIL: {case['question']} (got {table}, expected {case['expected_table']})")

    pool.close_all()
    print(f"\n{passed}/{len(EVAL_CASES)} cases passed")

if __name__ == "__main__":
    run_eval()
