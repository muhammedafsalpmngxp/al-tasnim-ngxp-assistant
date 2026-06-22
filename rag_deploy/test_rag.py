"""
AL TASNIM RAG System — Test Suite
Usage:
  python test_rag.py
  python test_rag.py --url http://localhost:8002
  python test_rag.py --url http://SERVER_IP:8000
"""

import sys
import time
import argparse
import requests
from requests.exceptions import ConnectionError, Timeout, JSONDecodeError

G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
B = "\033[1m"
E = "\033[0m"

TIMEOUT = 90
_results: list[tuple] = []


def ask(base: str, label: str, query: str, must_contain: str = "", must_not_contain: str = "") -> None:
    """Send one query, check expectations, append result to _results."""
    try:
        t0 = time.time()
        resp = requests.post(f"{base}/ask", json={"query": query}, timeout=TIMEOUT)
        ms = (time.time() - t0) * 1000

        if resp.status_code != 200 or not resp.content:
            _record(label, False, f"HTTP {resp.status_code}: {resp.text[:80]}", ms)
            return

        ans = resp.json().get("direct_answer", "")
        fail = ""
        if "does not exist" in ans or "could not be compiled" in ans:
            fail = "SQL error leaked into answer"
        elif must_contain and must_contain.lower() not in ans.lower():
            fail = f"missing: '{must_contain}'"
        elif must_not_contain and must_not_contain.lower() in ans.lower():
            fail = f"should not contain: '{must_not_contain}'"

        _record(label, not fail, fail, ms, ans)

    except (ConnectionError, Timeout) as exc:
        _record(label, False, str(exc), 0)


def _record(label: str, passed: bool, reason: str = "", ms: float = 0, ans: str = "") -> None:
    """Store result and print one line."""
    _results.append((label, passed, reason, ms))
    status = f"{G}PASS{E}" if passed else f"{R}FAIL{E}"
    print(f"  [{status}] {label:<40} {ms:>6.0f}ms")
    if not passed:
        print(f"           reason : {reason}")
        if ans:
            print(f"           answer : {ans[:120]}")


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{B}{'─'*65}{E}")
    print(f"{B}  {title}{E}")
    print(f"{B}{'─'*65}{E}")


def run_health(base: str) -> bool:
    """Check /healthz. Returns False if server is down or DB is empty."""
    section("1. HEALTH CHECK")
    try:
        h = requests.get(f"{base}/healthz", timeout=10).json()
        chunks = h.get("chunks_in_db", 0)
        if chunks == 0:
            print(f"  {R}FAIL{E}  DB empty — run: python scripts/09_universal_ingest.py")
            return False
        print(f"  {G}OK{E}   chunks_in_db={chunks}  bm25={h.get('chunks_in_bm25', 0)}")
        print(f"       embed={h.get('embed_model','')}  llm={h.get('llm_model','')}")
        return True
    except (ConnectionError, JSONDecodeError) as exc:
        print(f"  {R}FAIL{E}  Cannot reach server — {exc}")
        return False


def run_sql_counts(base: str) -> None:
    """Total and filtered count queries."""
    section("2. SQL — TOTAL COUNTS")
    ask(base, "total well count",           "How many wells are there in total?",                    must_contain="10632")
    ask(base, "total count alternate phrasing", "What is the total number of wells in the database?", must_contain="10632")

    section("3. SQL — FILTERED COUNT BY RIG")
    ask(base, "wells on SWER101",  "How many wells does SWER101 have?",          must_contain="682",  must_not_contain="10632")
    ask(base, "wells on SWER102",  "How many wells are on rig SWER102?",          must_contain="954",  must_not_contain="10632")
    ask(base, "wells on SWER103",  "How many wells does rig SWER103 have?",                           must_not_contain="10632")

    section("4. SQL — FILTERED COUNT BY STATUS")
    ask(base, "completed location prep",    "How many wells have completed location preparation?",    must_contain="610", must_not_contain="10632")
    ask(base, "in-progress flowline",       "How many wells have flowline construction in progress?",                     must_not_contain="10632")
    ask(base, "completed commissioning",    "How many wells have completed commissioning?",                               must_not_contain="10632")


def run_sql_breakdowns(base: str) -> None:
    """Status breakdown queries."""
    section("5. SQL — STATUS BREAKDOWN")
    ask(base, "location prep breakdown",     "What is the breakdown of location preparation status?", must_contain="completed")
    ask(base, "flowline const breakdown",    "What is the flowline construction status breakdown?",   must_contain="completed")
    ask(base, "commissioning breakdown",     "What is the commissioning status breakdown?",           must_contain="completed")


def run_sql_lists(base: str) -> None:
    """Ranked, range, and filtered list queries."""
    section("6. SQL — RANKED LISTS")
    ask(base, "top 5 wells by progress",    "List the top 5 wells by overall progress",              must_contain="swer")
    ask(base, "top 10 wells by progress",   "Show the top 10 wells by overall progress percentage",  must_contain="swer")
    ask(base, "bottom 5 wells",             "List 5 wells with the lowest progress",                 must_contain="swer")

    section("7. SQL — PROGRESS RANGE LIST")
    ask(base, "wells below 50 pct",         "List wells with progress below 50 percent",             must_not_contain="no records")
    ask(base, "wells below 30 pct",         "Show wells with progress below 30 percent",             must_not_contain="no records")
    ask(base, "wells above 80 pct",         "List wells with progress above 80 percent",             must_not_contain="no records")

    section("8. SQL — WELL LIST WITH FILTER")
    ask(base, "list wells on SWER102",      "List all wells on rig SWER102",                         must_contain="swer102")
    ask(base, "list completed wells",       "List wells that have completed location preparation",   must_not_contain="no records")


def run_sql_detail(base: str) -> None:
    """Well detail and activity lookups."""
    section("9. SQL — WELL DETAIL LOOKUP")
    ask(base, "detail by PDO ID 34568",     "What is the well location of PDO WELL ID 34568?",       must_contain="nimr",  must_not_contain="not found")
    ask(base, "detail by PDO ID 36106",     "Show full details for PDO well ID 36106",                                      must_not_contain="not found")
    ask(base, "detail by well name",        "Show details for well AMIN_1673188_OP31",               must_contain="amin",  must_not_contain="not found")

    section("10. SQL — ACTIVITY MASTER")
    ask(base, "list civil activities",      "List all civil activities",                             must_contain="civil")
    ask(base, "list mechanical activities", "List all mechanical activities",                        must_contain="mechanical")
    ask(base, "activity by code",           "What is the production norm for activity C-001?",                            must_not_contain="does not exist")

    section("11. SQL — PER-RIG SUMMARY")
    ask(base, "rig summary all rigs",       "Show well count and average progress per rig",          must_contain="swer")
    ask(base, "highest average progress",   "Which rig has the highest average progress?",           must_contain="swer")


def run_rag(base: str) -> None:
    """Document knowledge (RAG) queries."""
    section("12. RAG — DOCUMENT KNOWLEDGE")
    ask(base, "M-90 milestone",             "What is the M-90 milestone?",                           must_contain="location", must_not_contain="not found in context")
    ask(base, "M-60 milestone",             "What happens at the M-60 milestone?")
    ask(base, "FLAF procedure",             "What is the FLAF and when is it issued?",                                         must_not_contain="not found in context")
    ask(base, "rig-on steps",               "What are the key steps before rig-on?",                                           must_not_contain="not found in context")
    ask(base, "buffer status explained",    "What does buffer status mean in well delivery?",                                   must_not_contain="not found in context")

    section("13. RAG — MUST NOT HIT SQL")
    ask(base, "scope question → RAG",       "What is the scope of the AL TASNIM contract?",                                    must_not_contain="sql error")
    ask(base, "procedure question → RAG",   "What are the commissioning sign-off requirements?",                               must_not_contain="sql error")


def run_edge_cases(base: str) -> None:
    """Boundary and error-handling cases."""
    section("14. EDGE CASES")
    ask(base, "unknown PDO ID",             "What is the well location of PDO WELL ID 99999?",                                 must_not_contain="does not exist")
    ask(base, "count with zero result",     "How many wells have no flowline progress?",                                       must_not_contain="does not exist")
    ask(base, "vague progress phrasing",    "Which rig has the highest average overall progress?",  must_contain="swer")


def print_summary() -> None:
    """Print pass/fail totals and list failures."""
    passed = [r for r in _results if r[1]]
    failed = [r for r in _results if not r[1]]
    total  = len(_results)
    avg_ms = sum(r[3] for r in _results if r[3] > 0) / max(total, 1)
    pct    = (len(passed) / total * 100) if total else 0
    color  = G if pct >= 85 else (Y if pct >= 65 else R)

    section("SUMMARY")
    print(f"  Total   : {total}")
    print(f"  {G}Passed  : {len(passed)}{E}")
    if failed:
        print(f"  {R}Failed  : {len(failed)}{E}")
    print(f"  Avg lat : {avg_ms:.0f}ms")
    print(f"\n  {color}{B}Score: {len(passed)}/{total} ({pct:.0f}%){E}")

    if failed:
        print(f"\n  {R}Failed tests:{E}")
        for label, _, reason, _ in failed:
            print(f"    ✗  {label}")
            print(f"       {reason}")
    print()


def main() -> None:
    """Entry point — parse args, run all test groups, print summary."""
    parser = argparse.ArgumentParser(description="AL TASNIM RAG test suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the RAG server")
    base = parser.parse_args().url.rstrip("/")

    print(f"\n{B}AL TASNIM RAG — Test Suite{E}")
    print(f"Server : {base}\n")

    if not run_health(base):
        sys.exit(1)

    run_sql_counts(base)
    run_sql_breakdowns(base)
    run_sql_lists(base)
    run_sql_detail(base)
    run_rag(base)
    run_edge_cases(base)
    print_summary()

    sys.exit(0 if all(r[1] for r in _results) else 1)


if __name__ == "__main__":
    main()
