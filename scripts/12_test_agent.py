"""
Test suite for the AL TASNIM Agentic AI module (agent/).

Tests are split into two groups:
  A) Unit tests  — no external services needed (router, clarification, validator)
  B) Integration — requires prod_rag.py running on localhost:8000

Run all tests (unit only, no services needed):
    conda activate v12
    python scripts/12_test_agent.py

Run with integration tests (start prod_rag.py first on port 8000):
    python scripts/12_test_agent.py --integration
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
import time
from typing import Callable, List, Tuple

sys.path.insert(0, ".")

# Force config reload between test runs
import agent.config as _acfg
_acfg.get_config.cache_clear()

import agent.router as _ar
_ar._STATE["patterns"] = {}

from agent.config import get_config
from agent.models import RouteType, UserRole
from agent.router import check_clarification, rule_based_route
from agent.validator import (
    check_human_review_needed,
    determine_confidence,
    validate_answer,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
results: List[Tuple[str, str, str]] = []  # (group, name, status)


def check(group: str, name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((group, name, status))
    marker = "✓" if condition else "✗"
    suffix = f"  ← {detail}" if (not condition and detail) else ""
    print(f"  [{marker}] {name}{suffix}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── A. Config ─────────────────────────────────────────────────────────────────

def test_config() -> None:
    section("Config loading")
    cfg = get_config()
    check("config", "server port is 8001",    cfg["server"]["port"] == 8001)
    check("config", "rag_service base_url",    "localhost" in cfg["rag_service"]["base_url"])
    check("config", "cache enabled",           cfg["cache"]["enabled"] is True)
    check("config", "cache ttl is 300s",       cfg["cache"]["ttl_seconds"] == 300)
    check("config", "rag_priority_patterns exist",
          bool(cfg["router"].get("rag_priority_patterns")))
    check("config", "clarification vague_patterns exist",
          bool(cfg["clarification"].get("vague_patterns")))
    check("config", "human_review trigger_patterns exist",
          bool(cfg["human_review"].get("trigger_patterns")))


# ── B. Clarification gate ─────────────────────────────────────────────────────

def test_clarification() -> None:
    section("Clarification gate")

    vague_cases = [
        ("When will it finish?",        True),
        ("Give me an update",           True),
        ("What's happening?",           True),
        ("Is it delayed?",              True),
    ]
    specific_cases = [
        ("When will well NIMR-31722 finish the flowline activity?", False),
        ("What is the progress of well 30750?",                     False),
        ("How many wells are on RIG-101?",                          False),
        ("What is the SOP for FLAF approval?",                      False),
        ("When will it finish? Well ID 31722",                      False),
    ]

    for q, expected in vague_cases:
        needs, _ = check_clarification(q)
        check("clarification", f"vague: {q[:45]!r}", needs == expected,
              f"got {needs}, expected {expected}")

    for q, expected in specific_cases:
        needs, _ = check_clarification(q)
        check("clarification", f"specific: {q[:45]!r}", needs == expected,
              f"got {needs}, expected {expected}")

    # Clarification text is non-empty when needed
    needs, text = check_clarification("When will it finish?")
    check("clarification", "clarification text is non-empty", bool(text.strip()))


# ── C. Rule-based router ──────────────────────────────────────────────────────

def test_router() -> None:
    section("Rule-based router")

    cases = [
        # SQL queries
        ("How many wells are delayed?",                  RouteType.SQL),
        ("List all oil producer wells",                  RouteType.SQL),
        ("Show KPI counts for Nimr and Marmul",          RouteType.SQL),
        ("Current status of RIG-102",                    RouteType.SQL),
        ("Which rig has the most wells?",                RouteType.SQL),
        ("Total wells per rig breakdown",                RouteType.SQL),
        # RAG queries
        ("What is the SOP for handover?",                RouteType.RAG),
        ("What is the FLAF procedure?",                  RouteType.RAG),
        ("How to carry out FLAF commissioning?",         RouteType.RAG),
        ("Explain the process for rig-off",              RouteType.RAG),
        ("What was discussed in the Phase 1 meeting?",   RouteType.RAG),
        # Analytics
        ("When will the ALBRG cluster finish?",          RouteType.ANALYTICS),
        ("Forecast completion date for Nimr",            RouteType.ANALYTICS),
        ("What is the delay risk for cluster ALBRG?",    RouteType.ANALYTICS),
        # Multi (SQL + RAG)
        ("Why is well 30750 behind schedule?",           RouteType.MULTI),
        ("Root cause of the delay in flowline activity", RouteType.MULTI),
        # Recommend
        ("What should we do to recover the Nimr delay?", RouteType.RECOMMEND),
        ("Recommend actions for crew allocation",         RouteType.RECOMMEND),
        ("How can we speed up progress on cluster ALBRG?", RouteType.RECOMMEND),
    ]

    for q, expected in cases:
        route = rule_based_route(q)
        check("router", f"{expected.value}: {q[:50]!r}",
              route == expected, f"got {route}")


# ── D. Validator ──────────────────────────────────────────────────────────────

def test_validator() -> None:
    section("Evidence validator")

    # Valid answers
    long_sql_answer = "Well 30750 is currently at 45% overall progress on RIG-101."
    ok, reason = validate_answer(long_sql_answer, ["db"], "sql", "operations")
    check("validator", "valid SQL answer accepted", ok, reason)

    long_rag_answer = "The FLAF approval procedure requires the PDO engineer to sign off."
    ok, reason = validate_answer(long_rag_answer, ["doc.pdf"], "rag", "operations")
    check("validator", "valid RAG answer accepted", ok, reason)

    # Invalid answers
    ok, _ = validate_answer("I don't know.", [], "sql", "operations")
    check("validator", "hallucination phrase rejected", not ok)

    ok, _ = validate_answer("", [], "sql", "operations")
    check("validator", "empty answer rejected", not ok)

    ok, _ = validate_answer("No.", [], "sql", "operations")
    check("validator", "too-short answer rejected", not ok)

    # Confidence scoring
    sql_result_good = {"rows": [{"well": "A"}], "error": None}
    sql_result_bad  = {"rows": [], "error": "Connection refused"}
    rag_result_good = {"answer": "The FLAF procedure is documented in section 4.2.", "error": None}

    c = determine_confidence("sql",   sql_result_good, None)
    check("validator", "SQL with rows → High confidence", c == "High", c)

    c = determine_confidence("rag",   None, rag_result_good)
    check("validator", "RAG with answer → High confidence", c == "High", c)

    c = determine_confidence("sql",   sql_result_bad, None)
    check("validator", "SQL with error → Low confidence", c == "Low", c)

    c = determine_confidence("multi", sql_result_good, rag_result_good)
    check("validator", "multi with both sources → Medium", c == "Medium", c)

    # Human review triggers
    needs, reason = check_human_review_needed(
        "Can we approve overnight shift for RIG-101?", "High", "recommend", ""
    )
    check("validator", "night shift approval triggers review", needs, reason)

    needs, _ = check_human_review_needed(
        "How many wells are on RIG-101?", "High", "sql", ""
    )
    check("validator", "routine SQL query does NOT trigger review", not needs)

    needs, _ = check_human_review_needed(
        "What is the progress?", "Low", "sql", ""
    )
    check("validator", "Low confidence triggers review", needs)


# ── E. Integration (requires prod_rag.py on :8000) ────────────────────────────

async def test_integration() -> None:
    section("Integration — full agent pipeline (requires prod_rag.py on :8000)")

    from agent.orchestrator import run_agent

    int_cases = [
        {
            "query":    "How many wells are on each rig? Give a per-rig breakdown.",
            "role":     UserRole.OPERATIONS.value,
            "check_fn": lambda r: bool(r.get("direct_answer")),
            "label":    "per-rig well count",
            "expected_route": "sql",
        },
        {
            "query":    "What was discussed in the Phase 1 review meeting?",
            "role":     UserRole.OPERATIONS.value,
            "check_fn": lambda r: bool(r.get("direct_answer")),
            "label":    "Phase 1 transcript RAG",
            "expected_route": "rag",
        },
        {
            "query":    "When will it finish?",
            "role":     UserRole.OPERATIONS.value,
            "check_fn": lambda r: r.get("clarification_needed") is True,
            "label":    "vague query → clarification gate",
            "expected_route": "clarification",
        },
        {
            "query":    "What is the FLAF approval procedure?",
            "role":     UserRole.OPERATIONS.value,
            "check_fn": lambda r: bool(r.get("direct_answer")),
            "label":    "FLAF procedure → RAG",
            "expected_route": "rag",
        },
    ]

    for case in int_cases:
        t0 = time.monotonic()
        result = await run_agent(
            query      = case["query"],
            session_id = "test_integration",
            user_role  = case["role"],
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        ok = case["check_fn"](result)
        route_ok = (
            result.get("route", "").lower() == case["expected_route"]
            or case["expected_route"] == "any"
        )
        check("integration", f"{case['label']} (route={result.get('route')}  {elapsed}ms)",
              ok and route_ok,
              f"answer={result.get('direct_answer', '')[:80]!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true",
                        help="Also run integration tests (needs prod_rag.py on :8000)")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  AL TASNIM Agentic AI — Test Suite")
    print("═" * 60)

    test_config()
    test_clarification()
    test_router()
    test_validator()

    if args.integration:
        asyncio.run(test_integration())

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = len(results)
    passed  = sum(1 for _, _, s in results if s == PASS)
    failed  = total - passed
    groups: dict = {}
    for g, n, s in results:
        groups.setdefault(g, {"pass": 0, "fail": 0})
        groups[g]["pass" if s == PASS else "fail"] += 1

    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    for g, counts in groups.items():
        bar = "✓" * counts["pass"] + "✗" * counts["fail"]
        print(f"  {g:<20} {counts['pass']:>3} pass  {counts['fail']:>2} fail  [{bar}]")
    print(f"\n  Total: {passed}/{total} passed", end="")
    if failed:
        print(f"  ← {failed} FAILED")
        sys.exit(1)
    else:
        print("  — ALL PASSED ✓")


if __name__ == "__main__":
    main()
