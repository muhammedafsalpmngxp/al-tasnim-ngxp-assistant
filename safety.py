"""
safety.py — SQL safety validation; only SELECT statements are permitted.
"""
import re

_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE|MERGE|GRANT|REVOKE|DENY|BULK)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate that a SQL string is safe to execute.

    Returns:
        (True, "OK")              — query is a plain SELECT with no forbidden ops
        (False, "<reason>")       — query failed validation
    """
    stripped = sql.strip()

    if not stripped:
        return False, "Empty SQL"

    if _BLOCKED.search(stripped):
        return False, "SQL contains forbidden operations (only SELECT is allowed)"

    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        return False, "Only SELECT statements are permitted"

    return True, "OK"
