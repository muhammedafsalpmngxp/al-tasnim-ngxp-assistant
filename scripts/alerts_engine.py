#!/usr/bin/env python3
"""
AL TASNIM — Deterministic Smart Alerts Engine.

Evaluates pre-defined SQL rules against the PostgreSQL database every 5 minutes
WITHOUT using an LLM. Fires alerts via email (SMTP) and logs to the alerts_log
table in PostgreSQL.

Run continuously:
    conda activate v12
    python scripts/alerts_engine.py

Run once (for testing):
    from scripts.alerts_engine import run_once
    run_once()
"""
import json
import os
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
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

HERE    = Path(__file__).parent.parent
CFG     = HERE / "config" / "alerts_config.yaml"
PG_URL  = os.getenv("PG_URL", "postgresql://abhay@/altasnim?host=/var/run/postgresql")

# SMTP settings
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "altasnim-alerts@ngxptechnologies.com")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expand_env(value: str) -> str:
    """Replace ${VAR} patterns with their env values."""
    import re
    return re.sub(r'\$\{(\w+)\}', lambda m: os.getenv(m.group(1), ""), str(value))


def _get_conn():
    conn = psycopg2.connect(PG_URL)
    conn.autocommit = True
    return conn


def _ensure_alerts_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id          SERIAL PRIMARY KEY,
                rule_id     TEXT NOT NULL,
                rule_name   TEXT,
                severity    TEXT,
                message     TEXT,
                row_count   INT,
                fired_at    TIMESTAMP DEFAULT NOW(),
                notified_via TEXT[]
            )
        """)


def _log_alert(conn, rule_id: str, rule_name: str, severity: str,
               message: str, row_count: int, notified_via: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO alerts_log
               (rule_id, rule_name, severity, message, row_count, notified_via)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (rule_id, rule_name, severity, message, row_count, notified_via),
        )


def _resolve_recipients(cfg: dict, notify_roles: list[str]) -> list[str]:
    recip_cfg = cfg.get("recipients", {})
    emails: list[str] = []
    for role in notify_roles:
        raw = _expand_env(recip_cfg.get(role, ""))
        for addr in raw.split(","):
            addr = addr.strip()
            if addr and "@" in addr:
                emails.append(addr)
    return list(dict.fromkeys(emails))  # dedup, preserve order


def _send_email(to_addrs: list[str], subject: str, body: str) -> bool:
    """Send email via SMTP. Returns True if sent, False if SMTP not configured."""
    if not SMTP_HOST or not to_addrs:
        print(f"  [alerts] (no SMTP) Would email {to_addrs}:\n    Subject: {subject}")
        return False
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"]    = SMTP_FROM
        msg["To"]      = ", ".join(to_addrs)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as srv:
            srv.starttls()
            if SMTP_USER:
                srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_FROM, to_addrs, msg.as_string())
        print(f"  [alerts] Email sent → {to_addrs}")
        return True
    except Exception as e:
        print(f"  [alerts] Email failed: {e}")
        return False


def _format_message(template: str, count: int, rows: list[tuple]) -> str:
    summary_parts = []
    for row in rows[:5]:
        summary_parts.append(str(row))
    summary = "; ".join(summary_parts) if summary_parts else "see alerts_log"
    return template.format(count=count, summary=summary)


# ---------------------------------------------------------------------------
# Core: evaluate one rule
# ---------------------------------------------------------------------------
def _evaluate_rule(conn, rule: dict, cfg: dict) -> bool:
    rule_id   = rule["id"]
    rule_name = rule.get("name", rule_id)
    severity  = rule.get("severity", "info").upper()
    sql       = rule.get("sql", "").strip()
    table     = rule.get("table", "")

    print(f"  [alerts] Checking [{severity}] {rule_name} …", end=" ", flush=True)

    if not sql:
        print("SKIP (no sql defined)")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except Exception as e:
        print(f"ERROR ({e})")
        return False

    count = len(rows)
    if count == 0:
        print("OK")
        return False

    message = _format_message(rule.get("message", "Alert fired."), count, rows)
    print(f"FIRED ({count} rows)")

    notify_roles = rule.get("notify_roles", [])
    recipients   = _resolve_recipients(cfg, notify_roles)
    notified_via: list[str] = ["in_app"]

    subject = f"[AL TASNIM ALERT] {severity} — {rule_name}"
    body = (
        f"AL TASNIM Automated Alert\n"
        f"{'='*50}\n\n"
        f"Rule     : {rule_name}\n"
        f"Severity : {severity}\n"
        f"Fired At : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Rows     : {count}\n\n"
        f"Message  : {message}\n\n"
        f"Top results:\n"
        + "\n".join(f"  {r}" for r in rows[:10])
        + "\n\n"
        "This is an automated alert from the AL TASNIM Intelligence System.\n"
        "Do not reply to this email."
    )

    if recipients and _send_email(recipients, subject, body):
        notified_via.append("email")

    _log_alert(conn, rule_id, rule_name, severity, message, count, notified_via)
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_once() -> dict:
    """Evaluate all rules once. Returns {fired: int, ok: int, errors: int}."""
    if not CFG.exists():
        print(f"[alerts] Config not found: {CFG}")
        return {"fired": 0, "ok": 0, "errors": 1}

    with CFG.open() as f:
        cfg = yaml.safe_load(f) or {}

    rules = cfg.get("rules", [])
    print(f"\n[alerts] Pass started — {len(rules)} rules | {datetime.now().isoformat()}")

    try:
        conn = _get_conn()
    except Exception as e:
        print(f"[alerts] DB connect failed: {e}")
        return {"fired": 0, "ok": 0, "errors": 1}

    _ensure_alerts_table(conn)

    stats = {"fired": 0, "ok": 0, "errors": 0}
    for rule in rules:
        try:
            fired = _evaluate_rule(conn, rule, cfg)
            stats["fired" if fired else "ok"] += 1
        except Exception as e:
            print(f"  [alerts] Rule '{rule.get('id')}' crashed: {e}")
            stats["errors"] += 1

    conn.close()
    print(f"[alerts] Pass complete — fired:{stats['fired']} ok:{stats['ok']} errors:{stats['errors']}\n")
    return stats


# ---------------------------------------------------------------------------
# Continuous loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not CFG.exists():
        print(f"ERROR: {CFG} not found. Create it from config/alerts_config.yaml.")
        raise SystemExit(1)

    with CFG.open() as f:
        _top_cfg = yaml.safe_load(f) or {}

    interval = int(_top_cfg.get("schedule_minutes", 5)) * 60
    print(f"[alerts] Starting — interval={interval}s | DB={PG_URL[:40]}…")

    try:
        while True:
            run_once()
            print(f"[alerts] Sleeping {interval}s …")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[alerts] Stopped.")
