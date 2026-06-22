#!/usr/bin/env python3
"""
AL TASNIM — RSR / SCR Auto-Email Watcher.

Polls a local inbox folder for new Rig Sequence Reports (RSR) and Sequence
Change Reports (SCR) dropped by PDO, then auto-emails all stakeholders.

RSR = Rig Sequence Report  — PDO communicates the sequence of rig moves
SCR = Sequence Change Report — PDO changes an existing rig sequence

MVP: watches data/RSR_SCR_INBOX/ as a stand-in for the Autodesk ACC folder.
Production: configure an ACC webhook → call check_once() from the handler.

Run continuously:
    conda activate v12
    python scripts/rsr_scr_watcher.py

Run once (for testing / webhook handler):
    from scripts.rsr_scr_watcher import check_once
    check_once()
"""
import fnmatch
import json
import os
import shutil
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

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

HERE = Path(__file__).parent.parent
CFG  = HERE / "config" / "rsr_scr_config.yaml"

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "altasnim-alerts@ngxptechnologies.com")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expand_env(value: str) -> str:
    import re
    return re.sub(r'\$\{(\w+)\}', lambda m: os.getenv(m.group(1), ""), str(value))


def _load_manifest(processed_dir: Path) -> set[str]:
    manifest_path = processed_dir / "processed.json"
    if manifest_path.exists():
        try:
            return set(json.loads(manifest_path.read_text()))
        except Exception:
            pass
    return set()


def _save_manifest(processed_dir: Path, seen: set[str]) -> None:
    manifest_path = processed_dir / "processed.json"
    manifest_path.write_text(json.dumps(sorted(seen), indent=2))


def _resolve_recipients(cfg: dict) -> dict[str, list[str]]:
    """Return {role: [email, ...]} for all configured recipients."""
    raw = cfg.get("recipients", {})
    result: dict[str, list[str]] = {}
    for role, val in raw.items():
        expanded = _expand_env(val)
        emails = [e.strip() for e in expanded.split(",") if e.strip() and "@" in e]
        if emails:
            result[role] = emails
    return result


def _all_recipient_emails(recipients: dict[str, list[str]]) -> list[str]:
    seen: list[str] = []
    for emails in recipients.values():
        for e in emails:
            if e not in seen:
                seen.append(e)
    return seen


def _send_email(to_addrs: list[str], subject: str, body: str) -> bool:
    if not SMTP_HOST or not to_addrs:
        print(f"  [rsr_scr] (no SMTP) Would email {len(to_addrs)} recipients:")
        print(f"    Subject: {subject}")
        print(f"    Body preview: {body[:200]}")
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
        return True
    except Exception as e:
        print(f"  [rsr_scr] Email failed: {e}")
        return False


def _detect_doc_type(filename: str) -> str:
    upper = filename.upper()
    if upper.startswith("RSR"):
        return "RSR"
    if upper.startswith("SCR"):
        return "SCR"
    return "RSR/SCR"


def _process_file(fpath: Path, cfg: dict, processed_dir: Path) -> bool:
    filename  = fpath.name
    doc_type  = _detect_doc_type(filename)
    size_kb   = round(fpath.stat().st_size / 1024, 1)
    received  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    recipients_map = _resolve_recipients(cfg)
    all_emails     = _all_recipient_emails(recipients_map)
    email_cfg      = cfg.get("email", {})

    subject = _expand_env(email_cfg.get(
        "subject_template",
        "[AL TASNIM] New {doc_type} received — {filename}",
    )).format(doc_type=doc_type, filename=filename)

    body_intro = _expand_env(email_cfg.get("body_intro", "")).format(
        doc_type=doc_type, filename=filename,
    )

    recipient_lines = "\n".join(
        f"  {role.replace('_', ' ').title()}: {', '.join(emails)}"
        for role, emails in recipients_map.items()
    )

    body = (
        f"{body_intro}\n\n"
        f"Document Details\n"
        f"{'─'*40}\n"
        f"  File     : {filename}\n"
        f"  Type     : {doc_type}\n"
        f"  Received : {received}\n"
        f"  Size     : {size_kb} KB\n\n"
        f"Notified Parties\n"
        f"{'─'*40}\n"
        f"{recipient_lines}\n\n"
        "Action Required\n"
        "─────────────────────────────────────\n"
        "  Please download the document from the shared drive,\n"
        "  review the rig sequence changes, and update your plans\n"
        "  accordingly. Coordinate with the Construction Manager\n"
        "  for any permit or civil implications.\n\n"
        "─────────────────────────────────────────────────────────\n"
        "This is an automated notification from the AL TASNIM\n"
        "Intelligence System. Do not reply to this email.\n"
    )

    sent = _send_email(all_emails, subject, body)
    n = len(all_emails)
    print(f"  [rsr_scr] NEW {doc_type}: {filename} → {'emailed' if sent else 'logged'} {n} recipients")

    # Move to processed folder
    dest = processed_dir / filename
    if dest.exists():
        dest = processed_dir / f"{fpath.stem}_{received.replace(':', '').replace(' ', '_')}{fpath.suffix}"
    shutil.move(str(fpath), str(dest))
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_once() -> int:
    """Scan the inbox once. Returns number of new files processed."""
    if not CFG.exists():
        print(f"[rsr_scr] Config not found: {CFG}")
        return 0

    with CFG.open() as f:
        cfg = yaml.safe_load(f) or {}

    watch_dir     = HERE / cfg.get("watch_folder", "data/RSR_SCR_INBOX")
    processed_dir = HERE / cfg.get("processed_folder", "data/RSR_SCR_PROCESSED")
    patterns      = cfg.get("patterns", ["RSR*.pdf", "RSR*.xlsx", "SCR*.pdf", "SCR*.xlsx"])

    watch_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    seen = _load_manifest(processed_dir)
    processed = 0

    for fpath in watch_dir.iterdir():
        if not fpath.is_file():
            continue
        if fpath.name in seen:
            continue
        # Check against patterns (case-insensitive)
        matched = any(
            fnmatch.fnmatch(fpath.name.upper(), pat.upper())
            for pat in patterns
        )
        if not matched:
            continue

        try:
            _process_file(fpath, cfg, processed_dir)
            seen.add(fpath.name)
            processed += 1
        except Exception as e:
            print(f"  [rsr_scr] Error processing {fpath.name}: {e}")

    _save_manifest(processed_dir, seen)
    return processed


# ---------------------------------------------------------------------------
# Continuous loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not CFG.exists():
        print(f"ERROR: {CFG} not found.")
        raise SystemExit(1)

    with CFG.open() as f:
        _top_cfg = yaml.safe_load(f) or {}

    interval = int(_top_cfg.get("poll_interval_seconds", 60))
    watch    = HERE / _top_cfg.get("watch_folder", "data/RSR_SCR_INBOX")

    print(f"[rsr_scr] Watching {watch} every {interval}s …")
    print(f"[rsr_scr] SMTP: {'configured' if SMTP_HOST else 'not set (console mode)'}")

    try:
        while True:
            n = check_once()
            if n:
                print(f"[rsr_scr] {n} file(s) processed.")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[rsr_scr] Stopped.")
