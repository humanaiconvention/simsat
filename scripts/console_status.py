"""
BEAST Console Status — daily briefing and live updates on console open.

Reads from:
  - D:/humanai-convention/agents/control-plane/summary.json  (agent fleet state)
  - D:/humanai-convention/STATUS.md                          (project snapshot)
  - D:/humanai-convention/HANDOFF.md                        (recent changes)
  - D:/humanai-convention/agents/*/workspace/               (agent-specific data)
  - /tmp/simsat_access.log                                   (SimSat traffic)
  - SimSat API on :9005                                      (if running)
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
HAIC_ROOT    = Path("D:/humanai-convention")
AGENTS_DIR   = HAIC_ROOT / "agents"
SUMMARY_FILE = AGENTS_DIR / "control-plane" / "summary.json"
STATUS_FILE  = HAIC_ROOT / "STATUS.md"
HANDOFF_FILE = HAIC_ROOT / "HANDOFF.md"
STAMP_FILE   = Path("/tmp/beast_last_open")
ACCESS_LOG   = Path(os.environ.get("SIMSAT_ACCESS_LOG", "/tmp/simsat_access.log"))

SIM_PORT     = int(os.environ.get("SIM_PORT", "9005"))
SIM_URL      = f"http://localhost:{SIM_PORT}"

# ── ANSI ───────────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
W      = 60
SEP    = f"{DIM}{'-' * W}{RESET}"


# ── Helpers ────────────────────────────────────────────────────────────────

def _get(path: str, timeout: int = 3):
    try:
        with urllib.request.urlopen(f"{SIM_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _read_stamp() -> datetime:
    try:
        ts = float(STAMP_FILE.read_text().strip())
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=24)


def _write_stamp():
    try:
        STAMP_FILE.write_text(str(time.time()))
    except Exception:
        pass


def _time_ago(dt_str: str) -> str:
    """Convert ISO timestamp to human 'Xh ago'."""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        h = int(delta.total_seconds() // 3600)
        m = int((delta.total_seconds() % 3600) // 60)
        if h > 48:
            return f"{h//24}d ago"
        if h:
            return f"{h}h {m}m ago"
        return f"{m}m ago"
    except Exception:
        return "?"


def _md_first_section(path: Path, section: str) -> str:
    """Extract the first content paragraph after a heading containing `section`."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        in_section = False
        buf = []
        for line in lines:
            if re.match(r"^#{1,3}\s+.*" + re.escape(section), line, re.IGNORECASE):
                in_section = True
                continue
            if in_section:
                if re.match(r"^#{1,3}\s+", line) and buf:
                    break
                if line.strip():
                    buf.append(line.strip())
                    if len(buf) >= 3:
                        break
        return " ".join(buf)
    except Exception:
        return ""


def _handoff_today(last_open: datetime) -> list[str]:
    """Extract bullet points added to HANDOFF.md since last_open."""
    try:
        text = HANDOFF_FILE.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        items = []
        in_recent = False
        for line in lines:
            if "What Changed" in line or "This Session" in line:
                in_recent = True
                continue
            if in_recent:
                if re.match(r"^#{1,3}\s+", line) and items:
                    break
                if line.strip().startswith(("- ", "* ", "+ ")):
                    items.append(line.strip().lstrip("-*+ ").strip())
        return items[:8]
    except Exception:
        return []


def _simsat_log_stats() -> dict:
    stats = {"total": 0, "external": 0, "sessions": 0, "turns": 0, "receipts": 0, "ext_ips": set()}
    try:
        for line in ACCESS_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            if " - \"" not in line:
                continue
            stats["total"] += 1
            ip_part = line.split("INFO:")[1].strip().split(":")[0] if "INFO:" in line else ""
            if ip_part and ip_part not in ("127.0.0.1", "::1", "0.0.0.0", ""):
                stats["external"] += 1
                stats["ext_ips"].add(ip_part)
            if "POST /haic/session" in line and "/turn" not in line and "/close" not in line:
                stats["sessions"] += 1
            if "/turn" in line and "POST" in line:
                stats["turns"] += 1
            if "/receipt" in line and "GET" in line:
                stats["receipts"] += 1
    except FileNotFoundError:
        pass
    return stats


# ── Agent fleet ────────────────────────────────────────────────────────────

STATE_ICON = {
    "live":            f"{GREEN}*{RESET}",
    "working":         f"{CYAN}>{RESET}",
    "active":          f"{GREEN}+{RESET}",
    "paused":          f"{DIM}~{RESET}",
    "stopped":         f"{DIM}.{RESET}",
    "shutting-down":   f"{YELLOW}v{RESET}",
    "error":           f"{RED}!{RESET}",
}

AGENTS_OF_INTEREST = [
    "haic-supervisor",
    "haic-dispatch",
    "haic-envoy",
    "haic-security",
    "haic-maestro",
    "haic-prism",
    "haic-librarian",
    "haic-mapper",
]


def _load_fleet() -> list[dict]:
    try:
        return json.loads(SUMMARY_FILE.read_text(encoding="utf-8", errors="replace")).get("agents", [])
    except Exception:
        return []


def _agent_row(a: dict) -> str:
    state  = a.get("state", "?")
    icon   = STATE_ICON.get(state, f"{DIM}?{RESET}")
    name   = a["id"].replace("haic-", "").title().replace("-", " ")
    task   = a.get("current_task", "").strip()
    # Clean up garbled unicode from the encoding issue in summary.json
    # Normalise garbled bytes that appear when the JS agent writes latin1-encoded em-dashes
    task   = task.encode("latin-1", "replace").decode("utf-8", "replace")
    task   = re.sub(r"[\ufffd\x00-\x08\x0b-\x1f]", "", task).strip()
    # Truncate
    task   = (task[:36] + "..") if len(task) > 38 else task
    ago    = _time_ago(a["updated_at"]) if a.get("updated_at") else ""
    return f"  {icon} {BOLD}{name:<14}{RESET} {DIM}{task:<38}{RESET}  {DIM}{ago}{RESET}"


# ── Security agent details ─────────────────────────────────────────────────

def _security_summary(agent: dict) -> str:
    task = agent.get("current_task", "")
    # Parse "scans:N blocks:N warns:N escalations:N" from task string
    m = re.search(r"scans:(\d+).*?blocks:(\d+).*?warns:(\d+)", task)
    if m:
        scans, blocks, warns = m.group(1), m.group(2), m.group(3)
        color = RED if int(blocks) > 0 or int(warns) > 0 else DIM
        return f"  {color}Security  scans={scans}  blocks={blocks}  warns={warns}{RESET}"
    return ""


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    last_open = _read_stamp()
    _write_stamp()

    now      = datetime.now(timezone.utc)
    delta    = now - last_open
    h, m     = int(delta.total_seconds() // 3600), int((delta.total_seconds() % 3600) // 60)
    since    = f"{h}h {m}m ago" if h else f"{m}m ago"

    fleet    = _load_fleet()
    fleet_by_id = {a["id"]: a for a in fleet}

    # Count live/working
    live_n   = sum(1 for a in fleet if a.get("state") in ("live", "working", "active"))
    total_n  = len(fleet)

    print()
    print(SEP)
    print(f"{BOLD}  BEAST Console{RESET}  {DIM}{now.strftime('%a %d %b %Y  %H:%M UTC')}{RESET}")
    print(f"  {DIM}last open {since}{RESET}")
    print(SEP)

    # ── Agent fleet ──────────────────────────────────────────────────────
    print(f"\n{BOLD}  Agents  {DIM}({live_n} active / {total_n} total){RESET}")

    for aid in AGENTS_OF_INTEREST:
        a = fleet_by_id.get(aid)
        if a:
            print(_agent_row(a))

    # Security inline summary
    sec = fleet_by_id.get("haic-security")
    if sec:
        s = _security_summary(sec)
        if s:
            print(s)

    # ── What changed (HANDOFF) ───────────────────────────────────────────
    changes = _handoff_today(last_open)
    if changes:
        print(f"\n{BOLD}  Recent changes{RESET}  {DIM}(HANDOFF.md){RESET}")
        for item in changes[:5]:
            # Condense — show first line only
            short = item.split(" -- ")[0].split(": ", 1)[-1]
            short = (short[:55] + "..") if len(short) > 57 else short
            print(f"  {DIM}- {short}{RESET}")

    # ── SimSat pipeline ──────────────────────────────────────────────────
    print(f"\n{BOLD}  SimSat Pipeline{RESET}")
    sim_health = _get("/haic/health")
    if sim_health:
        pos = _get("/data/current/position")
        pos_str = ""
        if pos and "position" in pos:
            p = pos["position"]
            pos_str = f"  {p[1]:.1f}N {p[0]:.1f}E {p[2]:.0f}km"
        print(f"  {GREEN}* online{RESET}{DIM}{pos_str}  PRISM {sim_health.get('prism','?')}{RESET}")
        log = _simsat_log_stats()
        if log["external"] > 0:
            ips = ", ".join(sorted(log["ext_ips"]))
            print(f"  {YELLOW}! {log['external']} external visit(s): {ips}{RESET}")
        if log["sessions"] > 0:
            print(f"  {CYAN}  {log['sessions']} HAIC sessions  {log['turns']} turns  {log['receipts']} receipts{RESET}")
        wins = _get("/haic/windows?count=1")
        if wins and wins.get("windows"):
            w = wins["windows"][0]
            wt = w.get("start_time", "")[:16].replace("T", " ")
            print(f"  {DIM}next window {wt}Z  {w.get('region_description','')}{RESET}")
    else:
        print(f"  {DIM}. offline (watchdog will start on pipeline open){RESET}")

    # ── Project snapshot (one line from STATUS.md) ───────────────────────
    try:
        status_lines = STATUS_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in status_lines[3:12]:
            line = line.strip()
            if line.startswith("- **") and "active model" in line.lower():
                model = re.search(r"\*\*(.+?)\*\*", line)
                if model:
                    print(f"\n  {DIM}Model  {model.group(1)}{RESET}")
                break
    except Exception:
        pass

    # ── Training pipeline status ──────────────────────────────────────────
    try:
        training = json.loads((AGENTS_DIR / "control-plane" / "training-status.json").read_text(encoding="utf-8"))
        current  = training.get("lattice_sessions", "?")
        target   = training.get("pipeline_threshold", "?")
        model    = training.get("next_version", training.get("current_model", "?"))
        ready    = training.get("pipeline_ready", False)
        bar_len  = 20
        try:
            filled = int(bar_len * int(current) / int(target))
        except Exception:
            filled = 0
        bar   = f"{'#' * filled}{'.' * (bar_len - filled)}"
        color = CYAN if ready else DIM
        print(f"  {color}Training  [{bar}] {current}/{target} sessions  next={model}{RESET}")
    except Exception:
        pass

    print()
    print(SEP)
    print()


if __name__ == "__main__":
    main()
