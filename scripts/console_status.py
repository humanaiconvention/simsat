"""
BEAST Console Status — what actually happened since you were last here.

Reads agent logs, security patrol, supervisor relay, convention-hall inbox,
mapper/librarian activity. Surfaces observations, not infrastructure state.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────
# HAIC sister-repo path. Override via HAIC_REPO_PATH for non-default layouts.
HAIC          = Path(os.environ.get("HAIC_REPO_PATH", str(Path.home() / "humanai-convention")))
AGENTS        = HAIC / "agents"
STAMP_FILE    = Path("/tmp/beast_last_open")

# ── ANSI ───────────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"
W     = 62
SEP   = f"{DIM}{'-' * W}{RESET}"


# ── Stamp ──────────────────────────────────────────────────────────────────

def _read_stamp() -> datetime:
    try:
        return datetime.fromtimestamp(float(STAMP_FILE.read_text()), tz=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=24)

def _write_stamp():
    try:
        STAMP_FILE.write_text(str(time.time()))
    except Exception:
        pass

def _ago(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    h = int(delta.total_seconds() // 3600)
    m = int((delta.total_seconds() % 3600) // 60)
    if h > 48: return f"{h//24}d ago"
    if h:      return f"{h}h {m}m ago"
    return f"{m}m ago"

def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# ── Security ───────────────────────────────────────────────────────────────

def _security_report(since: datetime) -> list[str]:
    lines = []
    log = AGENTS / "haic-security" / "workspace" / "logs" / "patrol.log"
    try:
        raw = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return lines

    # Deduplicate consent warnings — report count, not each UUID
    consent_missing: set[str] = set()
    detections: list[str] = []
    blocks = 0
    warns = 0
    last_stats: dict = {}
    restarts = 0

    for line in raw:
        ts_m = re.match(r"\[(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\]", line)
        if not ts_m:
            continue
        ts = _parse_ts(ts_m.group(1))
        if ts and ts < since:
            continue

        if "worker-started" in line:
            restarts += 1
        elif "lattice-missing-consent" in line:
            sid_m = re.search(r"session_id=([a-f0-9-]+)", line)
            if sid_m:
                consent_missing.add(sid_m.group(1)[:8])
        elif ("blocked" in line.lower() or "inject" in line.lower() or "poison" in line.lower()) \
                and "shutdown" not in line.lower():
            detections.append(line.split("] ", 1)[-1].strip())
        elif "shutdown" in line and "stats=" in line:
            stats_m = re.search(r"stats=(\{.+\})", line)
            if stats_m:
                try:
                    last_stats = json.loads(stats_m.group(1))
                    blocks += last_stats.get("blocks", 0)
                    warns  += last_stats.get("warns", 0)
                except Exception:
                    pass

    if blocks > 0:
        lines.append(f"{RED}Security: {blocks} block(s) — review patrol.log{RESET}")
    if warns > 0:
        lines.append(f"{YELLOW}Security: {warns} warning(s){RESET}")
    if consent_missing:
        lines.append(f"{YELLOW}Security: {len(consent_missing)} lattice session(s) missing consent_hash{RESET}")
    for d in detections[:3]:
        lines.append(f"{RED}  detection: {d[:70]}{RESET}")
    if not lines:
        lines.append(f"{DIM}Security: clean — no detections, no blocks{RESET}")
    return lines


# ── Supervisor ────────────────────────────────────────────────────────────

def _supervisor_report(since: datetime) -> list[str]:
    lines = []
    log = AGENTS / "supervisor" / "logs" / "supervisor.log"
    try:
        raw = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return lines

    events: list[str] = []
    fleet_snapshots: list[tuple[datetime, int, int]] = []  # (ts, live, paused)

    for line in raw:
        ts_m = re.match(r"\[(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\]", line)
        if not ts_m:
            continue
        ts = _parse_ts(ts_m.group(1))
        if ts and ts < since:
            continue

        if "envoy-briefed" in line:
            pass  # captured via fleet_snapshots; skip duplicate narrative
        elif "relay-forwarded" in line:
            from_m = re.search(r"from=(\S+)", line)
            to_m   = re.search(r"to=(\S+)", line)
            if from_m and to_m:
                events.append(f"Relay: {from_m.group(1).replace('haic-','')} -> {to_m.group(1).replace('haic-','')}")
        elif "relay-blocked" in line or "relay-rejected" in line:
            events.append(f"{RED}Relay blocked: {line.split('] ',1)[-1].strip()[:60]}{RESET}")
        elif re.search(r"sync \|.*live=(\d+).*paused=(\d+)", line):
            m = re.search(r"live=(\d+).*paused=(\d+)", line)
            if m and ts:
                fleet_snapshots.append((ts, int(m.group(1)), int(m.group(2))))

    # Summarise fleet peak from snapshots
    if fleet_snapshots:
        peak_live = max(s[1] for s in fleet_snapshots)
        latest    = fleet_snapshots[-1]
        lines.append(f"{DIM}Supervisor: peak {peak_live} agents live, now {latest[1]} live / {latest[2]} paused{RESET}")

    for e in events[:5]:
        lines.append(f"  {DIM}{e}{RESET}")

    return lines


# ── Mapper ─────────────────────────────────────────────────────────────────

def _mapper_report(since: datetime) -> list[str]:
    lines = []
    log = AGENTS / "haic-mapper" / "workspace" / "logs" / "worker.log"
    try:
        raw = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return lines

    prompts_done = 0
    yielded_to: set[str] = set()
    last_active = None

    for line in raw:
        ts_m = re.match(r"\[(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\]", line)
        if not ts_m:
            continue
        ts = _parse_ts(ts_m.group(1))
        if ts and ts < since:
            continue
        if "prompt-complete" in line or "map-updated" in line:
            prompts_done += 1
            last_active = ts
        elif "compute-busy" in line:
            yt_m = re.search(r"yield_to=([^\|]+)", line)
            if yt_m:
                for a in yt_m.group(1).split(","):
                    yielded_to.add(a.strip().replace("haic-", ""))

    if prompts_done:
        lines.append(f"Mapper: {prompts_done} mapping task(s) completed")
    elif yielded_to:
        agents = ", ".join(sorted(yielded_to)[:4])
        lines.append(f"{DIM}Mapper: yielded to {agents} (idle — no mapping work done){RESET}")
    return lines


# ── Librarian ─────────────────────────────────────────────────────────────

def _librarian_report(since: datetime) -> list[str]:
    lines = []
    log = AGENTS / "haic-librarian" / "workspace" / "logs" / "worker.log"
    try:
        raw = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return lines

    papers_moved = 0
    deferred = False

    for line in raw:
        ts_m = re.match(r"\[(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\]", line)
        if not ts_m:
            continue
        ts = _parse_ts(ts_m.group(1))
        if ts and ts < since:
            continue
        if "paper-classified" in line or "paper-moved" in line or "batch-complete" in line:
            papers_moved += 1
        elif "compute-busy" in line:
            deferred = True

    if papers_moved:
        lines.append(f"Librarian: {papers_moved} article(s) classified/moved")
    elif deferred:
        lines.append(f"{DIM}Librarian: deferred — higher-priority agents were active{RESET}")
    return lines


# ── Convention Hall ────────────────────────────────────────────────────────

def _convention_report(since: datetime) -> list[str]:
    lines = []
    inbox = AGENTS / "convention-hall" / "workspace" / "INBOX.md"
    try:
        raw = inbox.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return lines

    sessions: list[str] = []
    for block in re.split(r"\n(?=\{)", raw):
        try:
            msg = json.loads(block.strip())
            ts = _parse_ts(msg.get("timestamp", ""))
            if ts and ts >= since:
                frm   = msg.get("from", "?").replace("haic-", "")
                topic = msg.get("content", "")[:60].replace("\n", " ")
                sessions.append(f"{frm}: {topic}")
        except Exception:
            pass

    if sessions:
        lines.append(f"Convention Hall: {len(sessions)} message(s)")
        for s in sessions[:3]:
            lines.append(f"  {DIM}{s}{RESET}")
    return lines


# ── Envoy ──────────────────────────────────────────────────────────────────

def _envoy_report(since: datetime) -> list[str]:
    lines = []
    log = AGENTS / "haic-envoy" / "workspace" / "logs" / "worker.log"
    try:
        raw = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return lines

    contacts = 0
    for line in raw:
        ts_m = re.match(r"\[(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\]", line)
        if not ts_m:
            continue
        ts = _parse_ts(ts_m.group(1))
        if ts and ts < since:
            continue
        if "outreach" in line or "contact" in line or "response-sent" in line:
            contacts += 1

    if contacts:
        lines.append(f"Envoy: {contacts} outreach/contact event(s)")
    return lines


# ── Pending inboxes ────────────────────────────────────────────────────────

def _pending_inboxes() -> list[str]:
    """Find agent inboxes with unprocessed content."""
    lines = []
    for agent_dir in sorted(AGENTS.iterdir()):
        inbox = agent_dir / "workspace" / "INBOX.md"
        if not inbox.exists():
            continue
        try:
            text = inbox.read_text(encoding="utf-8", errors="replace")
            # Look for JSON blocks that look like unprocessed relay messages
            pending = re.findall(r'"type":\s*"relay"', text)
            if pending:
                name = agent_dir.name.replace("haic-", "")
                lines.append(f"{YELLOW}  {name}: {len(pending)} pending relay(s){RESET}")
        except Exception:
            pass
    return lines


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    last_open = _read_stamp()
    _write_stamp()

    now   = datetime.now(timezone.utc)
    delta = now - last_open
    h, m  = int(delta.total_seconds() // 3600), int((delta.total_seconds() % 3600) // 60)
    since_str = f"{h}h {m}m ago" if h else f"{m}m ago"

    print()
    print(SEP)
    print(f"{BOLD}  BEAST  {RESET}{DIM}{now.strftime('%a %d %b  %H:%M UTC')}  |  since {since_str}{RESET}")
    print(SEP)

    any_output = False

    # Security
    sec = _security_report(last_open)
    if sec:
        print(f"\n{BOLD}  Security{RESET}")
        for l in sec: print(f"  {l}")
        any_output = True

    # Supervisor
    sup = _supervisor_report(last_open)
    if sup:
        print(f"\n{BOLD}  Supervisor{RESET}")
        for l in sup: print(f"  {l}")
        any_output = True

    # Convention Hall
    conv = _convention_report(last_open)
    if conv:
        print(f"\n{BOLD}  Convention Hall{RESET}")
        for l in conv: print(f"  {l}")
        any_output = True

    # Mapper
    mapper = _mapper_report(last_open)
    if mapper:
        print(f"\n{BOLD}  Mapper{RESET}")
        for l in mapper: print(f"  {l}")
        any_output = True

    # Librarian
    lib = _librarian_report(last_open)
    if lib:
        print(f"\n{BOLD}  Librarian{RESET}")
        for l in lib: print(f"  {l}")
        any_output = True

    # Envoy
    env = _envoy_report(last_open)
    if env:
        print(f"\n{BOLD}  Envoy{RESET}")
        for l in env: print(f"  {l}")
        any_output = True

    # Pending inbox relays
    pending = _pending_inboxes()
    if pending:
        print(f"\n{BOLD}  Pending{RESET}")
        for l in pending: print(l)
        any_output = True

    if not any_output:
        print(f"\n  {DIM}No agent activity since last open.{RESET}")

    print()
    print(SEP)
    print()


if __name__ == "__main__":
    main()
