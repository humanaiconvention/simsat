"""
SimSat Console Status — morning/session summary printed on console open.

Queries the live API and reads the access log to report:
  - API health
  - Sessions since last console open (or last 24h)
  - Any external (non-localhost) visits
  - PRISM health
  - Upcoming observation windows

Run by the SimSat Pipeline .bashrc block on startup.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PORT      = int(os.environ.get("SIM_PORT", "9005"))
BASE_URL  = f"http://localhost:{PORT}"
LOG_FILE  = os.environ.get("SIMSAT_ACCESS_LOG", "/tmp/simsat_access.log")
STAMP_FILE = "/tmp/simsat_last_open"

# ANSI
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
RESET  = "\033[0m"
SEP    = f"{DIM}{'-'*54}{RESET}"


def _get(path: str, timeout: int = 4):
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _read_last_open() -> datetime:
    """Return timestamp of last console open, defaulting to 24h ago."""
    try:
        ts = float(Path(STAMP_FILE).read_text().strip())
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=24)


def _write_stamp() -> None:
    try:
        Path(STAMP_FILE).write_text(str(time.time()))
    except Exception:
        pass


def _parse_access_log(since: datetime) -> dict:
    """Parse the uvicorn access log written by the watchdog."""
    stats = {
        "total_requests": 0,
        "external_visits": 0,
        "haic_sessions": 0,
        "haic_turns": 0,
        "receipts": 0,
        "external_ips": set(),
    }
    try:
        with open(LOG_FILE) as f:
            for line in f:
                # uvicorn log format: INFO:     <ip>:<port> - "METHOD /path HTTP/1.1" STATUS
                if " - \"" not in line:
                    continue
                stats["total_requests"] += 1
                ip = line.split("INFO:")[1].strip().split(":")[0] if "INFO:" in line else ""
                if ip and ip not in ("127.0.0.1", "::1", "0.0.0.0", ""):
                    stats["external_visits"] += 1
                    stats["external_ips"].add(ip)
                if "POST /haic/session" in line and "/turn" not in line and "/close" not in line:
                    stats["haic_sessions"] += 1
                if "/turn" in line and "POST" in line:
                    stats["haic_turns"] += 1
                if "/receipt" in line and "GET" in line:
                    stats["receipts"] += 1
    except FileNotFoundError:
        pass
    return stats


def main() -> None:
    # Read + update stamp before waiting for API
    last_open = _read_last_open()
    _write_stamp()

    # Wait briefly for watchdog to bring API up
    api_up = False
    for _ in range(12):
        h = _get("/haic/health")
        if h:
            api_up = True
            break
        time.sleep(1)

    now = datetime.now(timezone.utc)
    elapsed = now - last_open
    hours = int(elapsed.total_seconds() // 3600)
    mins  = int((elapsed.total_seconds() % 3600) // 60)
    since_str = f"{hours}h {mins}m ago" if hours else f"{mins}m ago"

    print(SEP)
    print(f"{BOLD}  SimSat Pipeline{RESET}  {DIM}{now.strftime('%a %d %b %Y  %H:%M UTC')}{RESET}")
    print(SEP)

    # API status
    if api_up:
        pos = _get("/data/current/position")
        pos_str = ""
        if pos and "position" in pos:
            p = pos["position"]
            pos_str = f"  sat {p[1]:.1f}°N {p[0]:.1f}°E  {p[2]:.0f}km"
        print(f"  {GREEN}● API online{RESET}{DIM}{pos_str}{RESET}")
        prism_str = f"  PRISM {h.get('prism','?')}" if h else ""
        print(f"  {DIM}bridge {h.get('bridge','?')}{prism_str}{RESET}")
    else:
        print(f"  {RED}● API not responding{RESET}")

    # Since-last-open summary
    print(f"\n  {BOLD}Since last open{RESET}  {DIM}({since_str}){RESET}")

    log_stats = _parse_access_log(last_open)

    if not api_up and log_stats["total_requests"] == 0:
        print(f"  {DIM}No activity — pipeline was offline{RESET}")
    else:
        sessions_total = 0
        if api_up:
            sess_data = _get("/haic/sessions")
            if sess_data:
                sessions_total = sess_data.get("total", 0)

        if log_stats["external_visits"] > 0:
            ips = ", ".join(sorted(log_stats["external_ips"]))
            print(f"  {YELLOW}⚡ {log_stats['external_visits']} external visit(s) from: {ips}{RESET}")
        else:
            print(f"  {DIM}No external visits{RESET}")

        if log_stats["haic_sessions"] > 0:
            print(f"  {CYAN}[+] {log_stats['haic_sessions']} convention session(s) created{RESET}")
            print(f"  {CYAN}    {log_stats['haic_turns']} interview turn(s)  "
                  f"{log_stats['receipts']} receipt(s) issued{RESET}")
        else:
            print(f"  {DIM}No HAIC sessions{RESET}")

        if sessions_total > 0:
            print(f"  {DIM}{sessions_total} session(s) in live store{RESET}")

    # Next observation window
    if api_up:
        wins = _get("/haic/windows?count=1")
        if wins and wins.get("windows"):
            w = wins["windows"][0]
            print(f"\n  {DIM}Next window  {w.get('start_time','?')[:16].replace('T',' ')}Z"
                  f"  {w.get('region_description','')}{RESET}")

    print(SEP)
    print()


if __name__ == "__main__":
    main()
