"""
SimSat API Watchdog — auto-starts and restarts the simulator API on port 9005.

Run once per console session (or at system startup):
    python scripts/watchdog_sim.py

The watchdog:
  1. Checks if the API is already running on port 9005.
  2. Spawns it if not, using a shared Manager dict.
  3. Polls every POLL_INTERVAL seconds; respawns on crash.
  4. Exits cleanly on Ctrl+C.

Environment variables forwarded to the API process:
  HAIC_PRISM_MODE     synthetic (default) or full
  HAIC_EPSILON        entropy reduction threshold (default 0.01)
  ANTHROPIC_API_KEY   for live Claude interviewer (optional)
  MAPBOX_ACCESS_TOKEN for Mapbox imagery (optional)
  SIM_PORT            API port (default 9005)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

POLL_INTERVAL = 5       # seconds between health checks
STARTUP_GRACE = 10      # seconds to wait after spawn before first health check
PORT = int(os.environ.get("SIM_PORT", "9005"))
HEALTH_URL = f"http://localhost:{PORT}/"

# ---- locate repo root (two levels up from this file) ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
SIM_DIR     = os.path.join(REPO_ROOT, "src", "sim")

_proc: subprocess.Popen | None = None


def _is_up() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _spawn() -> subprocess.Popen:
    """Launch the simulator API as a subprocess."""
    env = {**os.environ}
    env.setdefault("HAIC_PRISM_MODE", "synthetic")
    env.setdefault("PYTHONPATH", SIM_DIR)

    # Inline bootstrap — sets up shared data and starts uvicorn on SIM_PORT
    bootstrap = f"""
import sys, os, multiprocessing
sys.path.insert(0, {SIM_DIR!r})
os.environ.setdefault("HAIC_PRISM_MODE", "synthetic")
import api as api_module
from api import api
import uvicorn

manager = multiprocessing.Manager()
shared_data_dict = manager.dict()
shared_data_dict["satellite_position"] = (0.0, 0.0, 550.0)
shared_data_dict["last_updated"] = "startup"
api.state.shared_data = shared_data_dict

uvicorn.run(api, host="0.0.0.0", port={PORT})
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", bootstrap],
        cwd=SIM_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"[watchdog] spawned API process PID={proc.pid} on port {PORT}")
    return proc


def _drain(proc: subprocess.Popen) -> None:
    """Print any buffered output from the subprocess (non-blocking)."""
    if proc.stdout is None:
        return
    import select
    import io
    # On Windows select() doesn't work on pipes; use a reader thread instead
    # (output is printed via the thread started in _spawn_with_reader)


def main() -> None:
    global _proc
    print(f"[watchdog] SimSat API watchdog starting — target port {PORT}")
    print(f"[watchdog] Repo:    {REPO_ROOT}")
    print(f"[watchdog] Sim dir: {SIM_DIR}")
    print(f"[watchdog] Press Ctrl+C to stop.\n")

    restart_count = 0

    try:
        while True:
            # --- check if already up externally ---
            if _is_up():
                # still alive — just wait
                time.sleep(POLL_INTERVAL)
                continue

            # --- check if our subprocess is still alive ---
            if _proc is not None and _proc.poll() is None:
                # Process running but not responding — give it a bit more time
                time.sleep(POLL_INTERVAL)
                continue

            # --- (re)spawn ---
            if _proc is not None:
                retcode = _proc.poll()
                print(f"[watchdog] Process exited (code={retcode}), restarting...")
                restart_count += 1
            else:
                print(f"[watchdog] API not running, starting...")

            _proc = _spawn()

            # Wait for startup grace period, checking health
            deadline = time.time() + STARTUP_GRACE
            while time.time() < deadline:
                time.sleep(1)
                if _is_up():
                    print(f"[watchdog] API is up (restarts={restart_count})")
                    break
            else:
                print(f"[watchdog] WARNING: API didn't respond within {STARTUP_GRACE}s")

    except KeyboardInterrupt:
        print("\n[watchdog] Shutting down...")
        if _proc and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
        print("[watchdog] Done.")


if __name__ == "__main__":
    main()
