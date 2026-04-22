"""
SimSat API Watchdog — auto-starts and restarts the simulator API on port 9005.

Normally launched automatically by the SimSat Pipeline terminal profile.
Can also be run directly:
    python scripts/watchdog_sim.py

The watchdog:
  1. Checks if the API is already running on port 9005.
  2. Spawns it if not.
  3. Polls every POLL_INTERVAL seconds; respawns on crash.
  4. Handles SIGTERM cleanly — kills the child API process and exits.
     (The SimSat Pipeline .bashrc trap sends SIGTERM on console close.)

Environment variables forwarded to the API process:
  HAIC_PRISM_MODE     synthetic (default) or full
  HAIC_EPSILON        entropy reduction threshold (default 0.01)
  ANTHROPIC_API_KEY   for live Claude interviewer (optional)
  MAPBOX_ACCESS_TOKEN for Mapbox imagery (optional)
  SIM_PORT            API port (default 9005)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

POLL_INTERVAL = 5       # seconds between health checks
STARTUP_GRACE = 15      # seconds to wait after spawn before first health check
PORT          = int(os.environ.get("SIM_PORT", "9005"))
HEALTH_URL    = f"http://localhost:{PORT}/"

# ---- locate repo root ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
SIM_DIR     = os.path.join(REPO_ROOT, "src", "sim")

_proc: subprocess.Popen | None = None


# ---- Health check ----

def _is_up() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ---- Process spawn ----

def _spawn() -> subprocess.Popen:
    """
    Launch the simulator API as a child process.

    Uses a regular dict for shared_data (not multiprocessing.Manager) because
    the watchdog runs only the API — no separate sim process needs IPC.
    A plain dict is sufficient and avoids multiprocessing spawn issues on Windows.
    """
    env = {**os.environ}
    env.setdefault("HAIC_PRISM_MODE", "synthetic")

    # Write a small launcher script to a temp file so multiprocessing
    # freeze_support works correctly (avoids re-import loops with -c on Windows)
    import tempfile, textwrap
    access_log = os.environ.get("SIMSAT_ACCESS_LOG", "/tmp/simsat_access.log")

    launcher = textwrap.dedent(f"""
        import sys, os
        sys.path.insert(0, {SIM_DIR!r})
        os.environ.setdefault("HAIC_PRISM_MODE", "synthetic")

        import multiprocessing
        multiprocessing.freeze_support()

        if __name__ == "__main__":
            import api as api_module
            from api import api
            import uvicorn
            import logging

            # Append-mode file handler for access log (persists across restarts)
            access_log_path = {access_log!r}
            file_handler = logging.FileHandler(access_log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(levelname)-8s  %(message)s"))
            logging.getLogger("uvicorn.access").addHandler(file_handler)

            shared_data_dict = {{
                "satellite_position": (0.0, 0.0, 550.0),
                "last_updated": "watchdog-start",
            }}
            api.state.shared_data = shared_data_dict
            uvicorn.run(api, host="0.0.0.0", port={PORT}, log_level="warning")
    """)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_simsat_api.py", delete=False
    )
    tmp.write(launcher)
    tmp.flush()
    tmp.close()

    proc = subprocess.Popen(
        [sys.executable, tmp.name],
        cwd=SIM_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[watchdog] spawned API (PID={proc.pid}) on :{PORT}", flush=True)
    return proc


# ---- Shutdown handler ----

def _shutdown(signum=None, frame=None) -> None:
    global _proc
    print("\n[watchdog] received shutdown — stopping API...", flush=True)
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    print("[watchdog] exiting.", flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown)
try:
    signal.signal(signal.SIGBREAK, _shutdown)   # Windows Ctrl+Break
except AttributeError:
    pass


# ---- Main loop ----

def main() -> None:
    global _proc
    print(f"[watchdog] SimSat pipeline  port={PORT}  pid={os.getpid()}", flush=True)

    restart_count = 0

    try:
        while True:
            if _is_up():
                time.sleep(POLL_INTERVAL)
                continue

            if _proc is not None and _proc.poll() is None:
                # Running but not yet responding — keep waiting
                time.sleep(POLL_INTERVAL)
                continue

            if _proc is not None:
                print(f"[watchdog] API exited (code={_proc.poll()}) — restarting...", flush=True)
                restart_count += 1
            else:
                print("[watchdog] starting API...", flush=True)

            _proc = _spawn()

            deadline = time.time() + STARTUP_GRACE
            while time.time() < deadline:
                time.sleep(1)
                if _is_up():
                    print(f"[watchdog] API ready  (restarts={restart_count})", flush=True)
                    break
            else:
                print(f"[watchdog] WARNING: API unresponsive after {STARTUP_GRACE}s", flush=True)

    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
