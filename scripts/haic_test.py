"""
HAIC Integration Test — verifies the full convention session lifecycle
against a running SimSat simulator.

Usage:
    # With Docker running:
    python scripts/haic_test.py

    # Custom URL:
    python scripts/haic_test.py --url http://localhost:9005

    # Skip Mapbox (if MAPBOX_ACCESS_TOKEN not set):
    python scripts/haic_test.py --no-mapbox

    # Verbose output:
    python scripts/haic_test.py --verbose
"""

import argparse
import json
import sys
import time
from typing import Any, Dict

import requests

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---- Colour helpers ----
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg): print(f"  {GREEN}\u2713{RESET} {msg}")
def fail(msg): print(f"  {RED}\u2717{RESET} {msg}")
def info(msg): print(f"  {CYAN}\u00b7{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")
def section(msg): print(f"\n{BOLD}{msg}{RESET}")


class HAICTestClient:
    def __init__(self, base_url: str, verbose: bool = False):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 30
        self.verbose = verbose
        self.passed = 0
        self.failed = 0

    def get(self, path: str, **kwargs) -> requests.Response:
        r = self.session.get(f"{self.base}{path}", **kwargs)
        if self.verbose:
            info(f"GET {path} -> {r.status_code}")
        return r

    def post(self, path: str, json_body: Any = None, **kwargs) -> requests.Response:
        r = self.session.post(f"{self.base}{path}", json=json_body, **kwargs)
        if self.verbose:
            info(f"POST {path} -> {r.status_code}")
        return r

    def assert_ok(self, label: str, condition: bool, detail: str = "") -> bool:
        if condition:
            ok(label)
            self.passed += 1
        else:
            fail(f"{label}{' — ' + detail if detail else ''}")
            self.failed += 1
        return condition

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        if self.failed == 0:
            print(f"{GREEN}{BOLD}All {total} checks passed.{RESET}")
        else:
            print(f"{RED}{BOLD}{self.failed}/{total} checks failed.{RESET}")
        print(f"{'='*50}")
        return self.failed == 0


def run_tests(base_url: str, include_mapbox: bool, verbose: bool) -> bool:
    c = HAICTestClient(base_url, verbose)

    # ----------------------------------------------------------------
    section("1. Health check")
    # ----------------------------------------------------------------
    r = c.get("/")
    c.assert_ok("Simulator API online", r.status_code == 200)
    c.assert_ok("HAIC flag present", r.json().get("haic") == "enabled",
                f"got: {r.json()}")

    r = c.get("/haic/health")
    c.assert_ok("HAIC health endpoint", r.status_code == 200)
    health = r.json()
    c.assert_ok("Bridge initialised", health.get("bridge") in ("ready", "not_initialised"),
                str(health))
    info(f"Bridge: {health.get('bridge')}  PRISM: {health.get('prism')}")

    # ----------------------------------------------------------------
    section("2. Position endpoint")
    # ----------------------------------------------------------------
    r = c.get("/data/current/position")
    c.assert_ok("Position endpoint reachable", r.status_code == 200)
    pos_data = r.json()
    lon_lat_alt = pos_data.get("lon-lat-alt", [])
    c.assert_ok("Position has 3 components", len(lon_lat_alt) == 3,
                str(lon_lat_alt))
    info(f"Position: lon={lon_lat_alt[0]:.4f}  lat={lon_lat_alt[1]:.4f}  alt={lon_lat_alt[2]:.2f}km")

    # ----------------------------------------------------------------
    section("3. Stimulus build")
    # ----------------------------------------------------------------
    stimulus_req: Dict[str, Any] = {
        "include_sentinel": True,
        "include_mapbox": include_mapbox,
        "sentinel_bands": ["red", "green", "blue"],
        "size_km": 5.0,
    }
    r = c.post("/haic/stimulus", stimulus_req)
    c.assert_ok("Stimulus build 200", r.status_code == 200, str(r.text[:200]))
    stimulus = r.json()

    stimulus_id = stimulus.get("stimulus_id")
    c.assert_ok("Stimulus has ID", bool(stimulus_id))
    c.assert_ok("Stimulus has images list", isinstance(stimulus.get("images"), list))
    c.assert_ok("Stimulus has content_hash", bool(stimulus.get("content_hash")))

    images = stimulus.get("images", [])
    info(f"Stimulus {stimulus_id[:8]}…  images={len(images)}")
    for img in images:
        status = "✓" if img.get("has_image") else "—"
        size = f"{img.get('image_size_bytes', 0) // 1024}KB" if img.get("image_size_bytes") else "n/a"
        info(f"  [{status}] {img['stimulus_type']}  {size}")

    # ----------------------------------------------------------------
    section("4. Stimulus retrieval + image endpoint")
    # ----------------------------------------------------------------
    r = c.get(f"/haic/stimulus/{stimulus_id}")
    c.assert_ok("Stimulus retrieval 200", r.status_code == 200)
    c.assert_ok("Same stimulus_id returned", r.json().get("stimulus_id") == stimulus_id)

    # Try to fetch first available image
    available_idx = next(
        (i for i, img in enumerate(images) if img.get("has_image")), None
    )
    if available_idx is not None:
        r = c.get(f"/haic/stimulus/{stimulus_id}/image/{available_idx}")
        c.assert_ok(
            f"Image {available_idx} PNG served",
            r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image/png"),
            f"status={r.status_code} ct={r.headers.get('Content-Type')}",
        )
        info(f"Image size: {len(r.content) // 1024}KB")
    else:
        warn("No images available (Sentinel/Mapbox may be unavailable for current position)")

    # ----------------------------------------------------------------
    section("5. Session creation")
    # ----------------------------------------------------------------
    r = c.post("/haic/session", {"stimulus_id": stimulus_id, "auto_stimulus": False})
    c.assert_ok("Session creation 200", r.status_code == 200, str(r.text[:200]))
    session = r.json()

    session_id = session.get("session_id")
    c.assert_ok("Session has ID", bool(session_id))
    c.assert_ok(
        "Session status is stimulus_ready or pending",
        session.get("status") in ("stimulus_ready", "pending"),
        f"got: {session.get('status')}",
    )
    info(f"Session {session_id[:8]}…  status={session.get('status')}")

    # ----------------------------------------------------------------
    section("6. Interview turns")
    # ----------------------------------------------------------------
    turns = [
        "I can see a large landmass with distinct terrain features. There appear to be some mountainous regions in the upper portion of the image.",
        "The colour gradients suggest different land cover types — possibly agricultural fields in the lower section, with what looks like a river system running diagonally.",
        "What strikes me most is the scale. Those features that look like threads are probably roads or waterways that are hundreds of metres wide in reality.",
    ]

    last_response = None
    for i, turn_text in enumerate(turns):
        # Simulate typing telemetry
        start_ms = int(time.time() * 1000) - (len(turn_text) * 80)  # ~80ms per char
        end_ms = int(time.time() * 1000)
        # Synthetic keystroke intervals with natural variance
        import random
        intervals = [int(random.gauss(80, 20)) for _ in range(min(20, len(turn_text)))]
        intervals = [max(30, i) for i in intervals]

        r = c.post(f"/haic/session/{session_id}/turn", {
            "content": turn_text,
            "client_start_ms": start_ms,
            "client_end_ms": end_ms,
            "keystroke_intervals": intervals,
        })
        c.assert_ok(f"Turn {i+1} accepted", r.status_code == 200, str(r.text[:200]))
        if r.status_code == 200:
            result = r.json()
            last_response = result.get("assistant_response", "")
            info(f"Turn {i+1}: {turn_text[:50]}…")
            info(f"  → {last_response[:80]}…" if len(last_response) > 80 else f"  → {last_response}")

    c.assert_ok("Interviewer responded to last turn", bool(last_response))

    # ----------------------------------------------------------------
    section("7. Session state")
    # ----------------------------------------------------------------
    r = c.get(f"/haic/session/{session_id}")
    c.assert_ok("Session GET 200", r.status_code == 200)
    session_state = r.json()
    turn_count = len([t for t in session_state.get("interview_turns", []) if t["role"] == "user"])
    c.assert_ok(f"All {len(turns)} turns recorded", turn_count == len(turns), f"got {turn_count}")
    info(f"Status: {session_state.get('status')}  turns: {turn_count}")

    # ----------------------------------------------------------------
    section("8. Sessions list")
    # ----------------------------------------------------------------
    r = c.get("/haic/sessions")
    c.assert_ok("Sessions list 200", r.status_code == 200)
    sessions_list = r.json().get("sessions", [])
    our_session = next((s for s in sessions_list if s["session_id"] == session_id), None)
    c.assert_ok("Our session appears in list", our_session is not None)
    info(f"Total sessions: {r.json().get('total', 0)}")

    # ----------------------------------------------------------------
    section("9. Close session (PRISM + viability + receipt)")
    # ----------------------------------------------------------------
    r = c.post(f"/haic/session/{session_id}/close")
    c.assert_ok("Close session 200", r.status_code == 200, str(r.text[:300]))
    closed = r.json()

    final_status = closed.get("status")
    c.assert_ok(
        "Session closed (settled or failed)",
        final_status in ("settled", "failed", "settlement_pending", "prism_measured"),
        f"got: {final_status}",
    )

    pog = closed.get("pog_telemetry", {}) or {}
    pog_result = pog.get("result", {}) if isinstance(pog, dict) else {}
    pog_score = pog_result.get("provenance_score", 0)
    pog_verified = closed.get("pog_verified", False)
    c.assert_ok(f"PoG evaluated (score={pog_score:.3f})", pog_score > 0, "score was 0")
    info(f"PoG verified: {pog_verified}  score: {pog_score:.3f}")

    gates = closed.get("viability_gates", {})
    if gates:
        passed_gates = sum(1 for v in gates.values() if v)
        info(f"Viability gates: {passed_gates}/{len(gates)} passed")
        for gate, passed in gates.items():
            status_sym = "✓" if passed else "✗"
            info(f"  [{status_sym}] {gate}")

    # Entropy delta (synthetic mode will have values)
    delta = closed.get("entropy_delta")
    if delta:
        dse = delta.get("delta_spectral_entropy", 0)
        verified = delta.get("reduction_verified", False)
        info(f"Entropy delta ΔS={dse:+.4f}  verified={verified}")

    # ----------------------------------------------------------------
    section("10. Receipt")
    # ----------------------------------------------------------------
    r = c.get(f"/haic/session/{session_id}/receipt")
    if closed.get("receipt_merkle_root"):
        c.assert_ok("Receipt endpoint 200", r.status_code == 200, str(r.text[:200]))
        receipt_data = r.json()
        merkle_root = receipt_data.get("merkle_root")
        c.assert_ok("Merkle root present", bool(merkle_root), "no merkle_root in response")
        c.assert_ok("Merkle root is 64-char hex", len(merkle_root or "") == 64)
        info(f"Merkle root: {(merkle_root or '')[:24]}…")

        # Verify settlement_result in closed session
        sr = closed.get("settlement_result", {}) or {}
        leaves = sr.get("leaves", [])
        c.assert_ok("Receipt has 7 leaf commitments", len(leaves) == 7, f"got {len(leaves)}")
        c.assert_ok("All leaves are 64-char hex", all(len(l) == 64 for l in leaves))
        info(f"Summary: {sr.get('summary', '')[:120]}")
    else:
        warn("No receipt generated — session may have failed viability gates")
        c.assert_ok(
            "Session completed processing (settled or failed)",
            final_status in ("settled", "failed"),
            f"got: {final_status}",
        )

    # ----------------------------------------------------------------
    section("11. Observation windows")
    # ----------------------------------------------------------------
    r = c.get("/haic/windows?count=3")
    c.assert_ok("Windows endpoint 200", r.status_code == 200)
    windows = r.json().get("windows", [])
    c.assert_ok("Returns 3 windows", len(windows) == 3, f"got {len(windows)}")
    info(f"Next window: {windows[0].get('start_time', 'unknown')}")
    info(f"  Region: {windows[0].get('region_description', '?')}")

    return c.summary()


def main():
    parser = argparse.ArgumentParser(description="HAIC integration test")
    parser.add_argument("--url", default="http://localhost:9005", help="Simulator base URL")
    parser.add_argument("--no-mapbox", action="store_true", help="Skip Mapbox imagery")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose HTTP logging")
    args = parser.parse_args()

    print(f"{BOLD}SimSat HAIC Integration Test{RESET}")
    print(f"Target: {args.url}")
    print(f"Mapbox: {'disabled' if args.no_mapbox else 'enabled'}")

    # Check simulator is up
    try:
        r = requests.get(f"{args.url}/", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"\n{RED}Cannot reach simulator at {args.url}: {e}{RESET}")
        print("Make sure the simulator is running: docker-compose up  or  python src/sim/main.py")
        sys.exit(1)

    success = run_tests(
        base_url=args.url,
        include_mapbox=not args.no_mapbox,
        verbose=args.verbose,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
