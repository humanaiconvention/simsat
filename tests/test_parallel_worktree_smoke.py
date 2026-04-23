from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RUNTIME_ROOT = Path(r"D:\SimSat")
FORK_ROOT = Path(r"D:\SimSat\worktrees\SimSat_fork")

SCRIPT = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
from encounter.features import FeatureBuilder
from encounter.planner import AnalyticPlanner
from encounter.schemas import EncounterGeometry, EncounterPolicy, EncounterProbeResult, EncounterWindow, TargetSpec
from encounter.trust_model import WCLITrustModel

policy = EncounterPolicy()
window = EncounterWindow(
    window_id="window-1",
    target_id="target-1",
    encounter_type="imaging_window",
    start_time="2026-03-10T12:00:00Z",
    end_time="2026-03-10T12:01:00Z",
    peak_time="2026-03-10T12:00:30Z",
    satellite_position_peak=[0.0, 0.0, 500.0],
    target_position=[0.0, 0.0],
    geometry=EncounterGeometry(
        elevation_degrees=70.0,
        off_nadir_degrees=10.0,
        slant_range_km=530.0,
        target_visible=True,
        bearing=180.0,
        pitch=20.0,
    ),
    target_priority=0.9,
    duration_seconds=60.0,
)
target = TargetSpec(target_id="target-1", label="Target One", lon=0.0, lat=0.0, priority=0.9, size_km=5.0)
probe = EncounterProbeResult(
    window_id="window-1",
    target_id="target-1",
    mapbox_feasible=True,
    sentinel_available=True,
    sentinel_cloud_cover=5.0,
    sentinel_source="sentinel-2a",
    sentinel_datetime="2026-03-10T11:50:00Z",
)
features = FeatureBuilder().build(window, target, probe, policy)
decision = AnalyticPlanner(policy=policy, trust_model=WCLITrustModel(policy)).decide(window, features)
print(json.dumps({
    "action": decision.action,
    "trust_band": decision.trust_band,
    "reason_codes": decision.reason_codes,
}))
"""


def _run_core_smoke(src_sim_path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(src_sim_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip())


def test_runtime_and_fork_core_smoke_match():
    runtime_src = RUNTIME_ROOT / "src" / "sim"
    fork_src = FORK_ROOT / "src" / "sim"

    runtime = _run_core_smoke(runtime_src)
    fork = _run_core_smoke(fork_src)

    assert runtime["action"] == "accept"
    assert runtime["action"] == fork["action"]
    assert runtime["trust_band"] == fork["trust_band"]
    assert runtime["reason_codes"] == fork["reason_codes"]
