from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from challenge_offline_batch import build_manifest_notes
from challenge_run_manifest import ChallengeRunManifest, ChallengeRunManifestStore


def test_manifest_store_writes_to_configured_base_dir(tmp_path):
    store = ChallengeRunManifestStore(tmp_path / "challenge_runs")
    manifest = ChallengeRunManifest(
        scenario_policy="competition",
        scenario_hours={"maritime_chokepoints": 48.0},
        lanes=["matrix"],
    )

    store.save(manifest)

    assert store.sqlite_path.exists()
    assert (store.base_dir / f"{manifest.run_id}.json").exists()
    saved = store.list_recent(limit=1)[0]
    assert saved["run_id"] == manifest.run_id


def test_build_manifest_notes_captures_repro_metadata():
    notes = build_manifest_notes(
        policy="competition",
        scenario_hours={"maritime_chokepoints": 48.0},
        lanes=["matrix", "readiness"],
        outputs={"matrix": "CHALLENGE_TEST_STATUS.md"},
    )

    assert notes["offline_transport"] is True
    assert notes["policy"] == "competition"
    assert notes["output_count"] == 1
    assert notes["output_names"] == ["matrix"]
