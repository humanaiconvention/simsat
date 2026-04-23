from __future__ import annotations

from pathlib import Path

from encounter.store import EncounterStore
from haic.session_store import SessionStore
from haic.stimulus_store import StimulusStore
from observation_vla.store import ObservationStore


def test_observation_store_handles_malformed_json_gracefully(tmp_path):
    base = tmp_path / "observation_vla"
    base.mkdir(parents=True, exist_ok=True)
    for filename in ["assessments.json", "traces.json", "outcomes.json", "memory.json", "submission_cases.json"]:
        (base / filename).write_text("{bad json", encoding="utf-8")

    store = ObservationStore(base)

    assert store.list_records(limit=5) == []
    assert store.list_traces(limit=5) == []
    assert store.list_outcomes(limit=5) == []
    assert store.list_memory_states(limit=5) == []
    assert store.list_submission_cases() == []


def test_encounter_store_defaults_when_payload_files_are_missing_or_partial(tmp_path):
    base = tmp_path / "encounter"
    base.mkdir(parents=True, exist_ok=True)
    (base / "records.json").write_text("{}", encoding="utf-8")
    (base / "evaluations.json").write_text("{}", encoding="utf-8")

    store = EncounterStore(base)

    assert store.list_records(limit=5) == []
    assert store.list_evaluations(limit=5) == []
    assert store.get_record("missing") is None


def test_stimulus_and_session_store_fall_back_to_empty_on_corruption(tmp_path):
    stimulus_path = tmp_path / "stimuli.json"
    session_path = tmp_path / "sessions.json"
    stimulus_path.write_text("{broken", encoding="utf-8")
    session_path.write_text("{broken", encoding="utf-8")

    stimulus_store = StimulusStore(str(stimulus_path))
    session_store = SessionStore(str(session_path))

    assert stimulus_store.list_recent(limit=5) == []
    assert stimulus_store.get("missing") is None
    assert session_store.list_all() == []
    assert session_store.get("missing") is None
