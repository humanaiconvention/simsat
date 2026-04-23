from __future__ import annotations

from encounter.schemas import (
    EncounterDecision,
    EncounterFeatures,
    EncounterGeometry,
    EncounterProbeResult,
    EncounterRecord,
    EncounterWindow,
)
from encounter.store import EncounterStore
from haic.schemas import ConventionSession, GroundingStimulus, SessionStatus, StimulusImage, StimulusType
from haic.session_store import SessionStore
from haic.stimulus_store import StimulusStore


def _sample_stimulus() -> GroundingStimulus:
    stimulus = GroundingStimulus(
        stimulus_id="stim-1",
        created_at="2026-03-10T12:00:00Z",
        satellite_position=[1.0, 2.0, 500.0],
        simulation_timestamp="2026-03-10T12:00:00Z",
        images=[StimulusImage(stimulus_type=StimulusType.SENTINEL_RGB, image_b64="ZmFrZQ==")],
        location_description="Test site",
        observation_context="Regression fixture",
    )
    stimulus.compute_hash()
    return stimulus


def test_encounter_store_round_trip(tmp_path):
    store = EncounterStore(tmp_path)
    record = EncounterRecord(
        decision=EncounterDecision(
            decision_id="dec-1",
            window_id="window-1",
            target_id="target-1",
            created_at="2026-03-10T12:00:30Z",
            action="accept",
            effective_action="accept",
            combined_score=0.81,
            trust_score=0.76,
            trust_band="high",
        ),
        window=EncounterWindow(
            window_id="window-1",
            target_id="target-1",
            encounter_type="imaging_window",
            start_time="2026-03-10T12:00:00Z",
            end_time="2026-03-10T12:00:20Z",
            peak_time="2026-03-10T12:00:10Z",
            satellite_position_peak=[0.0, 0.0, 500.0],
            target_position=[0.0, 0.0],
            geometry=EncounterGeometry(
                elevation_degrees=65.0,
                off_nadir_degrees=12.0,
                slant_range_km=540.0,
                target_visible=True,
                bearing=180.0,
                pitch=25.0,
            ),
            target_priority=0.8,
            duration_seconds=30.0,
        ),
        probe=EncounterProbeResult(
            window_id="window-1",
            target_id="target-1",
            mapbox_feasible=True,
            sentinel_available=True,
            sentinel_cloud_cover=12.0,
            sentinel_source="sentinel-2a",
            sentinel_datetime="2026-03-10T11:50:00Z",
        ),
        features=EncounterFeatures(
            window_id="window-1",
            target_id="target-1",
            duration_seconds=30.0,
            peak_elevation_degrees=65.0,
            off_nadir_degrees=12.0,
            slant_range_km=540.0,
            target_priority=0.8,
            mapbox_feasible=True,
            sentinel_available=True,
            sentinel_cloud_cover=12.0,
            score_components={"priority": 0.8},
        ),
    )

    store.save_record(record)

    reloaded = store.get_record("dec-1")
    listed = store.list_records(limit=5)

    assert reloaded is not None
    assert reloaded.decision.decision_id == "dec-1"
    assert reloaded.window.window_id == "window-1"
    assert listed[0].decision.decision_id == "dec-1"


def test_stimulus_store_round_trip(tmp_path):
    path = tmp_path / "stimuli.json"
    store = StimulusStore(str(path))
    stimulus = _sample_stimulus()

    store.put(stimulus)
    reloaded = StimulusStore(str(path))

    loaded = reloaded.get("stim-1")
    assert loaded is not None
    assert loaded.content_hash == stimulus.content_hash
    assert reloaded.list_recent(limit=1)[0].stimulus_id == "stim-1"


def test_session_store_round_trip(tmp_path):
    path = tmp_path / "sessions.json"
    store = SessionStore(str(path))
    stimulus = _sample_stimulus()
    session = ConventionSession(
        session_id="session-1",
        created_at="2026-03-10T12:00:00Z",
        status=SessionStatus.STIMULUS_READY,
        stimulus=stimulus,
        participant_id="participant-1",
        pog_verified=True,
    )

    store.create(session)
    reloaded = SessionStore(str(path))

    loaded = reloaded.get("session-1")
    assert loaded is not None
    assert loaded.status == SessionStatus.STIMULUS_READY
    assert loaded.stimulus is not None
    assert loaded.stimulus.stimulus_id == "stim-1"
