from __future__ import annotations

from encounter.features import FeatureBuilder
from encounter.materialize import StimulusMaterializer
from encounter.planner import AnalyticPlanner
from encounter.schemas import EncounterGeometry, EncounterPolicy, EncounterProbeResult, EncounterWindow, EphemerisPoint, TargetSpec
from encounter.service import EncounterService
from encounter.store import EncounterStore
from encounter.targets import TargetRepository
from encounter.trust_model import WCLITrustModel
from encounter.windows import WindowDetector
from haic.schemas import StimulusType
from haic.stimulus_store import StimulusStore


class FakeEphemerisService:
    def propagate_series(self, start_time, hours, step_seconds):
        return [
            EphemerisPoint(timestamp="2026-03-10T12:00:00Z", satellite_position=[0.0, 0.0, 500.0]),
            EphemerisPoint(timestamp="2026-03-10T12:00:30Z", satellite_position=[0.0, 0.0, 500.0]),
        ]


class FakeProbeService:
    def probe(self, window, target):
        return EncounterProbeResult(
            window_id=window.window_id,
            target_id=target.target_id,
            mapbox_feasible=True,
            sentinel_available=True,
            sentinel_cloud_cover=5.0,
            sentinel_source="sentinel-2a",
            sentinel_datetime="2026-03-10T11:50:00Z",
        )


class FakeWindowDetector(WindowDetector):
    def enumerate_windows(self, ephemeris_points, targets):
        return [
            EncounterWindow(
                window_id=f"window-{target.target_id}",
                target_id=target.target_id,
                encounter_type="imaging_window",
                start_time="2026-03-10T12:00:00Z",
                end_time="2026-03-10T12:01:00Z",
                peak_time="2026-03-10T12:00:30Z",
                satellite_position_peak=[0.0, 0.0, 500.0],
                target_position=[target.lon, target.lat],
                geometry=EncounterGeometry(
                    elevation_degrees=70.0,
                    off_nadir_degrees=10.0,
                    slant_range_km=530.0,
                    target_visible=True,
                    bearing=180.0,
                    pitch=20.0,
                ),
                target_priority=target.priority,
                duration_seconds=60.0,
            )
            for target in targets
        ]


class FakeBridge:
    def __init__(self):
        self.calls = []

    def build_stimulus_at(self, **kwargs):
        from haic.schemas import GroundingStimulus, StimulusImage

        self.calls.append(kwargs)
        return GroundingStimulus(
            stimulus_id="stim-1",
            simulation_timestamp=kwargs["timestamp"],
            satellite_position=[kwargs["sat_lon"], kwargs["sat_lat"], kwargs["sat_alt"]],
            images=[StimulusImage(stimulus_type=StimulusType.SENTINEL_RGB, image_b64="ZmFrZQ==")],
            observation_context="Fake stimulus",
        )


def _encounter_service(tmp_path):
    policy = EncounterPolicy()
    target_repo = TargetRepository(tmp_path / "targets.json")
    target_repo.replace_targets(
        [TargetSpec(target_id="target-1", label="Target One", lon=0.0, lat=0.0, priority=0.9, metadata={"scenario_pack": "scope-1"})]
    )
    store = EncounterStore(tmp_path / "encounter_store")
    return EncounterService(
        shared_data={"last_updated": "2026-03-10T12:00:00Z"},
        target_repo=target_repo,
        ephemeris=FakeEphemerisService(),
        window_detector=FakeWindowDetector(),
        probe_service=FakeProbeService(),
        feature_builder=FeatureBuilder(),
        planner=AnalyticPlanner(policy=policy, trust_model=WCLITrustModel(policy)),
        store=store,
        materializer=type("NoopMaterializer", (), {"materialize": lambda *args, **kwargs: None})(),
        policy=policy,
        observation_vla=None,
    )


def test_plan_persists_records_and_artifacts(tmp_path):
    service = _encounter_service(tmp_path)

    records = service.plan(
        start_time="2026-03-10T12:00:00Z",
        hours=1.0,
        step_seconds=30,
        top_k=5,
        scenario_pack="scope-1",
    )

    assert len(records) == 1
    record = records[0]
    assert record.artifact is not None
    assert record.decision.artifact_id == record.artifact.artifact_id
    assert record.artifact.action == record.decision.action
    assert record.artifact.merkle_root
    saved = service.store.get_record(record.decision.decision_id)
    assert saved is not None
    assert saved.decision.decision_id == record.decision.decision_id
    assert service.list_decisions(limit=5)[0].artifact_id == record.artifact.artifact_id


def test_materializer_uses_fake_bridge_and_persists_stimulus(tmp_path):
    from encounter.schemas import EncounterDecision, EncounterFeatures, EncounterRecord, EncounterWindow

    bridge = FakeBridge()
    stimulus_store = StimulusStore(str(tmp_path / "stimuli.json"))
    materializer = StimulusMaterializer(bridge=bridge, stimulus_store=stimulus_store)
    record = EncounterRecord(
        decision=EncounterDecision(decision_id="dec-1", target_id="target-1"),
        window=EncounterWindow.from_dict(
            {
                "window_id": "window-1",
                "target_id": "target-1",
                "encounter_type": "imaging_window",
                "start_time": "2026-03-10T12:00:00Z",
                "end_time": "2026-03-10T12:01:00Z",
                "peak_time": "2026-03-10T12:00:30Z",
                "satellite_position_peak": [1.0, 2.0, 500.0],
                "target_position": [3.0, 4.0],
                "geometry": {
                    "elevation_degrees": 60.0,
                    "off_nadir_degrees": 10.0,
                    "slant_range_km": 530.0,
                    "target_visible": True,
                },
            }
        ),
        probe=EncounterProbeResult(window_id="window-1", target_id="target-1", mapbox_feasible=True),
        features=EncounterFeatures(
            window_id="window-1",
            target_id="target-1",
            duration_seconds=60.0,
            peak_elevation_degrees=60.0,
            off_nadir_degrees=10.0,
            slant_range_km=530.0,
            target_priority=0.9,
            mapbox_feasible=True,
            sentinel_available=True,
            sentinel_cloud_cover=5.0,
        ),
    )
    target = TargetSpec(target_id="target-1", label="Target One", lon=3.0, lat=4.0, size_km=6.0)

    stimulus = materializer.materialize(record, target, persist=True)

    assert bridge.calls[0]["lon"] == 3.0
    assert bridge.calls[0]["sat_lon"] == 1.0
    assert stimulus.images[0].stimulus_type == StimulusType.SENTINEL_RGB
    assert stimulus_store.get("stim-1") is not None
