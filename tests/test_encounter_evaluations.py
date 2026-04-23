from __future__ import annotations

from encounter.features import FeatureBuilder
from encounter.planner import AnalyticPlanner
from encounter.schemas import (
    EncounterGeometry,
    EncounterPolicy,
    EncounterProbeResult,
    EncounterWindow,
    EphemerisPoint,
    TargetSpec,
)
from encounter.service import EncounterService
from encounter.store import EncounterStore
from encounter.targets import TargetRepository
from encounter.trust_model import WCLITrustModel
from encounter.windows import WindowDetector


class FakeEphemerisService:
    def propagate_series(self, start_time, hours, step_seconds):
        return [
            EphemerisPoint(timestamp="2026-03-10T12:00:00Z", satellite_position=[0.0, 0.0, 500.0]),
            EphemerisPoint(timestamp="2026-03-10T12:00:30Z", satellite_position=[0.0, 0.0, 500.0]),
        ]


class FakeProbeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def probe(self, window: EncounterWindow, target: TargetSpec) -> EncounterProbeResult:
        self.calls.append((window.window_id, target.target_id))
        if target.target_id == "target-high":
            return EncounterProbeResult(
                window_id=window.window_id,
                target_id=target.target_id,
                mapbox_feasible=True,
                sentinel_available=True,
                sentinel_cloud_cover=5.0,
                sentinel_source="sentinel-2a",
                sentinel_datetime="2026-03-10T11:50:00Z",
            )
        return EncounterProbeResult(
            window_id=window.window_id,
            target_id=target.target_id,
            mapbox_feasible=False,
            sentinel_available=False,
            sentinel_cloud_cover=90.0,
        )


class FakeWindowDetector(WindowDetector):
    def enumerate_windows(self, ephemeris_points, targets):
        windows: list[EncounterWindow] = []
        for target in targets:
            if target.target_id == "target-high":
                windows.append(
                    EncounterWindow(
                        window_id="window-high",
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
                )
            else:
                windows.append(
                    EncounterWindow(
                        window_id="window-low",
                        target_id=target.target_id,
                        encounter_type="imaging_window",
                        start_time="2026-03-10T12:00:00Z",
                        end_time="2026-03-10T12:01:00Z",
                        peak_time="2026-03-10T12:00:30Z",
                        satellite_position_peak=[0.0, 0.0, 500.0],
                        target_position=[target.lon, target.lat],
                        geometry=EncounterGeometry(
                            elevation_degrees=80.0,
                            off_nadir_degrees=55.0,
                            slant_range_km=540.0,
                            target_visible=True,
                            bearing=180.0,
                            pitch=22.0,
                        ),
                        target_priority=target.priority,
                        duration_seconds=30.0,
                    )
                )
        return windows


class FakeMaterializer:
    def materialize(self, record, target, persist=True):
        raise AssertionError("materialize() should not be called when materialize_top_k=0")


def _encounter_service(tmp_path) -> EncounterService:
    policy = EncounterPolicy()
    target_repo = TargetRepository(tmp_path / "targets.json")
    target_repo.replace_targets(
        [
            TargetSpec(
                target_id="target-high",
                label="High Target",
                lon=0.0,
                lat=0.0,
                priority=0.95,
                metadata={"scenario_pack": "scope-1"},
            ),
            TargetSpec(
                target_id="target-low",
                label="Low Trust Target",
                lon=1.0,
                lat=1.0,
                priority=1.0,
                metadata={"scenario_pack": "scope-1"},
            ),
        ]
    )
    return EncounterService(
        shared_data={"last_updated": "2026-03-10T12:00:00Z"},
        target_repo=target_repo,
        ephemeris=FakeEphemerisService(),
        window_detector=FakeWindowDetector(),
        probe_service=FakeProbeService(),
        feature_builder=FeatureBuilder(),
        planner=AnalyticPlanner(policy=policy, trust_model=WCLITrustModel(policy)),
        store=EncounterStore(tmp_path / "encounter_store"),
        materializer=FakeMaterializer(),
        policy=policy,
        observation_vla=None,
    )


def test_evaluate_persists_evaluation_and_filters_by_scenario(tmp_path):
    service = _encounter_service(tmp_path)

    evaluation = service.evaluate(
        start_time="2026-03-10T12:00:00Z",
        hours=1.0,
        step_seconds=30,
        top_k=5,
        materialize_top_k=0,
        scenario_pack="scope-1",
    )

    loaded = service.store.list_evaluations(limit=5)
    filtered = service.list_evaluations(limit=5, scenario_pack="scope-1")

    assert evaluation.window_count == 2
    assert evaluation.scenario_pack == "scope-1"
    assert evaluation.scaffold_summary is not None
    assert evaluation.trust_summary is not None
    assert evaluation.scaffold_summary.materialization_attempts == 0
    assert evaluation.trust_summary.materialization_attempts == 0
    assert loaded[0].evaluation_id == evaluation.evaluation_id
    assert filtered[0].evaluation_id == evaluation.evaluation_id
    assert "accept->accept" in evaluation.action_transition_counts
    assert any(key.endswith("->refine") for key in evaluation.action_transition_counts)
    assert any(delta.changed_to_refine for delta in evaluation.decision_deltas)


def test_evaluation_summary_order_and_round_trip(tmp_path):
    service = _encounter_service(tmp_path)

    first = service.evaluate(
        start_time="2026-03-10T12:00:00Z",
        hours=1.0,
        step_seconds=30,
        top_k=5,
        materialize_top_k=0,
        scenario_pack="scope-1",
    )
    second = service.evaluate(
        start_time="2026-03-10T12:10:00Z",
        hours=1.0,
        step_seconds=30,
        top_k=5,
        materialize_top_k=0,
        scenario_pack="scope-1",
    )

    evaluations = service.store.list_evaluations(limit=10)
    ids = {evaluation.evaluation_id for evaluation in evaluations}

    assert first.evaluation_id in ids
    assert second.evaluation_id in ids
    assert all(evaluation.window_count == 2 for evaluation in evaluations[:2])
    assert all(evaluation.parameters["scenario_pack"] == "scope-1" for evaluation in evaluations[:2])
