from __future__ import annotations

from types import SimpleNamespace

import pytest

from encounter.features import FeatureBuilder
from encounter.materialize import StimulusMaterializer
from encounter.planner import AnalyticPlanner
from encounter.schemas import EncounterDecision, EncounterFeatures, EncounterGeometry, EncounterPolicy, EncounterProbeResult, EncounterRecord, EncounterWindow, TargetSpec
from encounter.service import EncounterService
from encounter.store import EncounterStore
from encounter.targets import TargetRepository
from encounter.trust_model import WCLITrustModel
from encounter.windows import WindowDetector
from haic.schemas import GroundingStimulus, StimulusImage, StimulusType
import haic.stimulus_store as stimulus_store_module
from observation_vla.schemas import ObservationAssessment, ObservationAssessmentRecord, ObservationEvidence, ObservationResidual


class FakeBridge:
    def build_stimulus_at(self, **kwargs):
        return GroundingStimulus(
            stimulus_id="stim-1",
            simulation_timestamp=kwargs["timestamp"],
            satellite_position=[kwargs["sat_lon"], kwargs["sat_lat"], kwargs["sat_alt"]],
            images=[StimulusImage(stimulus_type=StimulusType.SENTINEL_RGB, image_b64="ZmFrZQ==")],
            observation_context="Bridge-built stimulus",
        )


class FakeObservationVLA:
    def __init__(self):
        self.saved_assessments = []
        self.logged_traces = []

    def stimulus_has_renderable_image(self, stimulus):
        return stimulus is not None and any(bool(image.image_b64) for image in stimulus.images)

    def assess_record(self, *, record, target, stimulus, scenario_pack, persist):
        return ObservationAssessmentRecord(
            sample=SimpleNamespace(
                to_dict=lambda: {"target_label": target.label},
                target_label=target.label,
            ),
            assessment=ObservationAssessment(
                assessment_id="asm-1",
                sample_id="obs-1",
                model_id="fake-vlm",
                created_at="2026-03-10T12:00:00Z",
                assessment_mode="image_conditioned",
                runtime_mode="test_runtime",
                image_count=1,
                recommended_action="accept",
                evidence=ObservationEvidence(
                    usable_observation=True,
                    scene_match_score=0.8,
                    salience_score=0.7,
                    change_or_event_score=0.6,
                    occlusion_or_cloud_risk=0.1,
                    confidence=0.85,
                    rationale_tags=["image_backed"],
                ),
            ),
        )

    def is_image_backed_assessment(self, assessment_record):
        return True

    def save_assessment(self, assessment_record):
        self.saved_assessments.append(assessment_record)
        return assessment_record

    def build_residual(self, assessment_record, decision):
        return ObservationResidual(
            assessment_id=assessment_record.assessment.assessment_id,
            trust_delta=0.1,
            score_delta=0.05,
            recommended_action="accept",
            effective_action="accept",
            rationale_tags=["residual_applied"],
        )

    def log_trace(self, **kwargs):
        trace = SimpleNamespace(trace_id="trace-1")
        self.logged_traces.append(kwargs)
        return trace


def _service(tmp_path, observation_vla):
    policy = EncounterPolicy()
    target_repo = TargetRepository(tmp_path / "targets.json")
    target_repo.replace_targets(
        [TargetSpec(target_id="target-1", label="Target One", lon=3.0, lat=4.0, priority=0.9, metadata={"scenario_pack": "scope-1"})]
    )
    store = EncounterStore(tmp_path / "encounter_store")
    stimulus_store = stimulus_store_module.StimulusStore(str(tmp_path / "stimuli.json"))
    materializer = StimulusMaterializer(FakeBridge(), stimulus_store)
    service = EncounterService(
        shared_data={"last_updated": "2026-03-10T12:00:00Z"},
        target_repo=target_repo,
        ephemeris=SimpleNamespace(),
        window_detector=WindowDetector(),
        probe_service=SimpleNamespace(),
        feature_builder=FeatureBuilder(),
        planner=AnalyticPlanner(policy=policy, trust_model=WCLITrustModel(policy)),
        store=store,
        materializer=materializer,
        policy=policy,
        observation_vla=observation_vla,
    )
    record = EncounterRecord(
        decision=EncounterDecision(
            decision_id="dec-1",
            window_id="window-1",
            target_id="target-1",
            created_at="2026-03-10T12:00:00Z",
            action="accept",
            combined_score=0.8,
            trust_score=0.75,
            reason_codes=["high_visibility"],
        ),
        window=EncounterWindow(
            window_id="window-1",
            target_id="target-1",
            encounter_type="imaging_window",
            start_time="2026-03-10T12:00:00Z",
            end_time="2026-03-10T12:01:00Z",
            peak_time="2026-03-10T12:00:30Z",
            satellite_position_peak=[1.0, 2.0, 500.0],
            target_position=[3.0, 4.0],
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
        ),
        probe=EncounterProbeResult(
            window_id="window-1",
            target_id="target-1",
            mapbox_feasible=True,
            sentinel_available=True,
            sentinel_cloud_cover=5.0,
        ),
        features=EncounterFeatures(
            window_id="window-1",
            target_id="target-1",
            duration_seconds=60.0,
            peak_elevation_degrees=70.0,
            off_nadir_degrees=10.0,
            slant_range_km=530.0,
            target_priority=0.9,
            mapbox_feasible=True,
            sentinel_available=True,
            sentinel_cloud_cover=5.0,
            score_components={"priority": 0.3},
        ),
    )
    service.store.save_record(record)
    return service, stimulus_store


def test_assess_materialized_decision_full_flow(tmp_path, monkeypatch):
    observation_vla = FakeObservationVLA()
    service, stimulus_store = _service(tmp_path, observation_vla)
    monkeypatch.setattr(stimulus_store_module, "_store", stimulus_store)

    result = service.assess_materialized_decision("dec-1")
    updated = service.store.get_record("dec-1")

    assert result["decision_id"] == "dec-1"
    assert result["trace_id"] == "trace-1"
    assert result["effective_action"] == "accept"
    assert updated is not None
    assert updated.decision.stimulus_id == "stim-1"
    assert updated.decision.observation_assessment_id == "asm-1"
    assert updated.decision.effective_action == "accept"
    assert updated.decision.effective_trust_score is not None
    assert updated.decision.effective_combined_score is not None
    assert "observation:residual_applied" in updated.decision.reason_codes
    assert observation_vla.saved_assessments
    assert observation_vla.logged_traces


def test_assess_materialized_decision_rejects_missing_renderable_image(tmp_path, monkeypatch):
    class NoImageObservationVLA(FakeObservationVLA):
        def stimulus_has_renderable_image(self, stimulus):
            return False

    observation_vla = NoImageObservationVLA()
    service, stimulus_store = _service(tmp_path, observation_vla)
    monkeypatch.setattr(stimulus_store_module, "_store", stimulus_store)

    with pytest.raises(RuntimeError):
        service.assess_materialized_decision("dec-1")
