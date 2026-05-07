from __future__ import annotations

from types import SimpleNamespace

import pytest

from encounter.schemas import EncounterDecision, EncounterFeatures, EncounterProbeResult, EncounterRecord, EncounterWindow
from haic.schemas import GroundingStimulus, StimulusImage, StimulusType
from observation_vla.schemas import (
    ObservationAssessment,
    ObservationAssessmentRecord,
    ObservationEvidence,
    ObservationMemoryState,
    ObservationOutcome,
    ObservationResidual,
    ObservationSample,
    ObservationTraceRecord,
)
from observation_vla.service import ObservationVLAService
from observation_vla.store import ObservationStore


class FakeAssessor:
    def build_sample(self, *, record, target, stimulus, scenario_pack):
        return ObservationSample(
            sample_id="obs-1",
            decision_id=record.decision.decision_id,
            window_id=record.window.window_id,
            target_id=target.target_id,
            scenario_pack=scenario_pack,
            target_label=target.label,
            target_tags=list(target.tags),
            target_metadata=dict(target.metadata),
            geometry=record.window.geometry.to_dict(),
            probe=record.probe.to_dict(),
            images=[],
        )

    def extract_renderable_images(self, stimulus):
        if stimulus is None:
            return []
        return [image.image_b64 for image in stimulus.images if image.image_b64]

    def assess(self, sample, rendered_images):
        return ObservationAssessment(
            assessment_id="asm-1",
            sample_id=sample.sample_id,
            model_id="fake-vlm",
            created_at="2026-03-10T12:00:00Z",
            assessment_mode="image_conditioned" if rendered_images else "metadata_only",
            runtime_mode="test_runtime" if rendered_images else "stub",
            image_count=len(rendered_images),
            confidence_adjustment=0.1,
            calibration_tags=["calibrated"],
            recommended_action="accept" if rendered_images else "refine",
            evidence=ObservationEvidence(
                usable_observation=bool(rendered_images),
                confidence=0.8 if rendered_images else 0.2,
                rationale_tags=["image_backed" if rendered_images else "metadata_only"],
            ),
        )


class FakeMemory:
    def __init__(self):
        self.apply_calls = 0
        self.rebuild_calls = 0

    def apply(self, sample, assessment, mission_id):
        self.apply_calls += 1
        assessment.calibration_tags.append(f"mission:{mission_id}")
        return assessment

    def rebuild_from_store(self):
        self.rebuild_calls += 1
        return [
            ObservationMemoryState(
                state_id="state-1",
                scope="mission",
                scope_id="default",
                last_updated="2026-03-10T12:05:00Z",
                sample_count=1,
                useful_count=1,
                mean_usefulness=0.9,
                confidence_bias=0.1,
                refine_bias=-0.1,
                action_alignment=1.0,
            )
        ]


class FakeResidualBuilder:
    def build(self, assessment, decision):
        return ObservationResidual(
            assessment_id=assessment.assessment_id,
            trust_delta=0.1,
            score_delta=0.05,
            recommended_action="accept",
            effective_action="accept",
            rationale_tags=["residual_applied"],
        )


class FakeDatasetBuilder:
    def build_dataset(self, traces, outcomes_by_trace_id, labelled_only):
        return [
            {
                "trace_id": trace.trace_id,
                "labelled": trace.trace_id in outcomes_by_trace_id,
                "labelled_only": labelled_only,
            }
            for trace in traces
            if (not labelled_only) or (trace.trace_id in outcomes_by_trace_id)
        ]


class FakeMissionResponseService:
    def __init__(self):
        self.calls: list[tuple[str, float, str, str]] = []

    def get_trace_response(self, trace_id: str):
        return {"action": {"action_id": f"act-{trace_id}"}}

    def propose_from_trace_id(self, trace_id: str):
        return SimpleNamespace(to_dict=lambda: {"action_id": f"act-{trace_id}"})

    def register_action_outcome(self, *, action_id, utility_realized, execution_status, notes):
        self.calls.append((action_id, utility_realized, execution_status, notes))
        return SimpleNamespace(action_id=action_id)


def _record() -> EncounterRecord:
    return EncounterRecord(
        decision=EncounterDecision(
            decision_id="dec-1",
            window_id="window-1",
            target_id="target-1",
            created_at="2026-03-10T12:00:00Z",
            action="accept",
            effective_action="accept",
            combined_score=0.8,
            trust_score=0.75,
        ),
        window=EncounterWindow.from_dict(
            {
                "window_id": "window-1",
                "target_id": "target-1",
                "encounter_type": "imaging_window",
                "start_time": "2026-03-10T12:00:00Z",
                "end_time": "2026-03-10T12:00:30Z",
                "peak_time": "2026-03-10T12:00:10Z",
                "satellite_position_peak": [0.0, 0.0, 500.0],
                "target_position": [0.0, 0.0],
                "geometry": {
                    "elevation_degrees": 60.0,
                    "off_nadir_degrees": 10.0,
                    "slant_range_km": 530.0,
                    "target_visible": True,
                    "bearing": 180.0,
                    "pitch": 20.0,
                },
                "target_priority": 0.8,
                "duration_seconds": 30.0,
            }
        ),
        probe=EncounterProbeResult(
            window_id="window-1",
            target_id="target-1",
            mapbox_feasible=True,
            sentinel_available=True,
            sentinel_cloud_cover=5.0,
            sentinel_source="sentinel-2a",
            sentinel_datetime="2026-03-10T11:50:00Z",
        ),
        features=EncounterFeatures(
            window_id="window-1",
            target_id="target-1",
            duration_seconds=30.0,
            peak_elevation_degrees=60.0,
            off_nadir_degrees=10.0,
            slant_range_km=530.0,
            target_priority=0.8,
            mapbox_feasible=True,
            sentinel_available=True,
            sentinel_cloud_cover=5.0,
            score_components={"priority": 0.3},
        ),
    )


def _service(tmp_path):
    memory = FakeMemory()
    service = ObservationVLAService(
        assessor=FakeAssessor(),
        residual_builder=FakeResidualBuilder(),
        memory=memory,
        dataset_builder=FakeDatasetBuilder(),
        store=ObservationStore(tmp_path),
        mission_id="default",
    )
    return service, memory


def test_observation_contracts_and_assessment_persistence(tmp_path):
    service, memory = _service(tmp_path)
    record = _record()
    target = SimpleNamespace(target_id="target-1", label="Target One", tags=["review"], metadata={"scenario_pack": "scope-1"})
    stimulus = GroundingStimulus(
        stimulus_id="stim-1",
        images=[StimulusImage(stimulus_type=StimulusType.SENTINEL_RGB, image_b64="ZmFrZQ==")],
    )

    assessment_record = service.assess_record(
        record=record,
        target=target,
        stimulus=stimulus,
        scenario_pack="scope-1",
        persist=True,
    )

    assert service.stimulus_has_renderable_image(stimulus) is True
    assert service.stimulus_has_renderable_image(None) is False
    assert service.is_image_backed_assessment(assessment_record) is True
    assert memory.apply_calls == 1
    loaded = service.get_assessment("asm-1")
    assert loaded is not None
    assert loaded.assessment.model_id == "fake-vlm"
    assert loaded.assessment.image_count == 1


def test_observation_trace_outcome_sync_and_dataset_export(tmp_path):
    service, memory = _service(tmp_path)
    mission_response = FakeMissionResponseService()
    service.attach_context(mission_response_service=mission_response)

    assessment_record = ObservationAssessmentRecord(
        sample=ObservationSample(
            sample_id="obs-1",
            decision_id="dec-1",
            window_id="window-1",
            target_id="target-1",
            scenario_pack="scope-1",
            target_label="Target One",
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
            evidence=ObservationEvidence(confidence=0.9, rationale_tags=["image_backed"]),
        ),
    )
    trace = service.log_trace(
        scenario_pack="scope-1",
        record=_record(),
        assessment_record=assessment_record,
        residual=ObservationResidual(
            assessment_id="asm-1",
            trust_delta=0.1,
            score_delta=0.05,
            recommended_action="accept",
            effective_action="accept",
            rationale_tags=["residual_applied"],
        ),
        decision_before={"action": "defer"},
        decision_after={"action": "accept", "effective_action": "accept"},
    )

    outcome, states, updated_trace = service.register_operator_review(
        trace_id=trace.trace_id,
        reviewer="ben",
        operator_action="approve",
        useful=True,
        usefulness_score=0.92,
        outcome_tags=["high_signal"],
        notes="Looks good",
    )
    dataset = service.export_dataset(limit=10, labelled_only=True)

    assert outcome.label_source == "operator_review"
    assert updated_trace.outcome_id == outcome.outcome_id
    assert memory.rebuild_calls == 1
    assert states[0].scope == "mission"
    assert mission_response.calls[0][0] == f"act-{trace.trace_id}"
    assert mission_response.calls[0][2] == "operator_reviewed"
    assert dataset == [{"trace_id": trace.trace_id, "labelled": True, "labelled_only": True}]


def test_apply_trust_layer_ttt_blocks_when_error_bias_fires(tmp_path):
    """_apply_trust_layer_ttt skips online_update when error_bias gate fires (blocking gate)."""
    from types import SimpleNamespace
    from encounter.schemas import EncounterPolicy
    from encounter.trust_model import WCLITrustModel

    # Build a real trust model seeded with 10 same-sign errors so error_bias fires.
    policy = EncounterPolicy()
    trust_model = WCLITrustModel(policy=policy)
    biased_details = {
        "target_priority": 0.9, "geometry_margin": 0.9,
        "duration_margin": 0.9, "imagery_support": 0.9, "clarity_support": 0.9,
    }
    for _ in range(10):
        trust_model.online_update(biased_details, learned_score=0.50, realized_utility=0.95)

    weights_after_seed = dict(trust_model._learned_weights)
    count_after_seed = trust_model.update_count

    # Wire into a minimal encounter_service mock
    fake_planner = SimpleNamespace(trust_model=trust_model)
    fake_encounter_service = SimpleNamespace(planner=fake_planner)

    service, _ = _service(tmp_path)
    service.encounter_service = fake_encounter_service

    # Build a minimal trace with trust_details + learned_score in decision_after
    trace = ObservationTraceRecord(
        trace_id="trace-ttt",
        created_at="2026-03-10T12:00:00Z",
        scenario_pack="scope-1",
        decision_id="dec-1",
        target_id="target-1",
        window_id="window-1",
        sample=ObservationSample(target_label="Target"),
        assessment=ObservationAssessment(assessment_id="asm-ttt"),
        residual=ObservationResidual(),
        decision_after={
            "trust_details": biased_details,
            "learned_score": 0.50,
        },
    )
    outcome = ObservationOutcome(
        outcome_id="out-ttt",
        trace_id="trace-ttt",
        operator_action="accept",
        useful=True,
        usefulness_score=0.95,
        label_source="operator_review",
    )

    service._apply_trust_layer_ttt(trace, outcome)

    # Weights must NOT change (error_bias gate blocked the update)
    assert trust_model._learned_weights == weights_after_seed
    # update_count must NOT increment
    assert trust_model.update_count == count_after_seed
    # But the log must have a new blocked=True entry
    snap = trust_model.get_weight_snapshot()
    recent = snap["recent_updates"]
    assert recent[-1].get("blocked") is True


def test_apply_trust_layer_ttt_updates_weights_when_gate_passes(tmp_path):
    """_apply_trust_layer_ttt calls online_update when error_bias gate passes (balanced errors)."""
    from types import SimpleNamespace
    from encounter.schemas import EncounterPolicy
    from encounter.trust_model import WCLITrustModel

    policy = EncounterPolicy()
    trust_model = WCLITrustModel(policy=policy)
    # Seed with 5 positive + 5 negative errors → 50/50 split → gate passes
    trust_details = {
        "target_priority": 0.8, "geometry_margin": 0.8,
        "duration_margin": 0.8, "imagery_support": 0.8, "clarity_support": 0.8,
    }
    for i in range(10):
        realized = 0.95 if i % 2 == 0 else 0.45
        trust_model.online_update(trust_details, learned_score=0.70, realized_utility=realized)

    count_before = trust_model.update_count

    fake_planner = SimpleNamespace(trust_model=trust_model)
    fake_encounter_service = SimpleNamespace(planner=fake_planner)

    service, _ = _service(tmp_path)
    service.encounter_service = fake_encounter_service

    trace = ObservationTraceRecord(
        trace_id="trace-pass",
        created_at="2026-03-10T12:00:00Z",
        scenario_pack="scope-1",
        decision_id="dec-1",
        target_id="target-1",
        window_id="window-1",
        sample=ObservationSample(target_label="Target"),
        assessment=ObservationAssessment(assessment_id="asm-pass"),
        residual=ObservationResidual(),
        decision_after={
            "trust_details": trust_details,
            "learned_score": 0.70,
        },
    )
    outcome = ObservationOutcome(
        outcome_id="out-pass",
        trace_id="trace-pass",
        operator_action="accept",
        useful=True,
        usefulness_score=0.92,
        label_source="operator_review",
    )

    service._apply_trust_layer_ttt(trace, outcome)

    # Weight update should have fired
    assert trust_model.update_count == count_before + 1


def test_pin_submission_case_requires_operator_review(tmp_path):
    service, _ = _service(tmp_path)
    trace = ObservationTraceRecord(
        trace_id="trace-1",
        created_at="2026-03-10T12:00:00Z",
        scenario_pack="scope-1",
        decision_id="dec-1",
        target_id="target-1",
        window_id="window-1",
        sample=ObservationSample(target_label="Target One"),
        assessment=ObservationAssessment(assessment_id="asm-1"),
        residual=ObservationResidual(),
    )
    service.store.save_trace(trace)

    with pytest.raises(ValueError):
        service.pin_submission_case("scope-1", "trace-1")
