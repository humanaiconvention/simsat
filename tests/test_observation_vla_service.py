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
