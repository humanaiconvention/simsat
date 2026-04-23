from __future__ import annotations

import io

from encounter.schemas import EncounterDecision
from haic.bridge import StimulusBridge
from haic.schemas import StimulusType
from observation_vla.dataset import ObservationDatasetBuilder
from observation_vla.memory import ObservationTTTMemory
from observation_vla.residual import ObservationResidualBuilder
from observation_vla.schemas import (
    ObservationAssessment,
    ObservationEvidence,
    ObservationMemoryState,
    ObservationOutcome,
    ObservationResidual,
    ObservationSample,
    ObservationTraceRecord,
)
from observation_vla.store import ObservationStore


class FakeSentinelProvider:
    def get_single_image_lon_lat(self, lon, lat, timestamp, data_type="png", spectral_bands=None, size_km=5):
        return {
            "image": io.BytesIO(b"fake-png"),
            "metadata": {
                "image_available": True,
                "source": "sentinel-2a",
                "cloud_cover": 7.5,
                "footprint": [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1],
            },
        }


class FakeMapboxProvider:
    def get_target_image(self, sat_lon, sat_lat, sat_alt, target_lon, target_lat):
        return {
            "image": b"fake-mapbox",
            "metadata": {
                "target_visible": True,
                "image_available": True,
                "elevation_degrees": 65.0,
                "bearing": 180.0,
                "pitch": 20.0,
            },
        }


def _sample() -> ObservationSample:
    return ObservationSample(
        sample_id="obs-1",
        decision_id="dec-1",
        window_id="window-1",
        target_id="target-1",
        scenario_pack="scope-1",
        target_label="Target One",
    )


def _assessment(*, action: str, confidence: float, scene_match: float, usable: bool, cloud_risk: float = 0.1) -> ObservationAssessment:
    return ObservationAssessment(
        assessment_id="asm-1",
        sample_id="obs-1",
        model_id="fake-vlm",
        created_at="2026-03-10T12:00:00Z",
        assessment_mode="image_conditioned",
        runtime_mode="test_runtime",
        image_count=1,
        recommended_action=action,
        evidence=ObservationEvidence(
            usable_observation=usable,
            scene_match_score=scene_match,
            salience_score=0.7,
            change_or_event_score=0.6,
            occlusion_or_cloud_risk=cloud_risk,
            confidence=confidence,
            rationale_tags=["base_signal"],
        ),
    )


def _trace(predicted_action: str = "accept", confidence: float = 0.8) -> ObservationTraceRecord:
    return ObservationTraceRecord(
        trace_id="trace-1",
        created_at="2026-03-10T12:00:00Z",
        scenario_pack="scope-1",
        decision_id="dec-1",
        target_id="target-1",
        window_id="window-1",
        sample=_sample(),
        assessment=_assessment(action="accept", confidence=confidence, scene_match=0.8, usable=True),
        residual=ObservationResidual(
            assessment_id="asm-1",
            trust_delta=0.1,
            score_delta=0.05,
            recommended_action=predicted_action,
            effective_action=predicted_action,
            rationale_tags=["seed_residual"],
        ),
        decision_before={"action": "defer"},
        decision_after={"action": predicted_action, "effective_action": predicted_action},
        probe={},
        features={},
    )


def test_ttt_memory_downgrades_and_upgrades_actions(tmp_path):
    store = ObservationStore(tmp_path)
    store.save_memory_state(
        ObservationMemoryState(
            state_id="scenario:scope-1",
            scope="scenario",
            scope_id="scope-1",
            last_updated="2026-03-10T12:00:00Z",
            sample_count=3,
            useful_count=1,
            mean_usefulness=0.4,
            confidence_bias=-0.15,
            refine_bias=0.25,
        )
    )
    memory = ObservationTTTMemory(store)

    downgraded = memory.apply(
        _sample(),
        _assessment(action="accept", confidence=0.62, scene_match=0.55, usable=True),
        mission_id="default",
    )

    assert downgraded.recommended_action == "refine"
    assert "ttt_downgraded_accept" in downgraded.calibration_tags
    assert "scenario_ttt:scope-1" in downgraded.calibration_tags

    store.replace_memory_states(
        [
            ObservationMemoryState(
                state_id="scenario:scope-1",
                scope="scenario",
                scope_id="scope-1",
                last_updated="2026-03-10T12:01:00Z",
                sample_count=4,
                useful_count=4,
                mean_usefulness=0.9,
                confidence_bias=0.12,
                refine_bias=-0.2,
            )
        ]
    )

    upgraded = memory.apply(
        _sample(),
        _assessment(action="refine", confidence=0.75, scene_match=0.65, usable=True, cloud_risk=0.2),
        mission_id="default",
    )

    assert upgraded.recommended_action == "accept"
    assert "ttt_upgraded_refine" in upgraded.calibration_tags


def test_ttt_memory_register_and_rebuild_states(tmp_path):
    store = ObservationStore(tmp_path)
    memory = ObservationTTTMemory(store)
    trace = _trace(predicted_action="defer", confidence=0.85)
    store.save_trace(trace)
    outcome = ObservationOutcome(
        outcome_id="out-1",
        trace_id=trace.trace_id,
        assessment_id="asm-1",
        decision_id=trace.decision_id,
        registered_at="2026-03-10T12:05:00Z",
        mission_id="mission-1",
        operator_action="accept",
        useful=True,
        usefulness_score=0.9,
        label_source="operator_review",
    )
    store.save_outcome(outcome)

    states = memory.register_outcome(trace, outcome)
    rebuilt = memory.rebuild_from_store()

    assert len(states) == 3
    assert {state.scope for state in states} == {"pass", "scenario", "mission"}
    scenario_state = next(state for state in rebuilt if state.scope == "scenario")
    assert scenario_state.scope_id == "scope-1"
    assert scenario_state.sample_count == 1
    assert scenario_state.mean_usefulness == 0.9


def test_residual_builder_shapes_effective_action():
    builder = ObservationResidualBuilder()
    decision = EncounterDecision(action="defer", trust_score=0.6, combined_score=0.55)

    accept_residual = builder.build(
        _assessment(action="accept", confidence=0.85, scene_match=0.8, usable=True),
        decision,
    )
    refine_residual = builder.build(
        _assessment(action="refine", confidence=0.45, scene_match=0.45, usable=False, cloud_risk=0.6),
        decision,
    )

    assert accept_residual.effective_action == "accept"
    assert accept_residual.trust_delta > 0
    assert "assessment_mode:image_conditioned" in accept_residual.rationale_tags
    assert refine_residual.effective_action == "refine"
    assert refine_residual.rationale_tags[-1] == "observation_action:refine"


def test_dataset_builder_and_bridge_package_runtime_payloads():
    trace = _trace()
    outcome = ObservationOutcome(
        outcome_id="out-1",
        trace_id=trace.trace_id,
        assessment_id="asm-1",
        decision_id=trace.decision_id,
        registered_at="2026-03-10T12:05:00Z",
        operator_action="accept",
        useful=True,
        usefulness_score=0.9,
    )
    rows = ObservationDatasetBuilder().build_dataset([trace], {trace.trace_id: outcome}, labelled_only=True)

    bridge = StimulusBridge(
        shared_data={"satellite_position": [1.0, 2.0, 500.0], "last_updated": "2026-03-10T12:00:00Z"},
        sentinel_provider=FakeSentinelProvider(),
        mapbox_provider=FakeMapboxProvider(),
    )
    stimulus = bridge.build_stimulus(size_km=6.0)
    windows = bridge.predict_observation_windows(count=2, step_seconds=300)

    assert rows[0]["labelled"] is True
    assert rows[0]["outcome"]["outcome_id"] == "out-1"
    assert len(stimulus.images) == 2
    assert {image.stimulus_type for image in stimulus.images} == {StimulusType.SENTINEL_RGB, StimulusType.MAPBOX_PERSPECTIVE}
    assert stimulus.content_hash
    assert "Available perspectives:" in stimulus.observation_context
    assert len(windows) == 2
    assert windows[0].region_description
