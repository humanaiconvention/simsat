from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = ROOT / "src" / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from api import api  # noqa: E402

RUNTIME_FILES = [
    SIM_ROOT / "data" / "encounter" / "records.json",
    SIM_ROOT / "data" / "encounter" / "evaluations.json",
    SIM_ROOT / "data" / "observation_vla" / "assessments.json",
    SIM_ROOT / "data" / "observation_vla" / "traces.json",
    SIM_ROOT / "data" / "observation_vla" / "outcomes.json",
    SIM_ROOT / "data" / "observation_vla" / "memory.json",
    SIM_ROOT / "data" / "observation_vla" / "submission_cases.json",
    SIM_ROOT / "data" / "mission_response" / "actions.json",
    SIM_ROOT / "data" / "mission_response" / "outcomes.json",
    SIM_ROOT / "data" / "mission_response" / "snapshots.json",
    SIM_ROOT / "data" / "haic" / "stimuli.json",
]


@contextmanager
def restore_runtime_files(paths: list[Path]):
    backups: dict[Path, str | None] = {}
    for path in paths:
        backups[path] = path.read_text(encoding="utf-8") if path.exists() else None
    try:
        yield
    finally:
        for path, text in backups.items():
            if text is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")


def _expect_status(response, code: int, label: str) -> dict:
    if response.status_code != code:
        raise AssertionError(f"{label} failed with {response.status_code}: {response.text}")
    return response.json()


def run() -> None:
    with restore_runtime_files(RUNTIME_FILES):
        with TestClient(api) as client:
            eval_a = _expect_status(
                client.post(
                    "/encounter/evaluate",
                    json={
                        "scenario_pack": "maritime_chokepoints",
                        "hours": 6,
                        "step_seconds": 120,
                        "top_k": 4,
                        "materialize_top_k": 1,
                    },
                ),
                200,
                "evaluate maritime",
            )
            capabilities = _expect_status(client.get("/capabilities"), 200, "capabilities")
            assert capabilities["challenge_no_mapbox_safe"] is True, capabilities
            assert capabilities["sentinel_first_challenge_scoring"] is True, capabilities
            _expect_status(
                client.post(
                    "/encounter/evaluate",
                    json={
                        "scenario_pack": "urban_coastal_ambiguity",
                        "hours": 6,
                        "step_seconds": 120,
                        "top_k": 4,
                        "materialize_top_k": 1,
                    },
                ),
                200,
                "evaluate urban",
            )
            filtered = _expect_status(
                client.get("/encounter/evaluations", params={"scenario_pack": "maritime_chokepoints", "limit": 3}),
                200,
                "filtered evaluations",
            )
            scenarios = [item["scenario_pack"] for item in filtered.get("evaluations", [])]
            assert scenarios and all(s == "maritime_chokepoints" for s in scenarios), scenarios

            planned = _expect_status(
                client.post(
                    "/encounter/plan",
                    json={
                        "scenario_pack": "maritime_chokepoints",
                        "hours": 6,
                        "step_seconds": 120,
                        "top_k": 4,
                    },
                ),
                200,
                "plan",
            )
            decisions = planned.get("decisions", [])
            assert decisions, "expected at least one decision"
            decision_id = decisions[0]["decision_id"]

            assessed = _expect_status(
                client.post(f"/encounter/decision/{decision_id}/assess"),
                200,
                "assess decision",
            )
            assessment = assessed.get("assessment", {})
            residual = assessed.get("residual", {})
            assert assessed.get("trace_id", "").startswith("trace_"), assessed
            assert assessment.get("assessment_mode") in {"image_conditioned", "metadata_only"}, assessment
            assert residual.get("effective_action") == assessed.get("effective_action"), assessed
            assert assessed.get("effective_trust_score") is not None, assessed
            assert assessed.get("effective_combined_score") is not None, assessed

            traces = _expect_status(
                client.get("/observation-vla/traces", params={"limit": 3}),
                200,
                "list traces",
            )
            trace_ids = [trace["trace_id"] for trace in traces.get("traces", [])]
            assert assessed["trace_id"] in trace_ids, trace_ids

            outcome_payload = _expect_status(
                client.post(
                    f"/observation-vla/trace/{assessed['trace_id']}/outcome",
                    json={
                        "operator_action": "accept",
                        "useful": True,
                        "usefulness_score": 0.9,
                        "outcome_tags": ["regression"],
                        "notes": "regression labelled example",
                    },
                ),
                200,
                "register outcome",
            )
            assert outcome_payload["outcome"]["trace_id"] == assessed["trace_id"], outcome_payload
            assert outcome_payload["memory_states"], outcome_payload
            assert outcome_payload["outcome"]["label_source"] == "simulated", outcome_payload

            reviewed_payload = _expect_status(
                client.post(
                    f"/observation-vla/trace/{assessed['trace_id']}/operator-review",
                    json={
                        "reviewer": "regression_operator",
                        "operator_action": "accept",
                        "useful": True,
                        "usefulness_score": 0.95,
                        "outcome_tags": ["reviewed"],
                        "notes": "regression operator-reviewed example",
                    },
                ),
                200,
                "register operator review",
            )
            assert reviewed_payload["outcome"]["trace_id"] == assessed["trace_id"], reviewed_payload
            assert reviewed_payload["outcome"]["label_source"] == "operator_review", reviewed_payload
            assert reviewed_payload["outcome"]["reviewer"] == "regression_operator", reviewed_payload

            review_bundle = _expect_status(
                client.get(f"/observation-vla/review-bundle/{assessed['trace_id']}"),
                200,
                "review bundle",
            )
            assert review_bundle["trace"]["trace_id"] == assessed["trace_id"], review_bundle
            assert review_bundle["ready_for_submission_case"] is True, review_bundle
            assert review_bundle["window"]["window_id"] == decisions[0]["window_id"], review_bundle

            pinned_case = _expect_status(
                client.post(
                    "/observation-vla/submission-case/maritime_chokepoints",
                    json={
                        "trace_id": assessed["trace_id"],
                        "pinned_by": "regression_operator",
                        "notes": "regression pinned case",
                    },
                ),
                200,
                "pin submission case",
            )
            assert pinned_case["trace_id"] == assessed["trace_id"], pinned_case

            submission_cases = _expect_status(
                client.get("/observation-vla/submission-cases"),
                200,
                "list submission cases",
            )
            assert any(item["trace_id"] == assessed["trace_id"] for item in submission_cases.get("cases", [])), submission_cases

            memory = _expect_status(
                client.get("/observation-vla/memory", params={"scope": "scenario", "limit": 5}),
                200,
                "list memory",
            )
            assert any(state["scope"] == "scenario" for state in memory.get("states", [])), memory

            dataset = _expect_status(
                client.get("/observation-vla/dataset", params={"limit": 10, "labelled_only": True}),
                200,
                "export dataset",
            )
            assert any(example["trace_id"] == assessed["trace_id"] for example in dataset.get("examples", [])), dataset

            review_candidates = _expect_status(
                client.get("/observation-vla/review-candidates", params={"scenario_pack": "maritime_chokepoints", "limit": 5}),
                200,
                "review candidates",
            )
            matching_candidates = [item for item in review_candidates.get("candidates", []) if item["trace_id"] == assessed["trace_id"]]
            assert matching_candidates, review_candidates
            assert matching_candidates[0]["needs_operator_review"] is False, matching_candidates[0]

            evidence = _expect_status(
                client.get(
                    "/encounter/evidence",
                    params={
                        "scenario_pack": "maritime_chokepoints",
                        "case_limit": 2,
                        "trace_limit": 20,
                    },
                ),
                200,
                "scenario evidence",
            )
            assert evidence.get("case_summaries"), evidence
            assert any(case["trace_id"] == assessed["trace_id"] for case in evidence.get("case_summaries", [])), evidence
            matching_cases = [case for case in evidence.get("case_summaries", []) if case["trace_id"] == assessed["trace_id"]]
            assert matching_cases and matching_cases[0]["label_source"] == "operator_review", matching_cases
            assert matching_cases[0]["is_pinned_submission_case"] is True, matching_cases
            assert evidence.get("reviewed_submission_ready") is True, evidence

            current_outcomes = _expect_status(
                client.get("/observation-vla/outcomes", params={"limit": 10, "current_only": True}),
                200,
                "current outcomes",
            )
            current_matches = [item for item in current_outcomes.get("outcomes", []) if item["trace_id"] == assessed["trace_id"]]
            assert current_matches and current_matches[0]["label_source"] == "operator_review", current_matches

            mission_action = _expect_status(
                client.post(f"/mission-response/from-trace/{assessed['trace_id']}"),
                200,
                "propose mission response",
            )
            assert mission_action.get("recommended_action"), mission_action

            mission_outcome = _expect_status(
                client.post(
                    f"/mission-response/action/{mission_action['action_id']}/outcome",
                    json={
                        "utility_realized": 0.82,
                        "execution_status": "simulated",
                        "notes": "regression mission response",
                    },
                ),
                200,
                "register mission response outcome",
            )
            assert mission_outcome.get("utility_realized") == 0.82, mission_outcome

            mission_summary = _expect_status(
                client.get("/mission-response/summary", params={"mission_id": "default", "scenario_pack": "maritime_chokepoints"}),
                200,
                "mission response summary",
            )
            assert mission_summary.get("total_actions", 0) >= 1, mission_summary
            assert mission_summary.get("utility_realized", 0.0) > 0.0, mission_summary

            summary = {
                "evaluation_id": eval_a.get("evaluation_id"),
                "mapbox_enabled": capabilities.get("mapbox_enabled"),
                "filtered_scenarios": scenarios,
                "decision_id": decision_id,
                "effective_action": assessed.get("effective_action"),
                "assessment_mode": assessment.get("assessment_mode"),
                "trace_id": assessed.get("trace_id"),
                "outcome_id": reviewed_payload["outcome"]["outcome_id"],
                "label_source": reviewed_payload["outcome"]["label_source"],
                "dataset_examples": dataset.get("total"),
                "evidence_cases": len(evidence.get("case_summaries", [])),
                "mission_response_action": mission_action.get("recommended_action"),
                "mission_response_utility": mission_summary.get("utility_realized"),
            }
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
