from __future__ import annotations

import argparse
import sys
from pathlib import Path

from challenge_run_config import KNOWN_SCENARIO_PACKS, resolve_scenario_hours
from challenge_run_manifest import ChallengeRunManifest, ChallengeRunManifestStore
from challenge_test_matrix import (
    build_status_report,
    diagnose_refine_gap,
    recommend_horizon_policy,
    run_horizon_sweep,
    run_pytest,
    scorecard_row,
)
from review_backlog_report import build_backlog_report
from review_queue_casebook import build_review_queue
from submission_evidence import build_report as build_submission_report
from submission_evidence import ensure_case_for_scenario
from submission_readiness import build_readiness_report


ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = ROOT / "src" / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from runtime import build_runtime_bundle, build_runtime_capabilities  # noqa: E402


class OfflineClient:
    def __init__(self) -> None:
        self.bundle = build_runtime_bundle({})

    def get_json(self, path: str, **params) -> dict:
        service = self.bundle
        if path == "/capabilities":
            return build_runtime_capabilities(self.bundle)
        if path == "/encounter/targets":
            targets = service.encounter_service.list_targets()
            return {
                "targets": [target.to_dict() for target in targets],
                "total": len(targets),
                "scenario_packs": service.encounter_service.list_scenario_packs(),
            }
        if path == "/encounter/evaluations":
            evaluations = service.encounter_service.list_evaluations(
                limit=int(params.get("limit", 10)),
                scenario_pack=params.get("scenario_pack"),
            )
            return {"evaluations": [evaluation.to_dict() for evaluation in evaluations], "total": len(evaluations)}
        if path == "/encounter/evidence":
            evidence = service.encounter_service.build_scenario_evidence(
                scenario_pack=str(params["scenario_pack"]),
                case_limit=int(params.get("case_limit", 3)),
                trace_limit=int(params.get("trace_limit", 100)),
            )
            return evidence.to_dict()
        if path == "/observation-vla/review-candidates":
            candidates = service.observation_vla_service.list_review_candidates(
                scenario_pack=params.get("scenario_pack"),
                limit=int(params.get("limit", 20)),
            )
            return {"candidates": candidates, "total": len(candidates)}
        if path.startswith("/observation-vla/review-bundle/"):
            trace_id = path.rsplit("/", 1)[-1]
            return service.observation_vla_service.get_review_bundle(trace_id)
        if path == "/observation-vla/submission-cases":
            cases = service.observation_vla_service.list_submission_cases()
            return {"cases": [case.to_dict() for case in cases], "total": len(cases)}
        if path == "/observation-vla/traces":
            traces = service.observation_vla_service.list_traces(limit=int(params.get("limit", 20)))
            return {"traces": [trace.to_dict() for trace in traces], "total": len(traces)}
        if path == "/observation-vla/outcomes":
            outcomes = service.observation_vla_service.list_outcomes(
                limit=int(params.get("limit", 20)),
                current_only=bool(params.get("current_only", True)),
            )
            return {"outcomes": [outcome.to_dict() for outcome in outcomes], "total": len(outcomes)}
        if path.startswith("/mission-response/trace/"):
            trace_id = path.rsplit("/", 1)[-1]
            return service.mission_response_service.get_trace_response(trace_id)
        raise KeyError(f"Offline GET path not implemented: {path}")

    def post_json(self, path: str, payload: dict | None = None, **params) -> dict:
        payload = payload or {}
        service = self.bundle
        if path == "/encounter/evaluate":
            return service.encounter_service.evaluate(
                start_time=payload.get("start_time"),
                hours=float(payload.get("hours", 6.0)),
                step_seconds=int(payload.get("step_seconds", 30)),
                top_k=int(payload.get("top_k", 20)),
                materialize_top_k=int(payload.get("materialize_top_k", 3)),
                scenario_pack=payload.get("scenario_pack"),
            ).to_dict()
        if path == "/encounter/plan":
            records = service.encounter_service.plan(
                start_time=payload.get("start_time"),
                hours=float(payload.get("hours", 6.0)),
                step_seconds=int(payload.get("step_seconds", 30)),
                top_k=int(payload.get("top_k", 20)),
                scenario_pack=payload.get("scenario_pack"),
            )
            return {
                "policy_version": service.encounter_service.get_policy().policy_id,
                "scenario_pack": payload.get("scenario_pack") or "all",
                "decisions": [record.decision.to_dict() for record in records],
            }
        if path.startswith("/encounter/decision/") and path.endswith("/assess"):
            decision_id = path.split("/")[3]
            return service.encounter_service.assess_materialized_decision(decision_id)
        if path.startswith("/observation-vla/reassess/"):
            trace_id = path.rsplit("/", 1)[-1]
            return service.observation_vla_service.reassess_trace(
                trace_id,
                persist=bool(params.get("persist", False)),
            ).to_dict()
        if path.startswith("/observation-vla/trace/") and path.endswith("/outcome"):
            trace_id = path.split("/")[3]
            outcome, states, trace = service.observation_vla_service.register_outcome(
                trace_id=trace_id,
                operator_action=str(payload["operator_action"]),
                useful=bool(payload["useful"]),
                usefulness_score=float(payload["usefulness_score"]),
                outcome_tags=list(payload.get("outcome_tags", [])),
                notes=payload.get("notes"),
                mission_id=payload.get("mission_id"),
                label_source=str(payload.get("label_source", "simulated")),
                reviewer=payload.get("reviewer"),
                review_status=str(payload.get("review_status", "provisional")),
                replace_existing=bool(payload.get("replace_existing", True)),
            )
            return {
                "outcome": outcome.to_dict(),
                "memory_states": [state.to_dict() for state in states],
                "trace": trace.to_dict(),
            }
        if path.startswith("/mission-response/from-trace/"):
            trace_id = path.rsplit("/", 1)[-1]
            return service.mission_response_service.propose_from_trace_id(trace_id).to_dict()
        if path.startswith("/mission-response/action/") and path.endswith("/outcome"):
            action_id = path.split("/")[3]
            return service.mission_response_service.register_action_outcome(
                action_id=action_id,
                utility_realized=float(payload["utility_realized"]),
                execution_status=str(payload.get("execution_status", "simulated")),
                notes=payload.get("notes"),
            ).to_dict()
        raise KeyError(f"Offline POST path not implemented: {path}")


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def run_lanes(
    client: OfflineClient,
    *,
    policy: str,
    scenario_hours: dict[str, float],
    lanes: list[str],
    run_pytest_gate: bool,
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    if "matrix" in lanes:
        if run_pytest_gate:
            pytest_ok, pytest_summary = run_pytest(ROOT)
        else:
            pytest_ok, pytest_summary = True, "skipped"
        capabilities = client.get_json("/capabilities")
        rows: list[dict[str, str]] = []
        horizon_guidance: dict[str, dict[str, str]] = {}
        review_candidates: dict[str, list[dict]] = {}
        refine_diagnosis = diagnose_refine_gap(ROOT)
        for scenario in KNOWN_SCENARIO_PACKS:
            sweep_rows = run_horizon_sweep(
                client,
                scenario=scenario,
                step_seconds=120,
                top_k=8,
                materialize_top_k=2,
            )
            smoke_lane, competition_lane = recommend_horizon_policy(sweep_rows)
            strongest_row = next(
                (row for row in sweep_rows if f"{row['hours']:g}h ambiguity rehearsal" == competition_lane),
                None,
            )
            horizon_guidance[scenario] = {
                "smoke_lane": smoke_lane,
                "competition_lane": competition_lane,
                "rationale": str(strongest_row["top_reason"]) if strongest_row is not None else "-",
            }
            evaluation = client.post_json(
                "/encounter/evaluate",
                {
                    "scenario_pack": scenario,
                    "hours": scenario_hours[scenario],
                    "step_seconds": 120,
                    "top_k": 8,
                    "materialize_top_k": 2,
                },
            )
            evidence = client.get_json("/encounter/evidence", scenario_pack=scenario, case_limit=3, trace_limit=100)
            row = scorecard_row(evaluation, evidence)
            row["scenario"] = f"{scenario} ({scenario_hours[scenario]:g}h)"
            rows.append(row)
            review_candidates[scenario] = client.get_json(
                "/observation-vla/review-candidates",
                scenario_pack=scenario,
                limit=10,
            ).get("candidates", [])
        report = build_status_report(
            root=ROOT,
            pytest_ok=pytest_ok,
            pytest_summary=pytest_summary,
            capabilities=capabilities,
            rows=rows,
            horizon_guidance=horizon_guidance,
            review_candidates=review_candidates,
            refine_diagnosis=refine_diagnosis,
        )
        outputs["matrix"] = str(_write_text(ROOT / "CHALLENGE_TEST_STATUS.md", report))

    if "review" in lanes:
        outputs["review"] = str(
            build_review_queue(
                client,
                output_path=ROOT / "REVIEW_QUEUE.md",
                assets_dir=ROOT / "review_queue_assets",
                limit_per_scenario=2,
            )
        )

    if "submission" in lanes:
        capabilities = client.get_json("/capabilities")
        scenarios = client.get_json("/encounter/targets").get("scenario_packs", [])
        scenario_artifacts = {
            scenario: ensure_case_for_scenario(
                client,
                scenario_pack=scenario,
                hours=scenario_hours[scenario],
                step_seconds=120,
                top_k=8,
                materialize_top_k=2,
                reviewed_only=False,
            )
            for scenario in scenarios
        }
        report = build_submission_report(
            client,
            scenarios,
            capabilities,
            scenario_artifacts,
            scenario_hours,
            reviewed_only=False,
        )
        outputs["submission"] = str(_write_text(ROOT / "SUBMISSION_PACKET.md", report))

    if "readiness" in lanes:
        report = build_readiness_report(
            client,
            packet_path=ROOT / "SUBMISSION_PACKET.md",
            casebook_path=ROOT / "SUBMISSION_CASEBOOK.md",
            assets_dir=ROOT / "submission_assets",
            observation_eval_path=ROOT / "OBSERVATION_VLA_EVAL.md",
        )
        outputs["readiness"] = str(_write_text(ROOT / "SUBMISSION_READINESS.md", report))

    if "backlog" in lanes:
        outputs["backlog"] = str(
            build_backlog_report(
                client,
                assets_dir=ROOT / "review_queue_assets",
                output_path=ROOT / "REVIEW_BACKLOG.md",
            )
        )

    return outputs


def build_manifest_notes(
    *,
    policy: str,
    scenario_hours: dict[str, float],
    lanes: list[str],
    outputs: dict[str, str] | None = None,
    error: Exception | None = None,
) -> dict[str, object]:
    notes: dict[str, object] = {
        "workspace_root": str(ROOT),
        "policy": policy,
        "scenario_hours": scenario_hours,
        "lanes": lanes,
        "offline_transport": True,
        "manifest_version": 1,
    }
    if outputs:
        notes["output_count"] = len(outputs)
        notes["output_names"] = sorted(outputs.keys())
    if error is not None:
        notes["error"] = str(error)
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local SimSat challenge lanes directly against services and store a run manifest.")
    parser.add_argument("--policy", choices=["smoke", "competition"], default="competition")
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["matrix", "review", "submission", "readiness", "backlog"],
        choices=["matrix", "review", "submission", "readiness", "backlog"],
    )
    parser.add_argument("--run-pytest-gate", action="store_true", help="Include the full pytest fast gate inside the matrix lane.")
    args = parser.parse_args()

    scenario_hours = resolve_scenario_hours(
        scenarios=KNOWN_SCENARIO_PACKS,
        default_hours=8.0,
        policy=args.policy,
    )
    manifest_store = ChallengeRunManifestStore()
    manifest = ChallengeRunManifest(
        scenario_policy=args.policy,
        scenario_hours=scenario_hours,
        lanes=args.lanes,
        notes=build_manifest_notes(
            policy=args.policy,
            scenario_hours=scenario_hours,
            lanes=args.lanes,
        ),
    )
    manifest_store.save(manifest)

    client = OfflineClient()
    try:
        manifest.outputs = run_lanes(
            client,
            policy=args.policy,
            scenario_hours=scenario_hours,
            lanes=args.lanes,
            run_pytest_gate=args.run_pytest_gate,
        )
        manifest.status = "completed"
        manifest.notes = build_manifest_notes(
            policy=args.policy,
            scenario_hours=scenario_hours,
            lanes=args.lanes,
            outputs=manifest.outputs,
        )
    except Exception as exc:
        manifest.status = "failed"
        manifest.notes = build_manifest_notes(
            policy=args.policy,
            scenario_hours=scenario_hours,
            lanes=args.lanes,
            outputs=manifest.outputs,
            error=exc,
        )
        manifest_store.save(manifest)
        raise

    manifest_store.save(manifest)
    print(f"Offline challenge batch completed: {manifest.run_id}")


if __name__ == "__main__":
    main()
