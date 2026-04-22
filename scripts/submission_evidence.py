from __future__ import annotations

import argparse
from pathlib import Path
import sys

import requests


class HttpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_json(self, path: str, **params) -> dict:
        response = requests.get(f"{self.base_url}{path}", params=params or None, timeout=90)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def post_json(self, path: str, payload: dict | None = None) -> dict:
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()


class InProcessClient:
    def __init__(self) -> None:
        from fastapi.testclient import TestClient

        root = Path(__file__).resolve().parents[1]
        sim_root = root / "src" / "sim"
        if str(sim_root) not in sys.path:
            sys.path.insert(0, str(sim_root))
        from api import api  # type: ignore

        self._client = TestClient(api)
        self._context = self._client.__enter__()

    def close(self) -> None:
        self._client.__exit__(None, None, None)

    def get_json(self, path: str, **params) -> dict:
        response = self._context.get(path, params=params or None)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def post_json(self, path: str, payload: dict | None = None) -> dict:
        response = self._context.post(path, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()


def fetch_latest_evaluation(client, scenario_pack: str, preferred_id: str | None = None) -> dict:
    try:
        payload = client.get_json("/encounter/evaluations", limit=20, scenario_pack=scenario_pack)
    except Exception:
        return {}
    evaluations = payload.get("evaluations", [])
    if not evaluations:
        return {}
    non_empty = [evaluation for evaluation in evaluations if int(evaluation.get("window_count", 0)) > 0]
    if preferred_id:
        for evaluation in non_empty:
            if evaluation.get("evaluation_id") == preferred_id:
                return evaluation
    if non_empty:
        return non_empty[0]
    if preferred_id:
        for evaluation in evaluations:
            if evaluation.get("evaluation_id") == preferred_id:
                return evaluation
    return evaluations[0]


def planner_summary_from_evidence(evidence: dict) -> tuple[dict, dict]:
    planner_summary = evidence.get("planner_summary", {})
    scaffold_summary = {
        "action_counts": planner_summary.get("scaffold_actions", {}),
        "materialization_yield": planner_summary.get("scaffold_yield", 0.0),
    }
    trust_summary = {
        "action_counts": planner_summary.get("trust_actions", {}),
        "materialization_yield": planner_summary.get("trust_yield", 0.0),
    }
    return scaffold_summary, trust_summary


def select_case_decision(planned: dict, evaluation: dict) -> dict:
    decisions = list(planned.get("decisions", []))
    if not decisions:
        raise RuntimeError("No decisions available for scenario planning")
    by_window = {decision.get("window_id"): decision for decision in decisions}
    for delta in evaluation.get("decision_deltas", []):
        if delta.get("action_changed") and delta.get("window_id") in by_window:
            return by_window[delta["window_id"]]
    ranked = sorted(
        decisions,
        key=lambda decision: (
            1 if decision.get("action") in {"accept", "refine", "defer"} else 0,
            float(decision.get("combined_score", 0.0)),
        ),
        reverse=True,
    )
    return ranked[0]


def simulate_outcome_payload(assessed: dict, scenario_pack: str) -> dict:
    effective_action = assessed.get("effective_action") or assessed.get("assessment", {}).get("recommended_action") or "skip"
    assessment = assessed.get("assessment", {})
    evidence = assessment.get("evidence", {})
    confidence = float(evidence.get("confidence", 0.0))
    useful = effective_action != "skip"
    usefulness_score = {
        "accept": 0.92,
        "refine": 0.80,
        "defer": 0.68,
        "skip": 0.20,
    }.get(effective_action, max(min(confidence, 1.0), 0.0))
    return {
        "operator_action": effective_action,
        "useful": useful,
        "usefulness_score": usefulness_score,
        "outcome_tags": [
            "simulated_submission_case",
            f"scenario:{scenario_pack}",
        ],
        "notes": "Auto-generated submission case label for reproducible challenge evidence.",
    }


def simulate_mission_response_utility(action: dict, outcome_payload: dict) -> float:
    usefulness = float(outcome_payload.get("usefulness_score", 0.0))
    recommended_action = action.get("recommended_action", "drop")
    if recommended_action == "materialize_now":
        return min(1.0, usefulness)
    if recommended_action == "queue_refine_review":
        return min(1.0, usefulness * 0.95)
    if recommended_action == "schedule_revisit":
        return min(1.0, usefulness * 0.75)
    if recommended_action == "escalate_operator":
        return min(1.0, usefulness * 0.85)
    return min(1.0, usefulness * 0.25)


def fetch_case_response(client, trace_id: str) -> tuple[dict, dict | None]:
    try:
        response = client.get_json(f"/mission-response/trace/{trace_id}")
    except Exception:
        return {}, None
    return response.get("action", {}), response.get("outcome")


def ensure_case_for_scenario(
    client,
    scenario_pack: str,
    hours: float,
    step_seconds: int,
    top_k: int,
    materialize_top_k: int,
    reviewed_only: bool = False,
) -> dict:
    requested_hours = hours
    evidence = client.get_json(
        "/encounter/evidence",
        scenario_pack=scenario_pack,
        case_limit=3,
        trace_limit=100,
    )

    if reviewed_only:
        if not evidence.get("reviewed_submission_ready", False):
            raise RuntimeError(
                f"Scenario {scenario_pack} does not have a pinned operator-reviewed submission case yet"
            )
        return {
            "evaluation": fetch_latest_evaluation(
                client,
                scenario_pack,
                preferred_id=evidence.get("evaluation_id"),
            ),
            "decision": {},
            "assessed": {},
            "outcome": {},
            "mission_action": {},
            "mission_outcome": {},
            "evidence": evidence,
            "requested_hours": requested_hours,
            "used_hours": requested_hours,
            "historical_only": True,
        }

    candidate_hours: list[float] = []
    for candidate in (hours, max(hours * 2.0, hours + 6.0), 24.0, 48.0):
        bounded = min(48.0, round(candidate, 2))
        if bounded not in candidate_hours:
            candidate_hours.append(bounded)

    evaluation = None
    planned = None
    used_hours = requested_hours
    last_error: Exception | None = None
    for candidate_hours_value in candidate_hours:
        try:
            evaluation = client.post_json(
                "/encounter/evaluate",
                {
                    "scenario_pack": scenario_pack,
                    "hours": candidate_hours_value,
                    "step_seconds": step_seconds,
                    "top_k": top_k,
                    "materialize_top_k": materialize_top_k,
                },
            )
            planned = client.post_json(
                "/encounter/plan",
                {
                    "scenario_pack": scenario_pack,
                    "hours": candidate_hours_value,
                    "step_seconds": step_seconds,
                    "top_k": top_k,
                },
            )
        except Exception as exc:
            last_error = exc
            continue
        decisions = planned.get("decisions", [])
        used_hours = candidate_hours_value
        if evaluation.get("window_count", 0) > 0 and decisions:
            break

    if evaluation is None or planned is None:
        if evidence.get("case_summaries"):
            return {
                "evaluation": fetch_latest_evaluation(client, scenario_pack),
                "decision": {},
                "assessed": {},
                "outcome": {},
                "mission_action": {},
                "mission_outcome": {},
                "evidence": evidence,
                "requested_hours": requested_hours,
                "used_hours": requested_hours,
                "historical_only": True,
            }
        raise RuntimeError(
            f"Unable to generate evaluation for scenario {scenario_pack}"
            + (f": {last_error}" if last_error is not None else "")
        )

    decisions = planned.get("decisions", [])
    historical_only = False
    if decisions and not reviewed_only:
        decision = select_case_decision(planned, evaluation)
        assessed = client.post_json(f"/encounter/decision/{decision['decision_id']}/assess")
        outcome_payload = simulate_outcome_payload(assessed, scenario_pack=scenario_pack)
        outcome = client.post_json(
            f"/observation-vla/trace/{assessed['trace_id']}/outcome",
            outcome_payload,
        )
        mission_action = client.post_json(f"/mission-response/from-trace/{assessed['trace_id']}")
        mission_outcome = client.post_json(
            f"/mission-response/action/{mission_action['action_id']}/outcome",
            {
                "utility_realized": simulate_mission_response_utility(mission_action, outcome_payload),
                "execution_status": "simulated",
                "notes": "Auto-generated mission response outcome for submission evidence.",
            },
        )
    else:
        historical_only = True
        decision = {}
        assessed = {}
        outcome = {}
        mission_action = {}
        mission_outcome = {}
    return {
        "evaluation": evaluation,
        "decision": decision,
        "assessed": assessed,
        "outcome": outcome,
        "mission_action": mission_action,
        "mission_outcome": mission_outcome,
        "evidence": evidence,
        "requested_hours": requested_hours,
        "used_hours": used_hours,
        "historical_only": historical_only,
    }


def build_report(
    client,
    scenarios: list[str],
    capabilities: dict,
    scenario_artifacts: dict[str, dict],
    reviewed_only: bool = False,
) -> str:
    lines: list[str] = []
    lines.append("# SimSat Submission Packet")
    lines.append("")
    lines.append("This packet was generated from the live challenge stack.")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Sentinel enabled: `{capabilities.get('sentinel_enabled')}`")
    lines.append(f"- Mapbox enabled: `{capabilities.get('mapbox_enabled')}`")
    lines.append(f"- Challenge no-Mapbox-safe: `{capabilities.get('challenge_no_mapbox_safe')}`")
    lines.append(f"- Sentinel-first challenge scoring: `{capabilities.get('sentinel_first_challenge_scoring')}`")
    lines.append(f"- ObservationVLA runtime: `{capabilities.get('observation_vla_runtime_mode')}`")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| scenario | windows | scaffold_accept | trust_accept | trust_refine | scaffold_yield | trust_yield | labelled_cases |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for scenario in scenarios:
        artifact = scenario_artifacts[scenario]
        evaluation = artifact.get("evaluation", {})
        evidence = artifact.get("evidence", {})
        scaffold_summary = evaluation.get("scaffold_summary", {})
        trust_summary = evaluation.get("trust_summary", {})
        if not scaffold_summary and not trust_summary:
            scaffold_summary, trust_summary = planner_summary_from_evidence(evidence)
        window_count = evaluation.get("window_count")
        if window_count in {None, 0} and evidence.get("window_count"):
            window_count = evidence.get("window_count")
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario,
                    str(window_count or 0),
                    str(scaffold_summary.get("action_counts", {}).get("accept", 0)),
                    str(trust_summary.get("action_counts", {}).get("accept", 0)),
                    str(trust_summary.get("action_counts", {}).get("refine", 0)),
                    str(scaffold_summary.get("materialization_yield", 0.0)),
                    str(trust_summary.get("materialization_yield", 0.0)),
                    str(evidence.get("labelled_trace_count", 0)),
                ]
            )
            + " |"
        )
    lines.append("")

    for scenario in scenarios:
        artifact = scenario_artifacts[scenario]
        evaluation = artifact.get("evaluation", {})
        evidence = artifact.get("evidence", {})
        assessed = artifact.get("assessed", {})
        mission_action = artifact.get("mission_action", {})
        mission_outcome = artifact.get("mission_outcome", {})
        lines.append(f"## {scenario}")
        lines.append("")
        evaluation_id = evaluation.get("evaluation_id") or evidence.get("evaluation_id")
        lines.append(f"- Evaluation: `{evaluation_id}`")
        lines.append(
            f"- Horizon used: `{artifact.get('used_hours')}` hour(s)"
            + (" (expanded from requested horizon)" if artifact.get("used_hours") != artifact.get("requested_hours") else "")
        )
        transition_counts = evaluation.get("action_transition_counts", {}) or evidence.get("transition_counts", {})
        lines.append(f"- Transition counts: `{transition_counts}`")
        deltas = evaluation.get("decision_deltas", []) or evidence.get("delta_highlights", [])
        if deltas:
            top_delta = deltas[0]
            lines.append(
                f"- Top delta: `{top_delta.get('target_label') or top_delta.get('target_id')}` "
                f"`{top_delta.get('scaffold_action')} -> {top_delta.get('trust_action')}` "
                f"(reason: `{top_delta.get('refinement_reason') or ','.join(top_delta.get('trust_reason_codes', []))}`)"
            )
        cases = evidence.get("case_summaries", [])
        if cases:
            case = cases[0]
            case_trace_id = case.get("trace_id", "")
            case_action, case_outcome = ({}, None)
            if case_trace_id:
                case_action, case_outcome = fetch_case_response(client, case_trace_id)
            lines.append(f"- Curated case: {case.get('summary_line')}")
            lines.append(
                f"- Stored visual assessment: mode=`{case.get('assessment_mode')}`, runtime=`{case.get('runtime_mode')}`, "
                f"recommended=`{case.get('recommended_action')}`, useful=`{case.get('useful')}`, "
                f"usefulness_score=`{case.get('usefulness_score')}`"
            )
            if case_action:
                lines.append(
                    f"- Mission response: action=`{case_action.get('recommended_action')}`, "
                    f"utility_realized=`{case_outcome.get('utility_realized') if case_outcome is not None else None}`"
                )
            elif mission_action and case_trace_id == assessed.get("trace_id"):
                lines.append(
                    f"- Mission response: action=`{mission_action.get('recommended_action')}`, "
                    f"utility_realized=`{mission_outcome.get('utility_realized')}`"
                )
            lines.append(f"- Trace: `{case_trace_id or assessed.get('trace_id')}`")
            label_source = case.get("label_source", "simulated")
            reviewer = case.get("reviewer")
            review_status = case.get("review_status", "provisional")
            label_line = f"- Label source: `{label_source}`"
            if reviewer:
                label_line += f"; reviewer=`{reviewer}`"
            label_line += f"; status=`{review_status}`"
            lines.append(label_line)
            if case.get("is_pinned_submission_case"):
                lines.append("- Submission case: `pinned`")
            if artifact.get("historical_only"):
                lines.append("- Case source: `historical labelled trace`")
        else:
            lines.append("- Curated case: none available")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This packet is Sentinel-first and does not depend on Mapbox.")
    lines.append("- Submission cases are generated from the live planner and stored assessment stack.")
    lines.append("- Stored case traces may have been assessed under an earlier ObservationVLA runtime than the current configured backend.")
    lines.append("- The current backend-specific reviewed check lives in `OBSERVATION_VLA_EVAL.md` and should be used for tight model claims.")
    lines.append("- Operator-reviewed labels are preferred when they exist for a scenario case.")
    if reviewed_only:
        lines.append("- This packet was generated in `reviewed-only` mode and fails if any scenario lacks a pinned operator-reviewed case.")
    else:
        lines.append("- Outcome labels fall back to `simulated_submission_case` only when no operator-reviewed label is available.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a repeatable submission evidence packet from the live challenge stack.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true", help="Run against an in-process FastAPI app instead of a live server.")
    parser.add_argument("--scenario-pack", default="all")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--step-seconds", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--materialize-top-k", type=int, default=2)
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--output", default=str(Path("D:/SimSat/SUBMISSION_PACKET.md")))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        capabilities = client.get_json("/capabilities")
        targets_payload = client.get_json("/encounter/targets")
        scenarios = (
            targets_payload.get("scenario_packs", [])
            if args.scenario_pack == "all"
            else [args.scenario_pack]
        )
        scenario_artifacts: dict[str, dict] = {}
        for scenario in scenarios:
            scenario_artifacts[scenario] = ensure_case_for_scenario(
                client,
                scenario_pack=scenario,
                hours=args.hours,
                step_seconds=args.step_seconds,
                top_k=args.top_k,
                materialize_top_k=args.materialize_top_k,
                reviewed_only=args.reviewed_only,
            )
        report = build_report(client, scenarios, capabilities, scenario_artifacts, reviewed_only=args.reviewed_only)
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"Submission packet written to {output_path}")
    except Exception as exc:
        print(f"Submission evidence generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
