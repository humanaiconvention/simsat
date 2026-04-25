from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import requests

from challenge_run_config import KNOWN_SCENARIO_PACKS, parse_scenario_hours, resolve_scenario_hours


class HttpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_json(self, path: str, **params) -> dict:
        response = requests.get(f"{self.base_url}{path}", params=params or None, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def post_json(self, path: str, payload: dict | None = None) -> dict:
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=180)
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


def run_pytest(root: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--basetemp=.pytest_tmp3"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    lines = [line for line in output.splitlines() if line.strip()]
    summary = lines[-1] if lines else f"exit={proc.returncode}"
    return proc.returncode == 0, summary


def scorecard_row(evaluation: dict, evidence: dict) -> dict[str, str]:
    scaffold = evaluation.get("scaffold_summary", {})
    trust = evaluation.get("trust_summary", {})
    transitions = evaluation.get("action_transition_counts", {})
    deltas = evaluation.get("decision_deltas", [])
    top_delta = deltas[0] if deltas else {}
    top_reason = top_delta.get("refinement_reason") or ",".join(top_delta.get("trust_reason_codes", [])) or "-"
    return {
        "scenario": str(evaluation.get("scenario_pack", "all")),
        "windows": str(evaluation.get("window_count", 0)),
        "scaffold_accept": str(scaffold.get("action_counts", {}).get("accept", 0)),
        "trust_accept": str(trust.get("action_counts", {}).get("accept", 0)),
        "trust_refine": str(trust.get("action_counts", {}).get("refine", 0)),
        "changed": str(len(deltas)),
        "accept_to_refine": str(transitions.get("accept->refine", 0)),
        "trust_yield": f"{float(trust.get('materialization_yield', 0.0)):.2f}",
        "reviewed_ready": str(bool(evidence.get("reviewed_submission_ready", False))),
        "top_reason": top_reason,
    }


def diagnose_refine_gap(root: Path) -> dict[str, dict[str, int]]:
    sim_root = root / "src" / "sim"
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))
    from observation_vla.store import ObservationStore

    store = ObservationStore(root / "src" / "sim" / "data" / "observation_vla")
    summary: dict[str, dict[str, int]] = {}
    for record in store.list_traces(limit=10000):
        trace = record.to_dict()
        scenario = record.scenario_pack
        bucket = summary.setdefault(
            scenario,
            {
                "planner_refine": 0,
                "effective_refine": 0,
                "runtime_clip_local": 0,
                "runtime_stub": 0,
                "total": 0,
            },
        )
        bucket["total"] += 1
        before = trace.get("decision_before", {})
        after = trace.get("decision_after", {})
        assessment = trace.get("assessment", {})
        if before.get("action") == "refine":
            bucket["planner_refine"] += 1
        effective_action = after.get("effective_action") or after.get("action")
        if effective_action == "refine":
            bucket["effective_refine"] += 1
        runtime_mode = assessment.get("runtime_mode")
        if runtime_mode == "clip_local":
            bucket["runtime_clip_local"] += 1
        if runtime_mode in {"stub", "stub_fallback"}:
            bucket["runtime_stub"] += 1
    return summary


def run_adaptive_evaluation(
    client,
    *,
    scenario: str,
    requested_hours: float,
    step_seconds: int,
    top_k: int,
    materialize_top_k: int,
) -> tuple[dict, float]:
    candidate_hours: list[float] = []
    for candidate in (requested_hours, max(requested_hours * 2.0, requested_hours + 6.0), 24.0, 48.0):
        bounded = min(48.0, round(candidate, 2))
        if bounded not in candidate_hours:
            candidate_hours.append(bounded)

    last_evaluation: dict | None = None
    last_hours = requested_hours
    for hours in candidate_hours:
        evaluation = client.post_json(
            "/encounter/evaluate",
            {
                "scenario_pack": scenario,
                "hours": hours,
                "step_seconds": step_seconds,
                "top_k": top_k,
                "materialize_top_k": materialize_top_k,
            },
        )
        last_evaluation = evaluation
        last_hours = hours
        if int(evaluation.get("window_count", 0)) > 0:
            return evaluation, hours

    return last_evaluation or {}, last_hours


def run_horizon_sweep(
    client,
    *,
    scenario: str,
    step_seconds: int,
    top_k: int,
    materialize_top_k: int,
    candidate_hours: list[float] | None = None,
) -> list[dict[str, int | float | str]]:
    sweep_hours = candidate_hours or [8.0, 16.0, 24.0, 48.0]
    rows: list[dict[str, int | float | str]] = []
    for hours in sweep_hours:
        evaluation = client.post_json(
            "/encounter/evaluate",
            {
                "scenario_pack": scenario,
                "hours": hours,
                "step_seconds": step_seconds,
                "top_k": top_k,
                "materialize_top_k": materialize_top_k,
            },
        )
        trust = evaluation.get("trust_summary", {})
        action_counts = trust.get("action_counts", {})
        deltas = evaluation.get("decision_deltas", [])
        top_delta = deltas[0] if deltas else {}
        top_reason = top_delta.get("refinement_reason") or ",".join(top_delta.get("trust_reason_codes", [])) or "-"
        rows.append(
            {
                "hours": hours,
                "windows": int(evaluation.get("window_count", 0)),
                "trust_accept": int(action_counts.get("accept", 0)),
                "trust_defer": int(action_counts.get("defer", 0)),
                "trust_refine": int(action_counts.get("refine", 0)),
                "changed": len(deltas),
                "top_reason": top_reason,
            }
        )
    return rows


def recommend_horizon_policy(sweep_rows: list[dict[str, int | float | str]]) -> tuple[str, str]:
    smoke_row = next((row for row in sweep_rows if int(row.get("windows", 0)) > 0), None)
    competition_row = None
    for row in sweep_rows:
        if int(row.get("trust_refine", 0)) > 0 and int(row.get("windows", 0)) > 0:
            competition_row = row
            break
    if competition_row is None:
        scored = [row for row in sweep_rows if int(row.get("windows", 0)) > 0]
        if scored:
            competition_row = sorted(
                scored,
                key=lambda row: (
                    int(row.get("changed", 0)),
                    int(row.get("trust_defer", 0)) + int(row.get("trust_refine", 0)),
                    int(row.get("windows", 0)),
                ),
                reverse=True,
            )[0]

    smoke_text = "-"
    if smoke_row is not None:
        smoke_text = f"{smoke_row['hours']:g}h smoke check"

    competition_text = "-"
    if competition_row is not None:
        competition_text = f"{competition_row['hours']:g}h ambiguity rehearsal"

    return smoke_text, competition_text


def build_status_report(
    *,
    root: Path,
    pytest_ok: bool,
    pytest_summary: str,
    capabilities: dict,
    rows: list[dict[str, str]],
    horizon_guidance: dict[str, dict[str, str]],
    review_candidates: dict[str, list[dict]],
    refine_diagnosis: dict[str, dict[str, int]],
) -> str:
    packet_path = root / "SUBMISSION_PACKET.md"
    casebook_path = root / "SUBMISSION_CASEBOOK.md"
    readiness_path = root / "SUBMISSION_READINESS.md"
    obs_eval_path = root / "OBSERVATION_VLA_EVAL.md"

    lines: list[str] = []
    lines.append("# SimSat Challenge Test Status")
    lines.append("")
    lines.append("## Fast Gate")
    lines.append("")
    lines.append(f"- Pytest pass: `{pytest_ok}`")
    lines.append(f"- Pytest summary: `{pytest_summary}`")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Sentinel enabled: `{capabilities.get('sentinel_enabled')}`")
    lines.append(f"- Mapbox enabled: `{capabilities.get('mapbox_enabled')}`")
    lines.append(f"- Sentinel-first challenge scoring: `{capabilities.get('sentinel_first_challenge_scoring')}`")
    lines.append(f"- ObservationVLA runtime: `{capabilities.get('observation_vla_runtime_mode')}`")
    lines.append("")
    lines.append("## Scenario Matrix")
    lines.append("")
    headers = [
        "scenario",
        "windows",
        "scaffold_accept",
        "trust_accept",
        "trust_refine",
        "changed",
        "accept_to_refine",
        "trust_yield",
        "reviewed_ready",
        "top_reason",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    lines.append("")
    lines.append("## Horizon Guidance")
    lines.append("")
    lines.append("| scenario | smoke_lane | competition_lane | rationale |")
    lines.append("| --- | --- | --- | --- |")
    for scenario in KNOWN_SCENARIO_PACKS:
        item = horizon_guidance.get(scenario, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario,
                    item.get("smoke_lane", "-"),
                    item.get("competition_lane", "-"),
                    item.get("rationale", "-"),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Refine Diagnosis")
    lines.append("")
    lines.append("| scenario | stored_traces | planner_refine | effective_refine | clip_local_traces | stub_family_traces |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for scenario in KNOWN_SCENARIO_PACKS:
        item = refine_diagnosis.get(scenario, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario,
                    str(item.get("total", 0)),
                    str(item.get("planner_refine", 0)),
                    str(item.get("effective_refine", 0)),
                    str(item.get("runtime_clip_local", 0)),
                    str(item.get("runtime_stub", 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Review Focus")
    lines.append("")
    for scenario in KNOWN_SCENARIO_PACKS:
        candidates = review_candidates.get(scenario, [])
        lines.append(f"### {scenario}")
        lines.append("")
        if not candidates:
            lines.append("- No pending review candidates.")
            lines.append("")
            continue
        candidate = candidates[0]
        lines.append(f"- Trace: `{candidate.get('trace_id')}`")
        lines.append(f"- Target: `{candidate.get('target_label') or candidate.get('target_id')}`")
        lines.append(f"- Recommended action: `{candidate.get('recommended_action')}`")
        current = candidate.get("current_outcome") or {}
        lines.append(f"- Current label source: `{current.get('label_source', 'unlabelled')}`")
        lines.append(f"- Needs operator review: `{candidate.get('needs_operator_review')}`")
        lines.append("")
    lines.append("## Artifact Presence")
    lines.append("")
    lines.append(f"- Submission packet exists: `{packet_path.exists()}`")
    lines.append(f"- Submission casebook exists: `{casebook_path.exists()}`")
    lines.append(f"- Submission readiness exists: `{readiness_path.exists()}`")
    lines.append(f"- ObservationVLA eval exists: `{obs_eval_path.exists()}`")
    lines.append("")
    lines.append("## Recommended Next Action")
    lines.append("")
    if not pytest_ok:
        lines.append("- Fix the failing runtime/test baseline before challenge-facing review work.")
    elif any(row["reviewed_ready"] != "True" for row in rows):
        lines.append("- Regenerate reviewed evidence or pin operator-reviewed submission cases for the missing scenarios.")
    elif any("ambiguity rehearsal" in item.get("competition_lane", "") and item.get("smoke_lane") != item.get("competition_lane") for item in horizon_guidance.values()):
        lines.append("- Keep short horizons as a smoke lane, but use the scenario-specific ambiguity-rehearsal horizons below for competition-facing evaluation and operator review.")
    elif all(row["trust_refine"] == "0" for row in rows):
        lines.append("- Planner-side `refine` is absent in the current live sweep; investigate trust-threshold calibration versus observation-side effective `refine` behavior in the stored trace corpus.")
    elif any(int(row["trust_refine"]) == 0 for row in rows):
        lines.append("- Investigate why the trust layer is no longer producing meaningful refine pressure in one or more scenario packs.")
    else:
        lines.append("- Run the review queue and then rebuild submission evidence for a fresh competition rehearsal.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact competition-facing challenge test matrix for SimSat.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument(
        "--policy",
        choices=["uniform", "smoke", "competition"],
        default="competition",
        help="Use shared per-scenario challenge hours instead of one uniform horizon.",
    )
    parser.add_argument("--scenario-hours", action="append", default=[], help="Per-scenario override in the form scenario_pack=hours")
    parser.add_argument("--step-seconds", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--materialize-top-k", type=int, default=2)
    parser.add_argument("--review-limit", type=int, default=10)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "CHALLENGE_TEST_STATUS.md"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pytest_ok, pytest_summary = run_pytest(root)

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        capabilities = client.get_json("/capabilities")
        rows: list[dict[str, str]] = []
        horizon_guidance: dict[str, dict[str, str]] = {}
        review_candidates: dict[str, list[dict]] = {}
        refine_diagnosis = diagnose_refine_gap(root)
        scenario_hours = resolve_scenario_hours(
            scenarios=KNOWN_SCENARIO_PACKS,
            default_hours=args.hours,
            policy=None if args.policy == "uniform" else args.policy,
            overrides=parse_scenario_hours(args.scenario_hours),
        )

        for scenario in KNOWN_SCENARIO_PACKS:
            sweep_rows = run_horizon_sweep(
                client,
                scenario=scenario,
                step_seconds=args.step_seconds,
                top_k=args.top_k,
                materialize_top_k=args.materialize_top_k,
            )
            smoke_lane, competition_lane = recommend_horizon_policy(sweep_rows)
            strongest_row = next(
                (
                    row for row in sweep_rows
                    if f"{row['hours']:g}h ambiguity rehearsal" == competition_lane
                ),
                None,
            )
            rationale = strongest_row["top_reason"] if strongest_row is not None else "-"
            horizon_guidance[scenario] = {
                "smoke_lane": smoke_lane,
                "competition_lane": competition_lane,
                "rationale": str(rationale),
            }
            evaluation, used_hours = run_adaptive_evaluation(
                client,
                scenario=scenario,
                requested_hours=scenario_hours[scenario],
                step_seconds=args.step_seconds,
                top_k=args.top_k,
                materialize_top_k=args.materialize_top_k,
            )
            evidence = client.get_json(
                "/encounter/evidence",
                scenario_pack=scenario,
                case_limit=3,
                trace_limit=100,
            )
            row = scorecard_row(evaluation, evidence)
            row["scenario"] = f"{scenario} ({used_hours:g}h)"
            rows.append(row)
            payload = client.get_json(
                "/observation-vla/review-candidates",
                scenario_pack=scenario,
                limit=args.review_limit,
            )
            review_candidates[scenario] = payload.get("candidates", [])

        report = build_status_report(
            root=root,
            pytest_ok=pytest_ok,
            pytest_summary=pytest_summary,
            capabilities=capabilities,
            rows=rows,
            horizon_guidance=horizon_guidance,
            review_candidates=review_candidates,
            refine_diagnosis=refine_diagnosis,
        )
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"Challenge test status written to {output_path}")
    except Exception as exc:
        print(f"Challenge test matrix failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
