from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

import requests


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def predicted_usefulness_score(assessment: dict) -> float:
    evidence = assessment.get("evidence", {})
    return _clamp(
        (0.45 * float(evidence.get("scene_match_score", 0.0)))
        + (0.25 * float(evidence.get("salience_score", 0.0)))
        + (0.10 * float(evidence.get("change_or_event_score", 0.0)))
        + (0.20 * (1.0 - float(evidence.get("occlusion_or_cloud_risk", 0.0))))
    )


def action_bucket(action: str) -> str:
    if action == "accept":
        return "accept"
    if action in {"defer", "refine"}:
        return "uncertain"
    return "skip"


class HttpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_json(self, path: str, **params) -> dict:
        response = requests.get(f"{self.base_url}{path}", params=params or None, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def post_json(self, path: str, payload: dict | None = None, **params) -> dict:
        response = requests.post(f"{self.base_url}{path}", params=params or None, json=payload, timeout=180)
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

        self._api = api
        self._client = TestClient(api)
        self._context = self._client.__enter__()

    def close(self) -> None:
        self._client.__exit__(None, None, None)

    def get_json(self, path: str, **params) -> dict:
        response = self._context.get(path, params=params or None)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def post_json(self, path: str, payload: dict | None = None, **params) -> dict:
        response = self._context.post(path, params=params or None, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def list_all_traces(self) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [trace.to_dict() for trace in service.store.list_traces(limit=5000)]

    def list_all_outcomes(self, current_only: bool = True) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [outcome.to_dict() for outcome in service.store.list_outcomes(limit=5000, current_only=current_only)]


def _fmt_counter(counter: Counter) -> str:
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter)) or "none"


def build_report(capabilities: dict, rows: list[dict]) -> str:
    stored_actions = Counter(row["stored_action"] for row in rows)
    current_actions = Counter(row["current_action"] for row in rows)
    shift_counts = Counter(f"{row['stored_action']}->{row['current_action']}" for row in rows)
    runtime_counts = Counter(row["stored_runtime"] for row in rows)
    scenario_counts = Counter(row["scenario_pack"] for row in rows)

    reviewed_rows = [row for row in rows if row["label_source"] == "operator_review"]
    simulated_rows = [row for row in rows if row["label_source"] == "simulated"]
    unlabelled_rows = [row for row in rows if row["label_source"] == "unlabelled"]
    current_refine_rate = sum(1 for row in rows if row["current_action"] == "refine") / len(rows) if rows else 0.0
    stored_accept_refine = [
        row for row in rows
        if row["stored_action"] == "accept" and row["current_action"] == "refine"
    ]
    accept_shift_material = len(stored_accept_refine) >= max(2, int(round(stored_actions.get("accept", 0) * 0.2)))

    useful_reviewed = [row for row in reviewed_rows if row.get("reviewed_useful") is True]
    useful_but_refined = [
        row for row in useful_reviewed
        if row["current_action"] == "refine"
    ]
    reviewed_mae = (
        sum(abs(float(row["predicted_usefulness_score"]) - float(row["reviewed_usefulness_score"])) for row in reviewed_rows) / len(reviewed_rows)
        if reviewed_rows else 0.0
    )

    per_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        per_target[row["target_label"]].append(row)

    conservative_targets: list[tuple[str, float, int, int]] = []
    for target_label, target_rows in per_target.items():
        target_refines = sum(1 for row in target_rows if row["current_action"] == "refine")
        target_accepts = sum(1 for row in target_rows if row["current_action"] == "accept")
        rate = target_refines / len(target_rows)
        conservative_targets.append((target_label, rate, target_refines, target_accepts))
    conservative_targets.sort(key=lambda item: (item[1], item[2], -item[3], item[0]), reverse=True)

    review_priority = sorted(
        [row for row in rows if row["label_source"] != "operator_review"],
        key=lambda row: (
            1 if row["label_source"] == "unlabelled" else 0,
            1 if row["current_action"] in {"refine", "defer", "skip"} else 0,
            1 if row["stored_action"] != row["current_action"] else 0,
            float(row["current_confidence"]),
            row["target_label"],
            row["trace_id"],
        ),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("# ObservationVLA Corpus Drift and Calibration Report")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- ObservationVLA runtime: `{capabilities.get('observation_vla_runtime_mode')}`")
    lines.append(f"- ObservationVLA model: `{capabilities.get('observation_vla_model_id')}`")
    lines.append(f"- Total traces reassessed: `{len(rows)}`")
    lines.append(f"- Stored runtime cohorts: `{_fmt_counter(runtime_counts)}`")
    lines.append(f"- Scenario coverage: `{_fmt_counter(scenario_counts)}`")
    lines.append("")
    lines.append("## Corpus Summary")
    lines.append("")
    lines.append(f"- Stored action distribution: `{_fmt_counter(stored_actions)}`")
    lines.append(f"- Current `clip_local` action distribution: `{_fmt_counter(current_actions)}`")
    lines.append(f"- Current refine rate: `{current_refine_rate:.2f}`")
    lines.append(f"- Stored `accept -> refine` shifts: `{len(stored_accept_refine)}` / `{stored_actions.get('accept', 0)}`")
    lines.append(f"- Current label sources: `operator_review={len(reviewed_rows)}, simulated={len(simulated_rows)}, unlabelled={len(unlabelled_rows)}`")
    lines.append("")
    lines.append("## Drift Signals")
    lines.append("")
    top_shifts = sorted(shift_counts.items(), key=lambda item: (-item[1], item[0]))
    for shift, count in top_shifts[:8]:
        lines.append(f"- `{shift}`: `{count}`")
    if reviewed_rows:
        lines.append("")
        lines.append("## Reviewed Calibration")
        lines.append("")
        lines.append(f"- Reviewed sample size: `{len(reviewed_rows)}`")
        lines.append(f"- Useful-but-refined reviewed cases: `{len(useful_but_refined)}` / `{len(useful_reviewed)}`")
        lines.append(f"- Reviewed usefulness score MAE: `{reviewed_mae:.2f}`")
        lines.append(
            f"- Exact action agreement on reviewed cases: "
            f"`{sum(1 for row in reviewed_rows if row['current_action'] == row['reviewer_action'])}/{len(reviewed_rows)}`"
        )
        lines.append(
            f"- Bucketed agreement on reviewed cases: "
            f"`{sum(1 for row in reviewed_rows if action_bucket(row['current_action']) == action_bucket(row['reviewer_action']))}/{len(reviewed_rows)}`"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if accept_shift_material:
        lines.append(
            "- The dominant drift pattern is `accept -> refine`, which points to an over-conservative action threshold rather than a complete loss of scene usefulness."
        )
    elif current_actions.get("accept", 0) >= stored_actions.get("accept", 0):
        lines.append(
            "- The worst `accept -> refine` collapse is gone. The current action policy now preserves historical high-confidence accepts while still keeping many newly added targets in `refine`."
        )
    else:
        lines.append(
            "- No strong `accept -> refine` pattern is visible in the current corpus."
        )
    if reviewed_rows and useful_but_refined:
        lines.append(
            "- On the reviewed set, the model still marks the scenes as useful while downgrading them to `refine`, which suggests the scene/evidence scores are usable but the decision policy is too cautious."
        )
    elif reviewed_rows:
        lines.append(
            "- On the small reviewed set, current `clip_local` actions now match the operator labels, but this is still a low-N check rather than a broad validation."
        )
    lines.append(
        "- The safest current claim is that `clip_local` is functioning as a conservative image-conditioned scorer with a partially calibrated action policy. It is not yet broadly validated for autonomous `accept` decisions."
    )
    lines.append("")
    lines.append("## Most Conservative Targets")
    lines.append("")
    lines.append("| target | refine_rate | refine_count | accept_count |")
    lines.append("| --- | --- | --- | --- |")
    for target_label, rate, refine_count, accept_count in conservative_targets[:10]:
        lines.append(
            f"| {target_label} | {rate:.2f} | {refine_count} | {accept_count} |"
        )
    lines.append("")
    lines.append("## Priority Review Shortlist")
    lines.append("")
    lines.append("Best next labels if we want to stress-test the conservative bias.")
    lines.append("")
    lines.append("| scenario | target | trace | stored_action | current_action | current_confidence | label_source |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in review_priority[:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario_pack"],
                    row["target_label"],
                    row["trace_id"],
                    row["stored_action"],
                    row["current_action"],
                    f"{row['current_confidence']:.2f}",
                    row["label_source"],
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Tighter Claim")
    lines.append("")
    if accept_shift_material:
        lines.append(
            "- Current evidence supports a narrow claim: `clip_local` produces image-conditioned usefulness signals across the corpus, but its action policy still leans conservative and can demote some prior `accept` cases to `refine`."
        )
    else:
        lines.append(
            "- Current evidence supports a narrow claim: `clip_local` now preserves the historical high-confidence `accept` cases while remaining conservative on newly added targets that still need human labels."
        )
    lines.append(
        "- The next human-review pass should focus on first labels for the newly added targets and any remaining non-accept high-confidence cases, rather than adding more near-duplicate accepted scenes."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a corpus-level drift/calibration report for ObservationVLA.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true", help="Run against an in-process FastAPI app instead of a live server.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "OBSERVATION_VLA_CORPUS.md"))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        capabilities = client.get_json("/capabilities")
        if hasattr(client, "list_all_traces"):
            traces = client.list_all_traces()[: args.limit]
        else:
            traces = client.get_json("/observation-vla/traces", limit=args.limit).get("traces", [])
        outcome_limit = min(max(args.limit * 2, 100), 1000)
        if hasattr(client, "list_all_outcomes"):
            outcomes = client.list_all_outcomes(current_only=True)[:outcome_limit]
        else:
            outcomes = client.get_json("/observation-vla/outcomes", limit=outcome_limit, current_only=True).get("outcomes", [])
        outcomes_by_trace = {str(outcome.get("trace_id")): outcome for outcome in outcomes}

        rows: list[dict] = []
        for trace in traces:
            trace_id = str(trace.get("trace_id"))
            reassessment_record = client.post_json(f"/observation-vla/reassess/{trace_id}", persist=False)
            current_assessment = reassessment_record.get("assessment", {})
            stored_assessment = trace.get("assessment", {})
            outcome = outcomes_by_trace.get(trace_id, {})
            rows.append(
                {
                    "scenario_pack": trace.get("scenario_pack", "all"),
                    "target_label": trace.get("sample", {}).get("target_label", trace.get("target_id", "")),
                    "trace_id": trace_id,
                    "stored_action": str(stored_assessment.get("recommended_action", "unknown")),
                    "stored_runtime": str(stored_assessment.get("runtime_mode", "unknown")),
                    "current_action": str(current_assessment.get("recommended_action", "unknown")),
                    "current_runtime": str(current_assessment.get("runtime_mode", "unknown")),
                    "current_confidence": float(current_assessment.get("evidence", {}).get("confidence", 0.0)),
                    "predicted_usefulness_score": predicted_usefulness_score(current_assessment),
                    "label_source": str(outcome.get("label_source", "unlabelled")),
                    "reviewer_action": str(outcome.get("operator_action", "")),
                    "reviewed_useful": outcome.get("useful"),
                    "reviewed_usefulness_score": float(outcome.get("usefulness_score", 0.0)),
                }
            )

        report = build_report(capabilities, rows)
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"ObservationVLA corpus report written to {output_path}")
        if args.inprocess and str(capabilities.get("observation_vla_runtime_mode", "")).startswith("clip_local") and os.name == "nt":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    except Exception as exc:
        print(f"ObservationVLA corpus report failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
