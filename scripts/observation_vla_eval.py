from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import requests


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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

    def list_all_outcomes(self, current_only: bool = True) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [outcome.to_dict() for outcome in service.store.list_outcomes(limit=5000, current_only=current_only)]


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


def build_report(capabilities: dict, rows: list[dict]) -> str:
    exact_agreement = sum(1 for row in rows if row["exact_agreement"]) / len(rows) if rows else 0.0
    bucket_agreement = sum(1 for row in rows if row["bucket_agreement"]) / len(rows) if rows else 0.0
    useful_accuracy = sum(1 for row in rows if row["useful_agreement"]) / len(rows) if rows else 0.0
    mae = sum(row["usefulness_abs_error"] for row in rows) / len(rows) if rows else 0.0

    lines: list[str] = []
    lines.append("# ObservationVLA Reviewed Evaluation")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- ObservationVLA runtime: `{capabilities.get('observation_vla_runtime_mode')}`")
    lines.append(f"- ObservationVLA model: `{capabilities.get('observation_vla_model_id')}`")
    lines.append(f"- Reviewed sample size: `{len(rows)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Exact operator-action agreement: `{exact_agreement:.2f}`")
    lines.append(f"- Bucketed action agreement: `{bucket_agreement:.2f}`")
    lines.append(f"- Useful / not-useful agreement: `{useful_accuracy:.2f}`")
    lines.append(f"- Usefulness score MAE: `{mae:.2f}`")
    lines.append("")
    lines.append("## Reviewed Cases")
    lines.append("")
    lines.append("| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario_pack"],
                    row["target_label"],
                    row["trace_id"],
                    row["model_action"],
                    row["reviewer_action"],
                    str(row["predicted_useful"]),
                    str(row["reviewed_useful"]),
                    f"{row['predicted_usefulness_score']:.2f}",
                    f"{row['reviewed_usefulness_score']:.2f}",
                    f"{row['usefulness_abs_error']:.2f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Tight Claim")
    lines.append("")
    if not rows:
        lines.append("- No reviewed traces are available yet; no model-backed claim is warranted.")
    else:
        lines.append(
            "- Current claim: the ObservationVLA lane is now image-model-backed and shows "
            f"`{exact_agreement:.2f}` exact action agreement over `{len(rows)}` operator-reviewed Sentinel cases."
        )
        lines.append(
            "- Conservative claim: this is an early, low-N validation of image-conditioned usefulness scoring, not a broad benchmark."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current ObservationVLA backend against operator-reviewed traces.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true", help="Run against an in-process FastAPI app instead of a live server.")
    parser.add_argument("--scenario-pack", default="all")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default=str(Path("D:/SimSat/OBSERVATION_VLA_EVAL.md")))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        capabilities = client.get_json("/capabilities")
        if hasattr(client, "list_all_outcomes"):
            all_outcomes = client.list_all_outcomes(current_only=True)
        else:
            all_outcomes = client.get_json("/observation-vla/outcomes", limit=max(args.limit, 1000), current_only=True).get("outcomes", [])
        reviewed_outcomes = [outcome for outcome in all_outcomes if outcome.get("label_source") == "operator_review"]
        if args.scenario_pack not in {"", "all"}:
            reviewed_outcomes = [
                outcome for outcome in reviewed_outcomes
                if client.get_json(f"/observation-vla/trace/{outcome['trace_id']}").get("trace", {}).get("scenario_pack") == args.scenario_pack
            ]
        reviewed_outcomes = reviewed_outcomes[: args.limit]

        rows: list[dict] = []
        for outcome in reviewed_outcomes:
            trace_id = outcome["trace_id"]
            trace_payload = client.get_json(f"/observation-vla/trace/{trace_id}")
            trace = trace_payload.get("trace", {})
            assessment_record = client.post_json(f"/observation-vla/reassess/{trace_id}", persist=False)
            assessment = assessment_record.get("assessment", {})
            predicted_score = predicted_usefulness_score(assessment)
            predicted_useful = bool(assessment.get("evidence", {}).get("usable_observation", False))
            reviewer_action = str(outcome.get("operator_action", ""))
            model_action = str(assessment.get("recommended_action", ""))
            rows.append(
                {
                    "scenario_pack": trace.get("scenario_pack", "all"),
                    "target_label": trace.get("sample", {}).get("target_label", trace.get("target_id", "")),
                    "trace_id": trace_id,
                    "model_action": model_action,
                    "reviewer_action": reviewer_action,
                    "predicted_useful": predicted_useful,
                    "reviewed_useful": bool(outcome.get("useful", False)),
                    "predicted_usefulness_score": predicted_score,
                    "reviewed_usefulness_score": float(outcome.get("usefulness_score", 0.0)),
                    "usefulness_abs_error": abs(predicted_score - float(outcome.get("usefulness_score", 0.0))),
                    "exact_agreement": model_action == reviewer_action,
                    "bucket_agreement": action_bucket(model_action) == action_bucket(reviewer_action),
                    "useful_agreement": predicted_useful == bool(outcome.get("useful", False)),
                }
            )

        report = build_report(capabilities, rows)
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"ObservationVLA evaluation written to {output_path}")
        if args.inprocess and str(capabilities.get("observation_vla_runtime_mode", "")).startswith("clip_local") and os.name == "nt":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    except Exception as exc:
        print(f"ObservationVLA evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
