from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import requests

from submission_casebook import export_stimulus_image


def _sample_image_key(trace: dict) -> str:
    sample = trace.get("sample") or {}
    images = sample.get("images") or []
    target_id = trace.get("target_id") or sample.get("target_id") or "unknown_target"
    if images:
        image = images[0]
        return "|".join(
            [
                str(target_id),
                str(image.get("source") or "unknown_source"),
                str(image.get("timestamp") or "unknown_timestamp"),
                str(image.get("image_type") or "unknown_type"),
            ]
        )
    probe = trace.get("probe") or {}
    return "|".join(
        [
            str(target_id),
            str(probe.get("sentinel_source") or "unknown_source"),
            str(probe.get("sentinel_datetime") or "unknown_timestamp"),
            str(trace.get("window_id") or "unknown_window"),
        ]
    )


class HttpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_json(self, path: str, **params) -> dict:
        response = requests.get(f"{self.base_url}{path}", params=params or None, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def list_all_traces(self) -> list[dict]:
        return self.get_json("/observation-vla/traces", limit=1000).get("traces", [])

    def list_all_outcomes(self, current_only: bool = True) -> list[dict]:
        return self.get_json("/observation-vla/outcomes", limit=1000, current_only=current_only).get("outcomes", [])

    def list_submission_cases(self) -> list[dict]:
        return self.get_json("/observation-vla/submission-cases").get("cases", [])


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

    def list_all_traces(self) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [trace.to_dict() for trace in service.store.list_traces(limit=5000)]

    def list_all_outcomes(self, current_only: bool = True) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [outcome.to_dict() for outcome in service.store.list_outcomes(limit=5000, current_only=current_only)]

    def list_submission_cases(self) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [case.to_dict() for case in service.store.list_submission_cases()]


def _write_asset(trace: dict, assets_dir: Path) -> Path | None:
    stimulus_id = trace.get("stimulus_id")
    if not stimulus_id:
        return None
    scenario_pack = str(trace.get("scenario_pack") or "unknown")
    trace_id = str(trace.get("trace_id") or "unknown_trace")
    asset_path = assets_dir / f"{scenario_pack}_{trace_id}.png"
    if asset_path.exists():
        return asset_path.resolve()
    if export_stimulus_image(str(stimulus_id), asset_path):
        return asset_path.resolve()
    return None


def _candidate_rank(trace: dict, outcome: dict | None) -> tuple:
    assessment = trace.get("assessment") or {}
    evidence = assessment.get("evidence") or {}
    return (
        1 if (outcome or {}).get("label_source") == "unlabelled" else 0,
        1 if str(assessment.get("recommended_action")) == "accept" else 0,
        float(evidence.get("confidence", 0.0)),
        str(trace.get("created_at", "")),
        str(trace.get("trace_id", "")),
    )


def build_quality_report(
    client,
    *,
    output_path: Path,
    assets_dir: Path,
) -> Path:
    traces = client.list_all_traces()
    outcomes = client.list_all_outcomes(current_only=True)
    submission_cases = client.list_submission_cases()

    outcomes_by_trace = {str(outcome.get("trace_id")): outcome for outcome in outcomes}
    pinned_trace_ids = {str(case.get("trace_id")) for case in submission_cases}

    runtime_counts = Counter(str((trace.get("assessment") or {}).get("runtime_mode") or "unknown") for trace in traces)
    mode_counts = Counter(str((trace.get("assessment") or {}).get("assessment_mode") or "unknown") for trace in traces)

    traces_by_target: dict[str, list[dict]] = defaultdict(list)
    target_label_by_id: dict[str, str] = {}
    scenario_by_target: dict[str, str] = {}
    for trace in traces:
        target_id = str(trace.get("target_id") or (trace.get("sample") or {}).get("target_id") or "")
        if not target_id:
            continue
        traces_by_target[target_id].append(trace)
        target_label_by_id[target_id] = str((trace.get("sample") or {}).get("target_label") or target_id)
        scenario_by_target[target_id] = str(trace.get("scenario_pack") or "all")

    lines: list[str] = []
    lines.append("# ObservationVLA Corpus Quality Report")
    lines.append("")
    lines.append("## Runtime Mix")
    lines.append("")
    lines.append(f"- Total traces: `{len(traces)}`")
    lines.append(f"- Runtime distribution: `{', '.join(f'{key}={runtime_counts[key]}' for key in sorted(runtime_counts))}`")
    lines.append(f"- Assessment modes: `{', '.join(f'{key}={mode_counts[key]}' for key in sorted(mode_counts))}`")
    lines.append(f"- Current outcomes: `{len(outcomes)}`")
    lines.append(f"- Pinned submission cases: `{len(pinned_trace_ids)}`")
    lines.append("")
    lines.append("## Per-Target Coverage")
    lines.append("")
    lines.append("| scenario | target | traces | distinct_any | distinct_model_backed | clip_local | stub_fallback | operator_reviewed | pinned |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    shortlist: list[dict] = []
    for target_id in sorted(traces_by_target, key=lambda item: (scenario_by_target[item], target_label_by_id[item])):
        target_traces = traces_by_target[target_id]
        distinct_any = {_sample_image_key(trace) for trace in target_traces}
        model_backed_traces = [trace for trace in target_traces if str((trace.get("assessment") or {}).get("runtime_mode")) == "clip_local"]
        distinct_model_backed = {_sample_image_key(trace) for trace in model_backed_traces}
        operator_reviewed = 0
        for trace in target_traces:
            outcome = outcomes_by_trace.get(str(trace.get("trace_id")))
            if outcome and outcome.get("label_source") == "operator_review":
                operator_reviewed += 1
        pinned = any(str(trace.get("trace_id")) in pinned_trace_ids for trace in target_traces)
        lines.append(
            "| "
            + " | ".join(
                [
                    scenario_by_target[target_id],
                    target_label_by_id[target_id],
                    str(len(target_traces)),
                    str(len(distinct_any)),
                    str(len(distinct_model_backed)),
                    str(len(model_backed_traces)),
                    str(sum(1 for trace in target_traces if str((trace.get("assessment") or {}).get("runtime_mode")) == "stub_fallback")),
                    str(operator_reviewed),
                    str(pinned),
                ]
            )
            + " |"
        )

        pending_model_backed = [
            trace
            for trace in model_backed_traces
            if (outcomes_by_trace.get(str(trace.get("trace_id"))) or {}).get("label_source") != "operator_review"
        ]
        if pending_model_backed:
            best = max(
                pending_model_backed,
                key=lambda trace: _candidate_rank(trace, outcomes_by_trace.get(str(trace.get("trace_id")))),
            )
            shortlist.append(best)

    lines.append("")
    lines.append("## Model-Backed Morning Shortlist")
    lines.append("")
    lines.append("One pending `clip_local` case per target, avoiding the fallback-heavy tail.")
    lines.append("")

    if not shortlist:
        lines.append("- No pending model-backed review cases.")
    else:
        for trace in sorted(
            shortlist,
            key=lambda item: (
                str(item.get("scenario_pack") or "all"),
                str((item.get("sample") or {}).get("target_label") or item.get("target_id") or ""),
            ),
        ):
            assessment = trace.get("assessment") or {}
            evidence = assessment.get("evidence") or {}
            trace_id = str(trace.get("trace_id") or "")
            outcome = outcomes_by_trace.get(trace_id) or {}
            target_label = str((trace.get("sample") or {}).get("target_label") or trace.get("target_id") or trace_id)
            lines.append(f"### {target_label}")
            lines.append("")
            lines.append(f"- Scenario: `{trace.get('scenario_pack', 'all')}`")
            lines.append(f"- Trace: `{trace_id}`")
            lines.append(f"- Action: `{assessment.get('recommended_action', 'unknown')}`")
            lines.append(f"- Confidence: `{float(evidence.get('confidence', 0.0)):.2f}`")
            lines.append(f"- Label source: `{outcome.get('label_source', 'unlabelled')}`")
            asset_path = _write_asset(trace, assets_dir)
            if asset_path is not None:
                lines.append("")
                lines.append(f"![{target_label}]({asset_path})")
                lines.append("")
                lines.append(f"- Asset: [{asset_path}]({asset_path})")
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The corpus is now large enough that raw trace count is no longer the bottleneck; image-backed coverage is.")
    lines.append("- The key weak spots remain the historically pinned targets where repeated materializations mostly fall back instead of producing fresh `clip_local` cases.")
    lines.append("- The safest next human review should prioritize the model-backed shortlist above, not the full mixed-quality trace store.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ObservationVLA corpus quality and emit a model-backed review shortlist.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true")
    parser.add_argument("--output", default=str(Path("D:/SimSat/OBSERVATION_VLA_QUALITY.md")))
    parser.add_argument("--assets-dir", default=str(Path("D:/SimSat/review_queue_assets")))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        result = build_quality_report(
            client,
            output_path=Path(args.output),
            assets_dir=Path(args.assets_dir),
        )
        print(f"ObservationVLA quality report written to {result}")
    except Exception as exc:
        print(f"ObservationVLA quality report failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
