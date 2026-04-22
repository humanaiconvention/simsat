from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import requests


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

    def list_all_traces(self) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [trace.to_dict() for trace in service.store.list_traces(limit=5000)]

    def list_all_outcomes(self, current_only: bool = True) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [outcome.to_dict() for outcome in service.store.list_outcomes(limit=5000, current_only=current_only)]


def _all_review_candidates(client, scenario_packs: list[str]) -> list[dict]:
    all_candidates: list[dict] = []
    for scenario_pack in scenario_packs:
        payload = client.get_json("/observation-vla/review-candidates", scenario_pack=scenario_pack, limit=1000)
        all_candidates.extend(payload.get("candidates", []))
    return all_candidates


def build_backlog_report(
    client,
    *,
    assets_dir: Path,
    output_path: Path,
) -> Path:
    targets_payload = client.get_json("/encounter/targets")
    scenario_packs = targets_payload.get("scenario_packs", [])
    targets = targets_payload.get("targets", [])
    target_by_id = {str(target.get("target_id")): target for target in targets}

    if hasattr(client, "list_all_traces"):
        traces = client.list_all_traces()
    else:
        traces = client.get_json("/observation-vla/traces", limit=1000).get("traces", [])
    if hasattr(client, "list_all_outcomes"):
        outcomes = client.list_all_outcomes(current_only=True)
    else:
        outcomes = client.get_json("/observation-vla/outcomes", limit=1000, current_only=True).get("outcomes", [])
    submission_cases = client.get_json("/observation-vla/submission-cases").get("cases", [])

    outcomes_by_trace = {str(outcome.get("trace_id")): outcome for outcome in outcomes}
    pinned_trace_ids = {str(case.get("trace_id")) for case in submission_cases}

    traces_by_target: dict[str, list[dict]] = defaultdict(list)
    distinct_keys_by_target: dict[str, set[str]] = defaultdict(set)
    reviewed_count_by_target: Counter[str] = Counter()
    simulated_count_by_target: Counter[str] = Counter()

    for trace in traces:
        target_id = str(trace.get("target_id") or (trace.get("sample") or {}).get("target_id") or "")
        if not target_id:
            continue
        traces_by_target[target_id].append(trace)
        distinct_keys_by_target[target_id].add(_sample_image_key(trace))
        outcome = outcomes_by_trace.get(str(trace.get("trace_id")))
        if outcome is None:
            continue
        if outcome.get("label_source") == "operator_review":
            reviewed_count_by_target[target_id] += 1
        else:
            simulated_count_by_target[target_id] += 1

    review_candidates = _all_review_candidates(client, scenario_packs)
    shortlist_by_target: dict[str, dict] = {}
    for candidate in review_candidates:
        if not candidate.get("needs_operator_review"):
            continue
        target_id = str(candidate.get("target_id") or "")
        if not target_id:
            continue
        current = shortlist_by_target.get(target_id)
        rank = (
            1 if candidate.get("runtime_mode") == "clip_local" else 0,
            1 if (candidate.get("current_outcome") or {}).get("label_source") == "simulated" else 0,
            candidate.get("trace_id", ""),
        )
        if current is None:
            shortlist_by_target[target_id] = candidate
            continue
        current_rank = (
            1 if current.get("runtime_mode") == "clip_local" else 0,
            1 if (current.get("current_outcome") or {}).get("label_source") == "simulated" else 0,
            current.get("trace_id", ""),
        )
        if rank > current_rank:
            shortlist_by_target[target_id] = candidate

    lines: list[str] = []
    lines.append("# SimSat Review Backlog")
    lines.append("")
    lines.append(f"- Total stored traces: `{len(traces)}`")
    lines.append(f"- Current outcomes: `{len(outcomes)}`")
    lines.append(f"- Pinned submission traces: `{len(pinned_trace_ids)}`")
    lines.append(f"- Targets represented: `{len(traces_by_target)}`")
    lines.append("")
    lines.append("## Corpus Coverage")
    lines.append("")
    lines.append("| scenario | target | traces | distinct_cases | operator_reviewed | simulated | pinned |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for scenario_pack in scenario_packs:
        scenario_targets = [
            target for target in targets
            if (target.get("metadata") or {}).get("scenario_pack") == scenario_pack
        ]
        for target in scenario_targets:
            target_id = str(target.get("target_id"))
            pinned = any(
                case.get("trace_id") in {trace.get("trace_id") for trace in traces_by_target.get(target_id, [])}
                for case in submission_cases
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        scenario_pack,
                        str(target.get("label", target_id)),
                        str(len(traces_by_target.get(target_id, []))),
                        str(len(distinct_keys_by_target.get(target_id, set()))),
                        str(reviewed_count_by_target.get(target_id, 0)),
                        str(simulated_count_by_target.get(target_id, 0)),
                        str(pinned),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Next Review Shortlist")
    lines.append("")
    lines.append("One best pending candidate per target.")
    lines.append("")
    for scenario_pack in scenario_packs:
        lines.append(f"### {scenario_pack}")
        lines.append("")
        scenario_targets = [
            target for target in targets
            if (target.get("metadata") or {}).get("scenario_pack") == scenario_pack
        ]
        wrote_any = False
        for target in scenario_targets:
            target_id = str(target.get("target_id"))
            candidate = shortlist_by_target.get(target_id)
            if candidate is None:
                continue
            wrote_any = True
            trace_id = str(candidate.get("trace_id"))
            runtime_mode = str(candidate.get("runtime_mode"))
            recommended_action = str(candidate.get("recommended_action"))
            current_outcome = candidate.get("current_outcome") or {}
            asset_path = assets_dir / f"{scenario_pack}_{trace_id}.png"
            lines.append(f"- `{target.get('label', target_id)}`")
            lines.append(f"  trace=`{trace_id}` runtime=`{runtime_mode}` recommended=`{recommended_action}` label_source=`{current_outcome.get('label_source', 'unlabelled')}`")
            if asset_path.exists():
                lines.append(f"  asset=[{asset_path.resolve()}]({asset_path.resolve()})")
        if not wrote_any:
            lines.append("- No pending review candidates.")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the current SimSat review corpus and next-review shortlist.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true")
    parser.add_argument("--assets-dir", default=str(Path("D:/SimSat/review_queue_assets")))
    parser.add_argument("--output", default=str(Path("D:/SimSat/REVIEW_BACKLOG.md")))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        result = build_backlog_report(
            client,
            assets_dir=Path(args.assets_dir),
            output_path=Path(args.output),
        )
        print(f"Review backlog written to {result}")
    except Exception as exc:
        print(f"Review backlog generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
