from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import requests

from review_queue_casebook import build_review_queue_with_generated_ids


def _parse_utc(text: str) -> datetime:
    normalized = text.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_base_times(value: str) -> list[str]:
    base_times = [item.strip() for item in value.split(",") if item.strip()]
    return base_times or ["2026-04-10T00:00:00Z"]


def _parse_target_ids(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _build_sweep_times(
    base_times: list[str],
    *,
    direction: str,
    sweep_count: int,
    step_days: int,
) -> list[str]:
    sweep_map: dict[str, datetime] = {}
    for base_time in base_times:
        base_dt = _parse_utc(base_time)
        offsets: list[int]
        if direction == "backward":
            offsets = [-(step_days * idx) for idx in range(sweep_count)]
        elif direction == "forward":
            offsets = [step_days * idx for idx in range(sweep_count)]
        else:
            offsets = sorted({-(step_days * idx) for idx in range(sweep_count)} | {step_days * idx for idx in range(sweep_count)}, reverse=True)
        for offset_days in offsets:
            dt = base_dt + timedelta(days=offset_days)
            sweep_map[_format_utc(dt)] = dt
    return [
        stamp
        for stamp, _ in sorted(
            sweep_map.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


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


_STUB_RUNTIME_MODES = frozenset({"stub", "stub_fallback"})


def _is_model_backed(trace: dict) -> bool:
    """Return True if the trace was assessed by a real (non-stub) VLA backend.

    Mirrors the runtime_mode gate in ObservationVLAService.is_image_backed_assessment()
    so that any backend accepted by assess_materialized_decision() also passes here.
    Previously this checked for "clip_local" only, causing quality_skips to
    accumulate whenever a non-clip_local real backend (http_endpoint,
    transformers_vlm_local, gemma4_haic_local) was active.
    """
    assessment = trace.get("assessment") or {}
    runtime_mode = str(assessment.get("runtime_mode") or "")
    return bool(runtime_mode) and runtime_mode not in _STUB_RUNTIME_MODES


@dataclass
class BuildStats:
    existing: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    added: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    duplicate_skips: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    quality_skips: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    assess_errors: list[str] = field(default_factory=list)


class HttpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_json(self, path: str, **params) -> dict:
        response = requests.get(f"{self.base_url}{path}", params=params or None, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def post_json(self, path: str, payload: dict | None = None, **params) -> dict:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            params=params or None,
            timeout=240,
        )
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def list_all_traces(self) -> list[dict]:
        return self.get_json("/observation-vla/traces", limit=1000).get("traces", [])


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
        response = self._context.post(path, json=payload, params=params or None)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()

    def list_all_traces(self) -> list[dict]:
        service = self._api.state.observation_vla_service
        return [trace.to_dict() for trace in service.store.list_traces(limit=5000)]


def _scenario_targets(targets_payload: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for target in targets_payload.get("targets", []):
        scenario_pack = (target.get("metadata") or {}).get("scenario_pack", "default")
        grouped[str(scenario_pack)].append(target)
    return grouped


def _filter_scenario_targets(
    scenario_targets: dict[str, list[dict]],
    target_ids: set[str],
) -> dict[str, list[dict]]:
    if not target_ids:
        return scenario_targets
    filtered: dict[str, list[dict]] = {}
    for scenario_pack, targets in scenario_targets.items():
        selected = [target for target in targets if str(target.get("target_id")) in target_ids]
        if selected:
            filtered[scenario_pack] = selected
    return filtered


def _build_existing_index(
    client,
    scenario_targets: dict[str, list[dict]],
    quota_mode: str,
) -> tuple[dict[str, set[str]], BuildStats]:
    traces = client.list_all_traces()
    keys_by_target: dict[str, set[str]] = defaultdict(set)
    stats = BuildStats()
    tracked_target_ids = {
        str(target.get("target_id"))
        for targets in scenario_targets.values()
        for target in targets
    }
    for trace in traces:
        target_id = str(trace.get("target_id") or (trace.get("sample") or {}).get("target_id") or "")
        if target_id not in tracked_target_ids:
            continue
        if quota_mode == "model_backed" and not _is_model_backed(trace):
            continue
        key = _sample_image_key(trace)
        if key not in keys_by_target[target_id]:
            keys_by_target[target_id].add(key)
            stats.existing[target_id] += 1
    return keys_by_target, stats


def _best_decisions_by_target(decisions: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for decision in decisions:
        target_id = str(decision.get("target_id") or "")
        if not target_id:
            continue
        current = best.get(target_id)
        if current is None or float(decision.get("combined_score", 0.0)) > float(current.get("combined_score", 0.0)):
            best[target_id] = decision
    return best


def _write_summary(
    output_path: Path,
    *,
    base_start_times: list[str],
    direction: str,
    quota_mode: str,
    target_ids: set[str],
    sweep_times: list[str],
    target_quota: int,
    max_quality_skips_per_target: int,
    queue_limit_per_scenario: int,
    scenario_targets: dict[str, list[dict]],
    keys_by_target: dict[str, set[str]],
    stats: BuildStats,
) -> Path:
    lines: list[str] = []
    lines.append("# SimSat Review Set Build")
    lines.append("")
    lines.append(f"- Base start times: `{', '.join(base_start_times)}`")
    lines.append(f"- Sweep direction: `{direction}`")
    lines.append(f"- Quota mode: `{quota_mode}`")
    lines.append(f"- Target filter: `{', '.join(sorted(target_ids)) if target_ids else 'all'}`")
    lines.append(f"- Sweep count: `{len(sweep_times)}`")
    lines.append(f"- Sweep times: `{', '.join(sweep_times)}`")
    lines.append(f"- Target quota: `{target_quota}`")
    lines.append(f"- Max quality skips per target: `{max_quality_skips_per_target}`")
    lines.append(f"- Queue limit per scenario: `{queue_limit_per_scenario}`")
    lines.append("")
    lines.append("| scenario | target | existing_distinct | added_distinct | total_distinct | duplicate_skips | quality_skips |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for scenario_pack in sorted(scenario_targets):
        for target in scenario_targets[scenario_pack]:
            target_id = str(target.get("target_id"))
            label = str(target.get("label", target_id))
            lines.append(
                "| "
                + " | ".join(
                    [
                        scenario_pack,
                        label,
                        str(stats.existing.get(target_id, 0)),
                        str(stats.added.get(target_id, 0)),
                        str(len(keys_by_target.get(target_id, set()))),
                        str(stats.duplicate_skips.get(target_id, 0)),
                        str(stats.quality_skips.get(target_id, 0)),
                    ]
                )
                + " |"
            )
    if stats.assess_errors:
        lines.append("")
        lines.append("## Assess Errors")
        lines.append("")
        for error in stats.assess_errors:
            lines.append(f"- {error}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_review_set(
    client,
    *,
    base_times: list[str],
    direction: str,
    quota_mode: str,
    target_ids: set[str],
    sweep_count: int,
    step_days: int,
    hours: float,
    step_seconds: int,
    top_k: int,
    target_quota: int,
    max_quality_skips_per_target: int,
    queue_limit_per_scenario: int,
    queue_output: Path,
    queue_assets_dir: Path,
    summary_output: Path,
) -> tuple[Path, Path]:
    targets_payload = client.get_json("/encounter/targets")
    scenario_targets = _filter_scenario_targets(_scenario_targets(targets_payload), target_ids)
    keys_by_target, stats = _build_existing_index(client, scenario_targets, quota_mode)
    sweep_times = _build_sweep_times(
        base_times,
        direction=direction,
        sweep_count=sweep_count,
        step_days=step_days,
    )
    generated_trace_ids_total: set[str] = set()

    for sweep_time in sweep_times:
        for scenario_pack, targets in scenario_targets.items():
            quota_needed = {
                str(target.get("target_id")): max(0, target_quota - len(keys_by_target.get(str(target.get("target_id")), set())))
                for target in targets
            }
            if not any(quota_needed.values()):
                continue

            planned = client.post_json(
                "/encounter/plan",
                {
                    "scenario_pack": scenario_pack,
                    "start_time": sweep_time,
                    "hours": hours,
                    "step_seconds": step_seconds,
                    "top_k": top_k,
                },
            )
            best_by_target = _best_decisions_by_target(planned.get("decisions", []))
            for target in targets:
                target_id = str(target.get("target_id"))
                if quota_needed.get(target_id, 0) <= 0:
                    continue
                if quota_mode == "model_backed" and stats.quality_skips.get(target_id, 0) >= max_quality_skips_per_target:
                    continue
                decision = best_by_target.get(target_id)
                if decision is None:
                    continue
                try:
                    assessed = client.post_json(f"/encounter/decision/{decision['decision_id']}/assess")
                    trace_payload = client.get_json(f"/observation-vla/trace/{assessed['trace_id']}")
                except Exception as exc:  # pragma: no cover - build should keep going
                    stats.assess_errors.append(
                        f"{scenario_pack}/{target_id} @ {sweep_time}: {exc}"
                    )
                    continue
                trace = trace_payload.get("trace", {})
                if quota_mode == "model_backed" and not _is_model_backed(trace):
                    stats.quality_skips[target_id] += 1
                    continue
                key = _sample_image_key(trace)
                if key in keys_by_target[target_id]:
                    stats.duplicate_skips[target_id] += 1
                    continue
                keys_by_target[target_id].add(key)
                stats.added[target_id] += 1
                generated_trace_ids_total.add(str(trace.get("trace_id", assessed["trace_id"])))

    build_review_queue_with_generated_ids(
        client,
        output_path=queue_output,
        assets_dir=queue_assets_dir,
        limit_per_scenario=queue_limit_per_scenario,
        seed_generated_trace_ids=generated_trace_ids_total,
    )
    _write_summary(
        summary_output,
        base_start_times=base_times,
        direction=direction,
        quota_mode=quota_mode,
        target_ids=target_ids,
        sweep_times=sweep_times,
        target_quota=target_quota,
        max_quality_skips_per_target=max_quality_skips_per_target,
        queue_limit_per_scenario=queue_limit_per_scenario,
        scenario_targets=scenario_targets,
        keys_by_target=keys_by_target,
        stats=stats,
    )
    return queue_output, summary_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build out a wider SimSat review set across targets and sweep times.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true")
    parser.add_argument("--start-time", default="2026-04-10T00:00:00Z")
    parser.add_argument("--base-times", default="")
    parser.add_argument("--direction", choices=["backward", "forward", "both"], default="backward")
    parser.add_argument("--quota-mode", choices=["any", "model_backed"], default="any")
    parser.add_argument("--target-ids", default="")
    parser.add_argument("--sweep-count", type=int, default=12)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--hours", type=float, default=48.0)
    parser.add_argument("--step-seconds", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--target-quota", type=int, default=3)
    parser.add_argument("--max-quality-skips-per-target", type=int, default=12)
    parser.add_argument("--queue-limit-per-scenario", type=int, default=8)
    parser.add_argument("--queue-output", default=str(Path("D:/SimSat/REVIEW_QUEUE.md")))
    parser.add_argument("--queue-assets-dir", default=str(Path("D:/SimSat/review_queue_assets")))
    parser.add_argument("--summary-output", default=str(Path("D:/SimSat/REVIEW_SET_BUILD.md")))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        base_times = _parse_base_times(args.base_times) if args.base_times.strip() else [args.start_time]
        target_ids = _parse_target_ids(args.target_ids)
        queue_output, summary_output = build_review_set(
            client,
            base_times=base_times,
            direction=args.direction,
            quota_mode=args.quota_mode,
            target_ids=target_ids,
            sweep_count=args.sweep_count,
            step_days=args.step_days,
            hours=args.hours,
            step_seconds=args.step_seconds,
            top_k=args.top_k,
            target_quota=args.target_quota,
            max_quality_skips_per_target=args.max_quality_skips_per_target,
            queue_limit_per_scenario=args.queue_limit_per_scenario,
            queue_output=Path(args.queue_output),
            queue_assets_dir=Path(args.queue_assets_dir),
            summary_output=Path(args.summary_output),
        )
        print(f"Review queue written to {queue_output}")
        print(f"Review build summary written to {summary_output}")
    except Exception as exc:
        print(f"Review set build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
