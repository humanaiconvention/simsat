from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import requests

from submission_casebook import export_stimulus_image


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
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()


def _scenario_targets(targets_payload: dict, scenario_pack: str) -> list[dict]:
    return [
        target
        for target in targets_payload.get("targets", [])
        if (target.get("metadata") or {}).get("scenario_pack") == scenario_pack
    ]


def _list_scenario_trace_target_ids(client, scenario_pack: str, limit: int = 100) -> set[str]:
    payload = client.get_json("/observation-vla/traces", limit=limit)
    return {
        trace.get("target_id") or (trace.get("sample") or {}).get("target_id")
        for trace in payload.get("traces", [])
        if trace.get("scenario_pack") == scenario_pack
    }


def _generate_missing_target_traces(
    client,
    scenario_pack: str,
    target_ids: list[str],
    *,
    hours: float = 48.0,
    step_seconds: int = 120,
    top_k: int = 100,
) -> set[str]:
    if not target_ids:
        return set()
    planned = client.post_json(
        "/encounter/plan",
        {
            "scenario_pack": scenario_pack,
            "hours": hours,
            "step_seconds": step_seconds,
            "top_k": top_k,
        },
    )
    decisions = planned.get("decisions", [])
    best_by_target: dict[str, dict] = {}
    for decision in decisions:
        target_id = decision.get("target_id")
        if target_id not in target_ids:
            continue
        current = best_by_target.get(target_id)
        if current is None or float(decision.get("combined_score", 0.0)) > float(current.get("combined_score", 0.0)):
            best_by_target[target_id] = decision

    generated_trace_ids: set[str] = set()
    for target_id in target_ids:
        decision = best_by_target.get(target_id)
        if decision is None:
            continue
        assessed = client.post_json(f"/encounter/decision/{decision['decision_id']}/assess")
        trace_id = assessed.get("trace_id")
        if trace_id:
            generated_trace_ids.add(str(trace_id))
    return generated_trace_ids


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_enriched_candidate(
    client,
    scenario_pack: str,
    candidate: dict,
    assets_dir: Path,
    generated_trace_ids: set[str],
) -> dict:
    trace_id = candidate["trace_id"]
    bundle = client.get_json(f"/observation-vla/review-bundle/{trace_id}")
    trace = bundle.get("trace", {})
    target = bundle.get("target", {})
    current_outcome = bundle.get("current_outcome") or {}
    reassessed = None
    reassess_error = None
    try:
        reassessed = client.post_json(f"/observation-vla/reassess/{trace_id}", persist=False)
    except Exception as exc:  # pragma: no cover - best-effort review aid
        reassess_error = str(exc)

    stimulus_id = trace.get("stimulus_id")
    asset_path = assets_dir / f"{scenario_pack}_{trace_id}.png"
    image_written = bool(stimulus_id) and export_stimulus_image(str(stimulus_id), asset_path)
    image_hash = _sha256_file(asset_path) if image_written else None
    stored_action = trace.get("assessment", {}).get("recommended_action")
    current_backend_action = None
    current_backend_runtime = None
    current_confidence = None
    if reassessed is not None:
        reassessment = reassessed.get("assessment", {})
        evidence = reassessment.get("evidence", {})
        current_backend_action = reassessment.get("recommended_action")
        current_backend_runtime = reassessment.get("runtime_mode")
        current_confidence = evidence.get("confidence")

    return {
        "candidate": candidate,
        "bundle": bundle,
        "trace": trace,
        "target": target,
        "current_outcome": current_outcome,
        "reassessed": reassessed,
        "reassess_error": reassess_error,
        "asset_path": asset_path,
        "image_written": image_written,
        "image_hash": image_hash,
        "stored_action": stored_action,
        "current_backend_action": current_backend_action,
        "current_backend_runtime": current_backend_runtime,
        "current_confidence": current_confidence,
        "generated_for_queue": trace_id in generated_trace_ids,
        "action_disagreement": bool(
            stored_action
            and current_backend_action
            and stored_action != current_backend_action
        ),
    }


def _select_diverse_candidates(
    enriched_candidates: list[dict],
    limit_per_scenario: int,
) -> list[dict]:
    ranked = sorted(
        enriched_candidates,
        key=lambda item: (
            1 if item.get("generated_for_queue") else 0,
            1 if item.get("action_disagreement") else 0,
            1 if item.get("image_written") else 0,
            float(item.get("current_confidence") or 0.0),
            item.get("trace", {}).get("created_at", ""),
            item.get("trace", {}).get("trace_id", ""),
        ),
        reverse=True,
    )

    selected: list[dict] = []
    selected_trace_ids: set[str] = set()
    seen_targets: set[str] = set()
    seen_hashes: set[str] = set()

    def try_add(item: dict, *, require_new_target: bool = False, require_new_hash: bool = False) -> bool:
        trace_id = item.get("trace", {}).get("trace_id")
        target_id = item.get("target", {}).get("target_id") or item.get("trace", {}).get("target_id")
        image_hash = item.get("image_hash")
        if not trace_id or trace_id in selected_trace_ids:
            return False
        if require_new_target and target_id in seen_targets:
            return False
        if require_new_hash and image_hash is not None and image_hash in seen_hashes:
            return False
        selected.append(item)
        selected_trace_ids.add(trace_id)
        if target_id:
            seen_targets.add(str(target_id))
        if image_hash is not None:
            seen_hashes.add(str(image_hash))
        return True

    for item in ranked:
        if len(selected) >= limit_per_scenario:
            break
        try_add(item, require_new_target=True, require_new_hash=True)

    for item in ranked:
        if len(selected) >= limit_per_scenario:
            break
        try_add(item, require_new_target=False, require_new_hash=True)

    for item in ranked:
        if len(selected) >= limit_per_scenario:
            break
        try_add(item, require_new_target=False, require_new_hash=False)

    return selected


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

    def post_json(self, path: str, payload: dict | None = None, **params) -> dict:
        response = self._context.post(path, json=payload, params=params or None)
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
        return response.json()


def build_review_queue(client, output_path: Path, assets_dir: Path, limit_per_scenario: int) -> Path:
    return build_review_queue_with_generated_ids(
        client,
        output_path=output_path,
        assets_dir=assets_dir,
        limit_per_scenario=limit_per_scenario,
        seed_generated_trace_ids=set(),
    )


def build_review_queue_with_generated_ids(
    client,
    output_path: Path,
    assets_dir: Path,
    limit_per_scenario: int,
    seed_generated_trace_ids: set[str] | None = None,
) -> Path:
    targets_payload = client.get_json("/encounter/targets")
    scenario_packs = targets_payload.get("scenario_packs", [])
    seed_generated_trace_ids = set(seed_generated_trace_ids or set())
    lines: list[str] = []
    lines.append("# SimSat Review Queue")
    lines.append("")
    lines.append("Unreviewed or simulated-labelled traces prioritized for the next human review pass.")
    lines.append("")

    for scenario_pack in scenario_packs:
        scenario_targets = _scenario_targets(targets_payload, scenario_pack)
        traced_target_ids = _list_scenario_trace_target_ids(client, scenario_pack=scenario_pack)
        missing_target_ids = [
            str(target.get("target_id"))
            for target in scenario_targets
            if str(target.get("target_id")) not in traced_target_ids
        ]
        locally_generated_trace_ids = _generate_missing_target_traces(
            client,
            scenario_pack=scenario_pack,
            target_ids=missing_target_ids,
        )
        generated_trace_ids = seed_generated_trace_ids | locally_generated_trace_ids

        payload = client.get_json(
            "/observation-vla/review-candidates",
            scenario_pack=scenario_pack,
            limit=min(max(limit_per_scenario * 20, 36), 1000),
        )
        raw_candidates = [
            candidate for candidate in payload.get("candidates", [])
            if candidate.get("needs_operator_review") and not candidate.get("is_pinned_submission_case")
        ]
        enriched_candidates = [
            _build_enriched_candidate(
                client,
                scenario_pack=scenario_pack,
                candidate=candidate,
                assets_dir=assets_dir,
                generated_trace_ids=generated_trace_ids,
            )
            for candidate in raw_candidates
        ]
        candidates = _select_diverse_candidates(enriched_candidates, limit_per_scenario=limit_per_scenario)
        lines.append(f"## {scenario_pack}")
        lines.append("")
        if missing_target_ids:
            lines.append(f"- Scenario targets discovered: `{', '.join(target.get('label', target.get('target_id', 'unknown')) for target in scenario_targets)}`")
            lines.append(f"- Fresh traces generated for missing targets: `{len(locally_generated_trace_ids)}`")
            lines.append("")
        if not candidates:
            lines.append("- No pending review candidates.")
            lines.append("")
            continue
        for item in candidates:
            trace = item["trace"]
            target = item["target"]
            current_outcome = item["current_outcome"]
            trace_id = trace.get("trace_id", "")
            lines.append(f"### {target.get('label', trace.get('target_id', trace_id))}")
            lines.append("")
            lines.append(f"- Trace: `{trace_id}`")
            lines.append(f"- Target ID: `{target.get('target_id', trace.get('target_id', 'unknown'))}`")
            lines.append(f"- Current label source: `{current_outcome.get('label_source', 'unlabelled')}`")
            lines.append(f"- Stored suggested action: `{item.get('stored_action')}`")
            lines.append(f"- Stored runtime: `{trace.get('assessment', {}).get('runtime_mode')}`")
            lines.append(f"- Freshly generated for queue: `{item.get('generated_for_queue')}`")
            lines.append(f"- Action disagreement: `{item.get('action_disagreement')}`")
            if item.get("reassessed") is not None:
                lines.append(f"- Current backend action: `{item.get('current_backend_action')}`")
                lines.append(f"- Current backend runtime: `{item.get('current_backend_runtime')}`")
                lines.append(f"- Current confidence: `{item.get('current_confidence')}`")
            elif item.get("reassess_error"):
                lines.append(f"- Current backend reassessment: `unavailable` ({item.get('reassess_error')})")
            lines.append(f"- Review note: `{target.get('metadata', {}).get('challenge_note', '')}`")
            if item.get("image_written"):
                lines.append("")
                lines.append(f"![{target.get('label', 'Review candidate')}]({item['asset_path'].resolve()})")
                lines.append("")
                lines.append(f"- Image asset: [{item['asset_path'].resolve()}]({item['asset_path'].resolve()})")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the next SimSat review candidates into a markdown casebook.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true")
    parser.add_argument("--limit-per-scenario", type=int, default=2)
    _repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", default=str(_repo_root / "REVIEW_QUEUE.md"))
    parser.add_argument("--assets-dir", default=str(_repo_root / "review_queue_assets"))
    args = parser.parse_args()

    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        result = build_review_queue(
            client,
            output_path=Path(args.output),
            assets_dir=Path(args.assets_dir),
            limit_per_scenario=args.limit_per_scenario,
        )
        print(f"Review queue written to {result}")
    except Exception as exc:
        print(f"Review queue generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
