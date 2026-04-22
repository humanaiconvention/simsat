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


def _parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"y", "yes", "true", "1"}:
        return True
    if normalized in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"Could not parse boolean value from {text!r}")


def list_candidates(client, scenario_pack: str | None, limit: int) -> list[dict]:
    payload = client.get_json("/observation-vla/review-candidates", scenario_pack=scenario_pack, limit=limit)
    candidates = payload.get("candidates", [])
    if not candidates:
        print("No review candidates found.")
        return []

    print("Review candidates:")
    for item in candidates:
        current = item.get("current_outcome") or {}
        label_source = current.get("label_source", "none")
        reviewer = current.get("reviewer") or "-"
        print(
            f"- trace={item.get('trace_id')} scenario={item.get('scenario_pack')} "
            f"target={item.get('target_label') or item.get('target_id')} "
            f"recommended={item.get('recommended_action')} "
            f"needs_operator_review={item.get('needs_operator_review')} "
            f"label_source={label_source} reviewer={reviewer}"
        )
    return candidates


def show_review_bundle(client, trace_id: str) -> dict:
    bundle = client.get_json(f"/observation-vla/review-bundle/{trace_id}")
    trace = bundle.get("trace", {})
    outcome = bundle.get("current_outcome") or {}
    window = bundle.get("window") or {}
    planner_decision = bundle.get("planner_decision") or {}
    mission_response = bundle.get("mission_response") or {}
    print("")
    print(f"Review bundle for trace={trace_id}")
    print(f"- scenario: {trace.get('scenario_pack')}")
    print(f"- target: {bundle.get('target', {}).get('label', trace.get('target_id'))}")
    print(f"- window: {window.get('start_time')} -> {window.get('end_time')} (peak {window.get('peak_time')})")
    print(f"- sentinel source: {bundle.get('probe', {}).get('sentinel_source')}")
    print(f"- cloud cover: {bundle.get('probe', {}).get('sentinel_cloud_cover')}")
    print(f"- planner action: {planner_decision.get('effective_action') or planner_decision.get('action')}")
    print(f"- observation recommendation: {trace.get('assessment', {}).get('recommended_action')}")
    print(f"- current label source: {outcome.get('label_source')}")
    print(f"- ready for submission case: {bundle.get('ready_for_submission_case')}")
    action = mission_response.get('action') if isinstance(mission_response, dict) else None
    if action:
        print(f"- mission response: {action.get('recommended_action')}")
    if bundle.get("submission_case"):
        print(f"- pinned submission case: {bundle['submission_case'].get('scenario_pack')} -> {bundle['submission_case'].get('trace_id')}")
    return bundle


def pin_submission_case(client, scenario_pack: str, trace_id: str, pinned_by: str | None, notes: str | None) -> dict:
    payload = client.post_json(
        f"/observation-vla/submission-case/{scenario_pack}",
        {
            "trace_id": trace_id,
            "pinned_by": pinned_by,
            "notes": notes,
        },
    )
    print(
        "Pinned submission case:",
        f"scenario={payload.get('scenario_pack')}",
        f"trace={payload.get('trace_id')}",
        f"pinned_by={payload.get('pinned_by')}",
    )
    return payload


def prompt_if_missing(value: str | None, prompt: str) -> str:
    if value is not None and value != "":
        return value
    return input(prompt).strip()


def submit_review(
    client,
    trace_id: str,
    reviewer: str | None,
    operator_action: str | None,
    useful: bool | None,
    usefulness_score: float | None,
    notes: str | None,
    mission_id: str | None,
    outcome_tags: list[str],
) -> dict:
    reviewer = prompt_if_missing(reviewer, "Reviewer name: ")
    operator_action = prompt_if_missing(operator_action, "Operator action [accept/refine/defer/skip]: ")
    if useful is None:
        useful = _parse_bool(input("Useful? [y/n]: "))
    if usefulness_score is None:
        usefulness_score = float(prompt_if_missing(None, "Usefulness score [0.0-1.0]: "))
    if notes is None:
        notes = input("Notes (optional): ").strip() or None

    payload = client.post_json(
        f"/observation-vla/trace/{trace_id}/operator-review",
        {
            "reviewer": reviewer,
            "operator_action": operator_action,
            "useful": useful,
            "usefulness_score": usefulness_score,
            "outcome_tags": outcome_tags,
            "notes": notes,
            "mission_id": mission_id,
        },
    )
    outcome = payload.get("outcome", {})
    print(
        "Registered operator review:",
        f"trace={trace_id}",
        f"outcome={outcome.get('outcome_id')}",
        f"label_source={outcome.get('label_source')}",
        f"reviewer={outcome.get('reviewer')}",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace simulated ObservationVLA labels with operator-reviewed outcomes.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--inprocess", action="store_true", help="Run against an in-process FastAPI app instead of a live server.")
    parser.add_argument("--scenario-pack", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--show-bundle-only", action="store_true")
    parser.add_argument("--trace-id", default=None)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--operator-action", default=None)
    parser.add_argument("--useful", default=None, help="true/false")
    parser.add_argument("--usefulness-score", type=float, default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--mission-id", default=None)
    parser.add_argument("--pin-submission-case", action="store_true")
    parser.add_argument("--pinned-by", default=None)
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()

    useful = _parse_bool(args.useful) if args.useful is not None else None
    client = None
    try:
        client = InProcessClient() if args.inprocess else HttpClient(args.base_url)
        candidates: list[dict] = []
        if args.trace_id is None or args.list_only:
            candidates = list_candidates(client, scenario_pack=args.scenario_pack, limit=args.limit)
        if args.list_only:
            return
        if args.trace_id and args.show_bundle_only:
            show_review_bundle(client, args.trace_id)
            return
        if args.trace_id is None:
            if not candidates:
                return
            if not sys.stdin.isatty():
                print("")
                print("No interactive terminal detected. Re-run with --trace-id to submit a review.")
                return
            print("")
            print("Provide --trace-id to submit a review, or rerun and enter it interactively now.")
            chosen_trace_id = input("Trace ID to review (blank to exit): ").strip()
            if not chosen_trace_id:
                return
            trace_id = chosen_trace_id
        else:
            trace_id = args.trace_id

        bundle = show_review_bundle(client, trace_id)
        review_requested = any(
            value is not None
            for value in (args.reviewer, args.operator_action, args.useful, args.usefulness_score, args.notes)
        )
        if args.pin_submission_case and not review_requested:
            scenario_pack = bundle.get("trace", {}).get("scenario_pack") or args.scenario_pack
            if not scenario_pack:
                raise RuntimeError("Could not infer scenario pack for submission pin")
            pin_submission_case(
                client=client,
                scenario_pack=scenario_pack,
                trace_id=trace_id,
                pinned_by=args.pinned_by or args.reviewer,
                notes=args.notes,
            )
            return
        submit_review(
            client=client,
            trace_id=trace_id,
            reviewer=args.reviewer,
            operator_action=args.operator_action,
            useful=useful,
            usefulness_score=args.usefulness_score,
            notes=args.notes,
            mission_id=args.mission_id,
            outcome_tags=args.tag,
        )
        if args.pin_submission_case:
            scenario_pack = bundle.get("trace", {}).get("scenario_pack") or args.scenario_pack
            if not scenario_pack:
                raise RuntimeError("Could not infer scenario pack for submission pin")
            pin_submission_case(
                client=client,
                scenario_pack=scenario_pack,
                trace_id=trace_id,
                pinned_by=args.pinned_by or args.reviewer,
                notes=args.notes,
            )
    finally:
        if isinstance(client, InProcessClient):
            client.close()


if __name__ == "__main__":
    main()
