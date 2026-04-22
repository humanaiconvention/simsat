from __future__ import annotations

import argparse
import sys

import requests

SCENARIO_NOTES = {
    "maritime_chokepoints": "show that the trust layer protects decisive high-value passes while still surfacing refine when support is thin",
    "disaster_response_weather": "show that weather and imagery uncertainty increase refine pressure instead of silent over-acceptance",
    "urban_coastal_ambiguity": "show that visually dense ports and coastlines create plausible-but-ambiguous windows that should not all become accept",
}


def post_json(base_url: str, path: str, payload: dict) -> dict:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=90)
    if response.status_code != 200:
        raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
    return response.json()


def get_json(base_url: str, path: str) -> dict:
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"{path} failed with {response.status_code}: {response.text}")
    return response.json()


def print_eval_summary(evaluation: dict) -> None:
    scaffold = evaluation.get("scaffold_summary", {})
    trust = evaluation.get("trust_summary", {})
    print(f"Scenario: {evaluation.get('scenario_pack', 'all')}")
    note = SCENARIO_NOTES.get(evaluation.get("scenario_pack", "all"))
    if note:
        print(f"Demo focus: {note}")
    print(f"Windows: {evaluation.get('window_count', 0)}")
    print(f"Transitions: {evaluation.get('action_transition_counts', {})}")
    print(
        "Action counts:",
        f"scaffold={scaffold.get('action_counts', {})}",
        f"trust={trust.get('action_counts', {})}",
    )
    print(
        "Yield:",
        f"scaffold={scaffold.get('materialization_yield', 0.0):.2f}",
        f"trust={trust.get('materialization_yield', 0.0):.2f}",
    )
    print("Changed decisions:")
    for delta in evaluation.get("decision_deltas", [])[:5]:
        label = delta.get("target_label") or delta.get("target_id")
        print(
            f"  - {label}: {delta.get('scaffold_action')} -> {delta.get('trust_action')} | "
            f"final={delta.get('trust_combined_score', 0.0):.2f} | trust={delta.get('trust_score', 0.0):.2f}"
        )
        if delta.get("refinement_reason"):
            print(f"    refine={delta['refinement_reason']}")
    print("Narrate:")
    print("  1. These are the same candidate windows under both planners.")
    print("  2. The scaffold ranks cheaply; the trust layer decides whether to accept or refine.")
    print("  3. The refine path preserves promising windows without pretending they are already reliable.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SimSat challenge demo flow.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim")
    parser.add_argument("--scenario-pack", default="maritime_chokepoints")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--step-seconds", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--materialize-top-k", type=int, default=3)
    args = parser.parse_args()

    try:
        capabilities = get_json(args.base_url, "/capabilities")
        if not capabilities.get("mapbox_enabled", False):
            print("Runtime note: Mapbox is disabled. This challenge demo is designed to remain valid with Sentinel + geometry only.")
        evaluation = post_json(
            args.base_url,
            "/encounter/evaluate",
            {
                "scenario_pack": args.scenario_pack,
                "hours": args.hours,
                "step_seconds": args.step_seconds,
                "top_k": args.top_k,
                "materialize_top_k": args.materialize_top_k,
            },
        )
        print_eval_summary(evaluation)
    except Exception as exc:
        print(f"Challenge demo failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
