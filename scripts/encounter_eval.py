from __future__ import annotations

import argparse
import sys
from typing import Iterable

import requests

KNOWN_SCENARIO_PACKS = [
    "maritime_chokepoints",
    "disaster_response_weather",
    "urban_coastal_ambiguity",
]


def run_evaluation(
    base_url: str,
    hours: float,
    step_seconds: int,
    top_k: int,
    materialize_top_k: int,
    scenario_pack: str | None,
) -> dict:
    payload = {
        "hours": hours,
        "step_seconds": step_seconds,
        "top_k": top_k,
        "materialize_top_k": materialize_top_k,
    }
    if scenario_pack and scenario_pack != "all":
        payload["scenario_pack"] = scenario_pack
    response = requests.post(
        f"{base_url.rstrip('/')}/encounter/evaluate",
        json=payload,
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"evaluation failed with {response.status_code}: {response.text}")
    return response.json()


def _format_ratio(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _summary_row(evaluation: dict) -> dict[str, str]:
    scaffold = evaluation.get("scaffold_summary", {})
    trust = evaluation.get("trust_summary", {})
    transitions = evaluation.get("action_transition_counts", {})
    decision_deltas = evaluation.get("decision_deltas", [])
    top_delta = decision_deltas[0] if decision_deltas else {}
    top_reason = top_delta.get("refinement_reason") or ",".join(top_delta.get("trust_reason_codes", [])) or "-"
    return {
        "scenario": str(evaluation.get("scenario_pack", "all")),
        "windows": str(evaluation.get("window_count", 0)),
        "scaffold_accept": str(scaffold.get("action_counts", {}).get("accept", 0)),
        "trust_accept": str(trust.get("action_counts", {}).get("accept", 0)),
        "trust_refine": str(trust.get("action_counts", {}).get("refine", 0)),
        "changed": str(len(decision_deltas)),
        "accept_to_refine": str(transitions.get("accept->refine", 0)),
        "scaffold_yield": _format_ratio(scaffold.get("materialization_yield")),
        "trust_yield": _format_ratio(trust.get("materialization_yield")),
        "top_reason": top_reason,
    }


def print_summary(evaluation: dict) -> None:
    scaffold = evaluation.get("scaffold_summary", {})
    trust = evaluation.get("trust_summary", {})
    print(f"Evaluation: {evaluation.get('evaluation_id')}")
    print(f"Created: {evaluation.get('created_at')}")
    print(f"Scenario: {evaluation.get('scenario_pack', 'all')}")
    print(f"Windows: {evaluation.get('window_count')}")
    print("Scaffold actions:", scaffold.get("action_counts", {}))
    print("Trust actions:", trust.get("action_counts", {}))
    print(
        "Materialization yield:",
        {
            "scaffold": scaffold.get("materialization_yield"),
            "trust": trust.get("materialization_yield"),
        },
    )
    print("Action transitions:", evaluation.get("action_transition_counts", {}))
    deltas = evaluation.get("decision_deltas", [])
    if deltas:
        print("Top decision deltas:")
        for delta in deltas[:5]:
            print(
                " -",
                delta.get("target_label") or delta.get("target_id"),
                f"{delta.get('scaffold_action')} -> {delta.get('trust_action')}",
                f"delta={delta.get('score_delta', 0.0):+.2f}",
                f"trust={delta.get('trust_score', 0.0):.2f}",
                f"reason={delta.get('refinement_reason') or ','.join(delta.get('trust_reason_codes', []))}",
            )


def print_markdown_table(evaluations: Iterable[dict]) -> None:
    rows = [_summary_row(evaluation) for evaluation in evaluations]
    headers = [
        "scenario",
        "windows",
        "scaffold_accept",
        "trust_accept",
        "trust_refine",
        "changed",
        "accept_to_refine",
        "scaffold_yield",
        "trust_yield",
        "top_reason",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row[header] for header in headers) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scaffold vs WCLI-trust encounter planners.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/sim", help="Encounter API base URL")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--step-seconds", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--materialize-top-k", type=int, default=3)
    parser.add_argument("--scenario-pack", default="all")
    parser.add_argument("--scenario-sweep", action="store_true", help="Run one evaluation per seeded scenario pack.")
    parser.add_argument("--markdown", action="store_true", help="Print a compact markdown scorecard.")
    args = parser.parse_args()

    try:
        if args.scenario_sweep:
            evaluations = [
                run_evaluation(
                    base_url=args.base_url,
                    hours=args.hours,
                    step_seconds=args.step_seconds,
                    top_k=args.top_k,
                    materialize_top_k=args.materialize_top_k,
                    scenario_pack=scenario_pack,
                )
                for scenario_pack in KNOWN_SCENARIO_PACKS
            ]
            if args.markdown:
                print_markdown_table(evaluations)
            else:
                for evaluation in evaluations:
                    print_summary(evaluation)
                    print()
        else:
            evaluation = run_evaluation(
                base_url=args.base_url,
                hours=args.hours,
                step_seconds=args.step_seconds,
                top_k=args.top_k,
                materialize_top_k=args.materialize_top_k,
                scenario_pack=args.scenario_pack,
            )
            if args.markdown:
                print_markdown_table([evaluation])
            else:
                print_summary(evaluation)
    except Exception as exc:
        print(f"Encounter evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
