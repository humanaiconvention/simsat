from __future__ import annotations

import argparse
import sys

import requests


class EncounterTestClient:
    def __init__(self, base_url: str, verbose: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.verbose = verbose

    def get(self, path: str, **kwargs) -> requests.Response:
        response = requests.get(f"{self.base_url}{path}", timeout=15, **kwargs)
        if self.verbose:
            print(f"GET {path} -> {response.status_code}")
        return response

    def post(self, path: str, json_body=None, **kwargs) -> requests.Response:
        response = requests.post(f"{self.base_url}{path}", json=json_body, timeout=15, **kwargs)
        if self.verbose:
            print(f"POST {path} -> {response.status_code}")
        return response


def _expect_ok(response: requests.Response, label: str) -> dict:
    if response.status_code != 200:
        raise RuntimeError(f"{label} failed with {response.status_code}: {response.text}")
    return response.json()


def run_tests(base_url: str, verbose: bool = False) -> bool:
    client = EncounterTestClient(base_url, verbose=verbose)

    capabilities = _expect_ok(client.get("/capabilities"), "capabilities")
    policy = _expect_ok(client.get("/encounter/policy"), "policy")
    targets = _expect_ok(client.get("/encounter/targets"), "targets")
    windows = _expect_ok(client.get("/encounter/windows"), "windows")
    planned = _expect_ok(client.post("/encounter/plan", json_body={"hours": 2, "step_seconds": 60, "top_k": 5}), "plan")
    evaluation = _expect_ok(
        client.post(
            "/encounter/evaluate",
            json_body={"hours": 2, "step_seconds": 60, "top_k": 5, "materialize_top_k": 2},
        ),
        "evaluate",
    )
    decisions = _expect_ok(client.get("/encounter/decisions"), "decisions")

    print(
        "Capabilities:",
        {
            "sentinel_enabled": capabilities.get("sentinel_enabled"),
            "mapbox_enabled": capabilities.get("mapbox_enabled"),
            "challenge_no_mapbox_safe": capabilities.get("challenge_no_mapbox_safe"),
            "observation_vla_runtime_mode": capabilities.get("observation_vla_runtime_mode"),
        },
    )
    print(f"Policy: {policy.get('policy_id')}")
    print(f"Targets: {len(targets.get('targets', []))}")
    print(f"Preview windows: {len(windows.get('windows', []))}")
    print(f"Planned decisions: {len(planned.get('decisions', []))}")
    print(f"Evaluation windows: {evaluation.get('window_count')}")
    print(f"Stored decisions: {len(decisions.get('decisions', []))}")
    if planned.get("decisions"):
        first = planned["decisions"][0]
        print(
            "WCLI trust:",
            {
                "action": first.get("action"),
                "scaffold": first.get("scaffold_score"),
                "trust": first.get("trust_score"),
                "combined": first.get("combined_score"),
                "band": first.get("trust_band"),
            },
        )
    if evaluation.get("trust_summary"):
        print(
            "Evaluation delta:",
            {
                "scaffold_actions": evaluation.get("scaffold_summary", {}).get("action_counts", {}),
                "trust_actions": evaluation.get("trust_summary", {}).get("action_counts", {}),
                "transitions": evaluation.get("action_transition_counts", {}),
                "yield": {
                    "scaffold": evaluation.get("scaffold_summary", {}).get("materialization_yield"),
                    "trust": evaluation.get("trust_summary", {}).get("materialization_yield"),
                },
            },
        )

    if planned.get("decisions"):
        decision_id = planned["decisions"][0]["decision_id"]
        materialized = _expect_ok(client.post(f"/encounter/decision/{decision_id}/materialize"), "materialize")
        print(f"Materialized stimulus: {materialized.get('stimulus_id')}")
        assessed = _expect_ok(client.post(f"/encounter/decision/{decision_id}/assess"), "assess")
        print(
            "Observation residual:",
            {
                "effective_action": assessed.get("effective_action"),
                "assessment_mode": assessed.get("assessment_mode"),
                "trace_id": assessed.get("trace_id"),
            },
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the SimSat encounter service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        ok = run_tests(args.base_url, verbose=args.verbose)
    except Exception as exc:
        print(f"Encounter smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
