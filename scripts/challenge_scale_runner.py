from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from challenge_run_config import KNOWN_SCENARIO_PACKS, resolve_scenario_hours, scenario_hours_args


def _run(root: Path, command: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, output


def _tail(output: str, line_count: int = 12) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-line_count:]) if lines else "(no output)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SimSat challenge lanes with a shared scenario policy.")
    parser.add_argument(
        "--policy",
        choices=["smoke", "competition"],
        default="competition",
        help="Shared scenario-hour policy for challenge-facing scripts.",
    )
    parser.add_argument("--inprocess", action="store_true", help="Run scripts against the in-process API where supported.")
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["matrix", "observation", "review", "submission", "readiness", "backlog"],
        choices=["fast", "matrix", "observation", "review", "submission", "readiness", "backlog"],
        help="Subset of local challenge lanes to run.",
    )
    parser.add_argument("--hours", type=float, default=8.0, help="Fallback hour value for scenarios without a shared policy entry.")
    parser.add_argument("--step-seconds", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--materialize-top-k", type=int, default=2)
    parser.add_argument("--review-limit", type=int, default=10)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "CHALLENGE_SCALE_RUN.md"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    hours_by_scenario = resolve_scenario_hours(
        scenarios=KNOWN_SCENARIO_PACKS,
        default_hours=args.hours,
        policy=args.policy,
    )
    shared_scenario_args = scenario_hours_args(hours_by_scenario)
    inprocess_arg = ["--inprocess"] if args.inprocess else []

    lane_commands: dict[str, list[str]] = {
        "fast": [sys.executable, "-m", "pytest", "-q"],
        "matrix": [
            sys.executable,
            "scripts/challenge_test_matrix.py",
            *inprocess_arg,
            "--policy",
            args.policy,
            "--hours",
            f"{args.hours:g}",
            "--step-seconds",
            str(args.step_seconds),
            "--top-k",
            str(args.top_k),
            "--materialize-top-k",
            str(args.materialize_top_k),
            "--review-limit",
            str(args.review_limit),
            *shared_scenario_args,
        ],
        "observation": [
            sys.executable,
            "scripts/observation_vla_eval.py",
            *inprocess_arg,
        ],
        "review": [
            sys.executable,
            "scripts/review_queue_casebook.py",
            *inprocess_arg,
        ],
        "submission": [
            sys.executable,
            "scripts/submission_evidence.py",
            *inprocess_arg,
            "--policy",
            args.policy,
            "--hours",
            f"{args.hours:g}",
            "--step-seconds",
            str(args.step_seconds),
            "--top-k",
            str(args.top_k),
            "--materialize-top-k",
            str(args.materialize_top_k),
            *shared_scenario_args,
        ],
        "readiness": [
            sys.executable,
            "scripts/submission_readiness.py",
            *inprocess_arg,
        ],
        "backlog": [
            sys.executable,
            "scripts/review_backlog_report.py",
            *inprocess_arg,
        ],
    }

    lines: list[str] = []
    lines.append("# SimSat Challenge Scale Run")
    lines.append("")
    lines.append(f"- Policy: `{args.policy}`")
    lines.append(f"- Scenario hours: `{hours_by_scenario}`")
    lines.append(f"- In-process: `{args.inprocess}`")
    lines.append("")
    lines.append("## Lane Results")
    lines.append("")
    lines.append("| lane | ok | note |")
    lines.append("| --- | --- | --- |")

    for lane in args.lanes:
        command = lane_commands[lane]
        ok, output = _run(root, command)
        note = _tail(output).replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {lane} | {ok} | {note} |")
        lines.append("")
        lines.append(f"## {lane}")
        lines.append("")
        lines.append("```text")
        lines.append(output or "(no output)")
        lines.append("```")
        lines.append("")

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Challenge scale run written to {output_path}")


if __name__ == "__main__":
    main()
