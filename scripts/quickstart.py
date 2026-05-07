#!/usr/bin/env python3
"""
SimSat collaborator quickstart — verifies a fresh clone is wired correctly.

Runs in this order:
  1. Full pytest suite (tests/ + src/sim/tests/, ~200 tests; HF-download tests
     skip by default)
  2. ObservationVLA eval against pinned reviewed cases (in-process FastAPI,
     uses the default `clip_local` backend if torch+CLIP are installed,
     else falls back gracefully)

If both pass, you have a working baseline to compare your model against.
Total runtime: ~30-60 seconds (without CLIP model download).

Usage:
    python scripts/quickstart.py

Exits non-zero on any failure so it composes with CI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _step(num: int, total: int, label: str) -> None:
    print(f"\n[{num}/{total}] {label}")
    print("-" * 72)


def _run(label: str, args: list[str], cwd: Path = REPO_ROOT) -> int:
    print(f"  $ {' '.join(args)}")
    proc = subprocess.run(args, cwd=str(cwd))
    if proc.returncode != 0:
        print(f"  X {label} FAILED (exit code {proc.returncode})")
    else:
        print(f"  OK {label}")
    return proc.returncode


def main() -> int:
    print("=" * 72)
    print("SimSat collaborator quickstart")
    print("=" * 72)

    failures: list[str] = []

    _step(1, 2, "Run pytest test suite (tests/ + src/sim/tests/)")
    rc = _run(
        "pytest",
        [sys.executable, "-m", "pytest", "tests/", "src/sim/tests/", "--no-header", "-q"],
    )
    if rc != 0:
        failures.append("pytest")

    _step(2, 2, "ObservationVLA eval against pinned reviewed cases")
    print("  (uses in-process FastAPI; no live server required)")
    print("  (writes to .quickstart_observation_vla_eval.md to preserve")
    print("   the canonical OBSERVATION_VLA_EVAL.md submission evidence)")
    rc = _run(
        "observation_vla_eval",
        [
            sys.executable,
            "scripts/observation_vla_eval.py",
            "--inprocess",
            "--output",
            ".quickstart_observation_vla_eval.md",
        ],
    )
    if rc != 0:
        failures.append("observation_vla_eval")
        print("  Note: this step requires `torch` + `transformers` for the")
        print("        default clip_local backend. Install via:")
        print("          pip install -r requirements.txt")

    print("\n" + "=" * 72)
    if failures:
        print(f"FAIL — {len(failures)} step(s) failed: {', '.join(failures)}")
        print("=" * 72)
        return 1
    print("PASS — your clone is ready. Next:")
    print("  - Read COLLABORATOR_GUIDE.md to plug in your model backend")
    print("  - Set OBSERVATION_VLA_BACKEND=<your_backend> in .env")
    print("  - Re-run this quickstart to confirm your backend works")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
