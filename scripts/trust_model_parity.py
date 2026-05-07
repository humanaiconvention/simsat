#!/usr/bin/env python3
"""DEPRECATED — trust_model_parity.py is retired.

This script was a one-shot harness used during the 2026-04 trust-model
refactor (commits 8fe720f → 8a66633) to verify the consolidated
`WCLITrustModel` reproduced the scattered planner-side trust logic with
zero drift. The "scattered" implementation no longer exists in the
codebase; `WCLITrustModel` has been the canonical implementation for
the entire 2026-05 submission cycle.

Replaced by:
  - Formal pytest coverage: `tests/test_encounter_trust_model.py`
    (8 tests: refine/no-refine/compound-risk/online-update + L2 reg
    drift bounds + L2 reg recovery + record_skipped_observation)
  - Viability gate coverage: `tests/test_ttt_viability_gates.py`
    (16 tests on the three-gate evaluator)
  - Trust-model edge cases: `tests/test_trust_model_edge_cases.py`
    (12 tests on numerical extremes, log overflow, recovery)
  - Trust-layer TTT 10-seed stability analysis:
    `ttt_stability_analysis.md` (22.5% MAE improvement +/- 0.1%)
  - Viability gates exercise on the same model:
    `VIABILITY_GATES_EXERCISE.md` (1100 updates x 3 streams)

Run those instead. This file is kept as a stub so links/build configs
that reference it continue to resolve, but execution does nothing useful.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(__doc__.strip())
    print()
    print("Suggested commands:")
    print("  pytest tests/test_encounter_trust_model.py -v")
    print("  cat ttt_stability_analysis.md")
    print("  python scripts/viability_gates_exercise.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
