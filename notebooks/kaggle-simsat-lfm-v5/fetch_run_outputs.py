#!/usr/bin/env python3
"""Fetch run artifacts from the Kaggle kernel after it finishes.

Pulls /kaggle/working/ contents (adapter, predictions, eval report) into
.kaggle_output/ locally so we can fold the numbers into submission docs
and upload the adapter to HuggingFace.

Usage:
    python notebooks/kaggle-simsat-lfm-v1/fetch_run_outputs.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "benhaslam/simsat-lfm2-5-vl-v5-training"
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent.parent / ".kaggle_output_v5"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Fetching outputs from {REPO} -> {OUT_DIR}")
    rc = subprocess.run(
        ["kaggle", "kernels", "output", REPO, "-p", str(OUT_DIR)],
        check=False,
    ).returncode
    if rc != 0:
        print(f"  kaggle CLI exit {rc}")
        return rc

    files = sorted(OUT_DIR.iterdir())
    print(f"\nDownloaded {len(files)} files:")
    for f in files:
        size = f.stat().st_size
        print(f"  {f.name}  ({size:,} bytes)")

    # Show eval report headlines if present
    report_path = OUT_DIR / "holdout_eval_report.json"
    if report_path.exists():
        print("\n" + "=" * 60)
        print("HOLDOUT EVAL REPORT")
        print("=" * 60)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "base" in report and "tuned" in report:
            base = report["base"]
            tuned = report["tuned"]
            for k in ("parse_rate", "exact_action_agreement",
                      "useful_agreement", "score_mae"):
                b = base.get(k, 0.0)
                t = tuned.get(k, 0.0)
                print(f"  {k:<28} base={b:.3f}  tuned={t:.3f}  delta={t-b:+.3f}")
            if "per_class_accuracy" in tuned:
                print("\n  Per-class accuracy (tuned):")
                for cls, acc in sorted(tuned["per_class_accuracy"].items()):
                    print(f"    {cls:<10} {acc:.3f}")
        else:
            print(json.dumps(report, indent=2))

    # Show base & tuned prediction counts
    for name in ("base_predictions.json", "tuned_predictions.json"):
        p = OUT_DIR / name
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            valid = sum(1 for d in data if d.get("predicted"))
            print(f"\n  {name}: {len(data)} entries, {valid} non-empty")

    return 0


if __name__ == "__main__":
    sys.exit(main())
