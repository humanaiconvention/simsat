#!/usr/bin/env python3
"""
SimSat LFM-VL v1 Kaggle push script.

Steps:
  1. Run prepare_dataset.py (rebuilds JSONL + copies images into datasets/simsat-lfm-v1/)
  2. kaggle datasets push   (creates/updates benhaslam/simsat-lfm-v1)
  3. python build_notebook.py  (builds notebook.ipynb + validates)
  4. kaggle kernels push    (pushes kernel to Kaggle)

Usage:
    python push.py [--dataset-only] [--kernel-only] [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATASET_DIR = HERE.parent.parent / "datasets" / "simsat-lfm-v1"

assert (HERE / "kernel-metadata.json").exists(), f"kernel-metadata.json not found in {HERE}"
assert DATASET_DIR.exists(), f"Dataset dir not found: {DATASET_DIR}"


def run(cmd: list[str], dry_run: bool = False) -> int:
    print(f"\n$ {' '.join(cmd)}")
    if dry_run:
        print("  [dry-run, skipped]")
        return 0
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ERROR: command exited {result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Push SimSat LFM dataset + kernel to Kaggle")
    parser.add_argument("--dataset-only", action="store_true", help="Only push dataset, skip kernel")
    parser.add_argument("--kernel-only", action="store_true", help="Only push kernel, skip dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    push_dataset = not args.kernel_only
    push_kernel = not args.dataset_only

    if push_dataset:
        print("=" * 60)
        print("STEP 1: Prepare dataset")
        print("=" * 60)
        rc = run([sys.executable, str(DATASET_DIR / "prepare_dataset.py")], dry_run=args.dry_run)
        if rc != 0:
            print("Dataset preparation failed. Aborting.")
            sys.exit(rc)

        print("\n" + "=" * 60)
        print("STEP 2: Push dataset to Kaggle")
        print("=" * 60)
        rc = run(
            ["kaggle", "datasets", "version", "-p", str(DATASET_DIR), "-m", "simsat-lfm-v1 update"],
            dry_run=args.dry_run,
        )
        if rc != 0:
            print("  version failed (possibly first push) — trying create...")
            rc = run(["kaggle", "datasets", "create", "-p", str(DATASET_DIR)], dry_run=args.dry_run)
        if rc != 0:
            print("Dataset push failed. Check kaggle CLI credentials and dataset-metadata.json.")
            sys.exit(rc)
        print("Dataset pushed: benhaslam/simsat-lfm-v1")

    if push_kernel:
        print("\n" + "=" * 60)
        print("STEP 3: Build notebook.ipynb")
        print("=" * 60)
        rc = run([sys.executable, str(HERE / "build_notebook.py")], dry_run=args.dry_run)
        if rc != 0:
            print("Notebook build failed.")
            sys.exit(rc)

        print("\n" + "=" * 60)
        print("STEP 4: Push kernel to Kaggle")
        print("=" * 60)
        rc = run(["kaggle", "kernels", "push", "-p", str(HERE)], dry_run=args.dry_run)
        if rc != 0:
            print("Kernel push failed. Check kaggle CLI credentials.")
            sys.exit(rc)
        print("Kernel pushed: benhaslam/simsat-lfm2-5-vl-v1-training")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    if not args.dry_run:
        print("Monitor at: https://www.kaggle.com/code/benhaslam/simsat-lfm2-5-vl-v1-training")
        print("Dataset at: https://www.kaggle.com/datasets/benhaslam/simsat-lfm-v1")


if __name__ == "__main__":
    main()
