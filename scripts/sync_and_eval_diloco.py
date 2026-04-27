#!/usr/bin/env python3
"""Sync DiLoCo learner kernel output and run observation_vla_eval against the
continued adapter.

Picks up where the SimSat-side workflow leaves off after `kaggle kernels push`:
polls the kernel status until COMPLETE (or fails fast on ERROR/CANCELLED),
downloads outputs to a local weights dir, copies the outbox into
D:\\diloco_lab\\inbox, sets the OBSERVATION_VLM_* env vars, and invokes
scripts/observation_vla_eval.py --inprocess.

Usage:
    python scripts/sync_and_eval_diloco.py
    python scripts/sync_and_eval_diloco.py --kernel benhaslam/<other-slug>
    python scripts/sync_and_eval_diloco.py --skip-poll   # assume already COMPLETE
    python scripts/sync_and_eval_diloco.py --skip-eval   # download only
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL = "benhaslam/simsat-diloco-round-0-learner"
DEFAULT_OUT = REPO_ROOT / "weights" / "diloco-round0-continued"
DILOCO_INBOX = Path(os.environ.get("DILOCO_LAB_INBOX", r"D:\diloco_lab\inbox"))


def _kernel_status(kernel: str) -> str:
    proc = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True, text=True,
    )
    text = proc.stdout + proc.stderr
    for line in text.splitlines():
        if "KernelWorkerStatus." in line:
            tail = line.split("KernelWorkerStatus.", 1)[1]
            return tail.split('"', 1)[0].strip()
    return "UNKNOWN"


def _wait(kernel: str, poll_seconds: int) -> str:
    while True:
        status = _kernel_status(kernel)
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] status = {status}", flush=True)
        if status in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"}:
            return status
        if status not in {"RUNNING", "QUEUED"}:
            return status
        time.sleep(poll_seconds)


def _download(kernel: str, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading kernel output to {target}")
    return subprocess.call(
        ["kaggle", "kernels", "output", kernel, "-p", str(target)],
    )


def _find_adapter_dir(root: Path) -> Path | None:
    candidates = list(root.rglob("adapter_config.json"))
    if not candidates:
        return None
    for c in candidates:
        if "diloco_continued" in str(c.parent).lower():
            return c.parent
    return candidates[0].parent


def _find_outbox(root: Path) -> Path | None:
    direct = root / "diloco_outbox"
    if direct.exists():
        return direct
    matches = list(root.rglob("diloco_outbox"))
    return matches[0] if matches else None


def _eval(adapter_path: Path) -> int:
    env = os.environ.copy()
    env["OBSERVATION_VLA_BACKEND"] = "gemma4"
    env["OBSERVATION_VLM_BASE_MODEL"] = env.get("OBSERVATION_VLM_BASE_MODEL", "google/gemma-4-e2b-it")
    env["OBSERVATION_VLM_LORA_PATH"] = str(adapter_path)
    env["OBSERVATION_VLM_MODE"] = "lora"
    env["OBSERVATION_VLM_MODEL_LABEL"] = env.get("OBSERVATION_VLM_MODEL_LABEL", "diloco-round0-continued")
    print(f"\nRunning observation_vla_eval against {adapter_path}")
    return subprocess.call(
        [sys.executable, str(REPO_ROOT / "scripts" / "observation_vla_eval.py"), "--inprocess"],
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--skip-poll", action="store_true",
                        help="Skip status polling; assume kernel is COMPLETE.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Download + outbox-copy only, skip eval.")
    parser.add_argument("--no-inbox-copy", action="store_true",
                        help="Skip copying outbox fragments to D:/diloco_lab/inbox.")
    args = parser.parse_args()

    print(f"Kernel:    {args.kernel}")
    print(f"Target:    {args.target_dir}")

    if not args.skip_poll:
        status = _wait(args.kernel, args.poll_seconds)
        print(f"\nFinal status: {status}")
        if status != "COMPLETE":
            print("Kernel did not complete cleanly; aborting before download.")
            return 1

    if _download(args.kernel, args.target_dir) != 0:
        print("Download failed.")
        return 1

    adapter = _find_adapter_dir(args.target_dir)
    if adapter is None:
        print("Could not locate adapter_config.json under download root.")
        return 1
    print(f"Adapter: {adapter}")

    if not args.no_inbox_copy:
        outbox = _find_outbox(args.target_dir)
        if outbox is None:
            print("Note: no diloco_outbox/ in download — skipping inbox copy.")
        else:
            DILOCO_INBOX.mkdir(parents=True, exist_ok=True)
            print(f"Copying outbox: {outbox} -> {DILOCO_INBOX}")
            for item in outbox.iterdir():
                dest = DILOCO_INBOX / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

    if args.skip_eval:
        print("\nSkip-eval set; stopping here.")
        return 0

    return _eval(adapter)


if __name__ == "__main__":
    sys.exit(main())
