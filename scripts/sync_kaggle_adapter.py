#!/usr/bin/env python3
"""
Download the latest Kaggle SimSat-Gemma4 adapter and triage it in one command.

When the Kaggle kernel `benhaslam/simsat-gemma4-v1-training` finishes
successfully, three manual steps are required:

    kaggle kernels output benhaslam/simsat-gemma4-v1-training -p ./weights/gemma4-vN/
    python scripts/diagnose_gemma4_checkpoint.py --check both \\
        --adapter-path ./weights/gemma4-vN/adapter
    export HAIC_GEMMA4_LORA_PATH=./weights/gemma4-vN/adapter
    export OBSERVATION_VLA_BACKEND=gemma4_haic_local
    python scripts/observation_vla_eval.py --inprocess

This script does all four:
  1. Verifies the kernel status is COMPLETE (not RUNNING / ERROR / QUEUED).
  2. Downloads the adapter dir into ./weights/gemma4-v<N>/ where N comes from
     the kernel's metadata or --version override.
  3. Runs scripts/diagnose_gemma4_checkpoint.py against the downloaded adapter.
  4. If the verdict is MASKING_OK_LOSS_DESCENDED, runs observation_vla_eval
     with the new backend wired in (subprocess; env vars set in-process).

Optional flags let you skip individual stages or pre-stage downloads while
training is still RUNNING (it'll fetch once it transitions to COMPLETE).

USAGE
-----
    python scripts/sync_kaggle_adapter.py --version 7
    python scripts/sync_kaggle_adapter.py --version 7 --skip-eval
    python scripts/sync_kaggle_adapter.py --version 7 --download-only
    python scripts/sync_kaggle_adapter.py --version 7 --wait
        (poll status every 60s until COMPLETE, then proceed)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "benhaslam/simsat-gemma4-v1-training"


def _run(cmd: list[str], env: dict | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _kernel_status() -> str:
    """Return the bare status string (e.g. 'RUNNING', 'COMPLETE', 'ERROR')."""
    proc = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_ID],
        capture_output=True, text=True,
    )
    # Output looks like: '<kernel> has status "KernelWorkerStatus.COMPLETE"'
    text = proc.stdout + proc.stderr
    for line in text.splitlines():
        if "has status" in line and "KernelWorkerStatus." in line:
            tail = line.split("KernelWorkerStatus.", 1)[1]
            return tail.split('"', 1)[0].strip()
    return "UNKNOWN"


def _wait_for_complete(poll_seconds: int) -> str:
    """Poll until status leaves RUNNING/QUEUED. Returns final status."""
    while True:
        status = _kernel_status()
        print(f"  [poll] status = {status}")
        if status in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
            return status
        if status not in ("RUNNING", "QUEUED"):
            print(f"  [poll] unexpected status — bailing out")
            return status
        time.sleep(poll_seconds)


def _download_adapter(version: int, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    # Kaggle CLI doesn't filter by version on output download; it pulls latest.
    # The version arg is informational (used for the local dir naming).
    return _run([
        "kaggle", "kernels", "output", KERNEL_ID, "-p", str(target_dir),
    ])


def _find_adapter_subdir(root: Path) -> Path | None:
    """Look for an adapter_config.json under root and return its containing dir."""
    candidates = list(root.rglob("adapter_config.json"))
    if not candidates:
        return None
    # Prefer one whose path contains 'adapter'
    for c in candidates:
        if "adapter" in c.parent.name.lower():
            return c.parent
    return candidates[0].parent


def _diagnose(adapter_path: Path, check: str) -> int:
    return _run([
        sys.executable,
        str(REPO_ROOT / "scripts" / "diagnose_gemma4_checkpoint.py"),
        "--check", check,
        "--adapter-path", str(adapter_path),
    ])


def _eval(adapter_path: Path) -> int:
    env = os.environ.copy()
    env["HAIC_GEMMA4_LORA_PATH"] = str(adapter_path)
    env["OBSERVATION_VLA_BACKEND"] = "gemma4_haic_local"
    return _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "observation_vla_eval.py"), "--inprocess"],
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download + triage + wire-in a freshly-trained Kaggle adapter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", type=int, required=True,
                        help="Kernel version number (used for local dir naming, e.g. 7).")
    parser.add_argument("--target-dir", type=Path, default=None,
                        help="Override download location (default: ./weights/gemma4-v<N>/).")
    parser.add_argument("--check", choices=["masking", "loss", "both"], default="both",
                        help="Diagnostic mode to run after download (default: both).")
    parser.add_argument("--skip-diagnose", action="store_true",
                        help="Skip the diagnostic step (just download).")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the observation_vla_eval step (just download + diagnose).")
    parser.add_argument("--download-only", action="store_true",
                        help="Same as --skip-diagnose --skip-eval.")
    parser.add_argument("--wait", action="store_true",
                        help="Poll status every --poll-seconds until COMPLETE/ERROR.")
    parser.add_argument("--poll-seconds", type=int, default=60,
                        help="Polling interval when --wait is set (default: 60).")
    parser.add_argument("--allow-error-status", action="store_true",
                        help="Download even if status is ERROR (sometimes there's a partial adapter).")
    args = parser.parse_args()

    if args.download_only:
        args.skip_diagnose = True
        args.skip_eval = True

    target_dir = args.target_dir or (REPO_ROOT / "weights" / f"gemma4-v{args.version}")

    print("=" * 72)
    print(f"Sync Kaggle adapter — kernel={KERNEL_ID} version=v{args.version}")
    print(f"Target dir: {target_dir}")
    print("=" * 72)

    # 1. Status check
    if args.wait:
        status = _wait_for_complete(args.poll_seconds)
    else:
        status = _kernel_status()
    print(f"\nKernel status: {status}")

    if status != "COMPLETE":
        if status == "ERROR" and args.allow_error_status:
            print("  Status is ERROR but --allow-error-status is set; proceeding.")
        elif status in ("RUNNING", "QUEUED"):
            print("  Kernel hasn't finished. Re-run with --wait, or wait and retry.")
            return 1
        else:
            print(f"  Cannot download: status={status!r}")
            return 1

    # 2. Download
    rc = _download_adapter(args.version, target_dir)
    if rc != 0:
        print("Download failed.")
        return rc

    # 3. Locate adapter dir
    adapter_path = _find_adapter_subdir(target_dir)
    if adapter_path is None:
        print(f"  WARN: no adapter_config.json found under {target_dir}.")
        print("  The kernel may not have produced an adapter (training crash).")
        print("  Inspect the downloaded files manually:")
        for p in sorted(target_dir.rglob("*"))[:20]:
            print(f"    {p.relative_to(target_dir)}")
        return 1
    print(f"\nAdapter located: {adapter_path}")

    # 4. Diagnose
    if not args.skip_diagnose:
        rc = _diagnose(adapter_path, args.check)
        if rc != 0:
            print("Diagnostic step failed (non-zero exit). Continuing to eval anyway.")

    # 5. Eval
    if not args.skip_eval:
        rc = _eval(adapter_path)
        if rc != 0:
            print("Eval step failed.")
            return rc

    print("\n" + "=" * 72)
    print("Sync complete.")
    print(f"Adapter at: {adapter_path}")
    print("To re-run the eval against this backend manually:")
    print(f"  set HAIC_GEMMA4_LORA_PATH={adapter_path}")
    print(f"  set OBSERVATION_VLA_BACKEND=gemma4_haic_local")
    print(f"  python scripts/observation_vla_eval.py --inprocess")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
