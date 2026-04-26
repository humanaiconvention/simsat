#!/usr/bin/env python3
"""
Diagnose a trained Gemma-4 checkpoint from the v4 (or later) Kaggle kernel.

The v3 Kaggle run had flat loss (3.9454) because ~90% of input tokens were
prompt tokens that diluted the gradient signal. v4 introduced
DataCollatorForCompletionOnlyLM (Fix #14) to mask prompt tokens with -100
so loss is computed only on the assistant turn.

This script triages two questions in one pass:

  1. **Is the masking working?** Apply the same chat template + collator
     used in training. Report the percentage of tokens masked (-100) vs
     trained (real label) per sample. Healthy range: 60–90% masked. Below
     ~40% means the response_template isn't matching → loss signal is being
     diluted by prompt tokens (v3-style failure). Above ~95% means almost
     no gradient signal at all → likely an over-aggressive template match.

  2. **Did the adapter actually learn?** Optionally load the LoRA adapter
     onto the base model and run a forward pass over a handful of training
     samples. Compare the average loss to the v3 baseline (3.9454) and to a
     "random initialization" baseline. If the loss is meaningfully below 3.0,
     the adapter has learned task-specific structure.

USAGE
-----
Quick masking check (no adapter, no GPU, ~30s):
    python scripts/diagnose_gemma4_checkpoint.py --check masking

Full check with adapter (requires GPU, downloads base model on first run):
    python scripts/diagnose_gemma4_checkpoint.py \
        --adapter-path ./weights/haic-v35-gov/adapter \
        --check both

Outputs a verdict line — one of:
    VERDICT: MASKING_OK_LOSS_DESCENDED    → ship the checkpoint
    VERDICT: MASKING_OK_LOSS_FLAT          → masking is fine; training failed for another reason
    VERDICT: MASKING_BROKEN                → response_template needs fixing → write Fix #16
    VERDICT: MASKING_TOO_AGGRESSIVE        → response_template is matching too much
    VERDICT: INCONCLUSIVE                  → check the printed numbers manually
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Defaults match what the v4 Kaggle notebook uses.
DEFAULT_BASE_MODEL = "google/gemma-4-e2b-it"
DEFAULT_RESPONSE_TEMPLATE = "<start_of_turn>model\n"
DEFAULT_DATASET_PATH = REPO_ROOT / "datasets" / "simsat-gemma4-v1" / "simsat_train.jsonl"

# Bands for verdict classification.
MASK_RATIO_MIN_HEALTHY = 0.40   # below this → masking is broken (v3 failure mode)
MASK_RATIO_MAX_HEALTHY = 0.95   # above this → almost no signal to train on
LOSS_DESCENDED_THRESHOLD = 3.0  # v3 baseline was 3.9454; meaningful learning lands below this


def _load_dataset(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Pass --dataset-path explicitly."
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _check_masking(
    dataset_path: Path,
    base_model: str,
    response_template: str,
    n_samples: int,
) -> tuple[float, dict]:
    """Apply chat template + collator; report mask ratio statistics.

    Returns (mean_mask_ratio, details_dict).
    """
    from transformers import AutoTokenizer
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        try:
            from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        except ImportError:
            from trl.data_utils import DataCollatorForCompletionOnlyLM

    print(f"Loading tokenizer for {base_model} (no model weights downloaded)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = _load_dataset(dataset_path, n_samples)
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_path}")
    print(f"Loaded {len(rows)} sample row(s) from {dataset_path.name}")

    # Apply chat template per row and tokenize. We mimic SFTTrainer's pipeline.
    examples = []
    for row in rows:
        messages = row.get("messages") or row.get("conversation") or []
        if not messages:
            continue
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(text, truncation=True, max_length=2048, return_tensors=None)
        examples.append(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        )

    if not examples:
        raise ValueError("No examples produced — dataset rows may be missing 'messages' key")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    batch = collator(examples)
    labels = batch["labels"]  # tensor of shape (B, T)
    input_ids = batch["input_ids"]

    # For each example, mask ratio = (#positions == -100) / (total non-pad positions)
    import torch
    per_example_ratios = []
    for i in range(labels.shape[0]):
        lbl = labels[i]
        ids = input_ids[i]
        non_pad = (ids != tokenizer.pad_token_id)
        total = int(non_pad.sum().item())
        if total == 0:
            continue
        masked = int(((lbl == -100) & non_pad).sum().item())
        per_example_ratios.append(masked / total)

    if not per_example_ratios:
        raise RuntimeError("Collator produced no usable batches")

    mean_ratio = sum(per_example_ratios) / len(per_example_ratios)
    details = {
        "n_examples": len(per_example_ratios),
        "mean_mask_ratio": mean_ratio,
        "min_mask_ratio": min(per_example_ratios),
        "max_mask_ratio": max(per_example_ratios),
        "response_template": response_template,
        "first_example_unmasked_token_count": int(
            ((labels[0] != -100) & (input_ids[0] != tokenizer.pad_token_id)).sum().item()
        ),
        "first_example_total_tokens": int((input_ids[0] != tokenizer.pad_token_id).sum().item()),
    }

    return mean_ratio, details


def _check_loss(
    adapter_path: Path,
    base_model: str,
    dataset_path: Path,
    n_samples: int,
) -> tuple[float, dict]:
    """Load adapter, run forward pass, return mean loss."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model {base_model} (this can take a few minutes)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    rows = _load_dataset(dataset_path, n_samples)
    losses = []
    for row in rows:
        messages = row.get("messages") or row.get("conversation") or []
        if not messages:
            continue
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        losses.append(float(out.loss.item()))

    if not losses:
        raise RuntimeError("No losses computed — dataset may be empty")

    mean_loss = sum(losses) / len(losses)
    details = {
        "n_samples": len(losses),
        "mean_loss": mean_loss,
        "min_loss": min(losses),
        "max_loss": max(losses),
        "v3_flat_loss_baseline": 3.9454,
    }
    return mean_loss, details


def _emit_verdict(mask_ratio: float | None, loss: float | None) -> str:
    masking_ok = (
        mask_ratio is not None
        and MASK_RATIO_MIN_HEALTHY <= mask_ratio <= MASK_RATIO_MAX_HEALTHY
    )
    masking_broken = mask_ratio is not None and mask_ratio < MASK_RATIO_MIN_HEALTHY
    masking_too_aggressive = mask_ratio is not None and mask_ratio > MASK_RATIO_MAX_HEALTHY
    loss_descended = loss is not None and loss < LOSS_DESCENDED_THRESHOLD
    loss_flat = loss is not None and loss >= LOSS_DESCENDED_THRESHOLD

    if masking_broken:
        return "MASKING_BROKEN"
    if masking_too_aggressive:
        return "MASKING_TOO_AGGRESSIVE"
    if masking_ok and loss_descended:
        return "MASKING_OK_LOSS_DESCENDED"
    if masking_ok and loss_flat:
        return "MASKING_OK_LOSS_FLAT"
    if masking_ok and loss is None:
        return "MASKING_OK_LOSS_NOT_CHECKED"
    return "INCONCLUSIVE"


def _next_step_for(verdict: str, adapter_path: Path | None) -> str:
    if verdict == "MASKING_OK_LOSS_DESCENDED":
        path_hint = str(adapter_path) if adapter_path else "<your adapter path>"
        return (
            "Ship it. Wire the adapter into the local backend:\n"
            f"  export OBSERVATION_VLM_LORA_PATH={path_hint}\n"
            "  export OBSERVATION_VLM_BASE_MODEL=google/gemma-4-e2b-it\n"
            "  export OBSERVATION_VLM_MODE=lora\n"
            "  export OBSERVATION_VLA_BACKEND=gemma4\n"
            "  python scripts/observation_vla_eval.py --inprocess\n"
            "Compare action-agreement and MAE numbers against the clip_local baseline.\n"
            "(NOTE: backend='gemma4' routes to TransformersVLMAdapter w/ the SimSat\n"
            " fine-tune. Don't use 'gemma4_haic_local' — that's the legacy v35-gov model.)"
        )
    if verdict == "MASKING_OK_LOSS_FLAT":
        return (
            "Masking is healthy but the model didn't learn task structure.\n"
            "Investigate: LoRA rank/alpha, learning rate, training epochs.\n"
            "Inspect the Kaggle kernel's loss curve — if it never moved, the\n"
            "issue is hyperparameters, not data. Consider Fix #16 = bump\n"
            "learning rate from 5e-5 → 2e-4 and increase epochs."
        )
    if verdict == "MASKING_OK_LOSS_NOT_CHECKED":
        return (
            "Masking is healthy. Re-run with --check both --adapter-path <path>\n"
            "to verify the adapter actually learned (loss < 3.0 = success)."
        )
    if verdict == "MASKING_BROKEN":
        return (
            "response_template isn't matching the assistant turn boundary.\n"
            "Most common cause: the template string differs from how Gemma\n"
            "tokenizes the chat. Try inspecting one example:\n"
            "  text = tokenizer.apply_chat_template(messages, tokenize=False)\n"
            "  print(repr(text))\n"
            "Find the EXACT bytes that mark the assistant turn and pass that\n"
            "as response_template. This is Fix #16 — write _patch_notebook_v5.py."
        )
    if verdict == "MASKING_TOO_AGGRESSIVE":
        return (
            "Almost everything is masked → barely any tokens contribute to loss.\n"
            "response_template may be matching multiple times in the input.\n"
            "Use response_template_ids= (not response_template=) and pass the\n"
            "tokenized version explicitly to avoid sub-string matches."
        )
    return (
        "Inconclusive — re-read the per-sample numbers above and decide manually."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Triage a trained Gemma-4 checkpoint: masking + loss.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check",
        choices=["masking", "loss", "both"],
        default="masking",
        help="What to check (default: masking — runs without GPU or adapter).",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Path to the LoRA adapter directory (required for --check loss/both).",
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help=f"HF model id of the base (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--response-template",
        default=DEFAULT_RESPONSE_TEMPLATE,
        help=f"Collator response template (default: {DEFAULT_RESPONSE_TEMPLATE!r}).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Training dataset jsonl (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of dataset rows to inspect (default: 20).",
    )
    args = parser.parse_args()

    if args.check in ("loss", "both") and args.adapter_path is None:
        print("ERROR: --check loss/both requires --adapter-path", file=sys.stderr)
        return 2

    mask_ratio = None
    loss = None

    print("=" * 72)
    print("Gemma-4 checkpoint diagnostic")
    print("=" * 72)
    print(f"  base_model:        {args.base_model}")
    print(f"  response_template: {args.response_template!r}")
    print(f"  dataset:           {args.dataset_path}")
    print(f"  n_samples:         {args.n_samples}")
    print(f"  adapter_path:      {args.adapter_path or '<not provided>'}")
    print(f"  check:             {args.check}")
    print()

    if args.check in ("masking", "both"):
        print("--- MASKING CHECK ---")
        mask_ratio, details = _check_masking(
            args.dataset_path, args.base_model,
            args.response_template, args.n_samples,
        )
        for k, v in details.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"  --> mean_mask_ratio = {mask_ratio:.2%}")
        print(f"      healthy band: [{MASK_RATIO_MIN_HEALTHY:.0%}, {MASK_RATIO_MAX_HEALTHY:.0%}]")
        print()

    if args.check in ("loss", "both"):
        print("--- LOSS CHECK ---")
        loss, details = _check_loss(
            args.adapter_path, args.base_model,
            args.dataset_path, args.n_samples,
        )
        for k, v in details.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"  --> mean_loss = {loss:.4f}")
        print(f"      v3 baseline (flat): {details['v3_flat_loss_baseline']:.4f}")
        print(f"      'descended' threshold: <{LOSS_DESCENDED_THRESHOLD}")
        print()

    verdict = _emit_verdict(mask_ratio, loss)
    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print("=" * 72)
    print()
    print("NEXT STEP:")
    print(_next_step_for(verdict, args.adapter_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
