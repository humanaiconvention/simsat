#!/usr/bin/env python3
"""
Build notebook.ipynb from simsat_lfm_v1_training.py.

Splits on CELL banners (# ====... / # CELL N: ... / # ====...).
Validates LFM-specific invariants before writing.

Usage:
    python build_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "simsat_lfm_v4_training.py"
OUT = Path(__file__).parent / "notebook.ipynb"

src = SRC.read_text(encoding="utf-8")

# Strip shebang + module docstring
src = re.sub(r"^#!.*?\n", "", src)
src = re.sub(r'^"""[\s\S]*?"""\n', "", src, count=1)

# ---- Validation checks ------------------------------------------------------
# Strip comments + docstrings so heuristics don't match prose.
_code_only = re.sub(r"#[^\n]*", "", src)
_code_only = re.sub(r'""".*?"""', "", _code_only, flags=re.DOTALL)
_code_only = re.sub(r"'''.*?'''", "", _code_only, flags=re.DOTALL)

# Note: comparisons against the empty string check that a literal/keyword is
# present in the cleaned source code (not inside a comment or docstring).
checks = {
    "CUDA_VISIBLE_DEVICES=0": "CUDA_VISIBLE_DEVICES" in src,
    "bfloat16 (LFM native dtype)": "bfloat16" in _code_only,
    "no fp16=True (would conflict with bf16)": "fp16=True" not in _code_only,
    "Authoritative LFM_MODULES list": (
        "LFM_MODULES" in _code_only
        and '"q_proj"' in _code_only
        and '"k_proj"' in _code_only
        and '"v_proj"' in _code_only
        and '"out_proj"' in _code_only
        and '"in_proj"' in _code_only
    ),
    "VISION_TOWER_MODULES (fc1, fc2)": (
        "VISION_TOWER_MODULES" in _code_only
        and '"fc1"' in _code_only
        and '"fc2"' in _code_only
    ),
    "MULTI_MODAL_PROJECTOR_MODULES (linear_1, linear_2)": (
        "MULTI_MODAL_PROJECTOR_MODULES" in _code_only
        and '"linear_1"' in _code_only
        and '"linear_2"' in _code_only
    ),
    "Pre-train target match audit": "Pre-train target_modules match audit" in src,
    "Post-save adapter sanity gate": "Adapter sanity check" in src,
    "lora_B non-zero check": "n_lora_b_nonzero" in _code_only,
    "enable_input_require_grads after get_peft_model": (
        src.index("get_peft_model(model, peft_config)")
        < src.index("enable_input_require_grads")
    ),
    "use_reentrant=False": '"use_reentrant": False' in src,
    "no Unsloth import (TRL only)": "from unsloth" not in src.lower(),
    "SFTConfig (not TrainingArguments)": "SFTConfig" in src,
    "AutoModelForImageTextToText": "AutoModelForImageTextToText" in src,
    "remove_unused_columns=False": "remove_unused_columns=False" in src,
    "skip_prepare_dataset True": '"skip_prepare_dataset": True' in src,
    "image_token_id mask in collator": "image_token_id" in _code_only,
    "Base-model holdout eval before LoRA": "BASE-model hold-out inference" in src,
    "Tuned-model holdout eval after training": "TUNED-model hold-out inference" in src,
    "Base/tuned metrics report": "BASE vs TUNED" in src,
    "Holdout eval report saved as JSON": "holdout_eval_report.json" in src,
    "score_predictions function defined": "def score_predictions" in _code_only,
    "Stratified holdout dataset path": "simsat_lfm_holdout.jsonl" in src,
}
print("Build checks:")
all_pass = True
for label, ok in checks.items():
    icon = "OK" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{icon}]  {label}")

if not all_pass:
    raise SystemExit("Build validation failed — fix issues above before pushing.")

# ---- Split into cells on CELL banners --------------------------------------
cell_pat = re.compile(r"# =+\n# CELL \d+:[^\n]*\n# =+\n", re.MULTILINE)
parts = [p.strip() for p in cell_pat.split(src) if p.strip()]

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# SimSat LFM2.5-VL-450M v1 — Kaggle T4\n",
            "\n",
            "LoRA fine-tune of `LiquidAI/LFM2.5-VL-450M` on SimSat operator-reviewed\n",
            "Sentinel-2 encounter assessments.\n",
            "\n",
            "**Stack:** raw transformers + TRL SFTTrainer + PEFT LoRA r=8 ·\n",
            "bf16 · single T4 (CUDA_VISIBLE_DEVICES=0)\n",
            "\n",
            "**LoRA target_modules** are taken verbatim from\n",
            "`Liquid4All/leap-finetune/src/leap_finetune/training_configs/peft_configs.py:DEFAULT_VLM_LORA`\n",
            "— the same set Liquid AI uses for their official VLM SFT examples.\n",
            "\n",
            "**Expected runtime:** 30–60 min training + ~5 min smoke eval\n",
            "\n",
            "Inputs:\n",
            "- Dataset: `benhaslam/simsat-lfm-v1` (141 train / 4 eval, ~165 MB)\n",
            "- Model: downloaded from HF at runtime (`LiquidAI/LFM2.5-VL-450M`)\n",
        ],
    },
]

for part in parts:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": part.splitlines(keepends=True),
    })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"\nWrote {OUT} ({len(cells)} cells, {OUT.stat().st_size:,} bytes)")
