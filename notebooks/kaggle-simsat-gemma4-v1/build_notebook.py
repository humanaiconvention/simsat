#!/usr/bin/env python3
"""
Build notebook.ipynb from simsat_gemma4_v1_training.py.

Splits on CELL banners (# ====... / # CELL N: ... / # ====...).
Prepends the environment/install cell as cell 0.
Writes notebook.ipynb ready for `kaggle kernels push`.

Usage:
    python build_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "simsat_gemma4_v1_training.py"
OUT = Path(__file__).parent / "notebook.ipynb"

src = SRC.read_text(encoding="utf-8")

# Strip shebang + module docstring
src = re.sub(r"^#!.*?\n", "", src)
src = re.sub(r'^"""[\s\S]*?"""\n', "", src, count=1)

# ── Validation checks (print on build so CI can spot regressions) ───────────
# Strip line/block comments first so heuristics don't match strings inside
# explanatory prose (e.g. the audit reference at line 173).
_code_only = re.sub(r"#[^\n]*", "", src)
_code_only = re.sub(r'""".*?"""', "", _code_only, flags=re.DOTALL)

checks = {
    "CUDA_VISIBLE_DEVICES=0": "CUDA_VISIBLE_DEVICES" in src,
    "float16 compute dtype": "bnb_4bit_compute_dtype=torch.float16" in src,
    "no bfloat16 in code": "bfloat16" not in _code_only,
    "fp16=False": "fp16=False" in src,
    # Fix #6 (corrected 2026-04-27): language_model.layers regex, NOT q_proj.linear.
    # The .linear suffix matched only multimodal towers. See
    # GEMMA4_LORA_NULL_TRAINING_AUDIT.md.
    "language_model.layers regex target": (
        "language_model" in _code_only
        and "layers" in _code_only
        and 'target_modules=r"' in _code_only
    ),
    "no .linear target in code (would re-introduce bug)": (
        '"q_proj.linear"' not in _code_only
    ),
    "processing_class=": "processing_class=tokenizer" in src,
    # Fix #8: SFTTrainer must use processing_class=, not tokenizer=.
    # DataCollator legitimately uses tokenizer=tokenizer (Fix #14), so look for
    # tokenizer= specifically inside the SFTTrainer( ... ) block.
    "no tokenizer= kwarg in SFTTrainer": (
        "tokenizer=tokenizer," not in (
            re.search(r"SFTTrainer\((.*?)\)", _code_only, flags=re.DOTALL)
            or type("X", (), {"group": lambda *_: ""})
        ).group(1)
    ),
    "no Unsloth import": "from unsloth" not in src.lower(),
    "SFTConfig (not TrainingArguments)": "SFTConfig" in src,
    "adamw_torch": '"adamw_torch"' in src,
    "enable_input_require_grads after get_peft_model": (
        src.index("get_peft_model(model, peft_config)")
        < src.index("enable_input_require_grads")
    ),
    "use_reentrant=False": '"use_reentrant": False' in src,
    # Sanity gate added 2026-04-27 — must not silently strip out.
    "post-save adapter sanity gate": "Adapter sanity check" in src,
}
print("Build checks:")
all_pass = True
for label, ok in checks.items():
    icon = "✓" if ok else "✗ FAIL"
    if not ok:
        all_pass = False
    print(f"  {icon}  {label}")

if not all_pass:
    raise SystemExit("Build validation failed — fix issues above before pushing.")

# ── Split into cells on CELL banners ────────────────────────────────────────
cell_pat = re.compile(r"# =+\n# CELL \d+:[^\n]*\n# =+\n", re.MULTILINE)
parts = [p.strip() for p in cell_pat.split(src) if p.strip()]

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# SimSat Gemma-4-E2B v2 — Kaggle T4\n",
            "\n",
            "QLoRA fine-tune on SimSat satellite encounter-assessment data.\n",
            "\n",
            "**Stack:** raw transformers + bitsandbytes NF4 4-bit + PEFT LoRA r=64 + "
            "TRL SFTTrainer · `google/gemma-4-E2B-it` · single T4 (CUDA_VISIBLE_DEVICES=0)\n",
            "\n",
            "All 12 hard-won T4 fixes applied. v2 adds Fix #11 (enable_input_require_grads "
            "after get_peft_model) and Fix #12 (use_reentrant=False), resolving the "
            "grad_norm=0.0 / zero-learning failure from v1.\n",
            "\n",
            "**Expected runtime:** ~30–45 min training + ~5 min eval\n",
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
