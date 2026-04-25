"""Patch notebook to v4: Fix #15 — TRL 0.15+ removed DataCollatorForCompletionOnlyLM.

Cap TRL to <0.15.0 in cell 1 and add defensive import fallback in cell 7.
"""
import json, pathlib

nb_path = pathlib.Path("notebooks/kaggle-simsat-gemma4-v1/notebook.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


# ── cell 0 ── update description to v4
set_src(cells[0], """\
# SimSat Gemma-4-E2B v4 — Kaggle T4

Full-precision bfloat16 fine-tune on SimSat satellite encounter-assessment data.

**Stack:** raw transformers + PEFT LoRA r=64 + TRL SFTTrainer + DataCollatorForCompletionOnlyLM\
 · `google/gemma-4-E2B-it` · single T4 (CUDA_VISIBLE_DEVICES=0)

All 15 hard-won T4 fixes applied. v4 key change:
- **Fix #15**: TRL 0.15.0 removed `DataCollatorForCompletionOnlyLM` in favour of an internal\
 completion-mask parameter. Pinning `trl<0.15.0` restores it. Defensive fallback import also\
 added (tries `trl.trainer.utils` and `trl.data_utils` before raising).

**Expected runtime:** ~35–50 min training + ~5 min eval
""")

# ── cell 1 ── pin TRL to <0.15.0
s1 = src(cells[1]).replace(
    "    \"'trl>=0.12.0' \"\n",
    "    \"'trl>=0.12.0,<0.15.0' \"  # Fix #15: 0.15.0 removed DataCollatorForCompletionOnlyLM\n",
)
set_src(cells[1], s1)

# ── cell 7 ── defensive DataCollatorForCompletionOnlyLM import
s7 = src(cells[7]).replace(
    "from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM",
    "from trl import SFTTrainer, SFTConfig\n"
    "# Fix #15: defensive import — TRL ≥0.15 moved/removed the collator from top-level.\n"
    "try:\n"
    "    from trl import DataCollatorForCompletionOnlyLM\n"
    "except ImportError:\n"
    "    try:\n"
    "        from trl.trainer.utils import DataCollatorForCompletionOnlyLM\n"
    "    except ImportError:\n"
    "        from trl.data_utils import DataCollatorForCompletionOnlyLM",
)
# Also update summary/output_dir for v4
s7 = s7.replace(
    'OUTPUT_DIR = "/kaggle/working/simsat-gemma4-v3-adapter"',
    'OUTPUT_DIR = "/kaggle/working/simsat-gemma4-v4-adapter"',
)
set_src(cells[7], s7)

# ── cell 9 ── version strings to v4
s9 = src(cells[9])
s9 = s9.replace('print("SIMSAT GEMMA-4-E2B v3 COMPLETE")', 'print("SIMSAT GEMMA-4-E2B v4 COMPLETE")')
s9 = s9.replace('"version": "simsat-gemma4-v3"', '"version": "simsat-gemma4-v4"')
s9 = s9.replace('"simsat-gemma4-v3-adapter"', '"simsat-gemma4-v4-adapter"')
s9 = s9.replace('"simsat_gemma4_v3_summary.json"', '"simsat_gemma4_v4_summary.json"')
set_src(cells[9], s9)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook written.")

# ── verify ──
nb2 = json.loads(nb_path.read_text(encoding="utf-8"))
c = nb2["cells"]

def s(cell):
    x = cell["source"]
    return "".join(x) if isinstance(x, list) else x

checks = [
    ("cell0 v4 title",                     "v4" in s(c[0])),
    ("cell0 Fix #15 mentioned",            "Fix #15" in s(c[0])),
    ("cell1 trl<0.15.0 pin",              "trl>=0.12.0,<0.15.0" in s(c[1])),
    ("cell7 defensive import try/except",  "from trl import DataCollatorForCompletionOnlyLM" in s(c[7]) and "except ImportError" in s(c[7])),
    ("cell7 v4 output_dir",               "simsat-gemma4-v4-adapter" in s(c[7])),
    ("cell9 v4 COMPLETE",                 "v4 COMPLETE" in s(c[9])),
    ("cell9 simsat-gemma4-v4",            "simsat-gemma4-v4" in s(c[9])),
    ("cell4 bfloat16 preserved",          "dtype=torch.bfloat16" in s(c[4])),
    ("cell7 data_collator arg preserved", "data_collator=collator" in s(c[7])),
]

all_ok = True
for name, result in checks:
    mark = "✓" if result else "✗ FAIL"
    print(f"  {mark}  {name}")
    if not result:
        all_ok = False

if all_ok:
    print("\nAll checks passed ✓")
else:
    print("\nSOME CHECKS FAILED")
