"""Patch notebook to v5: Fix #16 — pre-format ChatML messages into a 'text' column.

v4 errored on Kaggle with `KeyError: 'text'` at SFTTrainer line 416. SFTTrainer's
default `dataset_text_field='text'` looks for that column on each example, but
the dataset only has 'messages' (ChatML). Pre-format via apply_chat_template so
each row has a 'text' field; the DataCollatorForCompletionOnlyLM then masks
everything before the response_template.

This is THE fix that should unblock training. Combined with v4's bf16 +
DataCollatorForCompletionOnlyLM + TRL pin, the loss should finally descend.

The fix is inserted in cell 6 (the cell that runs `dataset = load_dataset(...)`)
RIGHT BEFORE the SFTTrainer is constructed in cell 7. Tokenizer is loaded in
cell 5, so it's available.
"""
import json, pathlib, sys

nb_path = pathlib.Path("notebooks/kaggle-simsat-gemma4-v1/notebook.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def find_cell(predicate):
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and predicate(src(c)):
            return i, c
    return None, None


# ── cell 0 ── update description to v5
set_src(cells[0], """\
# SimSat Gemma-4-E2B v5 — Kaggle T4

Full-precision bfloat16 fine-tune on SimSat satellite encounter-assessment data.

**Stack:** raw transformers + PEFT LoRA r=64 + TRL SFTTrainer + DataCollatorForCompletionOnlyLM\
 · `google/gemma-4-E2B-it` · single T4 (CUDA_VISIBLE_DEVICES=0)

All 16 hard-won T4 fixes applied. v5 key change:
- **Fix #16**: SFTTrainer's default `dataset_text_field='text'` triggered\
 `KeyError: 'text'` on the v4 run because the dataset has 'messages' (ChatML),\
 not 'text'. Pre-format each row via `tokenizer.apply_chat_template(...)` so\
 the dataset has a 'text' column; the DataCollatorForCompletionOnlyLM then\
 correctly masks everything before `<start_of_turn>model\\n`.

**Expected runtime:** ~35–50 min training + ~5 min eval
""")

# ── locate dataset-prep cell (`dataset = load_dataset(...)`) ──
ds_idx, ds_cell = find_cell(lambda s: "dataset = load_dataset(" in s)
if ds_cell is None:
    sys.exit("ERROR: could not find dataset = load_dataset(...) cell")

# Append the chat-template formatting step to the same cell so the next cell
# (SFTTrainer construction) sees a 'text' column.
extra = """

# Fix #16: SFTTrainer expects a 'text' column by default. Our dataset has
# 'messages' (ChatML). Apply the chat template here so each row gains a
# pre-formatted 'text' string. DataCollatorForCompletionOnlyLM then masks
# everything before '<start_of_turn>model\\n' so loss is computed only on
# the assistant turn.
def _apply_chat_template(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(
    _apply_chat_template,
    remove_columns=[c for c in dataset.column_names if c != "weight"],
)
print(f"Post-template columns: {dataset.column_names}")
print(f"Sample text head: {dataset[0]['text'][:200]}...")
assert "text" in dataset.column_names, "Fix #16 failed: 'text' column not produced"
"""
set_src(ds_cell, src(ds_cell) + extra)

# ── cell 7 ── update output_dir to v5
s7_cell_idx, s7 = find_cell(lambda s: "OUTPUT_DIR =" in s and "simsat-gemma4-v4-adapter" in s)
if s7 is not None:
    new_s7 = src(s7).replace(
        "/kaggle/working/simsat-gemma4-v4-adapter",
        "/kaggle/working/simsat-gemma4-v5-adapter",
    )
    set_src(s7, new_s7)
else:
    print("WARN: could not find OUTPUT_DIR=simsat-gemma4-v4-adapter — skipping rename")

# ── final cell ── version strings to v5
for c in cells:
    if c.get("cell_type") != "code":
        continue
    s = src(c)
    new_s = s
    new_s = new_s.replace('SIMSAT GEMMA-4-E2B v4 COMPLETE', 'SIMSAT GEMMA-4-E2B v5 COMPLETE')
    new_s = new_s.replace('SIMSAT GEMMA-4-E2B v3 COMPLETE', 'SIMSAT GEMMA-4-E2B v5 COMPLETE')
    new_s = new_s.replace('"version": "simsat-gemma4-v4"', '"version": "simsat-gemma4-v5"')
    new_s = new_s.replace('"version": "simsat-gemma4-v3"', '"version": "simsat-gemma4-v5"')
    new_s = new_s.replace('"simsat-gemma4-v4-adapter"', '"simsat-gemma4-v5-adapter"')
    new_s = new_s.replace('"simsat-gemma4-v3-adapter"', '"simsat-gemma4-v5-adapter"')
    new_s = new_s.replace('simsat_gemma4_v4_summary.json', 'simsat_gemma4_v5_summary.json')
    new_s = new_s.replace('simsat_gemma4_v3_summary.json', 'simsat_gemma4_v5_summary.json')
    if new_s != s:
        set_src(c, new_s)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook written.")

# ── verify ──
nb2 = json.loads(nb_path.read_text(encoding="utf-8"))
checks = []
all_src = "\n".join(
    ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
    for c in nb2["cells"]
)
checks.append(("v5 in title", "SimSat Gemma-4-E2B v5" in all_src))
checks.append(("Fix #16 marker", "Fix #16" in all_src))
checks.append(("apply_chat_template call", "_apply_chat_template" in all_src))
checks.append(("v5 adapter dir", "simsat-gemma4-v5-adapter" in all_src))
checks.append(("v5 summary file", "simsat_gemma4_v5_summary.json" in all_src))
checks.append(("v4 adapter dir gone", "simsat-gemma4-v4-adapter" not in all_src))
checks.append(("v3 adapter dir gone", "simsat-gemma4-v3-adapter" not in all_src))
checks.append(("dataset.map call", "dataset.map(" in all_src))
checks.append(("text column assertion", "Fix #16 failed" in all_src))

passed = sum(1 for _, ok in checks if ok)
for label, ok in checks:
    print(f"  {'OK' if ok else 'X '} {label}")
print(f"{passed}/{len(checks)} checks passed")
sys.exit(0 if passed == len(checks) else 1)
