"""Patch notebook to v7: Fix #18 — pre-tokenize, not just pre-format.

v6 still errored at SFTTrainer line 416 with the same ValueError but the
fuller message reveals the real cause:

    ValueError: Unable to create tensor... Perhaps your features (`text`
    in this case) have excessive nesting (inputs type `list` where type
    `int` is expected).

DataCollatorForCompletionOnlyLM (subclass of DataCollatorForLanguageModeling)
expects each example to already contain `input_ids` (list of ints). It calls
`tokenizer.pad(...)` to align lengths — that requires int IDs, not raw text.

When SFTTrainer is passed a custom data_collator, it skips its own
auto-tokenization step. So the text column never gets converted to
input_ids. The collator then receives `{"text": "<string>", "weight": 1.0}`
and crashes.

Fix #18 (v7): tokenize the dataset explicitly in the same .map() pass that
applies the chat template. The dataset then carries `input_ids` and
`attention_mask` columns the collator can pad.
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


# ── cell 0: bump description ──
set_src(cells[0], """\
# SimSat Gemma-4-E2B v7 — Kaggle T4

Full-precision bfloat16 fine-tune on SimSat satellite encounter-assessment data.

**Stack:** raw transformers + PEFT LoRA r=64 + TRL SFTTrainer + DataCollatorForCompletionOnlyLM\
 · `google/gemma-4-E2B-it` · single T4 (CUDA_VISIBLE_DEVICES=0)

All 18 hard-won T4 fixes applied. v7 key change:
- **Fix #18**: SFTTrainer skips its auto-tokenization step when a custom\
 data_collator is passed. DataCollatorForCompletionOnlyLM expects pre-tokenized\
 `input_ids` (not raw text). v6 errored with "excessive nesting" because the\
 collator received `{"text": "<string>"}` and called `tokenizer.pad()` on a\
 string. Fix: tokenize the `text` column in the same `.map()` that applies the\
 chat template, so the dataset carries `input_ids` + `attention_mask`.

**Expected runtime:** ~35–50 min training + ~5 min eval
""")

# ── locate the dataset prep cell (has Fix #16 marker) and replace its mapping ──
ds_idx, ds_cell = find_cell(lambda s: "Fix #16" in s and "_apply_chat_template" in s)
if ds_cell is None:
    sys.exit("ERROR: could not find Fix #16 dataset prep cell")

# Replace the inner _apply_chat_template + dataset.map block with a combined
# template-then-tokenize map.
old_block_marker = "# Fix #16:"
s = src(ds_cell)
prefix = s.split(old_block_marker, 1)[0]

new_tail = """# Fix #16 + #18: pre-format AND pre-tokenize. SFTTrainer skips its own
# auto-tokenization when a custom data_collator is provided (we use
# DataCollatorForCompletionOnlyLM). That collator expects each row to have
# `input_ids` (list of ints), not raw `text`. Build both in one map pass so
# the dataset gives the collator exactly what it needs.
def _format_and_tokenize(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    enc = tokenizer(
        text,
        truncation=True,
        max_length=1024,                # matches SFTConfig.max_seq_length
        padding=False,                  # collator pads per-batch
        add_special_tokens=False,       # apply_chat_template already added them
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }

dataset = dataset.map(
    _format_and_tokenize,
    remove_columns=[c for c in dataset.column_names if c != "weight"],
)
print(f"Post-tokenize columns: {dataset.column_names}")
print(f"Sample input_ids head ({len(dataset[0]['input_ids'])} tokens):"
      f" {dataset[0]['input_ids'][:25]}...")
assert "input_ids" in dataset.column_names, "Fix #18 failed: 'input_ids' column missing"
assert "attention_mask" in dataset.column_names, "Fix #18 failed: 'attention_mask' column missing"
"""
set_src(ds_cell, prefix + new_tail)

# ── version-string sweep: bump v6 → v7 everywhere ──
for c in cells:
    if c.get("cell_type") != "code":
        continue
    s = src(c)
    new_s = s
    new_s = new_s.replace('SIMSAT GEMMA-4-E2B v6 COMPLETE', 'SIMSAT GEMMA-4-E2B v7 COMPLETE')
    new_s = new_s.replace('"version": "simsat-gemma4-v6"', '"version": "simsat-gemma4-v7"')
    new_s = new_s.replace('simsat-gemma4-v6-adapter', 'simsat-gemma4-v7-adapter')
    new_s = new_s.replace('simsat_gemma4_v6_summary.json', 'simsat_gemma4_v7_summary.json')
    if new_s != s:
        set_src(c, new_s)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook written.")

# ── verify ──
nb2 = json.loads(nb_path.read_text(encoding="utf-8"))
all_src = "\n".join(
    ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
    for c in nb2["cells"]
)
checks = [
    ("v7 in title", "SimSat Gemma-4-E2B v7" in all_src),
    ("Fix #18 marker", "Fix #18" in all_src),
    ("_format_and_tokenize", "_format_and_tokenize" in all_src),
    ("input_ids assertion", "Fix #18 failed" in all_src),
    ("v7 adapter dir", "simsat-gemma4-v7-adapter" in all_src),
    ("v7 summary file", "simsat_gemma4_v7_summary.json" in all_src),
    ("v6 adapter dir gone", "simsat-gemma4-v6-adapter" not in all_src),
    ("max_seq_length retained", "max_seq_length=1024" in all_src),
    ("padding_side retained", 'padding_side = "right"' in all_src),
]
passed = sum(1 for _, ok in checks if ok)
for label, ok in checks:
    print(f"  {'OK' if ok else 'X '} {label}")
print(f"{passed}/{len(checks)} checks passed")
sys.exit(0 if passed == len(checks) else 1)
