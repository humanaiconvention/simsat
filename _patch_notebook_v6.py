"""Patch notebook to v6: Fix #17 — set max_seq_length so SFTTrainer truncates.

v5 got further than v4 — model loaded, LoRA wrapped (0.44% trainable), dataset
pre-formatted via apply_chat_template (Fix #16 worked). But training crashed
at SFTTrainer line 416 with:

    ValueError: Unable to create tensor, you should probably activate
    truncation and/or padding with 'padding=True' or
    'truncation=True' to have batched tensors with the same length.

Root cause: the source comment claimed "TRL infers from
tokenizer.model_max_length" — but Gemma's model_max_length is effectively
infinity (10^18). SFTTrainer auto-tokenizes the 'text' column with no
truncation; sequences end up ragged; the collator can't tensor-ify them.

Fix #17: explicitly set max_seq_length=1024 in SFTConfig (Gemma-4-E2B
context is 8192; SimSat ChatML rows are <800 tokens so 1024 is comfortable
headroom that almost never truncates real data).

Also addresses the v5 warning about padding_side: explicitly set
tokenizer.padding_side = "right" before SFTTrainer construction.
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


# ── cell 0 ── update description to v6
set_src(cells[0], """\
# SimSat Gemma-4-E2B v6 — Kaggle T4

Full-precision bfloat16 fine-tune on SimSat satellite encounter-assessment data.

**Stack:** raw transformers + PEFT LoRA r=64 + TRL SFTTrainer + DataCollatorForCompletionOnlyLM\
 · `google/gemma-4-E2B-it` · single T4 (CUDA_VISIBLE_DEVICES=0)

All 17 hard-won T4 fixes applied. v6 key changes:
- **Fix #17**: SFTConfig now sets `max_seq_length=1024` explicitly. v5 errored\
 with `ValueError: Unable to create tensor` because Gemma's\
 `tokenizer.model_max_length` is effectively infinity, so SFTTrainer's\
 auto-tokenization didn't truncate and the collator got ragged sequences.
- Set `tokenizer.padding_side = "right"` to silence the v5 warning and match\
 SFTTrainer's expected layout for half-precision training.

**Expected runtime:** ~35–50 min training + ~5 min eval
""")

# ── locate SFTConfig cell and inject max_seq_length ──
sft_idx, sft_cell = find_cell(lambda s: "training_args = SFTConfig(" in s)
if sft_cell is None:
    sys.exit("ERROR: could not find training_args = SFTConfig(...) cell")

s = src(sft_cell)
if "max_seq_length=" not in s:
    s = s.replace(
        "    remove_unused_columns=False,",
        "    remove_unused_columns=False,\n"
        "    max_seq_length=1024,               # Fix #17: Gemma's model_max_length is effectively infinity;\n"
        "                                       # SFTTrainer needs an explicit cap or batches end up ragged\n"
        "                                       # and the collator's tensor conversion fails.",
    )
    # Also strip the obsolete 'max_seq_length excluded' comment block.
    s = s.replace(
        "    # max_seq_length excluded — removed from SFTTrainer in TRL >=0.12 AND from\n"
        "    # SFTConfig in later TRL versions. TRL infers from tokenizer.model_max_length.\n",
        "",
    )
    set_src(sft_cell, s)

# ── inject padding_side fix near tokenizer load (cell 3, model+tokenizer load) ──
tok_idx, tok_cell = find_cell(lambda s: "tokenizer.pad_token = tokenizer.eos_token" in s)
if tok_cell is not None:
    ts = src(tok_cell)
    if 'padding_side' not in ts:
        ts = ts.replace(
            "tokenizer.pad_token = tokenizer.eos_token",
            "tokenizer.pad_token = tokenizer.eos_token\n"
            "tokenizer.padding_side = \"right\"  # Fix #17: SFTTrainer expects right-padding for half-precision training",
        )
        set_src(tok_cell, ts)

# ── version-string sweep: bump v5 → v6 everywhere ──
for c in cells:
    if c.get("cell_type") != "code":
        continue
    s = src(c)
    new_s = s
    new_s = new_s.replace('SIMSAT GEMMA-4-E2B v5 COMPLETE', 'SIMSAT GEMMA-4-E2B v6 COMPLETE')
    new_s = new_s.replace('"version": "simsat-gemma4-v5"', '"version": "simsat-gemma4-v6"')
    new_s = new_s.replace('"simsat-gemma4-v5-adapter"', '"simsat-gemma4-v6-adapter"')
    new_s = new_s.replace('/kaggle/working/simsat-gemma4-v5-adapter', '/kaggle/working/simsat-gemma4-v6-adapter')
    new_s = new_s.replace('simsat_gemma4_v5_summary.json', 'simsat_gemma4_v6_summary.json')
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
    ("v6 in title", "SimSat Gemma-4-E2B v6" in all_src),
    ("Fix #17 marker", "Fix #17" in all_src),
    ("max_seq_length=1024", "max_seq_length=1024" in all_src),
    ("padding_side right", 'tokenizer.padding_side = "right"' in all_src),
    ("v6 adapter dir", "simsat-gemma4-v6-adapter" in all_src),
    ("v6 summary file", "simsat_gemma4_v6_summary.json" in all_src),
    ("v5 adapter dir gone", "simsat-gemma4-v5-adapter" not in all_src),
    ("obsolete comment gone",
     "max_seq_length excluded" not in all_src),
]
passed = sum(1 for _, ok in checks if ok)
for label, ok in checks:
    print(f"  {'OK' if ok else 'X '} {label}")
print(f"{passed}/{len(checks)} checks passed")
sys.exit(0 if passed == len(checks) else 1)
