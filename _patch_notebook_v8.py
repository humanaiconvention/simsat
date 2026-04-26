"""Patch notebook to v8: Fix #19 — correct response_template for Gemma-4.

v7 trained successfully and produced an adapter, but all 7 eval cases failed
JSON parsing — the model emits unstructured text instead of the JSON contract.

Root cause discovered post-hoc by manually replicating the masking logic on
BEAST: the response_template `<start_of_turn>model\n` tokenizes to 9 tokens
(`<`, `start`, `_`, `of`, `_`, `turn`, `>`, `model`, `\n`) — none of which
appear in the actual chat-templated output. Gemma-4 uses a SPECIAL TOKEN
`<|turn>` (id 105) for turn boundaries, NOT the literal string
`<start_of_turn>`. The chat template emits `[<|turn>(105), model(4368), \\n(107)]`
as the assistant-turn marker.

Result: DataCollatorForCompletionOnlyLM never found the template, so its
masking either no-op'd or masked everything — either way, training never had
the assistant-only-loss signal it was supposed to. The model learned to
mimic the entire conversational structure (system + user + assistant) rather
than just emit the JSON in the assistant slot.

Fix #19 (v8): change response_template to `<|turn>model\n`. Verified by
re-encoding: `tok.encode('<|turn>model\\n', add_special_tokens=False)` yields
`[105, 4368, 107]`, which matches the 3-token marker found in the chat-templated
output stream.

If v8 succeeds, eval should produce structured JSON parseable by the existing
TransformersVLMAdapter, and the 7 'JSON parse failed' warnings should disappear.
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
# SimSat Gemma-4-E2B v8 — Kaggle T4

Full-precision bfloat16 fine-tune on SimSat satellite encounter-assessment data.

**Stack:** raw transformers + PEFT LoRA r=64 + TRL SFTTrainer + DataCollatorForCompletionOnlyLM\
 · `google/gemma-4-E2B-it` · single T4 (CUDA_VISIBLE_DEVICES=0)

All 19 hard-won T4 fixes applied. v8 key change:
- **Fix #19**: response_template was `<start_of_turn>model\\n` — that string\
 tokenizes to 9 word-piece tokens that never appear in the chat-templated\
 output. Gemma-4 uses special token `<|turn>` (id 105) for turn boundaries.\
 Correct template is `<|turn>model\\n` (encodes to 3 tokens [105, 4368, 107]\
 that DO appear). Without this fix DataCollatorForCompletionOnlyLM never\
 located the assistant-turn marker, masking was effectively broken, and the\
 model trained on the entire sequence — explaining v7's 7/7 JSON parse\
 failures despite successful training.

**Expected runtime:** ~35–50 min training + ~5 min eval
""")

# ── locate SFTConfig + collator cell and fix the template ──
collator_idx, collator_cell = find_cell(lambda s: 'response_template="<start_of_turn>model\\n"' in s)
if collator_cell is None:
    sys.exit("ERROR: could not find collator cell with response_template")

s = src(collator_cell)
s = s.replace(
    'response_template="<start_of_turn>model\\n"',
    'response_template="<|turn>model\\n"  # Fix #19: Gemma-4 uses <|turn> special token, not <start_of_turn>',
)
set_src(collator_cell, s)

# ── version-string sweep: bump v7 → v8 everywhere ──
for c in cells:
    if c.get("cell_type") != "code":
        continue
    s = src(c)
    new_s = s
    new_s = new_s.replace('SIMSAT GEMMA-4-E2B v7 COMPLETE', 'SIMSAT GEMMA-4-E2B v8 COMPLETE')
    new_s = new_s.replace('"version": "simsat-gemma4-v7"', '"version": "simsat-gemma4-v8"')
    new_s = new_s.replace('simsat-gemma4-v7-adapter', 'simsat-gemma4-v8-adapter')
    new_s = new_s.replace('simsat_gemma4_v7_summary.json', 'simsat_gemma4_v8_summary.json')
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
    ("v8 in title", "SimSat Gemma-4-E2B v8" in all_src),
    ("Fix #19 marker", "Fix #19" in all_src),
    ("correct response_template",
     'response_template="<|turn>model\\n"' in all_src),
    ("old response_template gone",
     'response_template="<start_of_turn>model\\n"' not in all_src),
    ("v8 adapter dir", "simsat-gemma4-v8-adapter" in all_src),
    ("v8 summary file", "simsat_gemma4_v8_summary.json" in all_src),
    ("v7 adapter dir gone", "simsat-gemma4-v7-adapter" not in all_src),
]
passed = sum(1 for _, ok in checks if ok)
for label, ok in checks:
    print(f"  {'OK' if ok else 'X '} {label}")
print(f"{passed}/{len(checks)} checks passed")
sys.exit(0 if passed == len(checks) else 1)
