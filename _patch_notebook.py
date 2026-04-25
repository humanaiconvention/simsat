"""Patch notebook to v3: Fix #13 (permanent bfloat16) + Fix #14 (DataCollatorForCompletionOnlyLM)."""
import json, pathlib

nb_path = pathlib.Path("notebooks/kaggle-simsat-gemma4-v1/notebook.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


# ── cell 0 ── markdown: v3 description
set_src(cells[0], """\
# SimSat Gemma-4-E2B v3 — Kaggle T4

Full-precision bfloat16 fine-tune on SimSat satellite encounter-assessment data.

**Stack:** raw transformers + PEFT LoRA r=64 + TRL SFTTrainer + DataCollatorForCompletionOnlyLM\
 · `google/gemma-4-E2B-it` · single T4 (CUDA_VISIBLE_DEVICES=0)

All 14 hard-won T4 fixes applied. v3 key additions:
- **Fix #13**: bfloat16 full-precision loading — NF4 4-bit quantization blocks Gemma-4's backward\
 pass through `Linear4bit` layers → `grad_norm=0.0` throughout. Full-precision avoids this;\
 5.1 B params @ bf16 ≈ 10.2 GB fits the T4 (16 GB).
- **Fix #14**: `DataCollatorForCompletionOnlyLM` — loss computed **only** on assistant response\
 tokens (`response_template="<start_of_turn>model\\n"`). Without masking, ~650 prompt tokens\
 dilute the gradient from ~75 JSON response tokens to near-zero → model learns to predict the\
 prompt but never learns the task → flat loss, majority-class bias (v2: 0/3 reviewed, 2/10 shortlist).

**Expected runtime:** ~35–50 min training + ~5 min eval
""")

# ── cell 1 ── add torchao to uninstall line
s1 = src(cells[1]).replace(
    'get_ipython().system("pip uninstall -y unsloth unsloth_zoo 2>&1 | tail -3")  # noqa: F821',
    '# Also uninstall torchao: PEFT 0.12+ dispatches into it and raises ImportError\n'
    '# in non-NF4 mode ("Found version 0.10.0, but only versions above 0.16.0 are supported").\n'
    'get_ipython().system("pip uninstall -y unsloth unsloth_zoo torchao 2>&1 | tail -3")  # noqa: F821',
)
set_src(cells[1], s1)

# ── cell 4 ── Fix #13 permanent: remove NF4, load bfloat16
set_src(cells[4], """\
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("\\n" + "=" * 60)
print("LOADING MODEL (bfloat16 full precision — Fix #13)")
print("=" * 60)

# Fix #13: bfloat16 full precision — NF4 4-bit quantization breaks the backward pass
# through Gemma-4's custom Linear4bit attention wrappers. All LoRA gradients become zero
# (grad_norm=0.0) regardless of target_modules, giving zero learning throughout.
# Full-precision bfloat16 avoids the quantized-layer backward-path issue entirely.
#
# Memory: Gemma4ForConditionalGeneration at bfloat16 = 5.1B * 2 bytes ≈ 10.2 GB VRAM;
# T4 has 16 GB — comfortable margin. T4 (Turing SM7.5) has no native bf16 ALUs so
# ops fall back to fp32 internally, but this is preferable to grad_norm=0.
#
# dtype= (not deprecated torch_dtype=) per Fix #9.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,      # Fix #13 + Fix #9: full precision bfloat16, correct kwarg
    device_map="cuda:0",
)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {model.__class__.__name__}")
print(f"Parameters: {param_count:,}")
print(f"Device: {next(model.parameters()).device}")
print(f"Dtype: {next(model.parameters()).dtype}")
""")

# ── cell 5 ── LoRA: update Fix #6 comment + add peft_config guard
s5 = src(cells[5])
s5 = s5.replace(
    "# Fix #6: target inner `.linear` module, NOT the outer Gemma4ClippableLinear wrapper.\n"
    "# PEFT can only inject LoRA into torch.nn.Linear / Linear4bit, not the Gemma-4 custom\n"
    "# outer class. Targeting `q_proj` directly raises ValueError from PEFT.",
    "# Fix #6: target inner `.linear` module — Gemma-4 wraps nn.Linear inside\n"
    "# Gemma4ClippableLinear. PEFT injects LoRA into the inner .linear attribute (a plain\n"
    "# nn.Linear) in both NF4 and bfloat16 modes. Targeting the outer `q_proj` directly\n"
    "# raises ValueError from PEFT's module walker.",
)
s5 = s5.replace(
    "# Custom kbit prep: freeze base WITHOUT fp32 upcast (prepare_model_for_kbit_training\n"
    "# would upcast norms to fp32, causing OOM on Gemma-4-E2B's large embedding tensors).\n"
    "for p in model.parameters():\n"
    "    p.requires_grad = False\n"
    "\n"
    "model = get_peft_model(model, peft_config)",
    "# Freeze all base parameters before PEFT wrapping.\n"
    "for p in model.parameters():\n"
    "    p.requires_grad = False\n"
    "\n"
    "# Guard: a warm Kaggle kernel that previously ran a failed get_peft_model() call may\n"
    "# leave a stale `peft_config` attribute on the model. PEFT then creates a second\n"
    '# adapter ("default_1") on top of the existing one — doubling LoRA memory and\n'
    "# confusing gradient attribution. Clear it before wrapping.\n"
    "if hasattr(model, \"peft_config\"):\n"
    "    del model.peft_config\n"
    "\n"
    "model = get_peft_model(model, peft_config)",
)
set_src(cells[5], s5)

# ── cell 7 ── Training: Fix #14 DataCollatorForCompletionOnlyLM
s7 = src(cells[7])
s7 = s7.replace(
    "from trl import SFTTrainer, SFTConfig",
    "from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM",
)
s7 = s7.replace(
    'OUTPUT_DIR = "/kaggle/working/simsat-gemma4-v2-adapter"',
    'OUTPUT_DIR = "/kaggle/working/simsat-gemma4-v3-adapter"',
)
s7 = s7.replace(
    "# Fix #10: fp16=False — THE showstopper for QLoRA. QLoRA + fp16=True triggers\n"
    "# GradScaler assertion because LoRA params (fp32) bypass GradScaler's inf hooks.\n"
    "# The model already computes in fp16 via bnb_4bit_compute_dtype — AMP not needed.",
    "# Fix #10: fp16=False — do NOT enable AMP. The model is in bfloat16; enabling fp16\n"
    "# AMP triggers GradScaler which conflicts with bfloat16 LoRA params. bf16=False\n"
    "# because T4 (SM7.5) has no native bfloat16 ALUs — AMP adds no benefit.",
)
s7 = s7.replace(
    "    fp16=False,                        # Fix #10: disable AMP with QLoRA\n"
    "    bf16=False,                        # T4 has no native bf16",
    "    fp16=False,                        # Fix #10: no fp16 AMP (bfloat16 model)\n"
    "    bf16=False,                        # T4 has no native bf16 ALUs",
)
s7 = s7.replace(
    "# Fix #7: max_seq_length NOT in SFTTrainer kwargs (moved to SFTConfig in TRL >=0.12)\n"
    "# Fix #8: processing_class= (not deprecated tokenizer= in TRL >=0.16)\n"
    "trainer = SFTTrainer(\n"
    "    model=model,\n"
    "    train_dataset=dataset,\n"
    "    processing_class=tokenizer,        # Fix #8: processing_class=, NOT tokenizer=\n"
    "    args=training_args,\n"
    "    # peft_config=None because we pre-wrapped with get_peft_model above\n"
    ")",
    "# Fix #14: DataCollatorForCompletionOnlyLM — compute loss ONLY on assistant tokens.\n"
    "#\n"
    "# Root cause of v2 flat loss (3.94 throughout, 0/3 reviewed eval accuracy):\n"
    "#   Default SFTTrainer computes cross-entropy over ALL sequence tokens.\n"
    "#   Each example has ~650 prompt tokens (system + user) and ~75 response tokens.\n"
    "#   Task-specific signal is diluted by 650/725 ≈ 90% non-signal tokens →\n"
    "#   effective gradient ≈ near-zero → model memorises prompt format, never learns\n"
    "#   the recommended_action discrimination → flat loss, majority-class output.\n"
    "#\n"
    "# Fix: mask prompt tokens with -100 (CrossEntropyLoss ignores them). Only the\n"
    "# assistant turn contributes to loss. response_template must exactly match Gemma-4's\n"
    '# chat format: "<start_of_turn>model\\n" is the delimiter after the user turn.\n'
    'collator = DataCollatorForCompletionOnlyLM(\n'
    '    response_template="<start_of_turn>model\\n",\n'
    "    tokenizer=tokenizer,\n"
    "    mlm=False,\n"
    ")\n"
    "\n"
    "# Fix #7: max_seq_length NOT in SFTTrainer kwargs (moved to SFTConfig in TRL >=0.12)\n"
    "# Fix #8: processing_class= (not deprecated tokenizer= in TRL >=0.16)\n"
    "trainer = SFTTrainer(\n"
    "    model=model,\n"
    "    train_dataset=dataset,\n"
    "    processing_class=tokenizer,        # Fix #8: processing_class=, NOT tokenizer=\n"
    "    data_collator=collator,            # Fix #14: response-only loss masking\n"
    "    args=training_args,\n"
    "    # peft_config=None because we pre-wrapped with get_peft_model above\n"
    ")",
)
set_src(cells[7], s7)

# ── cell 9 ── Summary: update version strings
s9 = src(cells[9])
s9 = s9.replace('print("SIMSAT GEMMA-4-E2B v1 COMPLETE")', 'print("SIMSAT GEMMA-4-E2B v3 COMPLETE")')
s9 = s9.replace('"version": "simsat-gemma4-v2"', '"version": "simsat-gemma4-v3"')
s9 = s9.replace('"simsat-gemma4-v2-adapter"', '"simsat-gemma4-v3-adapter"')
s9 = s9.replace('"bnb_compute_dtype": "float16"', '"precision": "bfloat16_full"')
s9 = s9.replace(
    'summary_path = "/kaggle/working/simsat_gemma4_v1_summary.json"',
    'summary_path = "/kaggle/working/simsat_gemma4_v3_summary.json"',
)
s9 = s9.replace('"simsat_gemma4_v1_summary.json"', '"simsat_gemma4_v3_summary.json"')
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
    ("cell0 v3 title",               "v3" in s(c[0])),
    ("cell0 Fix #14 mentioned",       "Fix #14" in s(c[0])),
    ("cell1 torchao uninstall",       "torchao" in s(c[1])),
    ("cell4 bfloat16 headline",       "bfloat16 full precision" in s(c[4])),
    ("cell4 no BitsAndBytesConfig",   "BitsAndBytesConfig" not in s(c[4])),
    ("cell4 no bnb_config",           "bnb_config" not in s(c[4])),
    ("cell4 dtype=torch.bfloat16",    "dtype=torch.bfloat16" in s(c[4])),
    ("cell5 peft_config guard",       "del model.peft_config" in s(c[5])),
    ("cell7 DataCollator import",     "DataCollatorForCompletionOnlyLM" in s(c[7])),
    ("cell7 collator instantiated",   "DataCollatorForCompletionOnlyLM(" in s(c[7])),
    ("cell7 response_template",       "start_of_turn>model" in s(c[7])),
    ("cell7 data_collator arg",       "data_collator=collator" in s(c[7])),
    ("cell7 v3 output_dir",           "simsat-gemma4-v3-adapter" in s(c[7])),
    ("cell9 v3 COMPLETE",             "v3 COMPLETE" in s(c[9])),
    ("cell9 simsat-gemma4-v3",        "simsat-gemma4-v3" in s(c[9])),
    ("cell9 bfloat16_full",           "bfloat16_full" in s(c[9])),
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
