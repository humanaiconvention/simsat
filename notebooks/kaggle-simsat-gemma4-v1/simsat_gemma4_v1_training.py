#!/usr/bin/env python3
"""
SimSat Gemma-4-E2B QLoRA Fine-tune — Kaggle T4 (text-only)
===========================================================
Trains Gemma-4-E2B on SimSat encounter-assessment data.
All 10 hard-won T4 fixes from GEMMA4_KAGGLE_NOTES.md applied.

Inputs (Kaggle-attached):
  - Model: google/gemma-4/transformers/gemma-4-e2b-it/1
  - Dataset: benhaslam/simsat-gemma4-v1

Expected runtime: ~30-45 min training + ~5 min eval = ~40-50 min total
"""

# ============================================================
# CELL 0: Environment + dependency install
# ============================================================
# Fix #2: pin to single T4. device_map="auto" across T4x2 fights with HF
# Trainer's nn.DataParallel wrap -> RuntimeError: all params must be on cuda:0.
# Hiding cuda:1 is cleaner than is_parallelizable hacks.
import os, shutil, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Fix #3: uninstall Unsloth if left from a prior warm session.
# Even without `import unsloth`, it monkey-patches SFTTrainer globally.
get_ipython().system("pip uninstall -y unsloth unsloth_zoo 2>&1 | tail -3")  # noqa: F821

# Purge warm-kernel module state so next imports are clean.
for _m in list(sys.modules):
    if _m.startswith(("unsloth", "trl", "peft", "transformers", "accelerate", "bitsandbytes", "datasets")):
        del sys.modules[_m]

shutil.rmtree("/kaggle/working/unsloth_compiled_cache", ignore_errors=True)

# Fix #4: NO Unsloth. Gemma-4 detected as multimodal -> silent JIT hang.
# Raw transformers + bitsandbytes + peft + trl is the working stack.
get_ipython().system(  # noqa: F821
    "pip install -q -U "
    "'transformers>=4.51.0' "
    "'trl>=0.12.0' "
    "'peft>=0.12.0' "
    "'accelerate>=0.33.0' "
    "'bitsandbytes>=0.44.0' "
    "'datasets>=2.19.0'"
)

# ============================================================
# CELL 1: Detect paths
# ============================================================
import glob, json

print("=" * 60)
print("INPUT FILE LISTING")
print("=" * 60)
for d, _, fs in os.walk("/kaggle/input"):
    for f in fs:
        fp = os.path.join(d, f)
        print(f"  {fp}  ({os.path.getsize(fp):,} bytes)")

# Fix #1: glob the real Kaggle-attached model path
MODEL_ID = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"
if not os.path.exists(MODEL_ID):
    candidates = glob.glob("/kaggle/input/**/gemma*e2b*/1", recursive=True)
    if candidates:
        MODEL_ID = candidates[0]
    else:
        MODEL_ID = "google/gemma-4-E2B-it"
        print(f"WARNING: No local model found, falling back to HF: {MODEL_ID}")
print(f"\nModel: {MODEL_ID}")

# Dataset paths — simsat-gemma4-v1
DATASET_ROOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "simsat_train.jsonl" in files:
        DATASET_ROOT = root
        break

if DATASET_ROOT is None:
    raise RuntimeError("ERROR: simsat_train.jsonl not found in /kaggle/input. Is the dataset attached?")

TRAIN_PATH = os.path.join(DATASET_ROOT, "simsat_train.jsonl")
EVAL_REVIEWED_PATH = os.path.join(DATASET_ROOT, "simsat_eval_reviewed.jsonl")
EVAL_SHORTLIST_PATH = os.path.join(DATASET_ROOT, "simsat_eval_shortlist.jsonl")

print(f"Dataset root: {DATASET_ROOT}")
print(f"Train: {TRAIN_PATH} ({os.path.getsize(TRAIN_PATH):,} bytes)")

# ============================================================
# CELL 2: Load + validate training data
# ============================================================
print("\n" + "=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

with open(TRAIN_PATH) as f:
    train_rows = [json.loads(line) for line in f if line.strip()]

print(f"Training examples: {len(train_rows)}")

# Validate structure
first = train_rows[0]
assert "messages" in first, f"Expected 'messages' key, got: {list(first.keys())}"
assert len(first["messages"]) == 3, f"Expected 3 messages (sys/user/asst), got {len(first['messages'])}"
print(f"Format: ChatML messages ✓")
print(f"Sample user turn preview: {first['messages'][1]['content'][:120]}...")
print(f"Sample assistant turn: {first['messages'][2]['content'][:120]}...")

# Load eval sets
eval_reviewed = []
if os.path.exists(EVAL_REVIEWED_PATH):
    with open(EVAL_REVIEWED_PATH) as f:
        eval_reviewed = [json.loads(line) for line in f if line.strip()]
    print(f"Eval reviewed: {len(eval_reviewed)} examples")

eval_shortlist = []
if os.path.exists(EVAL_SHORTLIST_PATH):
    with open(EVAL_SHORTLIST_PATH) as f:
        eval_shortlist = [json.loads(line) for line in f if line.strip()]
    print(f"Eval shortlist: {len(eval_shortlist)} examples")

# ============================================================
# CELL 3: Load model with QLoRA (4-bit NF4)
# ============================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("\n" + "=" * 60)
print("LOADING MODEL (QLoRA NF4 4-bit)")
print("=" * 60)

# Fix #5: float16 (NOT bfloat16). T4 (Turing, CC 7.5) has no native bf16.
# bf16 on T4 falls back to slow emulation -> memory pressure + bnb kernel errors.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # Fix #5: float16, NOT bfloat16
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix #17: SFTTrainer expects right-padding for half-precision training

# Fix #9: use `dtype=` (not deprecated `torch_dtype=`) in from_pretrained.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda:0",
    dtype=torch.float16,   # Fix #9: `dtype=`, not `torch_dtype=`
)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {model.__class__.__name__}")
print(f"Parameters: {param_count:,}")
print(f"Device: {next(model.parameters()).device}")

# ============================================================
# CELL 4: Configure LoRA
# ============================================================
from peft import LoraConfig, get_peft_model

print("\n" + "=" * 60)
print("CONFIGURING LoRA")
print("=" * 60)

# Fix #6: target inner `.linear` module, NOT the outer Gemma4ClippableLinear wrapper.
# PEFT can only inject LoRA into torch.nn.Linear / Linear4bit, not the Gemma-4 custom
# outer class. Targeting `q_proj` directly raises ValueError from PEFT.
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear",
        "gate_proj.linear", "up_proj.linear", "down_proj.linear",
    ],
)

# Custom kbit prep: freeze base WITHOUT fp32 upcast (prepare_model_for_kbit_training
# would upcast norms to fp32, causing OOM on Gemma-4-E2B's large embedding tensors).
for p in model.parameters():
    p.requires_grad = False

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Fix #11: enable_input_require_grads AFTER get_peft_model, not before.
# PEFT wrapping overrides the base model's forward(); a hook registered on the
# base model before wrapping is lost. Must register on the PEFT-wrapped model so
# gradients can flow back through frozen base layers to the LoRA adapters.
# (v1 bug: called before get_peft_model → grad_norm=0.0 throughout, zero learning)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# ============================================================
# CELL 5: Load dataset
# ============================================================
from datasets import load_dataset

print("\n" + "=" * 60)
print("PREPARING DATASET")
print("=" * 60)

dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")
print(f"Dataset: {len(dataset)} examples")
print(f"Columns: {dataset.column_names}")

# Fix #16: SFTTrainer's default dataset_text_field="text" raised KeyError on
# v4 because the ChatML 'messages' column has no 'text' field. Pre-format
# each row by applying the chat template; the resulting 'text' column is
# what DataCollatorForCompletionOnlyLM masks against.
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

# ============================================================
# CELL 6: Train
# ============================================================
from trl import SFTTrainer, SFTConfig

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

OUTPUT_DIR = "/kaggle/working/simsat-gemma4-v6-adapter"

# Fix #10: fp16=False — THE showstopper for QLoRA. QLoRA + fp16=True triggers
# GradScaler assertion because LoRA params (fp32) bypass GradScaler's inf hooks.
# The model already computes in fp16 via bnb_4bit_compute_dtype — AMP not needed.
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,    # effective batch = 8
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    save_strategy="epoch",
    logging_steps=5,
    num_train_epochs=2,
    fp16=False,                        # Fix #10: disable AMP with QLoRA
    bf16=False,                        # T4 has no native bf16
    gradient_checkpointing=True,       # saves ~3-5 GiB activations on T4
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Fix #12: use_reentrant=True
    # (default) breaks gradient flow through frozen base layers to LoRA adapters.
    # use_reentrant=False is the PEFT-recommended mode for QLoRA + grad checkpointing.
    remove_unused_columns=False,
    max_seq_length=1024,               # Fix #17: Gemma's model_max_length is effectively infinity;
                                       # SFTTrainer needs an explicit cap or batches end up ragged
                                       # and the collator's tensor conversion fails.
    report_to="none",
    dataloader_num_workers=0,          # avoid multiprocessing issues on T4
)

# Fix #7: max_seq_length NOT in SFTTrainer kwargs (moved to SFTConfig in TRL >=0.12)
# Fix #8: processing_class= (not deprecated tokenizer= in TRL >=0.16)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,        # Fix #8: processing_class=, NOT tokenizer=
    args=training_args,
    # peft_config=None because we pre-wrapped with get_peft_model above
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

train_result = trainer.train()
print(f"\nTraining complete!")
print(f"  Final loss: {train_result.training_loss:.4f}")
print(f"  Steps: {train_result.global_step}")

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved: {OUTPUT_DIR}")

# ============================================================
# CELL 7: Evaluate — action accuracy on holdout cases
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION — ACTION ACCURACY")
print("=" * 60)

model.eval()

SYSTEM_PROMPT = (
    "You are a satellite encounter-assessment AI. "
    "Given metadata about a planned Earth observation, return a compact JSON assessment. "
    "Fields: usable_observation (bool), scene_match_score (float 0-1), "
    "salience_score (float 0-1), change_or_event_score (float 0-1), "
    "occlusion_or_cloud_risk (float 0-1), confidence (float 0-1), "
    "recommended_action (accept|defer|refine|skip), rationale_tags (list[str]). "
    "Return only valid JSON, no markdown, no explanation."
)


def assess_example(record: dict) -> dict:
    prompt_text = record.get("prompt_text", record.get("messages", [{}])[1].get("content", ""))
    target = record.get("target_json") or {}
    expected_action = target.get("recommended_action", "?")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=896)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    predicted_action = "?"
    try:
        parsed = json.loads(response)
        predicted_action = parsed.get("recommended_action", "?")
    except (json.JSONDecodeError, ValueError):
        # Try to extract action from partial JSON
        for action in ("accept", "defer", "refine", "skip"):
            if f'"recommended_action": "{action}"' in response or f'"recommended_action":"{action}"' in response:
                predicted_action = action
                break

    return {
        "trace_id": record.get("trace_id", "?"),
        "scenario_pack": record.get("scenario_pack", "?"),
        "target_label": record.get("target_label", "?"),
        "expected_action": expected_action,
        "predicted_action": predicted_action,
        "correct": predicted_action == expected_action,
        "response_preview": response[:200],
    }


def run_eval(examples: list, label: str) -> dict:
    results = []
    for ex in examples:
        r = assess_example(ex)
        results.append(r)
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} [{r['scenario_pack']}] {r['target_label']}: {r['expected_action']} → {r['predicted_action']}")

    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / len(results) if results else 0.0
    print(f"\n  {label}: {n_correct}/{len(results)} correct ({accuracy:.1%})")
    return {"label": label, "accuracy": accuracy, "n_correct": n_correct, "n_total": len(results), "results": results}


eval_results = {}

if eval_reviewed:
    print("\n--- Reviewed holdout (high-quality labels) ---")
    eval_results["reviewed"] = run_eval(eval_reviewed, "reviewed")

if eval_shortlist:
    print("\n--- Shortlist holdout (10 per-target representatives) ---")
    eval_results["shortlist"] = run_eval(eval_shortlist, "shortlist")

# ============================================================
# CELL 8: Summary + save results
# ============================================================
print("\n" + "=" * 60)
print("SIMSAT GEMMA-4-E2B v1 COMPLETE")
print("=" * 60)
print(f"  Training loss: {train_result.training_loss:.4f}")
print(f"  Training steps: {train_result.global_step}")
for name, res in eval_results.items():
    print(f"  Eval ({name}): {res['n_correct']}/{res['n_total']} = {res['accuracy']:.1%}")
print(f"  Adapter: {OUTPUT_DIR}")

summary = {
    "version": "simsat-gemma4-v6",
    "base_model": MODEL_ID,
    "training_loss": round(train_result.training_loss, 4),
    "training_steps": train_result.global_step,
    "training_examples": len(dataset),
    "eval": {
        name: {
            "accuracy": round(res["accuracy"], 4),
            "n_correct": res["n_correct"],
            "n_total": res["n_total"],
        }
        for name, res in eval_results.items()
    },
    "lora_config": {
        "r": 64,
        "lora_alpha": 128,
        "target_modules": [
            "q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear",
            "gate_proj.linear", "up_proj.linear", "down_proj.linear",
        ],
    },
    "training_config": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 8,
        "learning_rate": 2e-4,
        "epochs": 2,
        "fp16": False,
        "bnb_compute_dtype": "float16",
        "single_t4": True,
    },
}

summary_path = "/kaggle/working/simsat_gemma4_v6_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved: {summary_path}")
print(json.dumps(summary, indent=2))
