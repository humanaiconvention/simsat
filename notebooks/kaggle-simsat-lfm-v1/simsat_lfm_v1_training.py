#!/usr/bin/env python3
"""
SimSat LFM2.5-VL-450M LoRA Fine-tune + Base-vs-Tuned Eval — Kaggle T4
=====================================================================
Trains a LoRA adapter on LiquidAI/LFM2.5-VL-450M for SimSat encounter
assessment, then runs side-by-side eval (base vs tuned) on a stratified
32-row hold-out (8 per class). Produces the "measurable improvement over
the base model" number the Liquid Track rubric explicitly rewards.

Authoritative LoRA target_modules from `Liquid4All/leap-finetune`
(`src/leap_finetune/training_configs/peft_configs.py:DEFAULT_VLM_LORA`):

    LFM_MODULES                   = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]
    VISION_TOWER_MODULES          = ["fc1", "fc2"]
    MULTI_MODAL_PROJECTOR_MODULES = ["linear_1", "linear_2"]

    r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"

Inputs (Kaggle-attached):
  - Dataset: benhaslam/simsat-lfm-v1 mounted at /kaggle/input/simsat-lfm-v1/
        simsat_lfm_train.jsonl     (109 rows, balanced 4-class)
        simsat_lfm_holdout.jsonl   (32 rows, 8 per class — TRUE held-out)
        simsat_lfm_eval.jsonl      (4 legacy rows — defer/refine only)
        images/*.png

The model is downloaded from HF at runtime (~900 MB in bf16); we have
`enable_internet=True` in kernel-metadata.json.

Expected runtime on Kaggle T4: ~10 min base eval + 30-60 min training +
~10 min tuned eval = ~50-80 min total.
"""

# ============================================================
# CELL 0: Environment + dependency install
# ============================================================
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch as _torch
_cap = _torch.cuda.get_device_capability(0)
if _cap < (7, 0):
    raise RuntimeError(
        f"Wrong GPU: got sm_{_cap[0]}{_cap[1]} ({_torch.cuda.get_device_name(0)}). "
        "Minimum required is sm_70 (T4)."
    )
print(f"GPU: {_torch.cuda.get_device_name(0)} (sm_{_cap[0]}{_cap[1]})")

# Purge warm-kernel module state
for _m in list(sys.modules):
    if _m.startswith(("trl", "peft", "transformers", "accelerate", "datasets", "PIL", "torchvision")):
        del sys.modules[_m]

# Step 1 — Pin Pillow to an internally-consistent build BEFORE anything
# else is upgraded. The default Kaggle image was hitting an
# `ImportError: cannot import name '_Ink' from 'PIL._typing'` on Cell 1
# imports because the installed Pillow had `ImageText.py` from a newer
# release than `_typing.py`. --force-reinstall + --no-deps ensures all
# PIL/*.py files come from the same release.
get_ipython().system(  # noqa: F821
    "pip install -q --force-reinstall --no-deps 'pillow==11.3.0' 2>&1 | tail -3"
)

# Step 2 — Install the training stack. transformers <5 keeps us off the
# 5.0.0.dev wheel that the LFM config.json was built against; trust_remote_code
# loads the LFM modeling code from the model repo, so 4.46+ is sufficient.
get_ipython().system(  # noqa: F821
    "pip install -q -U "
    "'transformers>=4.46.0,<5.0.0' "
    "'trl>=0.12.0' "
    "'peft>=0.13.0' "
    "'accelerate>=1.0.0' "
    "'datasets>=3.0.0' "
    "'huggingface_hub>=0.26.0' "
    "2>&1 | tail -5"
)
print("Deps installed.")

# ============================================================
# CELL 1: Load processor + base model + dataset
# ============================================================
import copy
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
DTYPE = torch.bfloat16
DATA_DIR = Path("/kaggle/input/simsat-lfm-v1")
TRAIN_JSONL = DATA_DIR / "simsat_lfm_train.jsonl"
HOLDOUT_JSONL = DATA_DIR / "simsat_lfm_holdout.jsonl"
EVAL_JSONL = DATA_DIR / "simsat_lfm_eval.jsonl"

assert TRAIN_JSONL.exists(), f"Missing {TRAIN_JSONL}"
assert HOLDOUT_JSONL.exists(), f"Missing {HOLDOUT_JSONL}"

print(f"Loading {MODEL_ID} in bfloat16 on cuda:0 ...")
# `max_image_tokens` is a config-sourced kwarg supported by LFM's processor.
# Some transformers versions reject unknown kwargs in from_pretrained — fall
# back to a plain load if the kwarg variant errors.
try:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, max_image_tokens=256, trust_remote_code=True,
    )
except TypeError:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# `dtype` is the modern transformers kwarg (matches LFM's official model card
# example). Older transformers used `torch_dtype` — fall back if needed.
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map="cuda:0",
        trust_remote_code=True,
    )
except TypeError:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="cuda:0",
        trust_remote_code=True,
    )
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")


def _resolve_image_paths(rec: dict) -> dict:
    for msg in rec["messages"]:
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("type") == "image" and isinstance(item.get("image"), str):
                    item["image"] = str(DATA_DIR / item["image"])
    return rec


def load_messages_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(_resolve_image_paths(json.loads(line)))
    return out


def _load_image_in_messages(messages: list[dict]) -> list[dict]:
    out = copy.deepcopy(messages)
    for msg in out:
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("type") == "image" and isinstance(item.get("image"), str):
                    item["image"] = Image.open(item["image"]).convert("RGB")
    return out


train_records = load_messages_jsonl(TRAIN_JSONL)
holdout_records = load_messages_jsonl(HOLDOUT_JSONL)
eval_records = load_messages_jsonl(EVAL_JSONL) if EVAL_JSONL.exists() else []
print(f"Train: {len(train_records)}  Holdout: {len(holdout_records)}  Legacy-eval: {len(eval_records)}")

# Verify a sample image actually loads
sample_img_path = train_records[0]["messages"][1]["content"][0]["image"]
with Image.open(sample_img_path) as _img:
    print(f"Sample image: {sample_img_path}, size={_img.size}, mode={_img.mode}")


# ============================================================
# CELL 2: Base-model hold-out inference (BEFORE LoRA)
# ============================================================
# Capture base predictions on the 32-row stratified hold-out so we can compare
# them against tuned predictions later. Generation only — no parameter updates.

def generate_assessment(model, processor, messages_no_assistant: list[dict],
                        max_new_tokens: int = 256) -> str:
    """Run apply_chat_template + model.generate on a single conversation."""
    msgs = _load_image_in_messages(messages_no_assistant)
    inputs = processor.apply_chat_template(
        [msgs],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Decode only the newly-generated tokens
    new_tokens = out[0, prompt_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


print("Running BASE-model hold-out inference (32 samples)...")
base_predictions = []
for i, rec in enumerate(holdout_records):
    text = generate_assessment(model, processor, rec["messages"][:2])  # system + user only
    base_predictions.append({
        "trace_id": rec.get("trace_id"),
        "scenario_pack": rec.get("scenario_pack"),
        "target_label": rec.get("target_label"),
        "expected": rec["messages"][2]["content"][0]["text"],
        "predicted": text,
    })
    if (i + 1) % 8 == 0:
        print(f"  {i+1}/{len(holdout_records)} done")

# Persist immediately so a downstream training failure doesn't lose this work
Path("/kaggle/working").mkdir(exist_ok=True)
with open("/kaggle/working/base_predictions.json", "w", encoding="utf-8") as f:
    json.dump(base_predictions, f, indent=2)
print(f"Base predictions saved -> /kaggle/working/base_predictions.json")
print(f"Sample base prediction: {base_predictions[0]['predicted'][:200]}")


# ============================================================
# CELL 3: Apply LoRA (authoritative leap-finetune config)
# ============================================================
from peft import LoraConfig, get_peft_model, TaskType

LFM_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]
VISION_TOWER_MODULES = ["fc1", "fc2"]
MULTI_MODAL_PROJECTOR_MODULES = ["linear_1", "linear_2"]
TARGET_MODULES = LFM_MODULES + VISION_TOWER_MODULES + MULTI_MODAL_PROJECTOR_MODULES

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=TARGET_MODULES,
)

model = get_peft_model(model, peft_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# Pre-train target match audit — fail fast if the LoRA targets don't bind.
print("\nPre-train target_modules match audit:")
counts: dict[str, int] = {n: 0 for n in TARGET_MODULES}
for name, _module in model.named_modules():
    leaf = name.rsplit(".", 1)[-1]
    if leaf in counts:
        counts[leaf] += 1
for k, v in counts.items():
    print(f"  {k:14s} matched {v} modules")
total_matched = sum(counts.values())
if total_matched == 0:
    raise RuntimeError("No target modules matched — LoRA would train zero parameters.")
print(f"  TOTAL matched: {total_matched}")


# ============================================================
# CELL 4: Custom collator + smoke test
# ============================================================
def collate_fn(batch: list[dict]) -> dict:
    expanded = [_load_image_in_messages(rec["messages"]) for rec in batch]
    enc = processor.apply_chat_template(
        expanded,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
        padding=True,
    )
    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    image_token_id = getattr(model.config, "image_token_id", 396)
    labels[labels == image_token_id] = -100
    enc["labels"] = labels
    return enc


print("\nCollator smoke test on 1 sample:")
_b = collate_fn([train_records[0]])
print(f"  input_ids:    {tuple(_b['input_ids'].shape)}")
print(f"  labels:       {tuple(_b['labels'].shape)}")
if "pixel_values" in _b:
    print(f"  pixel_values: {tuple(_b['pixel_values'].shape)}")
print(f"  loss-token ratio: {(_b['labels'] != -100).float().mean().item():.3f}")


# ============================================================
# CELL 5: Configure trainer
# ============================================================
from trl import SFTTrainer, SFTConfig

OUTPUT_DIR = "/kaggle/working/simsat-lfm25vl-450m-v1"

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    max_length=2048,
    report_to="none",
    # NB: do NOT set dataset_text_field — when skip_prepare_dataset=True
    # is set, TRL hands the dataset to the data_collator unchanged and any
    # `dataset_text_field` value can trigger column-validation warnings/errors
    # that don't apply to multimodal records. Liquid AI's reference TRL
    # config does not set this field either.
)


class _ListDataset:
    def __init__(self, records: list[dict]):
        self._records = records

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]


trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=_ListDataset(train_records),
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)


# ============================================================
# CELL 6: Train
# ============================================================
print(f"Training for {sft_config.num_train_epochs} epochs on {len(train_records)} samples...")
print(f"Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
trainer.train()
print("Training complete.")


# ============================================================
# CELL 7: Save adapter + post-save sanity gate
# ============================================================
ADAPTER_DIR = OUTPUT_DIR + "-adapter"
trainer.model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved to: {ADAPTER_DIR}")

import safetensors.torch as st

adapter_path = Path(ADAPTER_DIR) / "adapter_model.safetensors"
assert adapter_path.exists(), f"Expected adapter file not found: {adapter_path}"
state = st.load_file(str(adapter_path))

n_total = len(state)
n_lora_b = sum(1 for k in state if "lora_B" in k)
n_lora_b_nonzero = sum(1 for k, t in state.items() if "lora_B" in k and t.abs().sum().item() > 0)
print("\nAdapter sanity check:")
print(f"  Total LoRA tensors:     {n_total}")
print(f"  lora_B tensors:         {n_lora_b}")
print(f"  lora_B tensors non-zero: {n_lora_b_nonzero}")

assert n_total > 0, "Adapter file contains zero tensors."
assert n_lora_b_nonzero > 0, "All lora_B tensors are zero — model trained nothing."

coverage = {"text_lm": False, "vision_tower": False, "projector": False}
for k in state:
    if any(m in k for m in LFM_MODULES) and "language_model" in k:
        coverage["text_lm"] = True
    if any(m in k for m in VISION_TOWER_MODULES):
        coverage["vision_tower"] = True
    if any(m in k for m in MULTI_MODAL_PROJECTOR_MODULES):
        coverage["projector"] = True
print(f"  Coverage: {coverage}")
missing = [k for k, v in coverage.items() if not v]
if missing:
    print(f"  WARNING: no LoRA weights for {missing}.")


# ============================================================
# CELL 8: Tuned-model hold-out inference (AFTER LoRA training)
# ============================================================
print("Running TUNED-model hold-out inference (32 samples)...")
trainer.model.eval()
tuned_predictions = []
for i, rec in enumerate(holdout_records):
    text = generate_assessment(trainer.model, processor, rec["messages"][:2])
    tuned_predictions.append({
        "trace_id": rec.get("trace_id"),
        "scenario_pack": rec.get("scenario_pack"),
        "target_label": rec.get("target_label"),
        "expected": rec["messages"][2]["content"][0]["text"],
        "predicted": text,
    })
    if (i + 1) % 8 == 0:
        print(f"  {i+1}/{len(holdout_records)} done")

with open("/kaggle/working/tuned_predictions.json", "w", encoding="utf-8") as f:
    json.dump(tuned_predictions, f, indent=2)
print("Tuned predictions saved -> /kaggle/working/tuned_predictions.json")
print(f"Sample tuned prediction: {tuned_predictions[0]['predicted'][:200]}")


# ============================================================
# CELL 9: Compute metrics + side-by-side report
# ============================================================
import re


def _extract_json(text: str) -> dict | None:
    """Robust JSON extractor — strips code fences, tries first {...} block."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    # First {...} block
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
    return None


def _parse_action(payload: dict | None) -> str | None:
    if payload is None:
        return None
    a = payload.get("recommended_action")
    if isinstance(a, str) and a.lower() in {"accept", "defer", "refine", "skip"}:
        return a.lower()
    return None


def _parse_useful(payload: dict | None) -> bool | None:
    if payload is None:
        return None
    # Approximation — the schema doesn't have a 'useful' field but we can
    # derive from confidence + recommended_action: accept ⇒ useful, skip ⇒ not.
    a = _parse_action(payload)
    if a == "accept" or a == "refine":
        return True
    if a == "skip":
        return False
    return None


def _parse_score(payload: dict | None) -> float | None:
    if payload is None:
        return None
    # Map recommended_action to a usefulness band the operator labels use.
    # accept -> ~0.85, refine -> ~0.55, defer -> ~0.40, skip -> ~0.20.
    band = {"accept": 0.85, "refine": 0.55, "defer": 0.40, "skip": 0.20}
    a = _parse_action(payload)
    if a in band:
        return band[a]
    # Fall back to confidence if available
    c = payload.get("confidence")
    if isinstance(c, (int, float)):
        return float(c)
    return None


def score_predictions(preds: list[dict]) -> dict:
    n = len(preds)
    parsed = 0
    correct_action = 0
    correct_useful = 0
    abs_errors: list[float] = []
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    for p in preds:
        try:
            expected = json.loads(p["expected"])
        except Exception:
            continue
        exp_action = expected.get("recommended_action", "").lower()
        exp_useful = exp_action in {"accept", "refine"}
        exp_score = expected.get("usefulness_score") or {
            "accept": 0.85, "refine": 0.55, "defer": 0.40, "skip": 0.20
        }.get(exp_action, 0.5)

        pred_payload = _extract_json(p["predicted"])
        if pred_payload is not None:
            parsed += 1

        pred_action = _parse_action(pred_payload)
        pred_useful = _parse_useful(pred_payload)
        pred_score = _parse_score(pred_payload)

        per_class_total[exp_action] = per_class_total.get(exp_action, 0) + 1
        if pred_action == exp_action:
            correct_action += 1
            per_class_correct[exp_action] = per_class_correct.get(exp_action, 0) + 1
        if pred_useful is not None and pred_useful == exp_useful:
            correct_useful += 1
        if pred_score is not None:
            abs_errors.append(abs(pred_score - exp_score))

    return {
        "n": n,
        "parsed": parsed,
        "parse_rate": parsed / n if n else 0.0,
        "exact_action_agreement": correct_action / n if n else 0.0,
        "useful_agreement": correct_useful / n if n else 0.0,
        "score_mae": (sum(abs_errors) / len(abs_errors)) if abs_errors else float("nan"),
        "per_class_accuracy": {
            cls: per_class_correct.get(cls, 0) / per_class_total.get(cls, 1)
            for cls in sorted(per_class_total)
        },
    }


base_metrics = score_predictions(base_predictions)
tuned_metrics = score_predictions(tuned_predictions)

print("\n" + "=" * 60)
print("BASE vs TUNED — 32-row stratified hold-out")
print("=" * 60)
print(f"{'Metric':<28} {'Base':>10} {'Tuned':>10} {'Delta':>10}")
print("-" * 60)
for k in ("parse_rate", "exact_action_agreement", "useful_agreement", "score_mae"):
    b = base_metrics[k]
    t = tuned_metrics[k]
    d = t - b
    print(f"{k:<28} {b:>10.3f} {t:>10.3f} {d:>+10.3f}")
print("\nPer-class accuracy:")
print(f"{'Class':<10} {'Base':>10} {'Tuned':>10} {'Delta':>10}")
print("-" * 42)
for cls in ("accept", "refine", "defer", "skip"):
    b = base_metrics["per_class_accuracy"].get(cls, 0.0)
    t = tuned_metrics["per_class_accuracy"].get(cls, 0.0)
    print(f"{cls:<10} {b:>10.3f} {t:>10.3f} {(t-b):>+10.3f}")

# Persist the eval report so it can be cited in the docs
with open("/kaggle/working/holdout_eval_report.json", "w", encoding="utf-8") as f:
    json.dump({"base": base_metrics, "tuned": tuned_metrics}, f, indent=2)
print("\nReport saved -> /kaggle/working/holdout_eval_report.json")


# ============================================================
# CELL 10: Smoke-test inference on legacy 4-row eval set
# ============================================================
if eval_records:
    print("\nLegacy 4-row eval (defer/refine only) sample:")
    sample = eval_records[0]
    text = generate_assessment(trainer.model, processor, sample["messages"][:2])
    print(f"Target: {sample.get('target_label')}")
    print(f"Tuned model output: {text[:400]}")
    print(f"Operator-reviewed: {sample['messages'][2]['content'][0]['text']}")


# ============================================================
# CELL 11: Optional — push adapter to HuggingFace Hub
# ============================================================
HF_REPO_ID = "HumanAIConvention/simsat-lfm25vl-450m-v1"

if False:  # set True after you've reviewed Cell 9 metrics
    from huggingface_hub import HfApi, login

    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("KAGGLE_USERNAME_HF_TOKEN")
    assert HF_TOKEN, "Set HF_TOKEN as a Kaggle secret to push."
    login(token=HF_TOKEN)
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=ADAPTER_DIR,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="SimSat LFM2.5-VL-450M v1 LoRA adapter — initial release",
    )
    # Also upload the eval report so the model card has the evidence inline
    api.upload_file(
        path_or_fileobj="/kaggle/working/holdout_eval_report.json",
        path_in_repo="holdout_eval_report.json",
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print(f"Pushed adapter to https://huggingface.co/{HF_REPO_ID}")
else:
    print(f"\nHF upload skipped. Edit Cell 11 (`if False:` -> `if True:`) once metrics look good.")
    print(f"Local adapter dir: {ADAPTER_DIR}")
