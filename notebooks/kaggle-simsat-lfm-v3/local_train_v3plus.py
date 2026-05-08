#!/usr/bin/env python3
"""v3+ recipe variant — overnight Phase 3.

The hypothesis after v4 + v5 negatives: maybe the data is fine, but the
recipe needs work. This variant explores a different point in
hyperparameter space against the SAME v2 dataset (165 train rows — exactly
what v3 used) to see if a recipe change alone moves the holdout.

Variant: longer schedule (8 epochs vs 5), gentler LR (1e-4 vs 2e-4),
slightly more dropout (0.10 vs 0.05) for regularization. All other
recipe knobs identical to v3.

If this beats v3 (>0.844 action), it's a candidate for v3.1. If it
matches or regresses, it's documented as a recipe-space negative
alongside v4 (data) and v5 (data).

Outputs:
  .kaggle_output_v3plus/holdout_eval_report.json
  .kaggle_output_v3plus/simsat-lfm25vl-450m-v3plus/
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "simsat-lfm-v2"  # 165 train, same as v3
TRAIN_PATH = DATA_DIR / "simsat_lfm_train.jsonl"
HOLDOUT_PATH = DATA_DIR / "simsat_lfm_holdout.jsonl"
IMAGES_DIR = DATA_DIR / "images"

OUT_DIR = ROOT / ".kaggle_output_v3plus"
OUT_DIR.mkdir(exist_ok=True)
ADAPTER_DIR = OUT_DIR / "simsat-lfm25vl-450m-v3plus"
ADAPTER_DIR.mkdir(exist_ok=True)

BASE_MODEL = "LiquidAI/LFM2.5-VL-450M"
GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=False,
    repetition_penalty=1.05,
    no_repeat_ngram_size=20,
)

# v3+ variant knobs (delta from v3 in comments)
LR = 1e-4               # was 2e-4
NUM_EPOCHS = 8          # was 5
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 4          # was 8 (T4); 4 fits 8GB; same as v5
LORA_DROPOUT = 0.10     # was 0.05
LORA_R = 16             # same as v3
LORA_ALPHA = 32         # same as v3
WARMUP_RATIO = 0.05     # was 0.03 (slightly more warmup for longer schedule)
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Common helpers (duplicated from local_train_v5 to keep file standalone)
def _load_image_in_messages(messages):
    from PIL import Image
    out = []
    for m in messages:
        new = {"role": m["role"], "content": []}
        content = m["content"]
        if isinstance(content, str):
            new["content"] = content
            out.append(new); continue
        for part in content:
            if part.get("type") == "image":
                p = part.get("image") or part.get("path")
                p = Path(p)
                if not p.is_absolute():
                    p = IMAGES_DIR / p.name
                new["content"].append({"type": "image", "image": Image.open(p).convert("RGB")})
            else:
                new["content"].append(part)
        out.append(new)
    return out


def _expected_action(rec):
    last = rec["messages"][-1]["content"]
    if isinstance(last, list):
        last = next((c.get("text", "") for c in last if c.get("type") == "text"), "")
    m = re.search(r'"recommended_action"\s*:\s*"(\w+)"', last)
    return m.group(1) if m else "?"


def _parse_pred(raw):
    s = raw.strip()
    if s.startswith("```json"): s = s[7:]
    if s.startswith("```"): s = s[3:]
    if s.endswith("```"): s = s[:-3]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except Exception: return None
        return None


def generate_assessment(model, processor, messages_no_assistant):
    msgs = _load_image_in_messages(messages_no_assistant)
    inputs = processor.apply_chat_template(
        [msgs], add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    new_tokens = out[0, prompt_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def run_eval(model, processor, holdout, label):
    print(f"\n=== {label} eval (n={len(holdout)}) ===", flush=True)
    preds = []
    t0 = time.time()
    for i, rec in enumerate(holdout):
        try:
            raw = generate_assessment(model, processor, rec["messages"][:-1])
        except Exception as e:
            raw = f"[ERROR: {type(e).__name__}: {e}]"
        parsed = _parse_pred(raw)
        preds.append({
            "trace_id": rec.get("trace_id"),
            "target_label": rec.get("target_label"),
            "expected_action": _expected_action(rec),
            "raw_output": raw,
            "predicted": parsed,
        })
        if (i + 1) % 4 == 0 or i + 1 == len(holdout):
            print(f"  [{i+1:>3}/{len(holdout)}]  elapsed={time.time()-t0:.0f}s", flush=True)
    return preds


_ACTION_TO_USEFUL = {"accept": True, "refine": True, "defer": False, "skip": False}
_BAND = {"accept": 0.85, "refine": 0.55, "defer": 0.40, "skip": 0.20}


def score(preds):
    n = len(preds) or 1
    parsed_ok = sum(1 for p in preds if isinstance(p.get("predicted"), dict))
    matches_action = matches_useful = 0
    mae = 0.0
    per_class_correct, per_class_total = {}, {}
    for p in preds:
        exp = p["expected_action"]
        per_class_total[exp] = per_class_total.get(exp, 0) + 1
        pred = p.get("predicted") or {}
        pa = pred.get("recommended_action") if isinstance(pred, dict) else None
        if pa == exp:
            matches_action += 1
            per_class_correct[exp] = per_class_correct.get(exp, 0) + 1
        if pa and exp in _ACTION_TO_USEFUL and pa in _ACTION_TO_USEFUL:
            if _ACTION_TO_USEFUL[pa] == _ACTION_TO_USEFUL[exp]:
                matches_useful += 1
        mae += abs(_BAND.get(exp, 0.5) - (_BAND.get(pa, 0.5) if pa else 0.5))
    return {
        "n": len(preds),
        "parse_rate": parsed_ok / n,
        "exact_action_agreement": matches_action / n,
        "useful_agreement": matches_useful / n,
        "score_mae": mae / n,
        "per_class_accuracy": {
            k: per_class_correct.get(k, 0) / per_class_total[k] for k in per_class_total
        },
    }


def main():
    print("=" * 60, flush=True)
    print("v3+ RECIPE VARIANT — overnight Phase 3", flush=True)
    print(f"  lr={LR}, epochs={NUM_EPOCHS}, eff_batch={GRAD_ACCUM}, dropout={LORA_DROPOUT}", flush=True)
    print("=" * 60, flush=True)

    train_records = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    holdout_records = [json.loads(l) for l in open(HOLDOUT_PATH, encoding="utf-8")]
    print(f"Train: {len(train_records)}  Holdout: {len(holdout_records)}", flush=True)

    print(f"\nLoading {BASE_MODEL} bf16 ...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0").eval()

    # ---- Base eval ----
    base_preds = run_eval(model, processor, holdout_records, "BASE")
    (OUT_DIR / "base_predictions.json").write_text(json.dumps(base_preds, indent=2), encoding="utf-8")
    base_metrics = score(base_preds)
    print(f"BASE: {json.dumps(base_metrics, indent=2)}", flush=True)

    # ---- LoRA wrap ----
    print(f"\nApplying LoRA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} ...", flush=True)
    from peft import LoraConfig, get_peft_model, TaskType
    LFM_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]
    VISION_TOWER_MODULES = ["fc1", "fc2"]
    MULTI_MODAL_PROJECTOR_MODULES = ["linear_1", "linear_2"]
    target_modules = list(set(LFM_MODULES + VISION_TOWER_MODULES + MULTI_MODAL_PROJECTOR_MODULES))
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: {trainable:,}", flush=True)

    # ---- Manual training loop ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    n_train = len(train_records)
    total_steps = (n_train * NUM_EPOCHS + GRAD_ACCUM - 1) // GRAD_ACCUM
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    print(f"total_steps={total_steps}  warmup_steps={warmup_steps}", flush=True)

    from torch.optim.lr_scheduler import LambdaLR
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    import random
    rng = random.Random(42)
    model.train()
    global_step, accum_count = 0, 0
    losses_recent = []
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        idx = list(range(n_train))
        rng.shuffle(idx)
        for sample_idx in idx:
            rec = train_records[sample_idx]
            full = _load_image_in_messages(rec["messages"])
            enc = processor.apply_chat_template(
                [full], add_generation_prompt=False, return_tensors="pt",
                return_dict=True, tokenize=True,
            ).to(model.device)
            prompt_only = _load_image_in_messages(rec["messages"][:-1])
            p_enc = processor.apply_chat_template(
                [prompt_only], add_generation_prompt=True, return_tensors="pt",
                return_dict=True, tokenize=True,
            )
            prompt_len = int(p_enc["input_ids"].shape[1])
            labels = enc["input_ids"].clone()
            labels[:, :prompt_len] = -100
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100
            try:
                out = model(**enc, labels=labels)
                loss = out.loss / GRAD_ACCUM
                loss.backward()
                losses_recent.append(float(loss.detach().item()) * GRAD_ACCUM)
                accum_count += 1
                if accum_count == GRAD_ACCUM:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accum_count = 0
                    global_step += 1
                    if global_step % 10 == 0 or global_step == total_steps:
                        avg = sum(losses_recent[-20:]) / max(1, min(20, len(losses_recent)))
                        elapsed = time.time() - t0
                        print(f"  step {global_step:>4}/{total_steps}  ep {epoch+1}/{NUM_EPOCHS}  loss={avg:.3f}  lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM at step {global_step+1}: {e}", flush=True)
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_count = 0
                continue

        # Per-epoch checkpoint
        ckpt = OUT_DIR / f"checkpoint-epoch-{epoch+1}"
        model.save_pretrained(str(ckpt))
        print(f"  ✓ checkpoint-epoch-{epoch+1}", flush=True)

    print(f"\nTraining complete in {(time.time()-t0)/60:.1f} min", flush=True)
    model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))

    # ---- Tuned eval ----
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    tuned_preds = run_eval(model, processor, holdout_records, "TUNED v3+")
    (OUT_DIR / "tuned_predictions.json").write_text(json.dumps(tuned_preds, indent=2), encoding="utf-8")
    tuned_metrics = score(tuned_preds)

    report = {
        "config": {
            "base_model": BASE_MODEL,
            "dataset": "simsat-lfm-v2 (165 train) — same as v3",
            "variant": "v3+",
            "lr": LR, "epochs": NUM_EPOCHS, "effective_batch": GRAD_ACCUM,
            "lora_dropout": LORA_DROPOUT,
        },
        "base": base_metrics,
        "tuned": tuned_metrics,
    }
    (OUT_DIR / "holdout_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== SUMMARY (v3+) ===", flush=True)
    print(f"  parse_rate              base={base_metrics['parse_rate']:.3f}  tuned={tuned_metrics['parse_rate']:.3f}", flush=True)
    print(f"  exact_action_agreement  base={base_metrics['exact_action_agreement']:.3f}  tuned={tuned_metrics['exact_action_agreement']:.3f}", flush=True)
    print(f"  useful_agreement        base={base_metrics['useful_agreement']:.3f}  tuned={tuned_metrics['useful_agreement']:.3f}", flush=True)
    print(f"  score_mae               base={base_metrics['score_mae']:.3f}  tuned={tuned_metrics['score_mae']:.3f}", flush=True)
    print(f"\n  vs v3 canonical (0.844 action / 0.055 mae): action delta = {tuned_metrics['exact_action_agreement']-0.844:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
