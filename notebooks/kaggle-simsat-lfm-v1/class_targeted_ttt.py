#!/usr/bin/env python3
"""Class-targeted TTT — does runtime adaptation LIFT a specific weak class?

The 30-step (v1) and 50-step (v2) extended TTT receipts both reached a
steady state where TTT preserves stability but doesn't lift performance —
because the stream is class-mixed and the gradients average out.

This experiment isolates the question: **if the stream is class-targeted
(only skip examples), does the gradient signal lift the model's skip-class
accuracy?**

If YES: direct empirical evidence that runtime TTT under the right stream
        curation lifts performance (not just preserves it).
If NO:  documented evidence that even skip-targeted streams don't lift the
        skip class on this architecture — would point to a deeper mismatch
        between teacher-forced loss and the action-classification objective.

Either way it's evidence; either way it's published.

Setup:
  - Load v3 adapter
  - Take 8 SKIP-class rows from holdout as the probe (same probe pre/post)
  - Take 16 SKIP-class rows from train as the TTT stream
  - Pre-eval probe → run 16 TTT steps → post-eval probe
  - Compare per-class skip accuracy and overall mae
"""
from __future__ import annotations

import gc
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
sys.path.insert(0, str(ROOT / "src" / "sim"))
from observation_vla.lfm_ttt import OnlineLoRAStepper

TRAIN_PATH = ROOT / "datasets" / "simsat-lfm-v1" / "simsat_lfm_train.jsonl"
HOLDOUT_PATH = ROOT / "datasets" / "simsat-lfm-v1" / "simsat_lfm_holdout.jsonl"
IMAGES_DIR = ROOT / "datasets" / "simsat-lfm-v1" / "images"
ADAPTER_DIR = ROOT / ".kaggle_output_v3" / "simsat-lfm25vl-450m-v3-adapter"
OUT_DIR = ROOT / ".kaggle_output"
RECEIPT_PATH = OUT_DIR / "class_targeted_ttt_receipt.json"

BASE_MODEL = "LiquidAI/LFM2.5-VL-450M"
TARGET_CLASS = "skip"
N_STREAM = 16  # how many target-class examples from train to use as TTT stream
LR = 1e-5

GEN_KWARGS = dict(
    max_new_tokens=256, do_sample=False,
    repetition_penalty=1.05, no_repeat_ngram_size=20,
)


def _load_image_in_messages(messages):
    from PIL import Image
    out = []
    for m in messages:
        new = {"role": m["role"], "content": []}
        content = m["content"]
        if isinstance(content, str):
            new["content"] = content; out.append(new); continue
        for part in content:
            if part.get("type") == "image":
                p = Path(part.get("image") or part.get("path"))
                if not p.is_absolute(): p = IMAGES_DIR / p.name
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


def _target_text(rec):
    last = rec["messages"][-1]["content"]
    if isinstance(last, list):
        return next((c.get("text", "") for c in last if c.get("type") == "text"), "")
    return last


def _generate_action(model, processor, messages_no_assistant):
    msgs = _load_image_in_messages(messages_no_assistant)
    inputs = processor.apply_chat_template(
        [msgs], add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    new_tokens = out[0, prompt_len:]
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    m = re.search(r'"recommended_action"\s*:\s*"(\w+)"', raw)
    return m.group(1) if m else None


_BAND = {"accept": 0.85, "refine": 0.55, "defer": 0.40, "skip": 0.20}


def measure_probe(model, processor, probe, label):
    model.eval()
    print(f"\n[{label}]", flush=True)
    matches = 0
    mae = 0.0
    pred_dist = {}
    for rec in probe:
        exp = _expected_action(rec)
        try: pa = _generate_action(model, processor, rec["messages"][:-1])
        except Exception: pa = None
        pred_dist[pa or "none"] = pred_dist.get(pa or "none", 0) + 1
        if pa == exp: matches += 1
        mae += abs(_BAND.get(exp, 0.5) - (_BAND.get(pa, 0.5) if pa else 0.5))
    n = len(probe) or 1
    metrics = {
        "exact_action_agreement": matches / n,
        "score_mae": mae / n,
        "prediction_distribution": pred_dist,
    }
    print(f"  action_agreement: {metrics['exact_action_agreement']:.3f}  ({matches}/{n})", flush=True)
    print(f"  score_mae: {metrics['score_mae']:.3f}", flush=True)
    print(f"  predictions: {pred_dist}", flush=True)
    model.train()
    return metrics


def _build_forward_loss_fn(processor):
    def forward_loss(model, _proc, messages, target_text):
        full = _load_image_in_messages(messages) + [
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        enc = processor.apply_chat_template([full], add_generation_prompt=False, return_tensors="pt", return_dict=True, tokenize=True).to(model.device)
        prompt_only = _load_image_in_messages(messages)
        p_enc = processor.apply_chat_template([prompt_only], add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True)
        prompt_len = int(p_enc["input_ids"].shape[1])
        labels = enc["input_ids"].clone()
        labels[:, :prompt_len] = -100
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        out = model(**enc, labels=labels)
        with torch.no_grad():
            argmax = out.logits.argmax(dim=-1)
            asst_ids = argmax[0, prompt_len - 1: prompt_len + 32]
            try: asst_text = processor.decode(asst_ids.cpu(), skip_special_tokens=True)
            except Exception: asst_text = ""
            m = re.search(r'"recommended_action"\s*:\s*"(\w+)"', asst_text)
            pa = m.group(1) if m else None
        return out.loss, pa
    return forward_loss


def main():
    print("=" * 60, flush=True)
    print(f"CLASS-TARGETED TTT — target class: {TARGET_CLASS!r}", flush=True)
    print("=" * 60, flush=True)

    print(f"\nLoading {BASE_MODEL} + v3 adapter ...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0").eval()

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), is_trainable=True)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n

    train_recs = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    holdout_recs = [json.loads(l) for l in open(HOLDOUT_PATH, encoding="utf-8")]

    holdout_target = [r for r in holdout_recs if _expected_action(r) == TARGET_CLASS]
    train_target = [r for r in train_recs if _expected_action(r) == TARGET_CLASS]
    print(f"\nTarget class examples — train: {len(train_target)}, holdout: {len(holdout_target)}", flush=True)

    probe = holdout_target[:8]
    stream = train_target[:N_STREAM]
    print(f"Probe: {len(probe)} ({TARGET_CLASS!r}-only holdout)", flush=True)
    print(f"Stream: {len(stream)} ({TARGET_CLASS!r}-only train)", flush=True)

    pre_metrics = measure_probe(model, processor, probe, "PRE-TTT probe")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    forward_loss = _build_forward_loss_fn(processor)
    stepper = OnlineLoRAStepper(model, processor, optimizer, forward_loss_fn=forward_loss)

    history = []
    print(f"\n--- TTT stream ({len(stream)} steps) ---", flush=True)
    for i, rec in enumerate(stream):
        op_action = _expected_action(rec)
        try:
            t0 = time.time()
            model.train()
            res = stepper.online_step(messages=rec["messages"][:-1], target_text=_target_text(rec))
            dur = time.time() - t0
            history.append({
                "step": i + 1, "trace_id": rec.get("trace_id"),
                "operator_action": op_action, "did_step": res.did_step,
                "loss": res.loss, "lora_delta_l2": res.lora_delta_l2,
                "duration_s": round(dur, 1),
            })
            tag = "STEP" if res.did_step else f"BLOCKED({res.blocked_by})"
            print(f"  [{i+1:>2}/{len(stream)}]  {tag}  loss={res.loss:.3f}  d_l2={res.lora_delta_l2:.4f}  ({dur:.0f}s)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i+1:>2}/{len(stream)}]  OOM", flush=True)
            torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"  [{i+1:>2}/{len(stream)}]  ERROR: {e}", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    post_metrics = measure_probe(model, processor, probe, "POST-TTT probe")

    delta_action = post_metrics["exact_action_agreement"] - pre_metrics["exact_action_agreement"]
    delta_mae = post_metrics["score_mae"] - pre_metrics["score_mae"]

    receipt = {
        "config": {
            "base_model": BASE_MODEL, "adapter": str(ADAPTER_DIR),
            "target_class": TARGET_CLASS, "n_stream": N_STREAM,
            "n_probe": len(probe), "lr": LR, "rep_penalty": GEN_KWARGS["repetition_penalty"],
        },
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "delta": {"action_agreement": delta_action, "score_mae": delta_mae},
        "history": history,
    }
    OUT_DIR.mkdir(exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"CLASS-TARGETED TTT RECEIPT — target {TARGET_CLASS!r}", flush=True)
    print("=" * 60, flush=True)
    print(f"  action_agreement   pre={pre_metrics['exact_action_agreement']:.3f}  post={post_metrics['exact_action_agreement']:.3f}  delta={delta_action:+.3f}", flush=True)
    print(f"  score_mae          pre={pre_metrics['score_mae']:.3f}  post={post_metrics['score_mae']:.3f}  delta={delta_mae:+.3f}", flush=True)
    print(f"\nReceipt: {RECEIPT_PATH}", flush=True)

    if delta_action > 0.05:
        print("\n*** RESULT: TTT LIFTS the target class. Architectural claim is empirically validated.", flush=True)
    elif delta_action < -0.05:
        print("\n*** RESULT: TTT REGRESSES the target class. Documented as honest negative.", flush=True)
    else:
        print("\n*** RESULT: TTT is FLAT on the target class. Stable but not lifting — same pattern as v2 50-step run.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
