#!/usr/bin/env python3
"""Extended TTT v2 — 50 steps, 16-row probe (4 per class).

Strengthens the 30-step receipt's caveat that the probe (8 rows) is too
small to distinguish noise from signal. With 16 rows, a 1-sample shift
moves the probe metric by 0.0625 instead of 0.125.

Same v3 adapter, same recipe, same memory hygiene as extended_ttt_run.py.
This is purely a probe-resolution upgrade.
"""
from __future__ import annotations

import gc
import json
import os
import random
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
OUT_DIR.mkdir(exist_ok=True)
RECEIPT_PATH = OUT_DIR / "extended_ttt_v2_receipt.json"

BASE_MODEL = "LiquidAI/LFM2.5-VL-450M"
N_TTT_STEPS = 50
PROBE_INTERVAL = 10
N_PROBE = 16  # 4 per class
LR = 1e-5
DOWNSTREAM_AGREE_RATE = 0.90

GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=False,
    repetition_penalty=1.05,
    no_repeat_ngram_size=20,
)


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


def measure_probe(model, processor, probe):
    model.eval()
    matches = parses = 0
    mae = 0.0
    pcc, pct = {}, {}
    for rec in probe:
        exp = _expected_action(rec)
        pct[exp] = pct.get(exp, 0) + 1
        try: pa = _generate_action(model, processor, rec["messages"][:-1])
        except Exception: pa = None
        if pa is not None: parses += 1
        if pa == exp:
            matches += 1
            pcc[exp] = pcc.get(exp, 0) + 1
        mae += abs(_BAND.get(exp, 0.5) - (_BAND.get(pa, 0.5) if pa else 0.5))
    n = len(probe) or 1
    model.train()
    return {
        "n": len(probe),
        "exact_action_agreement": matches / n,
        "parse_rate": parses / n,
        "score_mae": mae / n,
        "per_class_accuracy": {k: pcc.get(k, 0) / pct[k] for k in pct},
    }


def _build_forward_loss_fn(processor):
    def forward_loss(model, _processor, messages, target_text):
        full = _load_image_in_messages(messages) + [
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        enc = processor.apply_chat_template(
            [full], add_generation_prompt=False, return_tensors="pt",
            return_dict=True, tokenize=True,
        ).to(model.device)
        prompt_only = _load_image_in_messages(messages)
        p_enc = processor.apply_chat_template(
            [prompt_only], add_generation_prompt=True, return_tensors="pt",
            return_dict=True, tokenize=True,
        )
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
    print(f"EXTENDED TTT v2 — {N_TTT_STEPS} steps, {N_PROBE}-row probe (4 per class)", flush=True)
    print("=" * 60, flush=True)

    rng = random.Random(42)
    rng_outcome = random.Random(7)

    print(f"\nLoading {BASE_MODEL} + v3 adapter ...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0").eval()

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), is_trainable=True)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable LoRA params: {trainable:,}", flush=True)

    train_recs = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    holdout_recs = [json.loads(l) for l in open(HOLDOUT_PATH, encoding="utf-8")]
    rng.shuffle(train_recs)
    ttt_stream = train_recs[:N_TTT_STEPS]

    by_class = {}
    for r in holdout_recs:
        by_class.setdefault(_expected_action(r), []).append(r)
    probe = []
    per_class = max(1, N_PROBE // max(1, len(by_class)))
    for cls in sorted(by_class.keys()):
        probe.extend(by_class[cls][:per_class])
    probe = probe[:N_PROBE]
    print(f"Probe: {len(probe)} rows ({per_class} per class)", flush=True)
    print(f"TTT stream: {len(ttt_stream)} steps", flush=True)

    print("\n[probe @ step 0]", flush=True)
    probe_history = []
    metrics0 = measure_probe(model, processor, probe)
    metrics0["step"] = 0
    probe_history.append(metrics0)
    print(f"  action={metrics0['exact_action_agreement']:.3f}  mae={metrics0['score_mae']:.3f}  parse={metrics0['parse_rate']:.3f}", flush=True)
    print(f"  per-class: {metrics0['per_class_accuracy']}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.0,
    )
    forward_loss = _build_forward_loss_fn(processor)
    stepper = OnlineLoRAStepper(model, processor, optimizer, forward_loss_fn=forward_loss)

    history = []
    print(f"\n--- Running TTT stream ---", flush=True)
    for i, rec in enumerate(ttt_stream):
        op_action = _expected_action(rec)
        msgs = rec["messages"][:-1]
        target_text = _target_text(rec)

        agrees = rng_outcome.random() < DOWNSTREAM_AGREE_RATE
        if not agrees:
            history.append({
                "step": i + 1, "operator_action": op_action,
                "did_step": False, "blocked_by": "downstream_outcome_disagreement",
            })
            print(f"  [{i+1:>3}/{N_TTT_STEPS}]  SKIP (downstream)", flush=True)
            continue

        try:
            t0 = time.time()
            model.train()
            res = stepper.online_step(messages=msgs, target_text=target_text)
            dur = time.time() - t0
            history.append({
                "step": i + 1, "trace_id": rec.get("trace_id"),
                "operator_action": op_action, "did_step": res.did_step,
                "blocked_by": res.blocked_by, "loss": res.loss,
                "lora_delta_l2": res.lora_delta_l2, "duration_s": round(dur, 1),
            })
            tag = "STEP" if res.did_step else f"BLOCKED({res.blocked_by})"
            print(f"  [{i+1:>3}/{N_TTT_STEPS}]  {tag:<22} loss={res.loss:.3f}  d_l2={res.lora_delta_l2:.4f}  ({dur:.0f}s)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i+1:>3}/{N_TTT_STEPS}]  OOM — empty_cache", flush=True)
            history.append({"step": i + 1, "blocked_by": "cuda_oom"})
            torch.cuda.empty_cache()
            gc.collect()
            optimizer.zero_grad()
        except Exception as e:
            print(f"  [{i+1:>3}/{N_TTT_STEPS}]  ERROR: {type(e).__name__}: {e}", flush=True)
            history.append({"step": i + 1, "error": f"{type(e).__name__}"})

        torch.cuda.empty_cache()
        gc.collect()

        if (i + 1) % PROBE_INTERVAL == 0:
            print(f"\n[probe @ step {i+1}]", flush=True)
            metrics = measure_probe(model, processor, probe)
            metrics["step"] = i + 1
            probe_history.append(metrics)
            print(f"  action={metrics['exact_action_agreement']:.3f}  mae={metrics['score_mae']:.3f}  parse={metrics['parse_rate']:.3f}", flush=True)
            print(f"  per-class: {metrics['per_class_accuracy']}", flush=True)
            partial = {
                "config": {
                    "base_model": BASE_MODEL, "adapter": str(ADAPTER_DIR),
                    "n_ttt_steps_planned": N_TTT_STEPS, "lr": LR,
                    "downstream_agree_rate": DOWNSTREAM_AGREE_RATE,
                    "probe_interval": PROBE_INTERVAL, "n_probe": N_PROBE,
                    "rep_penalty": GEN_KWARGS["repetition_penalty"],
                },
                "history": history, "probe_metrics_history": probe_history,
                "completed_steps": i + 1,
            }
            RECEIPT_PATH.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    n_attempted = sum(1 for h in history if "did_step" in h)
    n_did = sum(1 for h in history if h.get("did_step"))
    n_blocked = sum(1 for h in history if h.get("did_step") is False)
    n_oom = sum(1 for h in history if h.get("blocked_by") == "cuda_oom")
    blocked_reasons = {}
    for h in history:
        if h.get("did_step") is False:
            r = h.get("blocked_by") or "unknown"
            blocked_reasons[r] = blocked_reasons.get(r, 0) + 1

    receipt = {
        "config": {
            "base_model": BASE_MODEL, "adapter": str(ADAPTER_DIR),
            "n_ttt_steps_planned": N_TTT_STEPS, "n_probe": N_PROBE,
            "probe_interval": PROBE_INTERVAL, "lr": LR,
            "downstream_agree_rate": DOWNSTREAM_AGREE_RATE, "seed": 42,
            "rep_penalty": GEN_KWARGS["repetition_penalty"],
        },
        "stream_summary": {
            "attempted": n_attempted, "applied": n_did,
            "blocked": n_blocked, "oom": n_oom,
            "blocked_reasons": blocked_reasons,
        },
        "probe_metrics_history": probe_history,
        "history": history,
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print("EXTENDED TTT v2 RECEIPT", flush=True)
    print("=" * 60, flush=True)
    print(f"Attempted: {n_attempted}, Applied: {n_did}, Blocked: {n_blocked}, OOM: {n_oom}", flush=True)
    print(f"Blocked reasons: {blocked_reasons}", flush=True)
    print(f"\nProbe trajectory:", flush=True)
    for pm in probe_history:
        print(f"  step {pm['step']:>3}  action={pm['exact_action_agreement']:.3f}  mae={pm['score_mae']:.3f}  parse={pm['parse_rate']:.3f}", flush=True)
    print(f"\nReceipt: {RECEIPT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
