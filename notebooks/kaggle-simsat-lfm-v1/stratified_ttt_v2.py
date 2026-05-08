#!/usr/bin/env python3
"""Stratified TTT v2 — operator-balanced feedback stream.

The two single-class lifts (skip +37.5pp, defer +75pp) showed TTT lifts
the targeted class. The architecturally critical question is whether a
balanced operator-feedback stream (1 from each class, repeated) lifts
ALL classes — or whether per-class deltas trade off.

Setup:
  - Adapter:        v3 canonical (HumanAIConvention/simsat-lfm25vl-450m-v3)
  - Stream:         stratified mix from train, 4 per class × 4 classes = 16 steps
  - Probe:          FULL 32-row v3 holdout (8 per class), pre and post
  - Loss:           action-token-weighted CE
  - lr=1e-4, same OnlineLoRAStepper, same six viability gates

Hypothesis: stratified TTT preserves or lifts every class on the full holdout,
            whereas class-targeted TTT (skip-only stream) regresses other
            classes in exchange for the target lift.

Either result is publishable.
"""
from __future__ import annotations
import gc, json, os, random, re, sys, time
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
RECEIPT_PATH = OUT_DIR / "stratified_ttt_v2_receipt.json"

BASE_MODEL = "LiquidAI/LFM2.5-VL-450M"
N_PER_CLASS = 4  # 4 per class * 4 classes = 16-step stream
LR = 1e-4

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


def measure_full_holdout(model, processor, holdout, label):
    model.eval()
    print(f"\n[{label}]", flush=True)
    matches = 0
    mae = 0.0
    pred_dist = {}
    per_class_correct, per_class_total = {}, {}
    for rec in holdout:
        exp = _expected_action(rec)
        per_class_total[exp] = per_class_total.get(exp, 0) + 1
        try: pa = _generate_action(model, processor, rec["messages"][:-1])
        except Exception: pa = None
        pred_dist[pa or "none"] = pred_dist.get(pa or "none", 0) + 1
        if pa == exp:
            matches += 1
            per_class_correct[exp] = per_class_correct.get(exp, 0) + 1
        mae += abs(_BAND.get(exp, 0.5) - (_BAND.get(pa, 0.5) if pa else 0.5))
    n = len(holdout) or 1
    metrics = {
        "n": n,
        "exact_action_agreement": matches / n,
        "score_mae": mae / n,
        "prediction_distribution": pred_dist,
        "per_class_accuracy": {k: per_class_correct.get(k, 0) / per_class_total[k] for k in per_class_total},
    }
    print(f"  exact_action: {matches}/{n} = {metrics['exact_action_agreement']:.3f}", flush=True)
    print(f"  score_mae:    {metrics['score_mae']:.3f}", flush=True)
    print(f"  per-class:    {metrics['per_class_accuracy']}", flush=True)
    print(f"  predictions:  {pred_dist}", flush=True)
    model.train()
    return metrics


def _build_action_only_forward_loss(processor):
    tokenizer = processor.tokenizer

    def forward_loss(model, _proc, messages, target_text):
        full = _load_image_in_messages(messages) + [
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        enc = processor.apply_chat_template(
            [full], add_generation_prompt=False, return_tensors="pt", return_dict=True, tokenize=True,
        ).to(model.device)
        prompt_only = _load_image_in_messages(messages)
        p_enc = processor.apply_chat_template(
            [prompt_only], add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True,
        )
        prompt_len = int(p_enc["input_ids"].shape[1])

        assistant_ids = enc["input_ids"][0, prompt_len:].cpu()
        assistant_text = processor.decode(assistant_ids, skip_special_tokens=False)

        m = re.search(r'"recommended_action"\s*:\s*"(\w+)"', assistant_text)
        if m is None:
            labels = enc["input_ids"].clone()
            labels[:, :prompt_len] = -100
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
        else:
            value_start_char = m.start(1)
            value_end_char = m.end(1)
            try:
                pre_value_tokens = tokenizer(assistant_text[:value_start_char], add_special_tokens=False)["input_ids"]
                value_tokens = tokenizer(assistant_text[value_start_char:value_end_char], add_special_tokens=False)["input_ids"]
                value_start_idx = prompt_len + len(pre_value_tokens)
                value_end_idx = value_start_idx + len(value_tokens)
            except Exception:
                value_start_idx, value_end_idx = prompt_len, enc["input_ids"].shape[1]
            labels = enc["input_ids"].clone()
            labels[:, :] = -100
            labels[:, value_start_idx:value_end_idx] = enc["input_ids"][:, value_start_idx:value_end_idx]
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100

        out = model(**enc, labels=labels)
        with torch.no_grad():
            argmax = out.logits.argmax(dim=-1)
            asst_ids = argmax[0, prompt_len - 1: prompt_len + 32]
            try: asst_text = processor.decode(asst_ids.cpu(), skip_special_tokens=True)
            except Exception: asst_text = ""
            mp = re.search(r'"recommended_action"\s*:\s*"(\w+)"', asst_text)
            pa = mp.group(1) if mp else None
        return out.loss, pa
    return forward_loss


def main():
    print("=" * 60, flush=True)
    print(f"STRATIFIED TTT v2 — balanced operator feedback", flush=True)
    print(f"  {N_PER_CLASS} per class × 4 classes = {N_PER_CLASS*4}-step stream", flush=True)
    print(f"  Probe: FULL 32-row v3 holdout, pre and post", flush=True)
    print("=" * 60, flush=True)

    rng = random.Random(42)

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

    by_class_train = {}
    for r in train_recs:
        by_class_train.setdefault(_expected_action(r), []).append(r)

    # Build stratified stream: shuffle each class, take N_PER_CLASS, interleave
    rng.seed(42)
    stratified = []
    classes = sorted(by_class_train.keys())
    samples_per_class = []
    for cls in classes:
        rng.shuffle(by_class_train[cls])
        samples_per_class.append(by_class_train[cls][:N_PER_CLASS])
    # Interleave: round-robin across classes
    for i in range(N_PER_CLASS):
        for cls_samples in samples_per_class:
            if i < len(cls_samples):
                stratified.append(cls_samples[i])

    print(f"\nStream order (interleaved):", flush=True)
    for i, rec in enumerate(stratified):
        print(f"  {i+1:>2}. {_expected_action(rec)}", flush=True)

    print(f"\nFull holdout: {len(holdout_recs)} rows (8 per class)", flush=True)

    pre_metrics = measure_full_holdout(model, processor, holdout_recs, "PRE-stratified-TTT FULL holdout")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    forward_loss = _build_action_only_forward_loss(processor)
    stepper = OnlineLoRAStepper(model, processor, optimizer, forward_loss_fn=forward_loss)

    history = []
    print(f"\n--- Stratified TTT stream ({len(stratified)} steps, action-only loss) ---", flush=True)
    for i, rec in enumerate(stratified):
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
            })
            tag = "STEP" if res.did_step else f"BLOCKED({res.blocked_by})"
            print(f"  [{i+1:>2}/{len(stratified)}]  {tag:<22} class={op_action:<6}  loss={res.loss:.4f}  d_l2={res.lora_delta_l2:.4f}  ({dur:.0f}s)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i+1:>2}/{len(stratified)}]  OOM", flush=True)
            torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"  [{i+1:>2}/{len(stratified)}]  ERROR: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache(); gc.collect()

    post_metrics = measure_full_holdout(model, processor, holdout_recs, "POST-stratified-TTT FULL holdout")

    delta_action = post_metrics["exact_action_agreement"] - pre_metrics["exact_action_agreement"]
    delta_mae = post_metrics["score_mae"] - pre_metrics["score_mae"]
    per_class_deltas = {
        k: post_metrics["per_class_accuracy"].get(k, 0) - pre_metrics["per_class_accuracy"].get(k, 0)
        for k in pre_metrics["per_class_accuracy"]
    }

    receipt = {
        "config": {
            "base_model": BASE_MODEL, "adapter": str(ADAPTER_DIR),
            "stream_type": "stratified",
            "n_per_class": N_PER_CLASS,
            "n_stream": N_PER_CLASS * 4,
            "n_probe_full": 32,
            "lr": LR, "loss": "action-token-weighted CE",
        },
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "delta": {
            "exact_action_agreement": delta_action,
            "score_mae": delta_mae,
            "per_class": per_class_deltas,
        },
        "history": history,
    }
    OUT_DIR.mkdir(exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"STRATIFIED TTT v2 RECEIPT — full 32-row holdout", flush=True)
    print("=" * 60, flush=True)
    print(f"  pre  exact_action: {pre_metrics['exact_action_agreement']:.3f}", flush=True)
    print(f"  post exact_action: {post_metrics['exact_action_agreement']:.3f}", flush=True)
    print(f"  delta:             {delta_action:+.3f}  ({delta_mae:+.3f} mae)", flush=True)
    print(f"\nPer-class shifts:", flush=True)
    for cls in sorted(per_class_deltas.keys()):
        pre = pre_metrics["per_class_accuracy"].get(cls, 0)
        post = post_metrics["per_class_accuracy"].get(cls, 0)
        d = per_class_deltas[cls]
        marker = "↑" if d > 0.05 else ("↓" if d < -0.05 else "→")
        print(f"  {cls:<8} {pre:.3f} → {post:.3f}  ({d:+.3f}) {marker}", flush=True)
    print(f"\nReceipt: {RECEIPT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
