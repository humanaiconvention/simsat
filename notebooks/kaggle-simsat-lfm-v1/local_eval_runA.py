#!/usr/bin/env python3
"""Run A: local re-eval of v1 adapter with decode-time repetition_penalty.

Goal: confirm the 3 parse failures from Run 14 are decode-time rep loops
(numeric field looping `0000...`), not a representational regression. Apply
`repetition_penalty=1.05` + `no_repeat_ngram_size=20` symmetrically to base
AND tuned generation so the matched-pair stays fair.

Runs locally on BEAST (RTX 2080, 8 GB VRAM). LFM2.5-VL-450M fits comfortably.

Outputs:
    .kaggle_output/runA_base_predictions.json
    .kaggle_output/runA_tuned_predictions.json
    .kaggle_output/runA_holdout_eval_report.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_PATH = ROOT / "datasets" / "simsat-lfm-v1" / "simsat_lfm_holdout.jsonl"
IMAGES_DIR = ROOT / "datasets" / "simsat-lfm-v1" / "images"
ADAPTER_DIR = ROOT / ".kaggle_output" / "simsat-lfm25vl-450m-v1-adapter"
OUT_DIR = ROOT / ".kaggle_output"
OUT_DIR.mkdir(exist_ok=True)

BASE_MODEL = "LiquidAI/LFM2.5-VL-450M"
GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=False,
    repetition_penalty=1.05,
    no_repeat_ngram_size=20,
)


def _load_image_in_messages(messages: list[dict]) -> list[dict]:
    """Replace image-path strings with PIL images, resolving relative paths."""
    from PIL import Image
    out: list[dict] = []
    for m in messages:
        new = {"role": m["role"], "content": []}
        content = m["content"]
        if isinstance(content, str):
            new["content"] = content
            out.append(new)
            continue
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


def _expected_action(rec: dict) -> str:
    last = rec["messages"][-1]["content"]
    if isinstance(last, list):
        last = next((c.get("text", "") for c in last if c.get("type") == "text"), "")
    m = re.search(r'"recommended_action"\s*:\s*"(\w+)"', last)
    return m.group(1) if m else "?"


def _expected_scores(rec: dict) -> dict:
    last = rec["messages"][-1]["content"]
    if isinstance(last, list):
        last = next((c.get("text", "") for c in last if c.get("type") == "text"), "")
    try:
        # Strip markdown fences if any
        s = last.strip()
        if s.startswith("```json"):
            s = s[7:]
        if s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return json.loads(s.strip())
    except Exception:
        return {}


def _parse_pred(raw: str) -> dict | None:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # Try greedy {...}
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return None
        return None


def generate_assessment(model, processor, messages_no_assistant: list[dict]) -> str:
    msgs = _load_image_in_messages(messages_no_assistant)
    inputs = processor.apply_chat_template(
        [msgs], add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    new_tokens = out[0, prompt_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def run_eval(model, processor, holdout: list[dict], label: str) -> list[dict]:
    print(f"\n=== {label} eval (n={len(holdout)}) ===")
    preds = []
    t0 = time.time()
    for i, rec in enumerate(holdout):
        msgs_no_assistant = rec["messages"][:-1]
        try:
            raw = generate_assessment(model, processor, msgs_no_assistant)
        except Exception as e:
            raw = f"[ERROR: {type(e).__name__}: {e}]"
        parsed = _parse_pred(raw)
        preds.append({
            "trace_id": rec.get("trace_id"),
            "target_label": rec.get("target_label"),
            "expected_action": _expected_action(rec),
            "expected_scores": _expected_scores(rec),
            "raw_output": raw,
            "predicted": parsed,
        })
        elapsed = time.time() - t0
        if (i + 1) % 4 == 0 or i + 1 == len(holdout):
            print(f"  [{i+1:>3}/{len(holdout)}] elapsed={elapsed:.0f}s  rate={(i+1)/elapsed:.2f}/s")
    return preds


_ACTION_TO_USEFUL = {"accept": True, "refine": True, "defer": False, "skip": False}
_BAND = {"accept": 0.85, "refine": 0.55, "defer": 0.40, "skip": 0.20}


def score(preds: list[dict]) -> dict:
    parsed_ok = [p for p in preds if isinstance(p.get("predicted"), dict)]
    parse_rate = len(parsed_ok) / len(preds) if preds else 0.0

    matches_action = 0
    matches_useful = 0
    mae = 0.0
    per_class_correct = {}
    per_class_total = {}
    for p in preds:
        exp_act = p["expected_action"]
        per_class_total[exp_act] = per_class_total.get(exp_act, 0) + 1
        pred = p.get("predicted") or {}
        pa = pred.get("recommended_action") if isinstance(pred, dict) else None
        if pa == exp_act:
            matches_action += 1
            per_class_correct[exp_act] = per_class_correct.get(exp_act, 0) + 1
        if pa is not None and exp_act in _ACTION_TO_USEFUL and pa in _ACTION_TO_USEFUL:
            if _ACTION_TO_USEFUL[pa] == _ACTION_TO_USEFUL[exp_act]:
                matches_useful += 1
        # MAE: band-mapped predicted vs operator usefulness_score (exp_scores)
        exp_band = _BAND.get(exp_act, 0.5)
        pred_band = _BAND.get(pa, 0.5) if pa else 0.5
        mae += abs(exp_band - pred_band)

    n = len(preds) or 1
    return {
        "n": len(preds),
        "parse_rate": parse_rate,
        "exact_action_agreement": matches_action / n,
        "useful_agreement": matches_useful / n,
        "score_mae": mae / n,
        "per_class_accuracy": {
            k: per_class_correct.get(k, 0) / per_class_total[k]
            for k in per_class_total
        },
    }


def main() -> int:
    print(f"Run A: local re-eval with repetition_penalty=1.05, no_repeat_ngram_size=20")
    print(f"GEN_KWARGS = {GEN_KWARGS}")

    holdout = [json.loads(l) for l in open(HOLDOUT_PATH, encoding="utf-8")]
    print(f"Holdout records: {len(holdout)}")

    print(f"\nLoading base model: {BASE_MODEL}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0").eval()

    base_preds = run_eval(model, processor, holdout, "BASE")
    (OUT_DIR / "runA_base_predictions.json").write_text(
        json.dumps(base_preds, indent=2), encoding="utf-8"
    )
    base_metrics = score(base_preds)
    print(f"BASE metrics: {json.dumps(base_metrics, indent=2)}")

    print(f"\nApplying LoRA adapter from {ADAPTER_DIR}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()

    tuned_preds = run_eval(model, processor, holdout, "TUNED (v1 + rep_penalty)")
    (OUT_DIR / "runA_tuned_predictions.json").write_text(
        json.dumps(tuned_preds, indent=2), encoding="utf-8"
    )
    tuned_metrics = score(tuned_preds)
    print(f"TUNED metrics: {json.dumps(tuned_metrics, indent=2)}")

    report = {
        "config": {
            "base_model": BASE_MODEL,
            "adapter": "HumanAIConvention/simsat-lfm25vl-450m-v1 (local copy)",
            "gen_kwargs": GEN_KWARGS,
            "device": "cuda:0",
            "n_holdout": len(holdout),
        },
        "base": base_metrics,
        "tuned": tuned_metrics,
    }
    (OUT_DIR / "runA_holdout_eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\n=== SUMMARY ===")
    print(f"  parse_rate              base={base_metrics['parse_rate']:.3f}  tuned={tuned_metrics['parse_rate']:.3f}")
    print(f"  exact_action_agreement  base={base_metrics['exact_action_agreement']:.3f}  tuned={tuned_metrics['exact_action_agreement']:.3f}")
    print(f"  useful_agreement        base={base_metrics['useful_agreement']:.3f}  tuned={tuned_metrics['useful_agreement']:.3f}")
    print(f"  score_mae               base={base_metrics['score_mae']:.3f}  tuned={tuned_metrics['score_mae']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
