"""SimSat v11 evaluation on Kaggle T4 — single-cell.

Pulls v11 adapter + eval_pool.jsonl from Hugging Face, runs inference,
prints per-pack breakdown + baselines + confusion matrix, writes CSV
to /kaggle/working/v12_eval_results.csv and uploads back to HF.
"""
import os, sys, subprocess, time

# ---- HF token (Kaggle Secrets preferred; fallback to embedded) ------------
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as _e:
    print(f"  Kaggle Secrets HF_TOKEN unavailable: {_e}")

if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN unavailable: set the Kaggle Secret HF_TOKEN before running.")
    print(f"  Using fallback HF_TOKEN ({HF_TOKEN[:8]}...)")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# ---- Pin compatible deps (avoid surprise upgrades) -----------------------
# Gemma-4 needs transformers from main (or post-4.55 release). On the Kaggle
# image transformers is already 4.55+ which usually supports Gemma-4. We
# upgrade unconditionally to the bleeding-edge git build to guarantee
# AutoModel can load `model_type=gemma4`.
print("\n[1/6] installing/pinning deps...")
# peft 0.15.x calls hf_hub_download(use_auth_token=...) which is removed in
# huggingface_hub 0.29+. Use peft from main (which uses `token=` kwarg) and
# transformers from main (for gemma4 architecture support).
subprocess.run(
    [
        "pip", "install", "-q",
        "--no-warn-conflicts",
        "--upgrade",
        "git+https://github.com/huggingface/transformers.git",
        "git+https://github.com/huggingface/peft.git",
        "accelerate>=1.1.1",
        "torchao>=0.16.0",  # transformers bleeding-edge requires this
        "pandas",
    ],
    check=True,
)
# Print what we ended up with
import transformers as _tf, peft as _pf
print(f"  transformers={_tf.__version__}  peft={_pf.__version__}")

import torch
print(f"  torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
assert torch.cuda.is_available(), "GPU not available — change kernel runtime to T4"

# ---- Pull eval pool ------------------------------------------------------
print("\n[2/6] downloading eval_pool.jsonl + eval_holdout.jsonl from HF...")
from huggingface_hub import hf_hub_download
# Full pool (152 cases — note: ~141 are in v12 training set, so this is
# in-distribution leakage for v12. Reported alongside the held-out N=4
# subset below for honesty.)
EVAL_PATH = hf_hub_download(
    repo_id="HumanAIConvention/simsat-gemma4-v11",  # eval files live here
    filename="eval_pool.jsonl",
    token=HF_TOKEN,
)
HOLDOUT_PATH = hf_hub_download(
    repo_id="HumanAIConvention/simsat-gemma4-v11",
    filename="eval_holdout.jsonl",
    token=HF_TOKEN,
)
print(f"  eval_pool: {EVAL_PATH}")
print(f"  eval_holdout: {HOLDOUT_PATH}")

# ---- Locate base model ---------------------------------------------------
print("\n[3/6] locating base model...")
import glob
BASE_PATH = None
for cand in ("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1", "/kaggle/input/gemma-4-e2b-it"):
    if os.path.isdir(cand):
        BASE_PATH = cand
        break
if not BASE_PATH:
    matches = glob.glob("/kaggle/input/**/gemma*e2b*it*/1", recursive=True)
    BASE_PATH = matches[0] if matches else None
if not BASE_PATH:
    BASE_PATH = "google/gemma-4-e2b-it"  # download from HF as last resort
print(f"  base_model: {BASE_PATH}")

# ---- Load model + LoRA adapter ------------------------------------------
print("\n[4/6] loading base + v12 LoRA...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
    token=HF_TOKEN,
)
model = PeftModel.from_pretrained(model, "HumanAIConvention/simsat-gemma4-v12", token=HF_TOKEN)
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s  device={next(model.parameters()).device}")

# ---- Run inference -------------------------------------------------------
print("\n[5/6] running inference...")
import json, re
with open(EVAL_PATH) as f:
    full_rows = [json.loads(l) for l in f if l.strip()]
with open(HOLDOUT_PATH) as f:
    holdout_rows = [json.loads(l) for l in f if l.strip()]
holdout_ids = {r["trace_id"] for r in holdout_rows}
print(f"  full pool: {len(full_rows)} cases (in-distribution leakage: ~93% of these are in v12 training set)")
print(f"  held-out: {len(holdout_rows)} cases (true generalization, but small N)")
rows = full_rows  # iterate over the full pool; we'll subset for holdout later

def parse_action(text):
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None, None, None
    try:
        obj = json.loads(m.group())
    except Exception:
        return None, None, None
    pa = obj.get("recommended_action")
    pu = obj.get("usable_observation")
    sal = obj.get("salience_score") or 0
    sm = obj.get("scene_match_score") or 0
    score = max(0.0, min(1.0, 0.5 * (float(sal) + float(sm))))
    return pa, (bool(pu) if pu is not None else None), score

results = []
t0 = time.time()
for i, r in enumerate(rows, start=1):
    msgs = [
        {"role": "system", "content": r["system_prompt"]},
        {"role": "user", "content": r["user_prompt"]},
    ]
    # Two-step tokenize: render template to string, then tokenize. More
    # robust across transformers versions than apply_chat_template(...).to(),
    # which can return BatchEncoding or Tensor depending on version.
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    input_ids = enc["input_ids"]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    pa, pu, ps = parse_action(text)
    results.append({
        "trace_id": r["trace_id"],
        "scenario": r["scenario_pack"],
        "target": r["target_label"],
        "cloud": (r.get("metadata") or {}).get("cloud_cover_pct"),
        "pred_action": pa,
        "pred_useful": pu,
        "pred_score": ps,
        "op_action": r["operator_action"],
        "op_useful": r["operator_useful"],
        "op_score": r["operator_usefulness"],
        "parse_ok": pa is not None,
        "raw": text[:400],
    })
    if i % 10 == 0:
        elapsed = time.time() - t0
        print(f"  [{i}/{len(rows)}] elapsed={elapsed:.0f}s · {elapsed/i:.1f}s/case")

elapsed = time.time() - t0
print(f"\n  inference done in {elapsed:.0f}s ({elapsed/len(rows):.1f}s/case avg)")

# ---- Summary -------------------------------------------------------------
print("\n[6/6] summary, baselines, per-pack, confusion matrix...")
import pandas as pd
df = pd.DataFrame(results)
df_ok = df[df.parse_ok].copy()
n_ok, n_total = len(df_ok), len(df)
print(f"\nParse rate: {n_ok}/{n_total} = {n_ok/n_total:.2%}")

exact = (df_ok.pred_action == df_ok.op_action).mean()
useful = (df_ok.pred_useful == df_ok.op_useful).mean()
mae = (df_ok.pred_score.fillna(0) - df_ok.op_score.fillna(0)).abs().mean()
print(f"\nHEADLINE (over {n_ok} parsed):")
print(f"  Exact action agreement: {exact:.3f}")
print(f"  Useful agreement:       {useful:.3f}")
print(f"  Score MAE:              {mae:.3f}")

print("\nPer-pack:")
for pack, sub in df_ok.groupby("scenario"):
    e = (sub.pred_action == sub.op_action).mean()
    print(f"  {pack:32s} N={len(sub):3d}  exact={e:.3f}")

from collections import Counter
op_dist = Counter(df_ok.op_action)
maj_class, maj_n = op_dist.most_common(1)[0]
print(f"\nOperator action distribution: {dict(op_dist)}")
print(f"Always-majority ({maj_class}) baseline: {maj_n/n_ok:.3f}")
print("Random uniform (4-class) baseline: 0.250")

actions_order = ["accept", "refine", "defer", "skip"]
cm = pd.crosstab(df_ok.op_action, df_ok.pred_action).reindex(index=actions_order, columns=actions_order, fill_value=0)
print("\nConfusion matrix (rows=operator, cols=v12 prediction):")
print(cm.to_string())

# ---- Save and upload results --------------------------------------------
csv_path = "/kaggle/working/v12_eval_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

# ---- Held-out (N=4) subset ----------------------------------------------
df_holdout = df_ok[df_ok.trace_id.isin(holdout_ids)].copy()
n_holdout = len(df_holdout)
holdout_exact = (df_holdout.pred_action == df_holdout.op_action).mean() if n_holdout else float("nan")
holdout_useful = (df_holdout.pred_useful == df_holdout.op_useful).mean() if n_holdout else float("nan")
holdout_mae = (df_holdout.pred_score.fillna(0) - df_holdout.op_score.fillna(0)).abs().mean() if n_holdout else float("nan")
print(f"\nHELD-OUT (N={n_holdout}, true generalization):")
print(f"  Exact action agreement: {holdout_exact:.3f}" if n_holdout else "  (no parsed holdout cases)")
print(f"  Useful agreement:       {holdout_useful:.3f}" if n_holdout else "")
print(f"  Score MAE:              {holdout_mae:.3f}" if n_holdout else "")
if n_holdout:
    print("  Per-trace:")
    for _, row in df_holdout.iterrows():
        print(f"    {row.trace_id[:30]} {row.scenario:30s} pred={row.pred_action:7s} op={row.op_action:7s} match={'✓' if row.pred_action==row.op_action else '✗'}")

summary = {
    "version": "v12",
    "full_pool": {
        "n_total": int(n_total),
        "n_parsed": int(n_ok),
        "exact": float(exact),
        "useful": float(useful),
        "mae": float(mae),
        "per_pack": {p: float((sub.pred_action == sub.op_action).mean()) for p, sub in df_ok.groupby("scenario")},
        "majority_baseline": maj_n / n_ok,
        "majority_class": maj_class,
        "confusion_matrix": cm.to_dict(),
        "leakage_warning": "~93% of full-pool cases are in v12 training set; this is an in-distribution metric, not generalization",
    },
    "holdout": {
        "n_total": int(n_holdout),
        "exact": float(holdout_exact) if n_holdout else None,
        "useful": float(holdout_useful) if n_holdout else None,
        "mae": float(holdout_mae) if n_holdout else None,
        "note": "True generalization (N=4 held out from training); small sample, no statistical power",
    },
    "elapsed_seconds": elapsed,
}
import json as _j
sj = "/kaggle/working/v12_eval_summary.json"
with open(sj, "w") as f:
    _j.dump(summary, f, indent=2)
print(f"Saved {sj}")

# ---- Upload to HF -------------------------------------------------------
print("\nUploading results to HF...")
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=csv_path,
    path_in_repo=f"v12_eval_results_n{n_total}.csv",
    repo_id="HumanAIConvention/simsat-gemma4-v12",
    repo_type="model",
    commit_message=f"v11 eval results N={n_total} (exact={exact:.3f}, useful={useful:.3f}, MAE={mae:.3f})",
)
api.upload_file(
    path_or_fileobj=sj,
    path_in_repo=f"v12_eval_summary_n{n_total}.json",
    repo_id="HumanAIConvention/simsat-gemma4-v12",
    repo_type="model",
    commit_message=f"v11 eval summary N={n_total}",
)
print(f"Uploaded to https://huggingface.co/HumanAIConvention/simsat-gemma4-v12")
print("\nDONE.")
