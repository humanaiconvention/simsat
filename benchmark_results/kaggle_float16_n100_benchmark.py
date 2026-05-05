# float16 benchmark N=100 — Gemma-4-E2B on T4
# Fallback if int8/bitsandbytes unavailable
import os, sys, gc, time, json, statistics
from pathlib import Path

try: del model, processor
except NameError: pass
import torch as _t; _t.cuda.empty_cache(); gc.collect()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if '/tmp/tr_head' not in sys.path:
    sys.path.insert(0, '/tmp/tr_head')
for k in list(sys.modules.keys()):
    if 'transformers' in k:
        del sys.modules[k]

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

KAGGLE_MODEL = Path('/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1')
MODEL_ID = str(KAGGLE_MODEL) if KAGGLE_MODEL.exists() else 'google/gemma-4-E2B-it'
IMAGE_SIZE = 448
WARMUP = 3
TIMED = 100
MAX_NEW_TOK = 512

PROMPT = """You are a satellite image analyst. Assess this 448x448 satellite tile.
Return ONLY valid JSON with exactly these keys:
{"usable_observation": true, "scene_match_score": 0.85, "salience_score": 0.7,
 "change_or_event_score": 0.3, "occlusion_or_cloud_risk": 0.1,
 "confidence": 0.8, "recommended_action": "collect", "rationale_tags": ["clear","urban"]}"""

def make_tile(seed=42):
    rng = np.random.default_rng(seed)
    base = rng.random((IMAGE_SIZE, IMAGE_SIZE, 3))
    for _ in range(5):
        x, y = rng.integers(0, IMAGE_SIZE-64, size=2)
        base[y:y+64, x:x+64] = rng.uniform(0.3, 0.8)
    return Image.fromarray((base.clip(0,1)*255).astype(np.uint8))

print("Loading float16 model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True
)
model.eval()
print("Model loaded (float16)")

tile = make_tile(42)
msgs = [{"role":"user","content":[{"type":"image","image":tile},{"type":"text","text":PROMPT}]}]
inputs = processor.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=True,
    return_tensors="pt", return_dict=True
).to("cuda:0")

def run_once():
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOK, do_sample=False)
    torch.cuda.synchronize()
    n_in = inputs['input_ids'].shape[1]
    return processor.decode(out[0][n_in:], skip_special_tokens=True)

print(f"Warmup {WARMUP}...")
for i in range(WARMUP):
    t0 = time.perf_counter()
    run_once()
    print(f"  warmup {i+1}: {(time.perf_counter()-t0)*1000:.0f}ms")

print(f"Timed {TIMED}...")
latencies, tokens = [], []
for i in range(TIMED):
    t0 = time.perf_counter()
    raw = run_once()
    dt = (time.perf_counter() - t0) * 1000
    n_tok = len(processor.tokenize(raw))
    latencies.append(dt)
    tokens.append(n_tok)
    if (i+1) % 10 == 0:
        print(f"  p{i+1}: {dt:.1f}ms {n_tok}tok  (mean {statistics.mean(latencies):.1f}ms)")

gpu = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
lat_sorted = sorted(latencies)
results = {
    "gpu": gpu, "vram_gb": round(vram_gb, 1), "image_size": IMAGE_SIZE,
    "dtype": "float16", "timed_passes": TIMED,
    "mean_output_tokens": round(statistics.mean(tokens), 1),
    "latency_mean_ms": round(statistics.mean(latencies), 1),
    "latency_std_ms": round(statistics.stdev(latencies), 1),
    "latency_p50_ms": round(lat_sorted[int(TIMED*0.50)-1], 1),
    "latency_p90_ms": round(lat_sorted[int(TIMED*0.90)-1], 1),
    "latency_p95_ms": round(lat_sorted[int(TIMED*0.95)-1], 1),
    "latency_p99_ms": round(lat_sorted[int(TIMED*0.99)-1], 1),
    "tokens_per_sec": round(statistics.mean(tokens)/(statistics.mean(latencies)/1000), 1),
    "model": "google/gemma-4-E2B-it", "warmup_passes": WARMUP,
    "max_new_tokens": MAX_NEW_TOK, "contract": "simsat_observe_assess_v1",
}
for k, v in results.items():
    print(f"{k}: {v}")

out_path = Path('/kaggle/working/simsat_vla_benchmark_float16_n100.json')
out_path.write_text(json.dumps(results, indent=2))
print(f"Saved: {out_path}")
