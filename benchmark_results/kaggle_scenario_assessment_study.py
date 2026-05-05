"""
SimSat Gemma-4 Scenario Assessment Study
=========================================
Runs the SimSat observe/assess contract against 86 synthetic scenario contexts,
matching the real trace dataset (3 scenario packs × targets × conditions).

Output: scenario_assessment_study.jsonl  (one JSON record per assessment)
        scenario_assessment_summary.json (aggregate statistics)

Assign to Kaggle after int8 benchmark completes.
"""
import os, sys, gc, time, json, statistics, re
from pathlib import Path
import numpy as np
from PIL import Image

# ── env ─────────────────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if '/tmp/tr_head' not in sys.path:
    sys.path.insert(0, '/tmp/tr_head')
for k in list(sys.modules.keys()):
    if 'transformers' in k:
        del sys.modules[k]
try: del model, processor
except NameError: pass
import torch as _t; _t.cuda.empty_cache(); gc.collect()

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

KAGGLE_MODEL = Path('/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1')
MODEL_ID = str(KAGGLE_MODEL) if KAGGLE_MODEL.exists() else 'google/gemma-4-E2B-it'
IMAGE_SIZE = 448
MAX_NEW_TOK = 512

# ── scenario definitions (mirrors real trace dataset) ────────────────────────
SCENARIOS = [
    # urban_coastal_ambiguity (34 real traces)
    {"pack": "urban_coastal_ambiguity", "target": "rotterdam",       "label": "Port of Rotterdam",    "tags": ["port","europe","urban"]},
    {"pack": "urban_coastal_ambiguity", "target": "singapore_port",  "label": "Port of Singapore",    "tags": ["port","asia","urban"]},
    {"pack": "urban_coastal_ambiguity", "target": "la_port",         "label": "Port of Los Angeles",  "tags": ["port","north_america","urban"]},
    {"pack": "urban_coastal_ambiguity", "target": "shenzhen_bay",    "label": "Shenzhen Bay",         "tags": ["bay","china","urban"]},
    {"pack": "urban_coastal_ambiguity", "target": "dubai_port",      "label": "Port of Dubai",        "tags": ["port","middle_east","urban"]},
    {"pack": "urban_coastal_ambiguity", "target": "hamburg",         "label": "Port of Hamburg",      "tags": ["port","europe","river"]},
    {"pack": "urban_coastal_ambiguity", "target": "sydney_harbour",  "label": "Sydney Harbour",       "tags": ["harbour","australia","urban"]},
    {"pack": "urban_coastal_ambiguity", "target": "istanbul_bosphorus", "label": "Istanbul Bosphorus","tags": ["strait","europe","urban"]},
    # disaster_response_weather (27 real traces)
    {"pack": "disaster_response_weather", "target": "fort_myers_coast", "label": "Fort Myers Coast",  "tags": ["flood","hurricane","coastal"]},
    {"pack": "disaster_response_weather", "target": "maui_lahaina",     "label": "Lahaina, Maui",     "tags": ["wildfire","hawaii","coastal"]},
    {"pack": "disaster_response_weather", "target": "turkey_earthquake","label": "Turkey Earthquake Zone","tags": ["earthquake","infrastructure","urban"]},
    {"pack": "disaster_response_weather", "target": "pakistan_floods",  "label": "Pakistan Flood Zone","tags": ["flood","agriculture","infrastructure"]},
    {"pack": "disaster_response_weather", "target": "ukraine_odessa",   "label": "Odessa Ukraine",    "tags": ["conflict","port","infrastructure"]},
    {"pack": "disaster_response_weather", "target": "morocco_earthquake","label": "Morocco High Atlas","tags": ["earthquake","mountain","rural"]},
    {"pack": "disaster_response_weather", "target": "libya_floods",     "label": "Derna Libya Floods","tags": ["flood","coastal","infrastructure"]},
    {"pack": "disaster_response_weather", "target": "california_fire",  "label": "California Wildfire","tags": ["wildfire","urban","smoke"]},
    {"pack": "disaster_response_weather", "target": "bangladesh_flood", "label": "Bangladesh Delta",  "tags": ["flood","agriculture","delta"]},
    # maritime_chokepoints (25 real traces)
    {"pack": "maritime_chokepoints", "target": "suez_canal",        "label": "Suez Canal",           "tags": ["canal","shipping","chokepoint"]},
    {"pack": "maritime_chokepoints", "target": "strait_hormuz",     "label": "Strait of Hormuz",     "tags": ["strait","energy","chokepoint"]},
    {"pack": "maritime_chokepoints", "target": "strait_malacca",    "label": "Strait of Malacca",    "tags": ["strait","shipping","asia"]},
    {"pack": "maritime_chokepoints", "target": "panama_canal",      "label": "Panama Canal",         "tags": ["canal","americas","chokepoint"]},
    {"pack": "maritime_chokepoints", "target": "bab_el_mandeb",     "label": "Bab-el-Mandeb Strait", "tags": ["strait","red_sea","chokepoint"]},
    {"pack": "maritime_chokepoints", "target": "danish_straits",    "label": "Danish Straits",       "tags": ["strait","europe","chokepoint"]},
    {"pack": "maritime_chokepoints", "target": "english_channel",   "label": "English Channel",      "tags": ["channel","europe","shipping"]},
    {"pack": "maritime_chokepoints", "target": "taiwan_strait",     "label": "Taiwan Strait",        "tags": ["strait","asia","geopolitical"]},
    {"pack": "maritime_chokepoints", "target": "dover_strait",      "label": "Dover Strait",         "tags": ["strait","europe","shipping"]},
]

# Condition variants to get closer to N=86
CONDITIONS = [
    {"cloud_pct": 0,    "note": "clear"},
    {"cloud_pct": 15,   "note": "partly_cloudy"},
    {"cloud_pct": 45,   "note": "cloudy"},
    {"cloud_pct": 80,   "note": "heavy_cloud"},
]

ASSESS_PROMPT_TEMPLATE = """You are a satellite imagery assessment system for mission-critical observation.
Assess this satellite image for the following context:

Target: {label}
Location type: {tags}
Scenario pack: {pack}
Estimated cloud cover: {cloud_pct}%
Condition: {note}

Return ONLY a single valid JSON object with exactly these 8 keys (no markdown, no explanation):
{{
  "usable_observation": <true|false>,
  "scene_match_score": <0.0-1.0>,
  "salience_score": <0.0-1.0>,
  "change_or_event_score": <0.0-1.0>,
  "occlusion_or_cloud_risk": <0.0-1.0>,
  "confidence": <0.0-1.0>,
  "recommended_action": <"collect"|"skip"|"flag"|"revisit">,
  "rationale_tags": [<list of 1-4 short strings>]
}}"""

JSON_KEYS = ["usable_observation","scene_match_score","salience_score",
             "change_or_event_score","occlusion_or_cloud_risk",
             "confidence","recommended_action","rationale_tags"]

def make_synthetic_tile(rng, cloud_pct=0, seed_offset=0):
    """Generate a plausible synthetic 448×448 tile with optional cloud overlay."""
    base = rng.random((IMAGE_SIZE, IMAGE_SIZE, 3))
    # Add some structure (edges, patches)
    for _ in range(5):
        x, y = rng.integers(0, IMAGE_SIZE-64, size=2)
        intensity = rng.uniform(0.3, 0.8)
        base[y:y+64, x:x+64] = intensity
    # Cloud layer
    if cloud_pct > 0:
        cloud_mask = rng.random((IMAGE_SIZE, IMAGE_SIZE)) < (cloud_pct / 100)
        cloud_val = rng.uniform(0.8, 1.0, (IMAGE_SIZE, IMAGE_SIZE, 3))
        base[cloud_mask] = cloud_val[cloud_mask]
    return Image.fromarray((base.clip(0,1) * 255).astype(np.uint8))

def extract_json(text):
    """Extract first valid JSON dict from model output."""
    # Try direct parse
    text = text.strip()
    for pattern in [r'\{[^{}]*\}', r'\{.*?\}']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except: pass
    try: return json.loads(text)
    except: return None

def check_contract(parsed):
    """Return (compliance_score, missing_keys)."""
    if not parsed or not isinstance(parsed, dict):
        return 0.0, JSON_KEYS
    present = [k for k in JSON_KEYS if k in parsed]
    return len(present) / len(JSON_KEYS), [k for k in JSON_KEYS if k not in parsed]

# ── load model ───────────────────────────────────────────────────────────────
print("Loading float16 model for scenario study...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True
)
model.eval()
print("Model loaded")

# ── build run list (scenario × condition combos, target ~86) ─────────────────
run_list = []
for sc in SCENARIOS:
    for cond in CONDITIONS:
        run_list.append({**sc, **cond})
        if len(run_list) >= 90:
            break
    if len(run_list) >= 90:
        break

print(f"Running {len(run_list)} assessments across {len(SCENARIOS)} scenarios")

# ── run ──────────────────────────────────────────────────────────────────────
out_path = Path('/kaggle/working/scenario_assessment_study.jsonl')
results = []
contract_scores = []
latencies = []
rng_global = np.random.default_rng(2026)

for i, run in enumerate(run_list):
    prompt = ASSESS_PROMPT_TEMPLATE.format(**run)
    img = make_synthetic_tile(rng_global, cloud_pct=run["cloud_pct"], seed_offset=i)
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    ).to("cuda:0")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOK, do_sample=False)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000

    n_in = inputs['input_ids'].shape[1]
    raw = processor.decode(out[0][n_in:], skip_special_tokens=True)
    parsed = extract_json(raw)
    cscore, missing = check_contract(parsed)
    contract_scores.append(cscore)
    latencies.append(dt)

    rec = {
        "run_idx": i,
        "scenario_pack": run["pack"],
        "target": run["target"],
        "target_label": run["label"],
        "condition_note": run["note"],
        "cloud_pct": run["cloud_pct"],
        "latency_ms": round(dt, 1),
        "contract_compliance": round(cscore, 3),
        "missing_keys": missing,
        "assessment": parsed,
        "raw_response": raw[:500],
    }
    results.append(rec)

    with open(out_path, 'a') as f:
        f.write(json.dumps(rec) + '\n')

    if (i+1) % 10 == 0 or i < 3:
        print(f"[{i+1}/{len(run_list)}] {run['target']} ({run['note']}) "
              f"lat={dt:.0f}ms compliance={cscore:.2f} "
              f"action={parsed.get('recommended_action','?') if parsed else 'PARSE_ERR'}")

# ── summary ──────────────────────────────────────────────────────────────────
from collections import Counter

actions = Counter(r['assessment'].get('recommended_action','PARSE_ERR') for r in results if r['assessment'])
compliance_full = sum(1 for s in contract_scores if s == 1.0)

by_pack = {}
for pack in ['urban_coastal_ambiguity','disaster_response_weather','maritime_chokepoints']:
    pack_recs = [r for r in results if r['scenario_pack'] == pack]
    if pack_recs:
        confs = [r['assessment']['confidence'] for r in pack_recs if r['assessment'] and 'confidence' in r['assessment']]
        by_pack[pack] = {
            "n": len(pack_recs),
            "mean_confidence": round(statistics.mean(confs), 3) if confs else None,
            "action_dist": dict(Counter(r['assessment'].get('recommended_action') for r in pack_recs if r['assessment'])),
        }

summary = {
    "total_runs": len(results),
    "model": "google/gemma-4-E2B-it",
    "dtype": "float16",
    "gpu": torch.cuda.get_device_name(0),
    "image_size": IMAGE_SIZE,
    "latency_mean_ms": round(statistics.mean(latencies), 1),
    "latency_p95_ms": round(sorted(latencies)[int(len(latencies)*0.95)-1], 1),
    "contract_compliance_mean": round(statistics.mean(contract_scores), 3),
    "contract_compliance_full_rate": round(compliance_full / len(results), 3),
    "recommended_action_distribution": dict(actions),
    "by_scenario_pack": by_pack,
    "confidence_all": {
        "mean": round(statistics.mean(
            r['assessment']['confidence'] for r in results
            if r['assessment'] and 'confidence' in r['assessment']
        ), 3),
    },
}

for k, v in summary.items():
    print(f"{k}: {v}")

Path('/kaggle/working/scenario_assessment_summary.json').write_text(json.dumps(summary, indent=2))
print(f"\nSaved {len(results)} records to {out_path}")
print("Saved summary to /kaggle/working/scenario_assessment_summary.json")
