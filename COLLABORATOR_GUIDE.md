# SimSat Collaborator Guide — Plugging In Your Model

This guide explains how to connect your model to the SimSat VLA lane and run a comparison against
the baseline (`clip_local`) and the Gemma-4-E2B fine-tune.

---

## What the VLA lane does

SimSat treats every observation opportunity as a structured **encounter window**. For each
window, the VLA lane does one thing: it looks at a Sentinel tile plus metadata and outputs
a structured JSON assessment saying whether the window is worth materializing. That JSON
feeds into the WCLI trust layer, which then decides: `accept`, `defer`, `refine`, or `skip`.

Your model replaces the baseline CLIP scorer in this one position. Everything else —
the encounter planner, trust layer, viability gates, TTT loop, mission-response layer —
is unchanged.

---

## The JSON contract

Your model must produce a response that contains exactly this JSON object:

```json
{
  "usable_observation": true,
  "scene_match_score": 0.82,
  "salience_score": 0.75,
  "change_or_event_score": 0.60,
  "occlusion_or_cloud_risk": 0.15,
  "confidence": 0.78,
  "recommended_action": "accept",
  "rationale_tags": ["your_backend_tag", "sentinel_support"]
}
```

| Field | Type | Meaning |
|---|---|---|
| `usable_observation` | bool | Is this window worth materializing at all? |
| `scene_match_score` | float 0–1 | Does the tile visually match the target? |
| `salience_score` | float 0–1 | How much signal / detail is present? |
| `change_or_event_score` | float 0–1 | Is there a notable event or change? |
| `occlusion_or_cloud_risk` | float 0–1 | Fraction of tile obscured by cloud or noise |
| `confidence` | float 0–1 | Overall assessment confidence |
| `recommended_action` | string | `accept` / `defer` / `refine` / `skip` |
| `rationale_tags` | list[str] | Your backend's identity tag(s) + reason tags |

Prose, markdown fences, or anything outside the JSON object is tolerated — the parser
extracts the first `{...}` block. Nested JSON objects will confuse it, so emit a flat object.

---

## What the prompt looks like

The triage prompt sent to your model has this structure:

```
You are an Earth-observation triage model. Assess whether this encounter
window is useful for the mission target.

<<CONTEXT>>
Target: Suez Canal Northern Entrance
Scenario pack: maritime_chokepoints
Target tags: maritime, chokepoint, high_priority
Sentinel available: True
Sentinel source: sentinel-2-l2a
Cloud cover: 12.30%
Target visible: True
Elevation deg: 52.40
Off-nadir deg: 8.10
Image attached: True
<<END CONTEXT>>

Triage question: Should this observation window be materialized?

Respond ONLY with a single JSON object matching this schema. No commentary.
{
  "usable_observation": <bool>,
  "scene_match_score": <float 0-1>,
  ...
}
```

When a Sentinel tile is available and encoded as base64, it arrives alongside the metadata.
Whether your model is VL-capable (can see the image) or text-only (metadata reasoning) is
auto-detected via `AutoProcessor`. Text-only models still produce useful assessments from
the metadata alone.

---

## Setup — Genesis (Guilherme)

Genesis uses a **native loader** — it does not delegate to `TransformersVLMAdapter`.
The model architecture is custom (GLA+FoX hybrid, NeoX tokenizer, ChatML format).

```bash
# Step 1 — clone your genesis repo and point to it
export GENESIS_REPO_PATH=/path/to/orchOSModel-genesis-v3

# Step 2 — point to your weights file
export GENESIS_WEIGHTS_PATH=/path/to/genesis_152m_instruct.safetensors

# Step 3 — select the backend and run
export OBSERVATION_VLA_BACKEND=genesis
export OBSERVATION_VLA_DEVICE=cuda
python scripts/observation_vla_eval.py --inprocess
# You should see: runtime_mode=genesis_local in the output
```

Full setup details and architecture context: `src/sim/observation_vla/genesis_local.py`.

**Note on zero-shot JSON quality:** the 152M base model generates narrative text rather
than structured JSON in zero-shot. Fine-tuning on `benhaslam/simsat-gemma4-v1` (same
ChatML format) is recommended before connecting to the eval. The Gemma-4 adapter and
training notebook at `notebooks/kaggle-simsat-gemma4-v1/` are the reference.

Config is read from: `genesis_local.py` → `GenesisAdapter` (native, no HF AutoModel).

---

## Setup — [T^3](https://github.com/GMaN1911) (Garrett)

Two backends are wired for T^3 — pick the one that matches your model's
output format.

### Option A — JSON-output VLM (LLM-style, returns 8-key assessment)

Use this if your T^3 variant produces a structured JSON assessment directly
(same contract as the Gemma-4 fine-tune). Routed through the generic
TransformersVLMAdapter:

```bash
export OBSERVATION_VLA_BACKEND=tesseract_t3
export TESSERACT_T3_MODE=merged
export TESSERACT_T3_MERGED_PATH=/path/to/tesseract-t3-weights
export OBSERVATION_VLA_DEVICE=cuda
```

Config: `tesseract_t3_local.py` → `TesseractT3Adapter` → `TransformersVLMAdapter`.

### Option B — Heatmap-output model (recommended for the DOFA-backbone T³)

Use this if your model emits a per-pixel alarm/anomaly heatmap (the
output format Garrett demoed 2026-04-27: 89M params, DOFA backbone
transfer, ~40 ms inference, alarm heatmap with peak/concentration stats
mapped semantically to ObservationVLA's 8-key payload). No retraining
required — the adapter pools your heatmap into stats and rule-maps to
the assessment fields.

```bash
export OBSERVATION_VLA_BACKEND=tesseract_t3_heatmap
export TESSERACT_T3_WEIGHTS_PATH=/path/to/tesseract-t3.<format>
export TESSERACT_T3_DEVICE=cuda
# Optional tuning of stats→action thresholds:
# export TESSERACT_T3_PEAK_ACCEPT=0.80
# export TESSERACT_T3_PEAK_REFINE=0.45
# export TESSERACT_T3_PEAK_DEFER=0.20
```

**Garrett: this is your insertion point.** Two functions in
`src/sim/observation_vla/tesseract_t3_heatmap.py` ship as stubs and need
your implementation:

1. `_load_tesseract_t3(weights_path, device)` — load your model from
   the path, return whatever object your forward pass expects
2. `_predict_heatmap(model, tile, device)` — run forward on a
   `(3, H, W)` float32 [0,1] tile; return `(H', W')` float32 numpy
   alarm heatmap

Everything downstream (stats extraction, mapping to the 8-key
ObservationVLA payload, action ladder, fallback path) is concrete and
tested. The `(H, W)` heatmap → `(scene_match, salience, change_or_event,
recommended_action, ...)` mapping is documented in `_stats_to_payload()`.

Until your loader is plugged in, the adapter falls back to a stub
payload (`action=skip`, `runtime_mode=stub_fallback`) so the service
stays alive during integration. Stub mode logs the underlying load
failure so you can diagnose without grepping.

13 unit tests cover the stats and payload paths in
`tests/test_tesseract_t3_heatmap.py`.

---

## Setup — Heatmap-output models (any collaborator)

If your model emits a per-pixel alarm / anomaly / salience heatmap rather
than a JSON assessment, plug into the **generic heatmap backend**. The
adapter pools your heatmap into stats and rule-maps to ObservationVLA's
8-key payload — no JSON-output head required, no retraining when you ship
updated weights.

```bash
export OBSERVATION_VLA_BACKEND=heatmap
export HEATMAP_WEIGHTS_PATH=/path/to/your/weights.<format>
export HEATMAP_DEVICE=cuda
# Optional tuning of the action ladder (peak ≥ accept → accept,
# peak ≥ refine → refine, peak ≥ defer → defer, else → skip):
# export HEATMAP_PEAK_ACCEPT=0.80
# export HEATMAP_PEAK_REFINE=0.45
# export HEATMAP_PEAK_DEFER=0.20
# Optional: override the rationale_tag for your backend (default: "heatmap")
# export HEATMAP_BACKEND_TAG=my_model
```

**Two functions to implement** in `src/sim/observation_vla/heatmap_local.py`:

1. `_load_model(weights_path, device)` — load your model from the path,
   return whatever object your forward pass expects
2. `_predict_heatmap(model, tile, device)` — run forward on a
   `(3, H, W)` float32 [0,1] tile; return `(H', W')` float32 numpy
   alarm heatmap (higher = stronger signal). Output spatial dims need
   not match input.

Everything downstream is concrete and tested:

| Heatmap stat | ObservationVLA field |
|---|---|
| `peak` | `salience_score`, `change_or_event_score` |
| `hotspot_concentration + (1-center_offset)` | `scene_match_score` |
| `peak * concentration` | `confidence` |
| `probe.sentinel_cloud_cover` | `occlusion_or_cloud_risk` |
| `peak` ladder | `recommended_action` |

Until your loader is plugged in, the adapter falls back to a stub payload
(`action=skip`, `runtime_mode=stub_fallback`) so the service stays alive
during integration. Stub mode logs the underlying load failure so you can
diagnose without grepping.

11 unit tests cover the stats and payload paths in
`tests/test_heatmap_adapter.py`. Module docstring at
`src/sim/observation_vla/heatmap_local.py` has the full integration
contract.

---

## Docker Compose snippet

Add to `docker-compose.yaml` under the `sim` service environment:

```yaml
# Genesis
- OBSERVATION_VLA_BACKEND=genesis
- GENESIS_MODE=merged
- GENESIS_MERGED_PATH=/app/weights/genesis/base
- OBSERVATION_VLA_DEVICE=cuda

# Tesseract T^3
- OBSERVATION_VLA_BACKEND=tesseract_t3
- TESSERACT_T3_MODE=merged
- TESSERACT_T3_MERGED_PATH=/app/weights/tesseract-t3/base
- OBSERVATION_VLA_DEVICE=cuda
```

---

## Verifying your integration

**1. Quick runtime check:**
```bash
docker compose up -d
curl http://localhost:8000/sim/capabilities | python -m json.tool
# Look for: "observation_vla_runtime_mode": "genesis_local"
```

**2. Eval against the 37 operator-reviewed cases:**
```bash
python scripts/observation_vla_eval.py --inprocess
# Shows: exact action agreement, bucketed agreement, useful agreement,
# MAE vs. operator usefulness scores. The reference numbers to beat
# (v11 SimSat fine-tune): exact 0.86, bucketed 0.86, useful 0.97, MAE 0.13.
```

**3. Full planner comparison:**
```bash
python scripts/encounter_eval.py \
    --base-url http://127.0.0.1:8000/sim \
    --scenario-sweep --top-k 8 --materialize-top-k 2 --markdown
# Prints scaffold vs. trust transition counts per scenario pack
```

**4. Side-by-side backend comparison:**

Run once with `OBSERVATION_VLA_BACKEND=clip_local`, save results.
Run again with your backend, compare:
- `accept→refine` transitions (trust layer respecting the VLA signal)
- Materialization yield on top-k candidates
- MAE vs. pinned operator usefulness scores

**5. (Gemma-4 / fine-tune contributors) Triage your trained checkpoint
before wiring it into the eval:**

```bash
# Masking-only check — no GPU, no adapter required, ~30s
python scripts/diagnose_gemma4_checkpoint.py --check masking

# Full check — loads adapter onto base model, runs forward pass
python scripts/diagnose_gemma4_checkpoint.py \
    --check both \
    --adapter-path ./weights/your-adapter-dir
```

Emits one of: `MASKING_OK_LOSS_DESCENDED` (ship it), `MASKING_OK_LOSS_FLAT`
(hyperparameter problem), `MASKING_BROKEN` (response_template doesn't match
your chat format), `MASKING_TOO_AGGRESSIVE` (template matches multiple
positions), or `INCONCLUSIVE`. Each verdict prints the next concrete step
inline.

When the verdict is `MASKING_OK_LOSS_DESCENDED`, wire the adapter:

```bash
export OBSERVATION_VLA_BACKEND=gemma4                        # NOT gemma4_haic_local!
export OBSERVATION_VLM_BASE_MODEL=google/gemma-4-e2b-it
export OBSERVATION_VLM_LORA_PATH=./weights/your-adapter-dir
export OBSERVATION_VLM_MODE=lora
python scripts/observation_vla_eval.py --inprocess
```

> **Backend gotcha:** `gemma4_haic_local` routes to the legacy HAIC v35-gov
> Gemma-4 (human-interview model). For a fresh SimSat fine-tune use
> `gemma4` (which routes to `TransformersVLMAdapter` with the
> `OBSERVATION_VLM_*` env vars above).

For a one-command download + diagnose + eval roundtrip after the Kaggle
kernel finishes:

```bash
python scripts/sync_kaggle_adapter.py --version 10 --wait
```

`--wait` polls the kernel status every 60s until it's COMPLETE, then
downloads, diagnoses, and runs the eval automatically.

---

## What the viability gates do to your output

> **Naming note:** The six gates below are described in terms of what they mean for VLA-layer assessments (the vocabulary relevant to a collaborator plugging in a model). The same gates are described in operational/implementation terms in `CHALLENGE_ENTRY.md` as: information gain, observation quality, metadata consistency, update magnitude, update rate, and error balance. They are the same six checks viewed from different angles.

Every candidate update — including changes driven by your model's confidence scores —
passes through six non-compensatory viability gates before it can persist into system state:

1. **Entropy reduction** — your `confidence` and `scene_match_score` must actually reduce
   uncertainty vs. the prior. An assessment that adds nothing gets rejected.
2. **Extraction risk** — bulk-mode inference on many windows simultaneously is flagged.
   Normal operational cadence per window is fine.
3. **PRISM consistency** — your `occlusion_or_cloud_risk` must be consistent with the
   probe's measured `sentinel_cloud_cover`. Large disagreements are flagged.
4. **Participation covenant** — adaptations require a real operator stimulus signal.
   Fully automated loops without a human trigger are rejected.
5. **Federated exchange** — raw imagery stays on the satellite. Only derived scores leave.
   Your JSON output (scores, not raw pixels) is the correct output format by design.
6. **Epistemic alignment** — your `confidence` should track actual realized utility.
   The TTT loop tunes the trust layer's blend weights from operator feedback to correct
   systematic over/under-confidence.

Gate 3 means: if your model says `occlusion_or_cloud_risk=0.05` and the probe measured
`sentinel_cloud_cover=85%`, that's a PRISM consistency violation. Calibrate your risk scores
to the probe metadata that's in your prompt.

---

## Prompt customization

If your model uses a different conversation format (system prompt, different role names,
etc.), you can subclass `TransformersVLMAdapter` and override `_build_structured_prompt()`
or `_generate()`. The only constraint is that the final output must parse to the 8-key JSON
schema above.

See `src/sim/observation_vla/transformers_vlm_local.py` for the full base class.
The `genesis_local.py` / `tesseract_t3_local.py` thin wrappers show the minimal
override surface if you only need to change env vars and branding.

---

## Contact

Ben Haslam — benjamin.haslam@gmail.com  
Challenge deadline: **May 8, 2026**
