# COLLABORATOR_GUIDE.md

This guide is for teams plugging a new model into SimSat's ObservationVLA lane — the component that assesses each satellite observation window and recommends an operator action.

---

## The JSON contract

Every backend must return a dict with exactly these eight keys:

| Key | Type | Range | Meaning |
|---|---|---|---|
| `usable_observation` | bool | — | Does the image support a useful observation at all? |
| `scene_match_score` | float | 0–1 | How well does the imagery match the mission target? |
| `salience_score` | float | 0–1 | How operationally significant is what's visible? |
| `change_or_event_score` | float | 0–1 | Evidence of change or mission-relevant event? |
| `occlusion_or_cloud_risk` | float | 0–1 | Fraction of scene obscured (cloud, shadow, edge cut-off) |
| `confidence` | float | 0–1 | Model's own confidence in the above scores |
| `recommended_action` | str | `accept` \| `refine` \| `skip` \| `defer` | Suggested operator action |
| `rationale_tags` | list[str] | — | Short labels explaining the decision (free-form, used for logging) |

Example:
```json
{
  "usable_observation": true,
  "scene_match_score": 0.82,
  "salience_score": 0.75,
  "change_or_event_score": 0.50,
  "occlusion_or_cloud_risk": 0.12,
  "confidence": 0.78,
  "recommended_action": "accept",
  "rationale_tags": ["sentinel_support", "high_scene_match"]
}
```

### Action semantics

The WCLI trust layer maps scores → action using these rules (in order):

1. **`accept`** — `usable_observation=True`, `confidence≥0.75`, `scene_match_score≥0.75`
2. **`defer`** — `usable_observation=True` but `salience_score<0.55` (worth revisiting later)
3. **`refine`** — `usable_observation=True` (doesn't qualify for accept or defer — needs operator review)
4. **`skip`** — `usable_observation=False` (scene not usable; move to next window)

Your model's `recommended_action` is what the evaluator measures against operator labels. The trust layer may override it at runtime based on confidence thresholds and online weight updates (TTT), but the raw action from your model is what's scored.

---

## What the WCLI trust layer does to your output

After your backend returns, the WCLI (Weighted Confidence Latency Index) trust model applies:

1. **Score re-weighting** — per-score weights learned from operator feedback (online TTT). In fresh deployments these are near-uniform. As an operator labels observations, weights shift to match human preferences.
2. **Action override** — if the trust model's re-weighted score disagrees with your `recommended_action`, it will substitute its own. This is expected and is not a bug — it means the operator feedback loop has adapted.
3. **Viability gate check** — three non-compensatory TTT gates run after each online update: weight drift (<30% from policy defaults), update rate (<1000 updates before reset), and error bias (<70% same-sign errors in the last 10 updates). A failed gate logs a warning but does not stop inference.

Your model's direct output is always logged in full before any trust-layer adjustment, so evaluation scripts can compare your raw output against the trust-adjusted output.

---

## Env vars for your backend

Copy `.env.example` to `.env` and set:

```bash
OBSERVATION_VLA_BACKEND=<your_value>    # see table below
OBSERVATION_VLA_DEVICE=cpu              # or cuda, cuda:0, etc.
```

| `OBSERVATION_VLA_BACKEND` | Your backend |
|---|---|
| `genesis` | Genesis (Guilherme Mesquita) |
| `tesseract_t3` | Tesseract T3 (Garrett Sutherland) |
| `transformers_vlm` | Any HuggingFace VLM |

For Genesis, also set in `.env`:
```bash
GENESIS_MODE=lora                       # lora | merged | gguf
GENESIS_BASE_MODEL=your-hf-org/genesis-base
GENESIS_LORA_PATH=./weights/genesis/adapter   # if lora
GENESIS_WEIGHTS_DIR=./weights/genesis
GENESIS_MODEL_LABEL=genesis
```

For Tesseract T3:
```bash
TESSERACT_T3_MODE=lora
TESSERACT_T3_BASE_MODEL=your-hf-org/tesseract-t3
TESSERACT_T3_LORA_PATH=./weights/tesseract-t3/adapter
TESSERACT_T3_WEIGHTS_DIR=./weights/tesseract-t3
TESSERACT_T3_MODEL_LABEL=tesseract-t3
```

For a generic HuggingFace VLM:
```bash
OBSERVATION_VLM_MODE=lora              # lora | merged
OBSERVATION_VLM_BASE_MODEL=your-hf-org/model-id
OBSERVATION_VLM_LORA_PATH=./weights/your-model/adapter
OBSERVATION_VLM_MODEL_LABEL=your-label
```

---

## Registering a new backend

1. Add a loader in `src/sim/observation_vla/` that returns the 8-key dict above.
2. Add your `OBSERVATION_VLA_BACKEND` value to the dispatch table in `src/sim/observation_vla/assessor.py`.
3. Add env var docs to `.env.example`.
4. Run the smoke path to confirm:
   ```bash
   python scripts/quickstart.py
   ```
   This runs 33 tests + eval against 5 pinned reviewed cases. Exit 0 = wired correctly.
5. Run the eval against the full shortlist:
   ```bash
   python scripts/observation_vla_eval.py --inprocess
   ```
   This produces `OBSERVATION_VLA_EVAL.md` with your action-agreement and MAE numbers.

---

## What gets evaluated

The challenge evaluation compares `recommended_action` against operator-reviewed labels across 37 Sentinel cases. Metrics:

- **Action agreement** — exact match rate (target: ≥0.80)
- **Usefulness MAE** — mean absolute error on `usable_observation` (continuous proxy)
- **Useful/not-useful agreement** — binary classification accuracy

The baseline is `clip_local` (CLIP ViT-Base/32, no fine-tuning). Your numbers are compared against it.

---

## What not to do

- Do not return `recommended_action` values outside `{accept, refine, skip, defer}` — the assessor will raise on parse.
- Do not omit any of the 8 keys — the schema validator rejects partial responses.
- Do not mutate the `.env.example` file — add new vars to it, but don't remove existing ones.
