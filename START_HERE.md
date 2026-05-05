# START HERE — Plugging Your Model In

You're looking at the SimSat repo. **You probably don't need most of it.** Find
the section below that matches your model and follow the 3 steps. The rest of
the codebase is the broader satellite-encounter-planning pipeline you're
plugging into; it runs on its own and you don't need to read it.

---

## If your model emits a per-pixel HEATMAP

(Alarm / anomaly / salience / change-detection — anything where the output is a
2-D `(H, W)` float map and higher = stronger signal.)

**Files you touch (1):**
1. `src/sim/observation_vla/heatmap_local.py` — fill in two `NotImplementedError`
   stubs near the top:
   - `_load_model(weights_path, device)` — load your model
   - `_predict_heatmap(model, tile, device)` — return `(H', W')` float32 numpy

That's it. The stats pooling, action mapping, and 8-key ObservationVLA payload
are all concrete and tested.

**Run it:**
```bash
export OBSERVATION_VLA_BACKEND=heatmap
export HEATMAP_WEIGHTS_PATH=/path/to/your/weights
export HEATMAP_DEVICE=cuda
export HEATMAP_BACKEND_TAG=your_model_name   # appears in rationale_tags
python scripts/observation_vla_eval.py --inprocess
```

**Optional knobs** (tune the action ladder if defaults feel off):
```bash
export HEATMAP_PEAK_ACCEPT=0.80   # peak >= this AND scene_match >= 0.5 → accept
export HEATMAP_PEAK_REFINE=0.45   # peak >= this → refine
export HEATMAP_PEAK_DEFER=0.20    # peak >= this → defer (else skip)
```

Until your loader is plugged in, the adapter falls back to a stub payload
(`runtime_mode=stub_fallback`, `action=skip`) so nothing breaks.

**Tests:** `pytest tests/test_heatmap_adapter.py` (14 tests, all the stats and
mapping logic).

---

## If your model emits a JSON ASSESSMENT

(Like the SimSat Gemma-4 fine-tune — prompt in, structured 8-key JSON out.)

**Files you touch (0 if your model speaks our JSON contract; 1 thin wrapper otherwise):**
- Your model just needs to emit the 8-key payload below given the prompt
  shape documented in `COLLABORATOR_GUIDE.md` ("The JSON contract" + "What
  the prompt looks like"). 
- For a HuggingFace-style VLM, no code change needed:

```bash
export OBSERVATION_VLA_BACKEND=transformers_vlm
export OBSERVATION_VLM_BASE_MODEL=your-org/your-model
export OBSERVATION_VLM_LORA_PATH=/path/to/your/adapter   # if you have one
export OBSERVATION_VLM_MODE=lora
python scripts/observation_vla_eval.py --inprocess
```

If your model has a custom load path (like Genesis's native loader), see
`src/sim/observation_vla/genesis_local.py` as a reference and add a sibling
file plus a route in `backend_factory.py`.

---

## What you'll see when it works

`scripts/observation_vla_eval.py` prints:

- exact action agreement (your model's action vs. operator label)
- bucketed action agreement (groups synonymous actions)
- useful / not-useful agreement
- usefulness-score MAE
- per-case breakdown

**Reference numbers to beat** (Gemma-4 v12 SimSat fine-tune over N=37
operator-reviewed cases): exact 0.86, bucketed 0.86, useful 0.97, MAE 0.13.
Full eval at [`OBSERVATION_VLA_EVAL.md`](./OBSERVATION_VLA_EVAL.md).

---

## If you want to read more

- **`COLLABORATOR_GUIDE.md`** — full collaborator integration guide (JSON
  contract, prompt shape, all backend options)
- **`CHALLENGE_ENTRY.md`** — what the SimSat project is and how the VLA
  lane fits the competition framing
- **`README.md`** — the repo's standing landing page

You don't need any of those to plug your model in. They're context.
