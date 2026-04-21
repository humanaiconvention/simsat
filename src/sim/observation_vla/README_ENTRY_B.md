# Entry B backend: Gemma-4 + HAIC v35-gov

This folder now contains two ObservationVLA backends:

| Backend file | Class | `runtime_mode` | Purpose |
|---|---|---|---|
| `adapter.py` (existing) | `ObservationVLMAdapter` | `clip_local`, `http_endpoint`, `stub`, `stub_fallback` | Baseline CLIP + stub path for Entry A and the current submission |
| `gemma4_haic_local.py` (NEW) | `Gemma4HAICAdapter` | `gemma4_haic_local`, `stub_fallback` | Entry B — convention-grounded Gemma-4 + HAIC v35-gov LoRA |

Both classes expose the same public interface (`assess(prompt, images, response_schema)`, `.runtime_mode`, `.model_id`) so they are drop-in replacements for each other at the service level.

## Quick start

### 1. Download weights

```bash
# From repo root
bash scripts/download_haic_v35_gov.sh
# (or PowerShell: pwsh scripts\download_haic_v35_gov.ps1)
```

This populates `./weights/haic-v35-gov/` from the `benhaslam/haic-gemma4-v35-gov-unsloth` Kaggle kernel output.

### 2. Pick a load mode

Set in `docker-compose.yaml` (sim service) or your shell:

```yaml
environment:
  - OBSERVATION_VLA_BACKEND=gemma4_haic_local
  - HAIC_GEMMA4_MODE=lora   # or "merged" or "gguf"
  - HAIC_GEMMA4_WEIGHTS_DIR=/app/weights/haic-v35-gov
  - OBSERVATION_VLA_DEVICE=cpu  # or "cuda" / "cuda:0"
```

| Mode | Best for | Memory | Notes |
|---|---|---|---|
| `lora` | Local dev, judge reproducibility | ~6–10 GB on CPU (fp32), ~3–5 GB on GPU (fp16) | Loads base Gemma-4 + LoRA via peft |
| `merged` | Fastest inference on local GPU | same as lora | No peft call; loads the 5-shard merged safetensors |
| `gguf` | On-orbit Orin 16GB demo + Liquid Track comparison | ~2–5 GB, CPU-friendly | Uses `llama-cpp-python`; the in-space compute story |

### 3. Wire the factory

The service layer needs a tiny factory change. Find wherever `ObservationVLMAdapter(...)` is instantiated (likely `src/sim/observation_vla/service.py` or `src/sim/api.py`) and replace with:

```python
def _make_adapter():
    backend = os.environ.get("OBSERVATION_VLA_BACKEND", "auto").strip().lower()
    if backend == "gemma4_haic_local":
        from observation_vla.gemma4_haic_local import Gemma4HAICAdapter
        return Gemma4HAICAdapter()
    from observation_vla.adapter import ObservationVLMAdapter
    return ObservationVLMAdapter()
```

(Keeps `clip_local` as default — Entry B opts in explicitly.)

### 4. Verify against existing tests

```bash
docker compose up --build
python scripts/haic_test.py           # HAIC convention lifecycle
python scripts/observation_vla_eval.py --inprocess   # Re-run reviewed eval with new backend
```

The reviewed eval should produce a new `OBSERVATION_VLA_EVAL.md` with `runtime=gemma4_haic_local` and the current `1.00` agreement numbers (on the 3 pinned reviewed cases) re-computed against Gemma-4. If agreement drops, that's the signal to go back to CLIP before submission.

## Payload contract

`Gemma4HAICAdapter.assess()` returns the same payload shape as `ObservationVLMAdapter._assess_local_clip()`:

```python
{
    "usable_observation": bool,
    "scene_match_score": float,       # 0-1
    "salience_score": float,          # 0-1
    "change_or_event_score": float,   # 0-1
    "occlusion_or_cloud_risk": float, # 0-1
    "confidence": float,              # 0-1
    "recommended_action": "accept" | "defer" | "refine" | "skip",
    "rationale_tags": list[str],
    "raw_response_text": str,
    "_prompt_used": str,
    "_schema_used": dict,
    "_model_id": "gemma4-haic-v35-gov:{lora|merged|gguf}",
}
```

`rationale_tags` always includes `"gemma4_haic_local"` and `"haic_mode:{lora|merged|gguf}"` so downstream filters can distinguish backends.

## Known limits of this scaffold

1. **No image conditioning yet.** This scaffold reasons from sample metadata only (target, probe cloud cover, Sentinel source, geometry). The fine-tuned HAIC model was trained on convention-grounded reasoning, not on image pixels. For image conditioning, stack it with the existing CLIP features from `adapter.py` — left as a follow-up.
2. **Falls back to metadata heuristic on any failure.** `_fallback_payload` mirrors the stub behavior in `adapter.py`. `runtime_mode` will report `stub_fallback` in that case — same convention as the existing adapter.
3. **Generation is deterministic.** `temperature=0.0`, `do_sample=False`. If the fine-tune produces degenerate repetition, switch to a low temperature (0.1) in `_generate_transformers`.
4. **No streaming, no batching.** Single-request inference. For the 3-case eval that's fine; for the progressive-snapshot demo we should revisit.

## What to do before submission

- [ ] Run `haic_test.py` against the new backend — all 11 sections should pass
- [ ] Run `observation_vla_eval.py --inprocess` — confirm agreement on 3 pinned cases
- [ ] Run `submission_evidence.py --reviewed-only` — make sure packet still passes the gate
- [ ] Re-record demo video with `gemma4_haic_local` active

## Feature branch

This code lives on branch `entry-b/backend` in `github.com/humanaiconvention/simsat`. Merge to `main` only after the checks above pass.
