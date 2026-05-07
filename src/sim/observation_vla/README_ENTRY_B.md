# ObservationVLA backends — model-agnostic template

This folder ships two ObservationVLA backends:

| Backend file | Class | `runtime_mode` | Purpose |
|---|---|---|---|
| `adapter.py` | `ObservationVLMAdapter` | `clip_local`, `http_endpoint`, `stub`, `stub_fallback` | Baseline CLIP + stub path |
| `transformers_vlm_local.py` | `TransformersVLMAdapter` | `transformers_vlm_local`, `stub_fallback` | **Model-agnostic template** — any transformers VLM or GGUF |
| `genesis_local.py` | `GenesisAdapter` | `genesis_local`, `stub_fallback` | Guilherme's Genesis model (collaborator seat) |
| `tesseract_t3_local.py` | `TesseractT3Adapter` | `tesseract_t3_local`, `stub_fallback` | Garrett's Tesseract T3 (collaborator seat) |
| `gemma4_haic_local.py` | `Gemma4HAICAdapter` | `gemma4_haic_local`, `stub_fallback` | HAIC v35-gov fine-tune (reference / legacy) |
| `backend_factory.py` | `build_vla_adapter()` | — | **Factory** — reads `OBSERVATION_VLA_BACKEND`, returns the right adapter |

Both classes expose the same public interface (`assess(prompt, images, response_schema)`, `.runtime_mode`, `.model_id`).

## What changed from the original scaffold (previously `Gemma4HAICAdapter`)

- **Renamed** to `TransformersVLMAdapter`.
- **Decoupled from HAIC / Gemma-4 / v35-gov** — no hard-coded model names or paths.
- **Auto-detects vision capability** via `AutoProcessor` → if the model has an image processor, images flow through the chat template; otherwise falls back to text-only `AutoTokenizer`.
- **Generic env vars** (`OBSERVATION_VLM_*`) replace the `HAIC_GEMMA4_*` namespace.

Why: the v35-gov HAIC Gemma-4 model is a human-interview / consent-governance fine-tune — it pivots on emotional user input, not satellite imagery. It was the wrong shape for SimSat Entry B. The scaffold is retained as a **template** so the same structure can host whichever model Entry A and Entry B actually ship.

## Model assignments (as of 2026-04-24)

| Track | Entry | Backend key | Model | Status |
|---|---|---|---|---|
| Liquid Track | Entry A | `transformers_vlm` | **LFM2.5-VL** encoder → MuZero `h()` | Encoder stub wired; weight loader pending |
| General AI Track | Entry B (primary) | `gemma4` | **Gemma-4-E2B** SimSat fine-tune | Kernel v3 ready; GPU quota resets ~15:15 today |
| General AI Track | Entry B (collaborator) | `genesis` | **Genesis** (Guilherme Ferrari Brescia) | Adapter wired; weights pending collaborator |
| General AI Track | Entry B (collaborator) | `tesseract_t3` | **[T^3](https://github.com/GMaN1911)** (Garrett Sutherland) | Adapter wired; weights pending collaborator |

All General AI Track backends load through `TransformersVLMAdapter` by swapping env vars.
Model adaptability is an explicit part of the entry — the governed pipeline is
the contribution, not any single model checkpoint.

## Env vars

```
OBSERVATION_VLM_MODE          lora | merged | gguf             (default: lora)
OBSERVATION_VLM_BASE_MODEL    base model path or HF repo id
OBSERVATION_VLM_LORA_PATH     path to LoRA adapter dir
OBSERVATION_VLM_MERGED_PATH   path to merged safetensors dir
OBSERVATION_VLM_GGUF_PATH     path to .gguf file
OBSERVATION_VLM_WEIGHTS_DIR   convenience root (base/, adapter/, gguf/)
OBSERVATION_VLM_DEVICE        cpu | cuda | cuda:0              (default: cpu)
OBSERVATION_VLM_MODEL_LABEL   cosmetic tag for model_id        (optional)
```

### Example configs

**Entry A — LFM2.5-VL from HuggingFace (no weights on disk, pulls from hub):**
```yaml
services:
  sim:
    environment:
      - OBSERVATION_VLA_BACKEND=transformers_vlm_local
      - OBSERVATION_VLM_MODE=merged
      - OBSERVATION_VLM_MERGED_PATH=LiquidAI/LFM2-VL-1.6B   # or LFM2.5 when repo id is confirmed
      - OBSERVATION_VLM_MODEL_LABEL=lfm2-vl-base
      - OBSERVATION_VLM_DEVICE=cuda
```

**Entry A — after fine-tuning (LoRA adapter locally):**
```yaml
services:
  sim:
    environment:
      - OBSERVATION_VLA_BACKEND=transformers_vlm_local
      - OBSERVATION_VLM_MODE=lora
      - OBSERVATION_VLM_BASE_MODEL=LiquidAI/LFM2-VL-1.6B
      - OBSERVATION_VLM_LORA_PATH=/app/weights/lfm2-vl-sentinel-ft/adapter
      - OBSERVATION_VLM_MODEL_LABEL=lfm2-vl-sentinel-v1
      - OBSERVATION_VLM_DEVICE=cuda
```

**Entry B — Gemma-4 fine-tune (once trained):**
```yaml
services:
  sim:
    environment:
      - OBSERVATION_VLA_BACKEND=transformers_vlm_local
      - OBSERVATION_VLM_MODE=lora
      - OBSERVATION_VLM_BASE_MODEL=google/gemma-4-e2b-it   # or unsloth/gemma-4-E2B-it
      - OBSERVATION_VLM_LORA_PATH=/app/weights/gemma4-simsat/adapter
      - OBSERVATION_VLM_MODEL_LABEL=gemma4-simsat-v1
      - OBSERVATION_VLM_DEVICE=cuda
```

**GGUF path (for the Orin on-orbit demo):**
```yaml
services:
  sim:
    environment:
      - OBSERVATION_VLA_BACKEND=transformers_vlm_local
      - OBSERVATION_VLM_MODE=gguf
      - OBSERVATION_VLM_GGUF_PATH=/app/weights/{model}/model.gguf
```

## Wire the factory

`backend_factory.py` ships the full dispatch. `runtime.py` already uses it:

```python
from observation_vla.backend_factory import build_vla_adapter
adapter = build_vla_adapter()   # reads OBSERVATION_VLA_BACKEND env var
```

Supported `OBSERVATION_VLA_BACKEND` values: `clip_local` (default), `gemma4`,
`genesis`, `tesseract_t3`, `transformers_vlm`, `gemma4_haic`, `stub`.

## Payload contract (unchanged)

`TransformersVLMAdapter.assess()` returns exactly the same shape as `ObservationVLMAdapter._assess_local_clip()`:

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
    "_model_id": "transformers_vlm:{lora|merged|gguf}:{label}",
}
```

Always-present rationale tags:
- `transformers_vlm_local` — identifies the backend
- `mode:{lora|merged|gguf}` — identifies the load mode
- `image_conditioned` | `image_available_text_only_model` | `metadata_only` — indicates whether the image was actually used in inference

## Known limits

1. **VL chat templates vary by model.** LFM2-VL, Gemma-4-VL, and LLaVA all differ. The adapter calls `apply_chat_template` on either the processor or tokenizer; if a specific model's template disagrees with the `{"type":"image"}` convention, customize `_generate_vlm`.
2. **llama.cpp VL support is spotty.** GGUF mode defaults to text-only. If you need VL-over-GGUF, switch to a dedicated multimodal llama.cpp fork.
3. **Fallback is metadata-only** — `runtime_mode` becomes `stub_fallback` when the model fails to load or generate, and the assessor sees the same schema as the existing `clip_local` stub path.

## Pre-merge gate (unchanged)

Before merging this branch to `main`:

- [ ] A concrete model is picked and wired (env vars set, factory dispatch added).
- [ ] `docker compose up` starts cleanly with the new backend active.
- [ ] `python scripts/haic_test.py` passes all 11 sections.
- [ ] `python scripts/observation_vla_eval.py --inprocess` produces acceptable agreement on the 3 pinned reviewed cases.

## Feature branch

This code lives on `entry-b/backend` in `github.com/humanaiconvention/simsat`. Model-specific downloaders / config files will arrive in follow-up commits once Ben + Guilherme finalize the Liquid vs General track model assignments.
