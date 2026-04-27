# Gemma-4-E2B QLoRA on Kaggle — SimSat Hard-Won Notes

**Working as of kernel v3 (2026-04-24).** 12 fixes required; each was a distinct blocker.

## Working config summary

- **Hardware:** Kaggle GPU T4×2 (free), single T4 via `CUDA_VISIBLE_DEVICES=0`
- **Model:** `google/gemma-4-E2B-it`, loaded from Kaggle-attached model at `/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1`
- **Stack:** raw `transformers>=4.51` + `bitsandbytes>=0.44` NF4 4-bit + `peft>=0.12` LoRA + `trl>=0.12` SFTTrainer (NO Unsloth)
- **Training:** batch 1 × grad_accum 8 (effective 8), 2 epochs, fp16=False (no AMP), gradient_checkpointing=True with use_reentrant=False

## Twelve fixes

### 1. Use Kaggle-attached model path, not HF download
The attached model is at `/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1`.
Glob glob `"/kaggle/input/**/gemma*e2b*/1"` to find it; fall back to HF only as last resort.

### 2. Pin to single T4 (`CUDA_VISIBLE_DEVICES=0`)
`device_map="auto"` across T4×2 shards model across both GPUs, then HF Trainer wraps with
`nn.DataParallel` → `RuntimeError: module must have its parameters on cuda:0`. Hide cuda:1:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 3. Uninstall Unsloth before importing trl
If left from a prior warm session, Unsloth monkey-patches `SFTTrainer` globally even without
an explicit `import unsloth`. Causes `RuntimeError: Unsloth: You must specify a formatting_func`.
```python
!pip uninstall -y unsloth unsloth_zoo
shutil.rmtree("/kaggle/working/unsloth_compiled_cache", ignore_errors=True)
```

### 4. No Unsloth on Gemma-4 at all
Unsloth detects Gemma-4 as multimodal → wraps in processor mode → **silent JIT hang** (GPU at 0%,
no error, no progress). Use raw transformers stack instead.

### 5. Use float16, NOT bfloat16 (T4 is CC 7.5, no native bf16)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # NOT bfloat16
    bnb_4bit_use_double_quant=True,
)
```

### 6. Use a regex anchored to `language_model.layers` for LoRA
**SUPERSEDED 2026-04-27.** The original advice — `target_modules=["q_proj.linear", ...]` —
matches only Gemma-4-E2B-it's `vision_tower` and `audio_tower`, NOT the language model
decoder layers, because only the towers wrap projections in `Gemma4ClippableLinear`
(which has a `.linear` sub-module). The language model exposes q_proj/k_proj/etc as
direct `nn.Linear`. The original advice silently produced no-op adapters from v1
through v10 (224 vision + 72 audio + 0 language LoRA tensors, every `lora_B`=0.0).
See `notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md` for the audit.

Correct pattern (verified locally with `init_empty_weights` + PEFT: 245 LoRA
modules, 100% language, 0 towers):
```python
target_modules = r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj$"
```

PEFT accepts `target_modules` as a regex string; the anchored `$` and explicit
`language_model.layers.N` prefix exclude the towers.

Targeting bare `q_proj` (without the regex anchor) still raises `ValueError`
because PEFT can't wrap the outer `Gemma4ClippableLinear` class on the towers
even when matched. The regex bypass works because it skips them entirely.

The training script also adds a post-save sanity gate that fails loudly if the
adapter has zero language-model LoRA tensors or all `lora_B = 0.0`, so this
class of bug can never silently ship again.

### 7. Drop `max_seq_length` from both SFTTrainer and SFTConfig
TRL ≥0.12 moved it to `SFTConfig`; later TRL versions removed it entirely.
Just omit it — TRL infers from `tokenizer.model_max_length`.

### 8. Use `processing_class=`, not `tokenizer=`, in SFTTrainer
TRL ≥0.16 removed the `tokenizer` kwarg. Use `processing_class=tokenizer`.

### 9. Use `dtype=`, not `torch_dtype=`, in `from_pretrained`
transformers 4.51 deprecated `torch_dtype`. Use `dtype=torch.float16`.

### 10. Disable AMP fp16 (QLoRA showstopper)
`fp16=True` in `SFTConfig` triggers `AssertionError: No inf checks were recorded`.
LoRA params (fp32) bypass `GradScaler` hooks. The model already runs in fp16 via
`bnb_4bit_compute_dtype`; AMP is not needed.
```python
training_args = SFTConfig(..., fp16=False, bf16=False, ...)
```

### 11. Call `enable_input_require_grads()` AFTER `get_peft_model()`
**Root cause of v1/v2 zero-learning:** calling `enable_input_require_grads()` on the base
model before wrapping with PEFT registers a hook on the base model's `forward()`. After
`get_peft_model()`, the PEFT wrapper overrides `forward()` and the hook is lost. Result:
`grad_norm=0.0` every step — LoRA has 22M trainable params but receives zero gradients.
```python
# WRONG (v1 bug):
model.enable_input_require_grads()   # hook registered on base model
model = get_peft_model(model, ...)   # PEFT overrides forward → hook lost

# CORRECT:
model = get_peft_model(model, ...)
model.enable_input_require_grads()   # hook registered on PEFT-wrapped model
```

### 12. Use `gradient_checkpointing_kwargs={"use_reentrant": False}`
`gradient_checkpointing=True` with default `use_reentrant=True` breaks gradient flow
through frozen base layers to LoRA adapters. `use_reentrant=False` is the PEFT-recommended
mode for QLoRA + gradient checkpointing.
```python
training_args = SFTConfig(
    ...
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ...
)
```

## Kaggle quota

- Weekly GPU quota: 45 hours (free tier). Resets Monday 00:00 UTC.
- P100 won't run bitsandbytes 4-bit (CC 6.0 < 7.5 minimum). Must use T4.
- `kernel-metadata.json` sets `machine_shape: "NvidiaTeslaT4"` — verify in Kaggle UI before Run All.
- `kaggle kernels push` commits the script but auto-run is subject to quota.
- Quota exhausted mid-session? Push lands; run is deferred until Monday reset.

## Memory budget (single T4, 16 GiB)

| Component | VRAM |
|---|---|
| 4-bit base (Gemma-4-E2B) | ~3 GiB |
| LoRA r=64 + fp32 AdamW state | ~1.2 GiB |
| Activations (batch 1, seq ~800, grad_ckpt=True) | ~3–5 GiB |
| CUDA context + buffers | ~1.5 GiB |
| **Peak** | **~9–11 GiB — comfortable** |

## v1/v2 run history

| Version | Kernel | Error | Fix |
|---|---|---|---|
| v1 | kernel v2 | `TypeError: SFTConfig.__init__() got unexpected kwarg 'max_seq_length'` | Fix #7: removed `max_seq_length` from SFTConfig |
| v2 | kernel v2 | `loss=3.74, grad_norm=0.0 every step` — zero learning | Fix #11 + #12: reorder enable_input_require_grads, use_reentrant=False |
| **v2 (fixed)** | **kernel v3** | **Pushed; awaiting quota reset (2026-04-28)** | — |

## Files

- Training script: `notebooks/kaggle-simsat-gemma4-v1/simsat_gemma4_v1_training.py`
- Build script: `notebooks/kaggle-simsat-gemma4-v1/build_notebook.py` (12 validation checks)
- Push script: `notebooks/kaggle-simsat-gemma4-v1/push.py`
- Kernel slug: `benhaslam/simsat-gemma4-v1-training`
- Dataset: `benhaslam/simsat-gemma4-v1` (294 ChatML training rows)
- v1 results: `kaggle_output/simsat_gemma4_v1_summary.json`

## Things tried that DIDN'T work

- Unsloth on Gemma-4 — multimodal detector causes JIT hang
- `device_map="auto"` on T4×2 — fights HF Trainer DataParallel
- `bf16=True` on T4 — no native bf16, emulated, causes memory pressure
- `fp16=True` with QLoRA — GradScaler assertion (Fix #10)
- `max_seq_length` in `SFTConfig` — removed in newer TRL (Fix #7)
- `enable_input_require_grads()` before `get_peft_model()` — hook lost after PEFT wrap, grad_norm=0.0 (Fix #11)
- `gradient_checkpointing=True` without `use_reentrant=False` — breaks gradient flow to LoRA (Fix #12)
