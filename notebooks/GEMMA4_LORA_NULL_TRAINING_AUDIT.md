# Gemma-4 SimSat LoRA — Null-Training Audit (2026-04-27)

## Headline

Every SimSat Gemma-4-E2B-it adapter from v1 through v10 has trained
**zero language-model LoRA parameters**. The `target_modules` pattern in
the training script matched the multimodal towers only. All `lora_B`
tensors are exactly 0.0 (PEFT zero init), so the LoRA contributes
nothing to forward. **All SimSat eval numbers to date measure stock
`google/gemma-4-E2B-it`, not a fine-tune.**

## Evidence

Audit of `D:\SimSat\weights\simsat-gemma4-v10-adapter\adapter_model.safetensors`
and `D:\diloco_lab\state\global_round_000000\adapter_model.safetensors`
(seeded from v10):

| Bucket | Tensor count |
|---|---|
| `vision_tower.*.lora_*` | 224 |
| `audio_tower.*.lora_*` | 72 |
| **language model `*.lora_*`** | **0** |
| Total | 296 |

| Param family | Non-zero count |
|---|---|
| `lora_A` (Kaiming init) | 148 |
| **`lora_B` (PEFT zero init)** | **0** |

`lora_B = 0.0` everywhere means `lora_B @ lora_A @ x = 0` always — LoRA
is identity. v3+ instrumentation (kernel v5, run 2026-04-27) confirmed
the gradient symptom directly:

```json
{
  "lora_grads_present_after_first_backward": 0,
  "lora_grads_nonzero_after_first_backward": 0,
  "lora_changed_in_memory": 0,
  "loss_before": 2.274883158504963,
  "loss_after":  2.274883158504963,
  "verdict": "GRADIENTS_NEVER_REACH_LORA"
}
```

After `loss.backward()` on the first batch, **no LoRA parameter received
any gradient** because the text-only ChatML training input never traverses
the vision or audio towers — and those are the only places LoRA was
attached.

## Root cause

`notebooks/kaggle-simsat-gemma4-v1/GEMMA4_KAGGLE_NOTES.md` Fix #6 set:

```python
target_modules=[
    "q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear",
    "gate_proj.linear", "up_proj.linear", "down_proj.linear",
]
```

The `.linear` suffix was added because bare `"q_proj"` raised `ValueError`
(PEFT can't wrap the outer `Gemma4ClippableLinear` class). But in
Gemma-4-E2B-it, **`Gemma4ClippableLinear` lives on the vision_tower and
audio_tower modules, NOT on the language model decoder layers**. So the
suffixed pattern matches only multimodal towers; the language model gets
no LoRA.

## Implications

1. **Every SimSat eval number to date is stock Gemma-4-E2B-it** —
   v9 → v10's MAE=0.16 / 0.21, bucketed=0.57 / 0.51, exact=0.41, and
   the accept-bias we were trying to fix all describe the BASE model,
   not a fine-tune.

2. **REFINE_BOOST cannot fix this.** The model never learns from the
   training data because no parameters are updated. Pushing the dataset
   further toward refine examples is a no-op.

3. **DiLoCo continuation cannot fix this either.** The seed inherits
   the same broken layout; rounds N+1 just produce more zero-deltas.

4. **The bias we observed lives in `google/gemma-4-E2B-it` itself.**
   It's not a SimSat training artifact.

## What to do next

To get an actual SimSat fine-tune, we need a `target_modules` pattern
that catches the **language model decoder layers** in Gemma-4-E2B-it.
The right next step is a one-shot Kaggle cell that loads the base model
and dumps `for n, _ in model.named_modules(): print(n)` filtered to
likely target patterns, so we can see the actual module path that
contains the language-model q/k/v/o/up/down/gate projections.

Likely candidates to test (in order):

```python
# 1. Use the regex form so we exclude towers explicitly:
target_modules = "model\\.layers\\..*\\.(q|k|v|o|up|down|gate)_proj.*"

# 2. PEFT's all-linear shortcut, then exclude towers:
target_modules = "all-linear"
modules_to_save = []  # avoid saving anything else

# 3. After inspecting named_modules, the explicit list — most likely
#    something like: ["language_model.model.layers.X.self_attn.q_proj", ...]
```

Once the right pattern is identified, **a fresh training run from
google/gemma-4-E2B-it** is needed — the v10 seed cannot be salvaged
because it was initialized only on towers; the language model has no
LoRA modules to inherit weights into.

## Files involved (not modified by this audit)

- `D:\SimSat\weights\simsat-gemma4-v10-adapter\adapter_model.safetensors` — null-trained
- `D:\diloco_lab\state\global_round_000000\adapter_model.safetensors` — null-trained (copy of v10)
- `D:\SimSat\notebooks\kaggle-simsat-gemma4-v1\simsat_gemma4_v1_training.py` — produces the broken target_modules
- `D:\diloco_lab\kaggle\continue_gemma4_adapter.py` — continuation runner; correctly inherits target_modules from seed (so any fix at the SimSat training script propagates)

## Audit run reference

- Kernel: `benhaslam/simsat-diloco-round-0-learner` v5
- Diagnostic: `D:\SimSat\weights\diloco-round0-continued\diloco_continued_adapter\v5_diagnostic.json`
- Notebook: `D:\SimSat\notebooks\kaggle-simsat-diloco-round0\notebook.ipynb`
- Audit commit: see git log on branch `codex/runtime-test-baseline`

---

## Follow-up: v11 partial save (2026-04-27)

After the target_modules fix (commit `12bf0ab`), v11 produced a real fine-tune
but the saved adapter has a structured gap. Per-layer audit of
`D:\SimSat\weights\simsat-gemma4-v11-adapter\simsat-gemma4-v9-adapter\adapter_model.safetensors`:

- **Layers 0-14:** all 7 LoRA modules saved (q, k, v, o, gate, up, down) ✓
- **Layers 15-34:** only 5 saved (q, o, gate, up, down) — **k_proj and v_proj missing** ✗

Total saved: 410 / 490 expected (84%). Local PEFT verification with
`init_empty_weights` + `get_peft_model` matches all 245 expected target
modules across all 35 layers — so config-time matching is fine. The drop
happens at **Kaggle save time**.

Most-likely cause: Gemma-4-E2B has `num_key_value_heads=1` (GQA), and the
loaded 4-bit model calls `tie_weights()` which shares the k_proj/v_proj
parameter pool across some layers. PEFT's `state_dict` dedupes by tensor
identity at save, so the second-half LoRA pairs collapse into the first-half
entries.

Symptom (load-time warning seen in the v11 eval):
```
UserWarning: Found missing adapter keys while loading the checkpoint: [
  'base_model.model.model.language_model.layers.15.self_attn.k_proj.lora_A.default.weight',
  ...
  'base_model.model.model.language_model.layers.34.self_attn.v_proj.lora_B.default.weight',
]
```

PEFT zero-fills the missing keys at load, so the v11 adapter is functional
but partially trained. Eval at N=37 still shows a massive jump over base
Gemma-4-E2B (exact 0.41 → 0.86). The missing k/v on the second half MAY be
load-bearing and a v12 with the missing keys preserved could lift metrics
further — or they may have received zero gradient anyway.

**Cheap diagnostic (one T4 run):** v12 with `peft>=0.20` (which has
better `state_dict` handling for tied parameters) and an in-script
sanity check that fails if any layer is missing k/v LoRA. Compare v12 eval
to v11 to see whether the missing weights mattered.

**Sanity gate** added in `simsat_gemma4_v1_training.py` (commit `12bf0ab`)
catches the no-language-LoRA case but does NOT catch this partial-save case.
A follow-up commit could tighten it to require all 35 layers × 7 modules =
245 LoRA pairs, not just "any language LoRA".
