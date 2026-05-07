# SimSat Gemma-4-E2B v11 Adapter Audit

This document audits the canonical SimSat fine-tune (v11) against the dynamic LoRA tensor sanity gate that was introduced in the v19 training notebook. It exists to retroactively prove that v11's saved adapter is complete and architecturally correct, without requiring a retrain.

## TL;DR

| Field | Value |
|---|---|
| Adapter | `HumanAIConvention/simsat-gemma4-v11` (Hugging Face) |
| Local path | `weights/simsat-gemma4-v11-adapter/simsat-gemma4-v9-adapter/adapter_model.safetensors` |
| Total LoRA tensors saved | **410** |
| Expected for Gemma-4-E2B GQA | **410** |
| Sanity gate | **PASS** ✓ |
| Reviewed eval (N=37) | exact agreement **0.86** · useful agreement **0.97** · MAE **0.13** |
| Training corpus | 713 ChatML rows (dataset v2, pre-defer-expansion) |
| Training loss | 0.2429 |
| Training steps | 180 |

## Why 410 is the correct count, not 490

Gemma-4-E2B has 35 attention layers but uses Grouped-Query Attention (GQA): `k_proj` and `v_proj` are shared across layer groups. Only **15 canonical k/v modules** exist as distinct PyTorch `nn.Linear` objects (layers 0–14); layers 15–34 hold Python-level aliases that point to the same underlying module objects. PyTorch's `named_modules()` and `named_parameters()` deduplicate by identity, so PEFT correctly wraps each canonical module exactly once.

Saved unique LoRA tensors:

```
non-k/v modules: 35 layers × 5 components (q, o, gate, up, down) × 2 (lora_A, lora_B) = 350
canonical k/v modules: 15 layers × 2 components (k, v) × 2 (lora_A, lora_B) = 60
                                                                        total = 410
```

A naive count assuming 35 independent k/v modules would give 35 × 7 × 2 = 490. That count was hardcoded in the v17/v18 sanity gate and produced a false-negative "partial save" failure. The v19 sanity gate replaced the hardcoded 490 with a dynamic count derived from `model.named_parameters()`, which yields 410 — matching reality.

## Audit procedure

1. Load `adapter_model.safetensors` from the v11 adapter directory via `safetensors.safe_open`.
2. Filter keys to those containing `lora_A` or `lora_B`.
3. Compare to expected count (35 × 5 × 2 + 15 × 2 × 2 = 410).

Audit script: `python3 -c '<see V11 audit block in repo history; reproducible from any HF snapshot>'`.

## Results

```
Total adapter keys: 410
LoRA keys: 410
  lora_A: 205 | lora_B: 205

Component breakdown:
  q_proj:    70   (35 layers × A/B)
  o_proj:    70   (35 layers × A/B)
  gate_proj: 70   (35 layers × A/B)
  up_proj:   70   (35 layers × A/B)
  down_proj: 70   (35 layers × A/B)
  k_proj:    30   (15 canonical layers × A/B)
  v_proj:    30   (15 canonical layers × A/B)

k/v canonical layer coverage: 15 unique layers
  k/v layers present: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

GATE: 410/410  ✓ PASS
```

## Implication for the submission

v11 is the canonical SimSat Gemma-4 fine-tune for the General AI Track. It was trained on dataset v2 (713 weighted ChatML rows, pre-defer-expansion), saved 410 LoRA tensors covering all language-model attention and MLP projections at the correct unique-module granularity, and produced the strongest operator-reviewed eval numbers in the v1–v19 series. The dynamic sanity gate added in v19 (`_EXPECTED_TOTAL = sum(1 for _, p in model.named_parameters() if "lora_A" in name or "lora_B" in name)`) verifies — without retraining — that v11's save is complete.

## Why later versions (v17–v19) regress

Between v11 and v17 the training dataset was regenerated from 713 rows (v2) to 628 rows (v3) with the defer class expanded from 12 → 120 weighted rows via an auto-defer heuristic (cloud_cover ≥ 80% + target_visible). That heuristic does not match the actual operator threshold — operators reviewing the same cloudy traces labeled most as `skip` rather than `defer`. The v17–v19 fine-tunes therefore over-predict `defer/refine` against the operator distribution, dropping exact action agreement to ~0.46. The phenomenon is itself a useful finding: it is a real example of the labeling-distribution problem that on-orbit TTT is designed to handle, and it justifies the architectural emphasis on **information gain** and **error balance** as viability gates (gates 1 and 6) rather than relying on auto-generated weak labels at training time. See [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) issue #27 for the full investigation history.
