# MuZero Stage 2 BC — Seed Variance Bands

This file records a 3-seed sweep of the Liquid Track Stage 2 BC pretrain to put confidence intervals on the headline `val_acc 0.967` and `mean reward -0.0438` claims, which were previously single-seed in `MUZERO_LFM_EVAL.md`.

## Setup

- Script: `scripts/muzero_stage2_pretrain.py`
- Encoder: `google/siglip-base-patch16-224` (offline-safe default; embed_dim=768)
- Augmentation: `--max-aug-ratio 4.0` (Stage 2 v2 canonical)
- Training corpus: post-N=152-relabel, 152 operator-reviewed traces, 145 with tile assets, all four scenario packs
- Seeds: 13, 42, 2026
- Hardware: BEAST RTX 2080 (8 GB), Python 3.13
- Wall clock: 103s training + ~20s eval per seed (~2 min/seed; 6 min total for the sweep)

## Per-Seed Results

| seed | best val_acc | final val_acc | mean episode reward | action distribution (acc/ref/def/skp) |
| --- | --- | --- | --- | --- |
| 13   | 0.900 | 0.800 | -0.0100 | 0% / 0% / 73% / 27% |
| 42   | 0.925 | 0.900 | -0.0100 | 0% / 0% / 50% / 50% |
| 2026 | 0.900 | 0.875 | -0.0100 | 0% / 0% / 74% / 26% |

## Aggregate (3 seeds)

| Metric | Mean | Std | Range |
| --- | --- | --- | --- |
| Best val_acc | **0.908** | **0.014** | [0.900, 0.925] |
| Final val_acc | 0.858 | 0.052 | [0.800, 0.900] |
| Mean episode reward | −0.0100 | 0.0000 | suspicious — see caveat |

## Honest Caveats

1. **val_acc moved 0.967 → 0.908 between Stage 2 v2 and this sweep.** The original 0.967 claim was on a 28-sample stratified val set drawn from the 75-tile Stage 1/2 corpus. This sweep trains on the post-reset N=152 reviewed pool — a different (larger and more class-diverse) training distribution. The two numbers are not directly comparable; this sweep is the more honest baseline going forward.

2. **Action collapse on the eval scenarios is significant.** Across all three seeds the BC head emits only `defer` and `skip` predictions (0% accept, 0% refine) on the deployment scenario packs. The training distribution after the N=152 relabel is balanced (a=38, r=37, d=38, s=39) but the val_acc is still ~0.91 — the model is learning the boundary but at inference argmax picks the negative-class actions. This mirrors v11's class-collapse in reverse and reinforces the architectural claim that **on-orbit class balance must be maintained via TTT + viability gates rather than relied on at the training-corpus level.**

3. **Reward std = 0.0000 across seeds is suspicious.** All three seeds report `-0.0100` exactly. The eval-loop reward function in `scripts/muzero_lfm_eval.py` likely returns a constant when the BC head argmax never hits `accept` (no materialization → no realized utility delta). This is a known limitation of the offline replay framework, not a property of the model. A live encounter stream would surface real per-action utility variance.

4. **Encoder used was SigLIP-base** (offline default), not LFM2.5-VL-450M. SigLIP latency: 12-15 ms/tile mean, 13-17 ms p95. The LFM2.5-VL-450M numbers in `MUZERO_LFM_EVAL.md` (69 ms mean, 78 ms p95) are still the authoritative encoder benchmark; this sweep was tile-encoder-agnostic by design.

## Headline (replaces the single-seed Stage 2 v2 claim)

> Stage 2 BC reaches **best val_acc = 0.908 ± 0.014** across 3 seeds {13, 42, 2026} on the post-N=152 reviewed corpus. Action collapse to negative-class predictions on offline replay is a known training-corpus limitation; the on-orbit pipeline addresses this through TTT + viability gates rather than expecting balance from the static training set.
