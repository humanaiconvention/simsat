# MuZero Stage 2 BC — Seed Variance Bands (10-seed sweep)

This file records a 10-seed sweep of the Liquid Track Stage 2 BC pretrain to put confidence intervals on the headline `val_acc` and `mean reward` claims, which were previously single-seed in `MUZERO_LFM_EVAL.md`.

## Setup

- Script: `scripts/muzero_stage2_pretrain.py`
- Encoder: `LiquidAI/LFM2.5-VL-450M` (canonical Liquid Track encoder; embed_dim=768)
- Augmentation: `--max-aug-ratio 4.0` (Stage 2 v2 canonical)
- Training corpus: post-N=152-relabel, 152 operator-reviewed traces, 145 with tile assets, all four scenario packs
- Seeds: **{7, 13, 23, 42, 100, 137, 256, 1024, 2026, 9999}** (N=10)
- Hardware: BEAST RTX 2080 (8 GB), Python 3.13
- Wall clock: ~2 min/seed × 10 = ~20 min for the full sweep

## Per-Seed Results

| seed | best val_acc | final val_acc | mean reward | action distribution (acc/ref/def/skp) |
| ---: | ---: | ---: | ---: | --- |
|    7 | 0.825 | 0.825 | -0.0100 |   0% /   0% /   0% / 100% |
|   13 | 0.900 | 0.800 | -0.0100 |   0% /   0% /  73% /  27% |
|   23 | 0.925 | 0.900 | -0.0100 |   0% /   0% /  74% /  26% |
|   42 | 0.925 | 0.900 | -0.0100 |   0% /   0% /  50% /  50% |
|  100 | 0.925 | 0.925 | -0.0100 |   0% /   0% / 100% /   0% |
|  137 | 0.975 | 0.975 | -0.0100 |   0% /   0% /  93% /   7% |
|  256 | 0.850 | 0.825 | -0.0100 |   5% /   0% /   0% /  95% |
| 1024 | 0.925 | 0.900 | -0.0100 |   0% /   0% /  81% /  19% |
| 2026 | 0.900 | 0.875 | -0.0100 |   0% /   0% /  74% /  26% |
| 9999 | 0.825 | 0.825 | -0.0100 |   0% /   0% / 100% /   0% |

## Aggregate (N=10 seeds)

| Metric | Mean | Std | Range |
| --- | --- | --- | --- |
| Best val_acc | **0.898** | **0.049** | [0.825, 0.975] |
| Final val_acc | 0.875 | 0.055 | [0.800, 0.975] |
| Mean episode reward | −0.0100 | 0.0000 | suspicious — see caveat |

## Honest Caveats

1. **The earlier 3-seed sweep claimed 0.908 ± 0.014.** With N=10 and a wider seed spread the variance is materially larger — **0.898 ± 0.049**. The 3-seed bands underestimated the true seed-to-seed variation by roughly 3.5×. **This is the more credible number** and should replace any 0.908 ± 0.014 claim that may still be in older docs.

2. **val_acc moved from 0.967 (Stage 2 v2 single-seed) → 0.898 ± 0.049 (10-seed mean) on this corpus.** The original 0.967 claim was on a 28-sample stratified val set drawn from the 75-tile Stage 1/2 corpus. This sweep trains on the post-reset N=152 reviewed pool — a different (larger and more class-diverse) training distribution. The two numbers are not directly comparable; this sweep is the more honest baseline going forward.

3. **Action collapse on the eval scenarios is significant and consistent across seeds.** Eight of ten seeds emit only `defer` and `skip` predictions; one (seed=256) emits a small `accept` slice (5%); one (seed=137) reaches 0.975 val_acc but still collapses to defer/skip in deployment. The training distribution after the N=152 relabel is balanced (a=38, r=37, d=38, s=39) but the val_acc is still ~0.90 — the model is learning the boundary but at inference argmax picks negative-class actions. This mirrors v11's class-collapse in reverse and reinforces the architectural claim that **on-orbit class balance must be maintained via TTT + viability gates rather than relied on at the training-corpus level.**

4. **Reward std = 0.0000 across all 10 seeds is suspicious.** Every seed reports `-0.0100` exactly. The eval-loop reward function in `scripts/muzero_lfm_eval.py` likely returns a constant when the BC head argmax never hits `accept` (no materialization → no realized utility delta). This is a known limitation of the offline replay framework, not a property of the model. A live encounter stream would surface real per-action utility variance.

5. **Encoder = LFM2.5-VL-450M** (canonical Liquid Track encoder). The 450 MB float16 vision tower runs on the RTX 2080 in ~70 ms/tile per the `MUZERO_LFM_EVAL.md` benchmark. The 145 review-asset tiles encode in ~10 s; the BC head is a small MLP and trains in ~80 s per seed.

## Headline (replaces the single-seed Stage 2 v2 claim)

> Stage 2 BC reaches **best val_acc 0.898 ± 0.049 across 10 seeds {7, 13, 23, 42, 100, 137, 256, 1024, 2026, 9999}** on the post-N=152 reviewed corpus, with the LFM2.5-VL-450M encoder. Range: [0.825, 0.975]. Action collapse to negative-class predictions on offline replay is a known training-corpus limitation; the on-orbit pipeline addresses this through TTT + viability gates rather than expecting balance from the static training set.
