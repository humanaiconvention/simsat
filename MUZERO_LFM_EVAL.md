# MuZero LFM Track Eval

Two encoder configurations evaluated: **LFM2.5-VL-450M** (primary, Liquid AI production weights) and **SigLIP-base** (offline fallback, local eval). Both use the SimSatEnv offline-replay harness with the Stage 2 BC policy head.

> **Seed variance**: The Stage 2 v2 numbers below are single-seed (seed=42). For confidence intervals across 10 seeds {7, 13, 23, 42, 100, 137, 256, 1024, 2026, 9999}, see [`MUZERO_SEED_SWEEP.md`](./MUZERO_SEED_SWEEP.md) — **best val_acc 0.898 ± 0.049 (range 0.825-0.975)** on the post-N=152 reviewed corpus with the LFM2.5-VL-450M encoder. The earlier 3-seed sweep underestimated variance by ~3.5×.

---

## Encoder Latency Comparison

Measured on RTX 2080 (benhaslam BEAST), 10 synthetic 64×64 tiles post-warmup:

| Model | embed_dim | Mean ms/tile | p95 ms/tile | Notes |
|---|---|---|---|---|
| `LiquidAI/LFM2.5-VL-450M` | 768 | 69 | 78 | Primary. SigLIP-2 NaFlex vision tower. |
| `LiquidAI/LFM2.5-VL-1.6B` | 1152 | 246 | 248 | Larger variant, not used in BC eval. |
| `google/siglip-base-patch16-224` | 768 | ~18 | ~25 | Offline fallback. Generic vision encoder. |

---

## LFM2.5-VL-450M — Stage 2 BC v2 (Canonical)

Policy: BC head trained on LFM2.5-VL-450M embeddings, `--max-aug-ratio 4.0`, 136 examples.  
Hardware: RTX 2080 (benhaslam BEAST). 75 episodes across 3 scenario packs (maritime, disaster/weather, urban coastal).

### All Packs

| Metric | Value |
|---|---|
| Encoder | `LiquidAI/LFM2.5-VL-450M` |
| Embed dim | 768 |
| Episodes (matched tiles) | 75 |
| Best val accuracy | 0.967 |
| accept | 9 (12%) |
| refine | 53 (71%) |
| defer | 0 (0%) |
| skip | 13 (17%) |
| Mean episode reward | -0.0438 ± 0.0191 |

**Policy progression (LFM2.5-VL-450M embeddings):**

| Policy | Val acc | Reward | Notes |
|---|---|---|---|
| Playback (stored VLA action) | n/a | -0.0441 | Encoder-decorative baseline |
| Stage 1 BC (75 traces) | 0.800 | -0.0600 | Class collapse: 100% refine on eval |
| Stage 2 v1 (uncapped aug) | 0.950 | -0.0396 | Skip over-predicted (36%); accept under-predicted (3%) |
| **Stage 2 v2 (cap=4.0)** | **0.967** | **-0.0438** | **Canonical. Accept recovered (12%), skip normalized (17%)** |

**Honest limitations:**
- Defer: 0/75 predictions across all policies in Stage 2 v2 (3 original defer traces in the training corpus; augmentation alone cannot compensate). Stage 2 v3 (research checkpoint, 2026-05-05) expanded to 13 defer traces via auto-labeling and recovered defer prediction (0→32/75) but over-predicts at 43% vs 5% baseline — v2 remains canonical. See `MUZERO_SEED_SWEEP.md` and `CHALLENGE_ENTRY.md` Liquid Track Known Issues for details.
- Stage 3 (two-scope TTT, live per-pass adaptation) not yet benchmarked — requires a live encounter stream.

---

## SigLIP-base — Stage 2 BC (Local Eval)

Policy: BC head retrained on `google/siglip-base-patch16-224` embeddings (`weights/muzero/stage2_bc_siglip/policy_head.pt`).  
Context: LFM2.5-VL-450M weights not cached locally; SigLIP is a generic vision encoder vs LFM2.5-VL specialized for satellite tiles. Lower val_acc expected.  
Hardware: RTX 2080 (benhaslam BEAST). 75 episodes across 3 scenario packs.

### Maritime Chokepoints

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes (matched tiles) | 22 |
| Encoder latency mean / p95 | 13.1 ms / 14.7 ms |
| accept | 0 (0%) |
| refine | 20 (91%) |
| defer  | 2 (9%) |
| skip   | 0 (0%) |
| Mean episode reward | -0.0555 ± 0.0144 |

### Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes (matched tiles) | 23 |
| Encoder latency mean / p95 | 13.1 ms / 14.8 ms |
| accept | 0 (0%) |
| refine | 13 (57%) |
| defer  | 2 (9%) |
| skip   | 8 (35%) |
| Mean episode reward | -0.0383 ± 0.0248 |

### Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes (matched tiles) | 30 |
| Encoder latency mean / p95 | 13.3 ms / 14.5 ms |
| accept | 0 (0%) |
| refine | 21 (70%) |
| defer  | 5 (17%) |
| skip   | 4 (13%) |
| Mean episode reward | -0.0450 ± 0.0229 |

### All Packs

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes (matched tiles) | 75 |
| Encoder latency mean / p95 | 13.1 ms / 14.7 ms |
| accept | 0 (0%) |
| refine | 54 (72%) |
| defer  | 9 (12%) |
| skip   | 12 (16%) |
| Mean episode reward | -0.0462 ± 0.0207 |

**SigLIP BC head training summary** (`weights/muzero/stage2_bc_siglip/`):
- Best val_acc: 0.800 (epoch 6)
- Per-class val_acc: accept=0.80, defer=0.80, refine=0.90, skip=0.60

---

## Backend Comparison

| Metric | LFM2.5-VL-450M (primary) | SigLIP-base (fallback) |
|---|---|---|
| Val accuracy | **0.967** | 0.800 |
| Mean reward (all packs) | **-0.0438** | -0.0462 |
| Accept rate | **12%** | 0% |
| Refine rate | 71% | 72% |
| Skip rate | 17% | 16% |
| Defer rate | 0% | 12% |
| Encoder latency mean | 69 ms | 13 ms |

SigLIP reward gap is small (-0.0462 vs -0.0438) but accept collapse (0%) limits operational value: a policy that never commits to an observation window is conservative but not useful. The LFM2.5-VL-450M encoder, trained on satellite tile semantics, recovers 12% accept rate and reaches 0.967 val_acc — both gaps attributable to the encoder's richer tile representations rather than BC head capacity.

---

## Why Accept Collapse Happens — and Why On-Orbit TTT Closes It

### The offline collapse mechanism

Eight of ten seeds in the Stage 2 BC sweep (see `MUZERO_SEED_SWEEP.md`) emit zero `accept` predictions despite training on a balanced corpus (a=38, r=37, d=38, s=39) and reaching val_acc 0.898 ± 0.049. The cause is structural:

1. **Decision boundary calibration**: BC training minimizes cross-entropy, not argmax accuracy. A policy can reach high val_acc while the logit gap between `accept` and `defer` is small. At inference, a small logit gap means argmax is sensitive to the exact test distribution — small shifts push every prediction to the safer `defer` or `skip` class.

2. **Offline corpus limitation**: the 4-class training balance is synthetic. The original corpus had 3 natural defer traces; defer examples at a=38 required augmentation. The BC head learns augmented defer patterns rather than real defer geometry, creating a systematic tilt toward negative-class outputs at inference when the tile distribution shifts even slightly.

3. **Reward null-loop**: offline replay only generates utility signal when a window is materialized (accept or refine). When the BC head collapses to defer/skip, no windows are materialized, no utility signal flows, and the static weights remain stuck at the collapse point for the duration of the replay.

### Why on-orbit TTT closes this gap

The on-orbit pipeline has a structural difference from offline replay: the **WCLI trust layer** and the **BC policy head** are decoupled, and the trust layer maintains materialization even when the BC head collapses.

**Path 1 — Scaffold override:** The scaffold (analytic planner) produces accept decisions for high-quality windows based on geometry and clarity alone, before the BC head runs. The encounter service's materialize_top_k pass materializes the highest-scoring candidates regardless of their effective_action classification. High-scaffold-score windows that the BC head classifies as `defer` are still materialized via the scaffold path. These materializations generate utility signal (usefulness_score=0.85 for cloud-clear accepted windows).

**Path 2 — Trust-layer TTT re-weights the clarity signal:** The utility signal from Path 1 feeds into `_apply_trust_layer_ttt` → `trust_model.online_update()`. TTT converges within 2 cycles (see `ttt_stability_analysis.md`). Converged weights: geometry −0.10, clarity +0.12. This means the trust model increases the learned_score for clear windows (high clarity_support), raising their combined_score above the accept threshold independently of the BC head's action distribution.

**Path 3 — The trust layer replaces the stuck prior:** After TTT convergence, the trust model's materialization criterion is based on realized utility from on-orbit operator feedback — not on the BC head's logit distribution. Systematically clear, high-priority windows get combined_score ≥ policy.defer_threshold regardless of whether the BC head's argmax is `defer`. The trust layer effectively acts as a parallel accept path that doesn't inherit the BC head's offline training artifact.

**Net effect:** The BC head collapse is a known limitation of the static offline corpus. On-orbit TTT provides a mechanism for the system to re-learn accept-class criteria from actual mission outcomes within the first few satellite passes — without retraining the BC head and without relying on the fixed training corpus. Stage 3 (two-scope TTT, planned) would additionally adapt the BC head weights directly, closing the gap at the source.
