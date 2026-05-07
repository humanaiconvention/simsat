# MuZero LFM Track Eval

Two encoder configurations evaluated: **LFM2.5-VL-450M** (primary, Liquid AI production weights) and **SigLIP-base** (offline fallback, local eval). Both use the SimSatEnv offline-replay harness with the Stage 2 BC policy head.

> **Seed variance**: The Stage 2 v2 numbers below are single-seed (seed=42). For confidence intervals across 3 seeds {13, 42, 2026}, see [`MUZERO_SEED_SWEEP.md`](./MUZERO_SEED_SWEEP.md) — **best val_acc 0.908 ± 0.014** on the post-N=152 reviewed corpus.

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
- Defer: 0/75 predictions across all policies. Corpus has only 3 original defer traces. `scripts/build_defer_queue.py` generates a focused 20-candidate review queue to address this.
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
