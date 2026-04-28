# LFM Track Eval — MuZero + LFM2.5-VL-450M Tile Encoder

Pipeline: `Sentinel PNG → LFM2.5-VL-450M (768-dim) → SimSatEnv → trust action → reward`  
Policy: stored VLA `recommended_action` from the 86-trace corpus.  
Encoder swap: change `TILE_ENCODER_MODEL` env var or pass `--model-id` to `build_encoder("lfm2vl", ...)`.  

## Maritime Chokepoints

| Metric | Value |
|---|---|
| Encoder | `LiquidAI/LFM2.5-VL-450M` |
| Embed dim | 768 |
| Episodes (matched tiles) | 22 |
| Encoder latency mean / p95 | 74.6 ms / 77.6 ms |
| accept | 6 (27%) |
| refine | 15 (68%) |
| defer  | 0 (0%) |
| skip   | 1 (5%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0441 ± 0.0233 |

## Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `LiquidAI/LFM2.5-VL-450M` |
| Embed dim | 768 |
| Episodes (matched tiles) | 23 |
| Encoder latency mean / p95 | 73.2 ms / 77.1 ms |
| accept | 3 (13%) |
| refine | 16 (70%) |
| defer  | 1 (4%) |
| skip   | 3 (13%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0448 ± 0.0230 |

## Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `LiquidAI/LFM2.5-VL-450M` |
| Embed dim | 768 |
| Episodes (matched tiles) | 30 |
| Encoder latency mean / p95 | 77.8 ms / 77.3 ms |
| accept | 4 (13%) |
| refine | 20 (67%) |
| defer  | 3 (10%) |
| skip   | 3 (10%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0433 ± 0.0236 |

## All Packs

| Metric | Value |
|---|---|
| Encoder | `LiquidAI/LFM2.5-VL-450M` |
| Embed dim | 768 |
| Episodes (matched tiles) | 75 |
| Encoder latency mean / p95 | 75.2 ms / 77.3 ms |
| accept | 13 (17%) |
| refine | 51 (68%) |
| defer  | 4 (5%) |
| skip   | 7 (9%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0441 ± 0.0233 |

## Variant comparison — encoder-only latency (RTX 2080)

`benhaslam` BEAST measurements over 10 synthetic 64×64 tiles after warmup:

| Model | embed_dim | Mean ms/tile | p95 ms/tile | Notes |
|---|---|---|---|---|
| `google/siglip-base-patch16-224` (substitute) | 768 | ~18 | ~25 | Prior baseline; offline-safe default of `HFVisionTowerEncoder()` |
| `LiquidAI/LFM2.5-VL-450M` | 768 | 69 | 78 | Default of `build_encoder("lfm2vl")`; recommended for the on-orbit framing |
| `LiquidAI/LFM2.5-VL-1.6B` | 1152 | 246 | 248 | Matches Liquid AI's "sub-250ms edge inference" claim; opt-in via `model_id=` |

Action distribution and mean episode reward are identical across encoders
because the eval here uses the stored VLA `recommended_action` as the
policy, not a learned head. The encoder rows above isolate the encode-pass
latency cost — the metric that matters for the on-orbit budget.

## Stage 1 BC pretrain (2026-04-27)

`scripts/muzero_stage1_pretrain.py` trains a small MLP policy head on top
of LFM2.5-VL-450M embeddings, supervised by the per-trace label
(`operator_action` when `label_source == "operator_review"`, else the
stored `assessment.recommended_action`). This is the first stage of the
three-stage Liquid Track training chain (pretrain → scenario-pack
fine-tune → two-scope TTT) and the first time the encoder embedding
actually drives an action prediction in this codebase.

Training: 75 (trace, label, asset) rows (33 reviewed + 42 simulated),
stratified 80/20 split, AdamW, 80 epochs (~30s on RTX 2080 after
encoding):

| Metric | Value |
|---|---|
| embed_dim | 768 (LFM2.5-VL-450M vision tower) |
| Train accuracy | 0.917 |
| Best val accuracy | **0.800** (12/15 at epoch 3) |
| Val per-action: accept | 2/3 (0.667) |
| Val per-action: refine | 10/10 (1.000) |
| Val per-action: defer | 0/1 (corpus only has 3 defer total) |
| Val per-action: skip | 0/1 (corpus only has 7 skip total) |

Saved at `weights/muzero/stage1_bc/policy_head.pt` (gitignored). Val
defer/skip zeros aren't meaningful signal — N=1 each. Train→val gap of
~0.12 reflects modest overfit on the small dataset; Stage 2's scenario
augmentation is the next lever rather than additional Stage 1 tuning.

Re-running `muzero_lfm_eval.py` with this BC head wired in as the
policy (instead of stored-action playback) is the next step that would
break the per-encoder-equivalence above and produce encoder-dependent
rewards.
