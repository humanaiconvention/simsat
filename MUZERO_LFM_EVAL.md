# LFM Track Eval — MuZero + LFM2.5-VL-450M Tile Encoder

Pipeline: `Sentinel PNG → LFM2.5-VL-450M (768-dim) → SimSatEnv → trust action → reward`
Encoder swap: change `TILE_ENCODER_MODEL` env var or pass `--model-id` to
`build_encoder("lfm2vl", ...)`. Policy swap: `--policy {playback,bc}` on
`scripts/muzero_lfm_eval.py`.

## Playback policy (stored VLA `recommended_action`)

This is the encoder-decorative baseline: the encoder runs (latency captured)
but the action comes from the stored VLA assessment, so reward is independent
of the encoder choice.

### Maritime Chokepoints

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

### Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `LiquidAI/LFM2.5-VL-450M` |
| Embed dim | 768 |
| Episodes (matched tiles) | 23 |
| Encoder latency mean / p95 | 76.5 ms / 79.1 ms |
| accept | 3 (13%) |
| refine | 16 (70%) |
| defer  | 1 (4%) |
| skip   | 3 (13%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0448 ± 0.0230 |

### Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `LiquidAI/LFM2.5-VL-450M` |
| Embed dim | 768 |
| Episodes (matched tiles) | 30 |
| Encoder latency mean / p95 | 75.0 ms / 78.0 ms |
| accept | 4 (13%) |
| refine | 20 (67%) |
| defer  | 3 (10%) |
| skip   | 3 (10%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0433 ± 0.0236 |

### All Packs (playback)

| Metric | Value |
|---|---|
| Episodes | 75 |
| Encoder latency mean / p95 | 75.2 ms / 77.3 ms |
| accept | 13 (17%) |
| refine | 51 (68%) |
| defer  | 4  (5%) |
| skip   | 7  (9%) |
| Mean episode reward | -0.0441 ± 0.0233 |

## BC policy (Stage 1 head)

`--policy bc` loads `weights/muzero/stage1_bc/policy_head.pt` and predicts
each action from the LFM-450M embedding via the trained head. **First
end-to-end Liquid Track measurement where the encoder embedding actually
drives the reward.**

### All Packs (BC)

| Metric | Value |
|---|---|
| Episodes | 75 |
| Encoder latency mean / p95 | 74.1 ms / 75.4 ms |
| accept | 0 (0%) |
| refine | 75 (100%) |
| defer  | 0 (0%) |
| skip   | 0 (0%) |
| Mean episode reward | -0.0600 ± 0.0000 |

### Lesson — class collapse on borderline cases

The BC head trained to **val_acc 0.800** on a 15-sample stratified split
(refine 10/10, accept 2/3; defer/skip have N=1 each). When deployed on
the full 75-trace eval, predictions collapse to refine for every case.
Diagnostic on individual traces shows softmax outputs cluster around
`[accept=0.43, refine=0.47, defer=0.03, skip=0.06]` — the head learned
the right relative ordering but the **margin between accept and refine
is too thin for argmax to break the majority-class prior**.

Two reinforcing causes:
- **Corpus class imbalance**: 64% refine, 23% accept, 13% defer+skip.
  Standard cross-entropy on imbalanced data biases logits toward the
  majority class — the head's "uncertain" output defaults to refine.
- **Best-state selection by val_acc**: the saved checkpoint was the
  epoch with the highest 15-sample val accuracy (epoch 3). Later
  epochs had higher train accuracy (0.917) and might have produced
  sharper margins on minority classes, but `early_stopping_by_val`
  picked the safer choice.

This is exactly the gap **Stage 2 (scenario-pack augmentation)** is
meant to close. With ~3-5x more (tile, action) pairs from per-pack
synthetic perturbation — especially of accept/defer/skip cases — the
imbalance softens and the BC head's margins widen. Stage 1 numbers
above are the legitimate baseline to beat.

## Variant comparison — encoder-only latency (RTX 2080)

`benhaslam` BEAST measurements over 10 synthetic 64×64 tiles after warmup:

| Model | embed_dim | Mean ms/tile | p95 ms/tile | Notes |
|---|---|---|---|---|
| `google/siglip-base-patch16-224` (substitute) | 768 | ~18 | ~25 | Prior baseline; offline-safe default of `HFVisionTowerEncoder()` |
| `LiquidAI/LFM2.5-VL-450M` | 768 | 69 | 78 | Default of `build_encoder("lfm2vl")`; recommended for the on-orbit framing |
| `LiquidAI/LFM2.5-VL-1.6B` | 1152 | 246 | 248 | Matches Liquid AI's "sub-250ms edge inference" claim; opt-in via `model_id=` |

## Stage 1 BC pretrain — training run details (2026-04-27)

`scripts/muzero_stage1_pretrain.py` trains a small MLP policy head on top
of LFM2.5-VL-450M embeddings, supervised by the per-trace label
(`operator_action` when `label_source == "operator_review"`, else the
stored `assessment.recommended_action`). First stage of the three-stage
Liquid Track training chain (pretrain → scenario-pack fine-tune → two-scope TTT).

Training: 75 (trace, label, asset) rows (33 reviewed + 42 simulated),
stratified 80/20 split, AdamW, 80 epochs (~30s on RTX 2080 after encoding):

| Metric | Value |
|---|---|
| embed_dim | 768 (LFM2.5-VL-450M vision tower) |
| Train accuracy | 0.917 |
| Best val accuracy | **0.800** (12/15 at epoch 3) |
| Val per-action: accept | 2/3 (0.667) |
| Val per-action: refine | 10/10 (1.000) |
| Val per-action: defer | 0/1 (corpus has 3 defer total) |
| Val per-action: skip | 0/1 (corpus has 7 skip total) |

Saved at `weights/muzero/stage1_bc/policy_head.pt` (gitignored).
