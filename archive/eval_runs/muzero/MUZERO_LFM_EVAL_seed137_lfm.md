Note: SigLIP weights (~370 MB) will be downloaded on first run.
Set RUN_HF_TILE_ENCODER_TESTS=1 to suppress this message.

Loading tile encoder: google/siglip-base-patch16-224 on cuda ...
Encoder ready: embed_dim=768  load_time=17.8s
Loading BC head: weights\muzero\stage2_bc_seed137_lfm\policy_head.pt
  BC head ready (action vocab: ['accept', 'defer', 'refine', 'skip'])

## LFM Track — Maritime Chokepoints

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 25 |
| Encoder latency (mean / p95) | 22.3 ms / 31.9 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 20  (80%) |
| Action: skip   | 5   (20%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 26 |
| Encoder latency (mean / p95) | 19.8 ms / 24.8 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 26  (100%) |
| Action: skip   | 0   (0%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 33 |
| Encoder latency (mean / p95) | 17.9 ms / 20.2 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 32  (97%) |
| Action: skip   | 1   (3%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — All Packs

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 84 |
| Encoder latency (mean / p95) | 20.0 ms / 25.6 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 78  (93%) |
| Action: skip   | 6   (7%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |
