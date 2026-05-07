Note: SigLIP weights (~370 MB) will be downloaded on first run.
Set RUN_HF_TILE_ENCODER_TESTS=1 to suppress this message.

Loading tile encoder: google/siglip-base-patch16-224 on cuda ...
Encoder ready: embed_dim=768  load_time=12.5s
Loading BC head: weights\muzero\stage2_bc_seed7_lfm\policy_head.pt
  BC head ready (action vocab: ['accept', 'defer', 'refine', 'skip'])

## LFM Track — Maritime Chokepoints

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 25 |
| Encoder latency (mean / p95) | 15.7 ms / 17.7 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 0  (0%) |
| Action: skip   | 25   (100%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 26 |
| Encoder latency (mean / p95) | 15.4 ms / 17.7 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 0  (0%) |
| Action: skip   | 26   (100%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 33 |
| Encoder latency (mean / p95) | 16.1 ms / 19.5 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 0  (0%) |
| Action: skip   | 33   (100%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — All Packs

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 84 |
| Encoder latency (mean / p95) | 15.7 ms / 18.3 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 0  (0%) |
| Action: skip   | 84   (100%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |
