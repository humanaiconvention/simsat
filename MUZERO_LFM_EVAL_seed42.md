Note: SigLIP weights (~370 MB) will be downloaded on first run.
Set RUN_HF_TILE_ENCODER_TESTS=1 to suppress this message.

Loading tile encoder: google/siglip-base-patch16-224 on cuda ...
Encoder ready: embed_dim=768  load_time=11.7s
Loading BC head: weights\muzero\stage2_bc_seed42\policy_head.pt
  BC head ready (action vocab: ['accept', 'defer', 'refine', 'skip'])

## LFM Track — Maritime Chokepoints

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 25 |
| Encoder latency (mean / p95) | 13.2 ms / 15.4 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 11  (44%) |
| Action: skip   | 14   (56%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 26 |
| Encoder latency (mean / p95) | 13.2 ms / 16.1 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 12  (46%) |
| Action: skip   | 14   (54%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 33 |
| Encoder latency (mean / p95) | 12.4 ms / 14.3 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 19  (58%) |
| Action: skip   | 14   (42%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — All Packs

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 84 |
| Encoder latency (mean / p95) | 12.9 ms / 15.3 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 42  (50%) |
| Action: skip   | 42   (50%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |
