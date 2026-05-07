Note: SigLIP weights (~370 MB) will be downloaded on first run.
Set RUN_HF_TILE_ENCODER_TESTS=1 to suppress this message.

Loading tile encoder: google/siglip-base-patch16-224 on cuda ...
Encoder ready: embed_dim=768  load_time=13.0s
Loading BC head: weights\muzero\stage2_bc_seed23_lfm\policy_head.pt
  BC head ready (action vocab: ['accept', 'defer', 'refine', 'skip'])

## LFM Track — Maritime Chokepoints

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 25 |
| Encoder latency (mean / p95) | 23.1 ms / 38.1 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 15  (60%) |
| Action: skip   | 10   (40%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 26 |
| Encoder latency (mean / p95) | 24.7 ms / 31.5 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 23  (88%) |
| Action: skip   | 3   (12%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 33 |
| Encoder latency (mean / p95) | 32.9 ms / 50.9 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 24  (73%) |
| Action: skip   | 9   (27%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — All Packs

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 84 |
| Encoder latency (mean / p95) | 26.9 ms / 40.2 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 62  (74%) |
| Action: skip   | 22   (26%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |
