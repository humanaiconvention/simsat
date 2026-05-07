Note: SigLIP weights (~370 MB) will be downloaded on first run.
Set RUN_HF_TILE_ENCODER_TESTS=1 to suppress this message.

Loading tile encoder: google/siglip-base-patch16-224 on cuda ...
Encoder ready: embed_dim=768  load_time=12.4s
Loading BC head: weights\muzero\stage2_bc_seed13_lfm\policy_head.pt
  BC head ready (action vocab: ['accept', 'defer', 'refine', 'skip'])

## LFM Track — Maritime Chokepoints

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 25 |
| Encoder latency (mean / p95) | 12.4 ms / 13.3 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 18  (72%) |
| Action: skip   | 7   (28%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 26 |
| Encoder latency (mean / p95) | 15.1 ms / 19.6 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 18  (69%) |
| Action: skip   | 8   (31%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 33 |
| Encoder latency (mean / p95) | 14.9 ms / 18.2 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 25  (76%) |
| Action: skip   | 8   (24%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |

## LFM Track — All Packs

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes | 84 |
| Encoder latency (mean / p95) | 14.1 ms / 17.0 ms |
| Action: accept | 0 (0%) |
| Action: refine | 0 (0%) |
| Action: defer  | 61  (73%) |
| Action: skip   | 23   (27%) |
| Materialization yield | - |
| Mean episode reward | -0.0100 |
