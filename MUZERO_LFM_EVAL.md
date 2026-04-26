# LFM Track Eval — MuZero + SigLIP Tile Encoder

Pipeline: `Sentinel PNG → SigLIP-base (768-dim) → SimSatEnv → trust action → reward`  
Policy: stored VLA `recommended_action` from the 86-trace corpus.  
LFM2.5-VL encoder slot: one-line `model_id` swap in `build_encoder("lfm2vl", model_id=...)`.  

## Maritime Chokepoints

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes (matched tiles) | 22 |
| Encoder latency mean / p95 | 17.4 ms / 20.9 ms |
| accept | 6 (27%) |
| refine | 15 (68%) |
| defer  | 0 (0%) |
| skip   | 1 (5%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0441 ± 0.0233 |

## Disaster / Weather

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes (matched tiles) | 23 |
| Encoder latency mean / p95 | 20.4 ms / 29.8 ms |
| accept | 3 (13%) |
| refine | 16 (70%) |
| defer  | 1 (4%) |
| skip   | 3 (13%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0448 ± 0.0230 |

## Urban Coastal Ambiguity

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes (matched tiles) | 30 |
| Encoder latency mean / p95 | 17.8 ms / 22.1 ms |
| accept | 4 (13%) |
| refine | 20 (67%) |
| defer  | 3 (10%) |
| skip   | 3 (10%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0433 ± 0.0236 |

## All Packs

| Metric | Value |
|---|---|
| Encoder | `google/siglip-base-patch16-224` |
| Embed dim | 768 |
| Episodes (matched tiles) | 75 |
| Encoder latency mean / p95 | 18.5 ms / 24.3 ms |
| accept | 13 (17%) |
| refine | 51 (68%) |
| defer  | 4 (5%) |
| skip   | 7 (9%) |
| Materialization yield | 0.00 |
| Mean episode reward | -0.0441 ± 0.0233 |
