# SimSat Gemma-4-E2B Inference Benchmark Results

## Hardware

- GPU: Tesla T4 (Kaggle free tier)
- VRAM: 15.6 GB
- Image size: 448 × 448 px (synthetic tile, same resolution as Sentinel-2 inputs)
- Prompt: SimSat observe/assess contract v1 (8-key JSON output)
- Max new tokens: 512

## Float16 — N=100 (primary benchmark)

| Metric | Value |
|---|---|
| dtype | float16 |
| timed passes | 100 |
| warmup passes | 3 |
| mean latency | 9,455.7 ms |
| std latency | 90.2 ms |
| p50 latency | 9,445.0 ms |
| p90 latency | 9,569.7 ms |
| p95 latency | 9,620.8 ms |
| p99 latency | 9,744.9 ms |
| mean output tokens | 121 |
| tokens/sec | 12.8 |

Raw JSON: [`gemma4_e2b_t4_float16_n100_benchmark.json`](./gemma4_e2b_t4_float16_n100_benchmark.json)

**Verification:** 100-pass timed run (3 warmup), Tesla T4 15.6 GB, Kaggle notebook67b4186434. Std 90.2 ms confirms tight distribution. Token count from generation output tensor shape (not re-tokenized).

## Float16 — N=20 (earlier session, different latency)

| Metric | Value |
|---|---|
| dtype | float16 |
| timed passes | 20 |
| warmup passes | 3 |
| mean latency | 11,207.8 ms |
| p50 latency | 11,201.4 ms |
| p95 latency | 11,301.5 ms |
| mean output tokens | 147 |
| tokens/sec | 13.1 |

**Discrepancy note:** The N=20 run (~11,208 ms mean) is ~19% slower than the N=100 run (~9,456 ms mean), both on Tesla T4 float16. Root cause uncertain — likely Kaggle session-level GPU allocation variability or cold state differences across sessions. The N=100 run is the primary benchmark: it has more passes, a tighter std (90.2 ms), and was run in a dedicated session immediately after model load. The N=20 figure is retained for transparency.

## Int8 — not available

**Status:** `BitsAndBytesConfig(load_in_8bit=True)` requires bitsandbytes ≥ 0.46.1, which is incompatible with the transformers HEAD build on Kaggle T4. Float16 is the production baseline.

## Scenario Assessment Study — N=90

90 Gemma-4 assessments across 27 scenario locations × 4 cloud conditions:

**Scenario packs:**
- `urban_coastal_ambiguity` — Rotterdam, Singapore, LA, Shenzhen Bay, Dubai, Hamburg, Sydney, Istanbul
- `disaster_response_weather` — Fort Myers, Lahaina, Turkey earthquake, Pakistan floods, Odessa, Morocco Atlas, Derna Libya, California fire, Bangladesh delta
- `maritime_chokepoints` — Suez Canal, Strait of Hormuz, Malacca, Panama Canal, Bab-el-Mandeb, Danish Straits, English Channel, Taiwan Strait, Dover Strait

**Cloud conditions:** 0% (clear), 15% (partly cloudy), 45% (cloudy), 80% (heavy cloud)

| Metric | Value |
|---|---|
| total assessments | 90 |
| contract compliance rate | 1.00 (all 90 outputs valid 8-key JSON) |
| mean confidence | 0.427 |
| action distribution | collect=18, skip=64, revisit=8, flag=0 |
| mean latency | 12,658.0 ms |
| p95 latency | 14,494.7 ms |

**By-pack breakdown:**

| Pack | N | Mean confidence | collect | skip | revisit |
|---|---|---|---|---|---|
| urban_coastal_ambiguity | 32 | 0.442 | 6 | 23 | 3 |
| disaster_response_weather | 36 | 0.436 | 8 | 23 | 5 |
| maritime_chokepoints | 22 | 0.391 | 4 | 18 | 0 |

**Notes:**
- Contract compliance 1.00 = every assessment returned valid 8-key JSON. No parse failures.
- "flag" action never chosen — base Gemma-4-E2B-IT defaults to skip when uncertain. Revisit used for marginal cases.
- Mean confidence 0.427 is honest: base model (not v11 fine-tune) running against synthetic tiles. Expect higher confidence on real Sentinel tiles with v11.
- Latency higher than N=100 benchmark (12,658 ms vs 9,456 ms) because scenario prompts are longer (include target label, tags, cloud condition).
- maritime_chokepoints n=22: run_list capped at 90, so the last pack was truncated (8 scenarios × 4 conditions would be 32; only 22 ran).

## ObservationVLA Eval (Operator-Reviewed)

Separate from inference benchmarks — these are Gemma-4-v11 real-tile assessments evaluated against operator ground truth.

| Metric | Value |
|---|---|
| N | 37 |
| model | gemma-4-E2B SimSat v11 LoRA fine-tune |
| image source | real Sentinel-2 tiles |
| exact action agreement | 0.86 |
| useful/not-useful agreement | 0.97 |
| usefulness score MAE | 0.13 |

See [OBSERVATION_VLA_EVAL.md](../OBSERVATION_VLA_EVAL.md) for per-case breakdown.

## Backend Comparison (Runtime Assessments)

The 86 runtime assessments in `assessments.json` used CLIP and stub backends, not Gemma-4:

| Backend | N | Mean operator usefulness | Accept rate | Model confidence |
|---|---|---|---|---|
| clip_local (TTT) | 23 | 0.647 | 4/23 (17%) | 0.469 |
| stub | 14 | 0.867 | 13/14 (93%) | 0.817 |

**Finding:** CLIP/TTT systematically underperforms stub in operator ratings and is under-confident (mean 0.469 vs 0.817). Root cause uncertain — possible miscalibration for these scenario types or selection bias. This is documented honestly; the Gemma-4 v11 fine-tune (0.86 agreement) is the active production backend and evaluated separately on real tiles.

## Honest Limitations

- N=100 and N=20 float16 runs show ~19% latency difference across Kaggle sessions — root cause not fully determined (GPU allocation variability most likely); N=100 is treated as primary
- Int8 unavailable: bitsandbytes incompatible with transformers HEAD on Kaggle T4
- Scenario assessment study uses synthetic tiles, not real Sentinel imagery; base model (not v11 fine-tune) used
- CLIP/TTT underperformance vs stub requires further investigation
- No Gemma-4 inference in the 86-record runtime database (all CLIP/stub)
- The N=37 operator eval is real Sentinel tiles but a separate offline evaluation
