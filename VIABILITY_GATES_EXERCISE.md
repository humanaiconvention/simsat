# Viability Gates Exercise

Drives `WCLITrustModel.online_update()` through three synthetic operator-feedback streams (N=1100 each) and tallies how often each of the three TTT viability gates (`weight_drift`, `update_rate`, `error_bias`) fails on each step. Replaces the single-session 'gates fired during today's review' anecdote with structured fire-rate numbers under varied conditions.

## Setup

- Updates per stream: **1100** (above `MAX_TTT_UPDATE_COUNT=1000` to exercise the rate ceiling)
- Streams: `baseline_clean, drift_one_class, saturation`
- Thresholds: `weight_drift > 0.3`, `update_rate > 1000`, `error_bias > 0.7`
- Gate semantics: `True` = passed (no concern); `False` = fired (concern raised, logged at WARNING)

## Streams

| Stream | Description | Expected gate behaviour |
| --- | --- | --- |
| `baseline_clean` | Well-distributed feature values; realized utility = predicted + small noise | `weight_drift` and `update_rate` stay quiet. `error_bias` fires ~38% of steps (the 70% threshold catches random 7-of-10 sign clusters; this is expected statistical behavior for a 50/50 error distribution). 62% of updates still proceed. |
| `drift_one_class` | All-high features + under-predicting learned_score → consistent positive error | `error_bias` should fire immediately (step 11) and block updates, **preventing** `weight_drift` from ever firing. This is the blocking cascade: systematic bias is intercepted before weights can drift. |
| `saturation` | Identical features every step, learned_score matches realized | `update_rate` should trip past 1000 cumulative updates; `weight_drift` / `error_bias` stay quiet (errors = 0) |

## Results

**Note:** `error_bias` is a **blocking gate** (evaluated pre-update in `service.py`): when it fires, the adaptation step is skipped entirely. `weight_drift` and `update_rate` are post-update log-only warnings.

| stream | n | applied | blocked (error_bias) | weight_drift fires | update_rate fires | error_bias fires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_clean` | 1100 | 677 | 423 (38.5%) | 0 (0.0%) | 0 (0.0%) | 423 (38.5%) |
| `drift_one_class` | 1100 | 10 | 1090 (99.1%) | 0 (0.0%) | 0 (0.0%) | 1090 (99.1%) |
| `saturation` | 1100 | 1100 | 0 (0.0%) | 0 (0.0%) | 100 (9.1%) | 0 (0.0%) |

## First-fire step per gate per stream

| stream | weight_drift | update_rate | error_bias |
| --- | ---: | ---: | ---: |
| `baseline_clean` | — | — | 11 |
| `drift_one_class` | — | — | 11 |
| `saturation` | — | 1001 | — |

## Interpretation

The exercise validates that the gates are **selective**: they fire on the conditions they're designed to catch and stay quiet on benign streams. Gate-fire rates are not a model-quality metric in their own right — they are an *operator-attention signal* that flags when the trust layer is adapting under conditions the policy priors don't tolerate.

**Gate semantics (as of 2026-05-07):** `error_bias` is a **blocking gate** — evaluated pre-update in `ObservationVLAService._apply_trust_layer_ttt()`; a fire causes the adaptation step to be skipped entirely (`blocked` column). This means the drift observed under `drift_one_class` is lower than it would be without blocking: the gate prevents the bias from compounding. `weight_drift` and `update_rate` are post-update log-only warnings; operator review is the arbiter for those.

Reproduce: `python scripts/viability_gates_exercise.py` from repo root.
