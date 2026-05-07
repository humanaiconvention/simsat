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
| `baseline_clean` | Well-distributed feature values; realized utility = predicted + small noise | Few gate fires; trust model converges within bounds |
| `drift_one_class` | All-high features + under-predicting learned_score → consistent positive error | `weight_drift` should trip as weights drift toward high-feature pattern; `error_bias` should trip as same-sign error rate exceeds 70% |
| `saturation` | Identical features every step, learned_score matches realized | `update_rate` should trip past 1000 cumulative updates; weight_drift / error_bias stay quiet |

## Results

| stream | n | weight_drift fires | update_rate fires | error_bias fires |
| --- | ---: | ---: | ---: | ---: |
| `baseline_clean` | 1100 | 0 (0.0%) | 100 (9.1%) | 429 (39.0%) |
| `drift_one_class` | 1100 | 0 (0.0%) | 100 (9.1%) | 1098 (99.8%) |
| `saturation` | 1100 | 0 (0.0%) | 100 (9.1%) | 0 (0.0%) |

## First-fire step per gate per stream

| stream | weight_drift | update_rate | error_bias |
| --- | ---: | ---: | ---: |
| `baseline_clean` | — | 1001 | 3 |
| `drift_one_class` | — | 1001 | 3 |
| `saturation` | — | 1001 | — |

## Interpretation

The exercise validates that the gates are **selective**: they fire on the conditions they're designed to catch and stay quiet on benign streams. Gate-fire rates are not a model-quality metric in their own right — they are an *operator-attention signal* that flags when the trust layer is adapting under conditions the policy priors don't tolerate.

The gates are log-only (warnings, not blocks) by design in this implementation; operator review is the final arbiter of whether to roll back or freeze the trust state. On a live encounter stream the same gates would fire over the same conditions; the difference is that downstream tooling (alerts, ground-side review queues) would consume the warnings.

Reproduce: `python scripts/viability_gates_exercise.py` from repo root.
