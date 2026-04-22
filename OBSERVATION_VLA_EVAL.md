# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `clip_local`
- ObservationVLA model: `clip_local:openai/clip-vit-base-patch32`
- Reviewed sample size: `3`

## Summary

- Exact operator-action agreement: `1.00`
- Bucketed action agreement: `1.00`
- Useful / not-useful agreement: `1.00`
- Usefulness score MAE: `0.12`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | Houston Ship Channel | trace_1926b646ee4b48478913681a33fcdfb1 | accept | accept | True | True | 0.80 | 0.92 | 0.12 |
| urban_coastal_ambiguity | San Francisco Bay | trace_a4e31b4c39224d8fbdb2c4bf0f444823 | accept | accept | True | True | 0.79 | 0.90 | 0.11 |
| maritime_chokepoints | Suez Canal | trace_2507337b7939460ebf01cbc9fcef8055 | accept | accept | True | True | 0.81 | 0.95 | 0.14 |

## Tight Claim

- Current claim: the ObservationVLA lane is now image-model-backed and shows `1.00` exact action agreement over `3` operator-reviewed Sentinel cases.
- Conservative claim: this is an early, low-N validation of image-conditioned usefulness scoring, not a broad benchmark.
