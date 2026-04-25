# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `clip_local`
- ObservationVLA model: `clip_local:openai/clip-vit-base-patch32`
- Reviewed sample size: `7`

## Summary

- Exact operator-action agreement: `0.86`
- Bucketed action agreement: `0.86`
- Useful / not-useful agreement: `0.86`
- Usefulness score MAE: `0.30`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| urban_coastal_ambiguity | Port of Rotterdam | trace_4f65355f5c954fbf8db3fc684bb377af | refine | refine | True | True | 0.55 | 0.89 | 0.34 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_fccb18798b844f68b6d0ad583d1e3caf | refine | refine | True | True | 0.55 | 0.87 | 0.32 |
| disaster_response_weather | Fort Myers Coast | trace_f7c1153a370b4c828274566a2714a524 | refine | refine | False | True | 0.50 | 0.90 | 0.40 |
| maritime_chokepoints | Port of Singapore | trace_e9e45ad652ff4e74853f0c6d0c03a827 | accept | refine | True | True | 0.62 | 0.88 | 0.26 |
| urban_coastal_ambiguity | San Francisco Bay | trace_a4e31b4c39224d8fbdb2c4bf0f444823 | accept | accept | True | True | 0.66 | 0.90 | 0.24 |
| disaster_response_weather | Houston Ship Channel | trace_1926b646ee4b48478913681a33fcdfb1 | accept | accept | True | True | 0.66 | 0.92 | 0.26 |
| maritime_chokepoints | Suez Canal | trace_2507337b7939460ebf01cbc9fcef8055 | accept | accept | True | True | 0.65 | 0.95 | 0.30 |

## Tight Claim

- Current claim: the ObservationVLA lane is now image-model-backed and shows `0.86` exact action agreement over `7` operator-reviewed Sentinel cases.
- Conservative claim: this is an early, low-N validation of image-conditioned usefulness scoring, not a broad benchmark.
