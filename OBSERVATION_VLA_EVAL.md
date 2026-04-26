# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `transformers_vlm_local`
- ObservationVLA model: `transformers_vlm:lora:gemma4-simsat`
- Reviewed sample size: `7`

## Summary

- Exact operator-action agreement: `0.43`
- Bucketed action agreement: `0.43`
- Useful / not-useful agreement: `0.86`
- Usefulness score MAE: `0.21`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| urban_coastal_ambiguity | Port of Rotterdam | trace_4f65355f5c954fbf8db3fc684bb377af | refine | refine | True | True | 0.65 | 0.89 | 0.24 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_fccb18798b844f68b6d0ad583d1e3caf | refine | refine | True | True | 0.65 | 0.87 | 0.22 |
| disaster_response_weather | Fort Myers Coast | trace_f7c1153a370b4c828274566a2714a524 | skip | refine | False | True | 0.57 | 0.90 | 0.33 |
| maritime_chokepoints | Port of Singapore | trace_e9e45ad652ff4e74853f0c6d0c03a827 | refine | refine | True | True | 0.73 | 0.88 | 0.15 |
| urban_coastal_ambiguity | San Francisco Bay | trace_a4e31b4c39224d8fbdb2c4bf0f444823 | refine | accept | True | True | 0.74 | 0.90 | 0.16 |
| disaster_response_weather | Houston Ship Channel | trace_1926b646ee4b48478913681a33fcdfb1 | refine | accept | True | True | 0.74 | 0.92 | 0.18 |
| maritime_chokepoints | Suez Canal | trace_2507337b7939460ebf01cbc9fcef8055 | refine | accept | True | True | 0.75 | 0.95 | 0.20 |

## Tight Claim

- Current claim: the ObservationVLA lane is now image-model-backed and shows `0.43` exact action agreement over `7` operator-reviewed Sentinel cases.
- Conservative claim: this is an early, low-N validation of image-conditioned usefulness scoring, not a broad benchmark.
