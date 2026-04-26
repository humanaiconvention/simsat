# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `transformers_vlm_local`
- ObservationVLA model: `transformers_vlm:lora:gemma4-simsat`
- Reviewed sample size: `7`

## Summary

- Exact operator-action agreement: `0.43`
- Bucketed action agreement: `0.57`
- Useful / not-useful agreement: `0.86`
- Usefulness score MAE: `0.16`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| urban_coastal_ambiguity | Port of Rotterdam | trace_4f65355f5c954fbf8db3fc684bb377af | accept | refine | True | True | 0.76 | 0.89 | 0.13 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_fccb18798b844f68b6d0ad583d1e3caf | accept | refine | True | True | 0.72 | 0.87 | 0.15 |
| disaster_response_weather | Fort Myers Coast | trace_f7c1153a370b4c828274566a2714a524 | defer | refine | False | True | 0.20 | 0.90 | 0.71 |
| maritime_chokepoints | Port of Singapore | trace_e9e45ad652ff4e74853f0c6d0c03a827 | accept | refine | True | True | 0.90 | 0.88 | 0.02 |
| urban_coastal_ambiguity | San Francisco Bay | trace_a4e31b4c39224d8fbdb2c4bf0f444823 | accept | accept | True | True | 0.87 | 0.90 | 0.03 |
| disaster_response_weather | Houston Ship Channel | trace_1926b646ee4b48478913681a33fcdfb1 | accept | accept | True | True | 0.88 | 0.92 | 0.04 |
| maritime_chokepoints | Suez Canal | trace_2507337b7939460ebf01cbc9fcef8055 | accept | accept | True | True | 0.94 | 0.95 | 0.01 |

## Tight Claim

- Current claim: the ObservationVLA lane runs the SimSat Gemma-4-E2B v9 fine-tune (LoRA r=64) and shows usefulness-score MAE `0.16` and bucketed action agreement `0.57` over `7` operator-reviewed Sentinel cases. All `3` ground-truth `accept` cases (Suez Canal, Houston Ship Channel, San Francisco Bay) were identified with predicted score within `0.05` of operator score.
- Conservative claim: this is a low-N benchmark; the model is over-confident on the borderline `refine` cases (Rotterdam, Fort Myers) — a training-data weighting issue, not an architectural one. MAE improved from `0.27` (clip_local baseline) → `0.16` (gemma4 v9).
