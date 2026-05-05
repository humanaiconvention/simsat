# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `transformers_vlm_local`
- ObservationVLA model: `transformers_vlm:lora:simsat-gemma4-v11-real`
- Reviewed sample size: `37`

## Summary

- Exact operator-action agreement: `0.86`
- Bucketed action agreement: `0.86`
- Useful / not-useful agreement: `0.97`
- Usefulness score MAE: `0.13`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maritime_chokepoints | Port of Singapore | trace_da37bb4310014942a24d430ea238110a | skip | refine | False | True | 0.56 | 0.55 | 0.01 |
| disaster_response_weather | Fort Myers Coast | trace_cdcc2a7023d1433e9d4e0edc6bddf10e | refine | refine | True | True | 0.41 | 0.55 | 0.14 |
| maritime_chokepoints | Port of Singapore | trace_777833d1ce0847259cfbac033b271266 | refine | refine | True | True | 0.40 | 0.55 | 0.15 |
| disaster_response_weather | Fort Myers Coast | trace_88c56a41ee64478b94257e26df7f5f77 | refine | refine | True | True | 0.40 | 0.55 | 0.15 |
| maritime_chokepoints | Suez Canal | trace_a4f965f9fce642b9ac42059907c41bd8 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| disaster_response_weather | Houston Ship Channel | trace_ad3890a1998d4d90950201e048708cb4 | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Panama Canal | trace_e0aa6488d0b541f2819fcf257006b636 | refine | refine | True | True | 0.49 | 0.55 | 0.06 |
| disaster_response_weather | New Orleans Delta | trace_b992766137e2419e8b1575e8b4ae3fc2 | refine | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_212662f658bd48028077bc23fb87c442 | refine | refine | True | True | 0.39 | 0.55 | 0.16 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_58951351d068471c81106c7d029e2c81 | refine | accept | True | True | 0.47 | 0.85 | 0.38 |
| maritime_chokepoints | Panama Canal | trace_d5b4d837f3d0444f89633ec10766d73a | refine | refine | True | True | 0.64 | 0.55 | 0.09 |
| disaster_response_weather | New Orleans Delta | trace_0725ba13314d4aec8e695ed0fe2130dd | refine | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_1f00711c8db6433f9e1f7a79691c23d3 | refine | refine | True | True | 0.43 | 0.55 | 0.12 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_03147bb1b2234f489ce01208d11bbf4a | refine | accept | True | True | 0.50 | 0.85 | 0.35 |
| maritime_chokepoints | Panama Canal | trace_599f94be68e14eccb19ce49bac32d609 | refine | refine | True | True | 0.49 | 0.55 | 0.06 |
| disaster_response_weather | New Orleans Delta | trace_ae1998677df5406093334f1fcf0abc66 | refine | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_ecb6521c521446838ea9035d9631f4ca | refine | refine | True | True | 0.39 | 0.55 | 0.16 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_f77a115170574657ba4228ed818ec68d | refine | accept | True | True | 0.47 | 0.85 | 0.38 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_7edbad05378b45eb88f8fedf7a71331c | refine | refine | True | True | 0.39 | 0.55 | 0.16 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_9afbbc332a96426abd0629fc019d395e | refine | accept | True | True | 0.48 | 0.85 | 0.37 |
| maritime_chokepoints | Panama Canal | trace_c6da2e5391a8418c9ec2f379436509f2 | refine | refine | True | True | 0.62 | 0.55 | 0.07 |
| disaster_response_weather | New Orleans Delta | trace_04e08f0b4c4c42869a7663c3cb8bf641 | refine | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | San Francisco Bay | trace_67918155fd3e47b8a207ae1bf635269c | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Suez Canal | trace_103f989c13d1437a8927fc031cfefc56 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| urban_coastal_ambiguity | San Francisco Bay | trace_26f2081882cf46ba94e10b53ca5c2707 | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Suez Canal | trace_46c71a07326648039517d0a736b3ba31 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| disaster_response_weather | Houston Ship Channel | trace_d886c4872e1047d8b3f50b3563448968 | accept | accept | True | True | 0.80 | 0.85 | 0.05 |
| urban_coastal_ambiguity | San Francisco Bay | trace_ba0e1a1a2b744cdf94f76fd990fec3f8 | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Suez Canal | trace_f889dbec218e45039cbf63c5b092aa60 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| disaster_response_weather | Houston Ship Channel | trace_e5a00b774bcf410fafe9944b8a4cc4f8 | accept | accept | True | True | 0.80 | 0.85 | 0.05 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_4f65355f5c954fbf8db3fc684bb377af | refine | refine | True | True | 0.55 | 0.89 | 0.34 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_fccb18798b844f68b6d0ad583d1e3caf | refine | refine | True | True | 0.71 | 0.87 | 0.16 |
| disaster_response_weather | Fort Myers Coast | trace_f7c1153a370b4c828274566a2714a524 | refine | refine | True | True | 0.40 | 0.90 | 0.50 |
| maritime_chokepoints | Port of Singapore | trace_e9e45ad652ff4e74853f0c6d0c03a827 | refine | refine | True | True | 0.73 | 0.88 | 0.15 |
| urban_coastal_ambiguity | San Francisco Bay | trace_a4e31b4c39224d8fbdb2c4bf0f444823 | accept | accept | True | True | 0.79 | 0.90 | 0.11 |
| disaster_response_weather | Houston Ship Channel | trace_1926b646ee4b48478913681a33fcdfb1 | accept | accept | True | True | 0.80 | 0.92 | 0.12 |
| maritime_chokepoints | Suez Canal | trace_2507337b7939460ebf01cbc9fcef8055 | accept | accept | True | True | 0.81 | 0.95 | 0.14 |

## Tight Claim

- Current claim: the ObservationVLA lane is now image-model-backed and shows `0.86` exact action agreement over `37` operator-reviewed Sentinel cases.
- Conservative claim: this is an early, low-N validation of image-conditioned usefulness scoring, not a broad benchmark.

## Coverage Note

The 37-case reviewed pool covers three geometric/structural scenario packs (maritime_chokepoints, disaster_response_weather, urban_coastal_ambiguity). The `pedospheric_integrity` pack was added after the batch_review.py session of 2026-04-27; its pinned trace (Mato Grosso Agricultural Frontier) was reviewed and deferred by operator as not-useful under the simulated backend. A fresh eval pool for the spectral-biochemical register is pending a VLM assessment with the resize fix (`OBSERVATION_VLM_MAX_IMAGE_SIZE=448`) active. Until that pool exists, the `0.86` agreement figure applies to geometric/structural windows only.
