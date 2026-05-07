# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `transformers_vlm_local`
- ObservationVLA model: `transformers_vlm:lora:gemma4-v17`
- Reviewed sample size: `37`
- Note: v17 adapter has partial save (410/490 tensors; k/v_proj missing layers 15-34 due to GQA dedup bug). v18 (`data_ptr()` fix) targets 490/490.

## Summary

- Exact operator-action agreement: `0.49`
- Bucketed action agreement: `0.51`
- Useful / not-useful agreement: `0.65`
- Usefulness score MAE: `0.18`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maritime_chokepoints | Suez Canal | trace_73f56c878603406e868fcf95b164997c | accept | accept | True | True | 0.77 | 0.95 | 0.18 |
| maritime_chokepoints | Suez Canal | trace_f9b6e8e10cd84e32b64a62632887ebbc | accept | accept | True | True | 0.77 | 0.95 | 0.18 |
| disaster_response_weather | Houston Ship Channel | trace_efa9e41802814c429b102c2f77eb64fc | defer | skip | False | False | 0.61 | 0.20 | 0.41 |
| urban_coastal_ambiguity | San Francisco Bay | trace_b5870f83612a4a7798c4c342661efeab | defer | skip | False | False | 0.41 | 0.20 | 0.21 |
| disaster_response_weather | Fort Myers Coast | trace_85ddfa677bfd459d95dbed455dd38505 | refine | skip | True | False | 0.53 | 0.20 | 0.33 |
| maritime_chokepoints | Port of Singapore | trace_50c8eab02a1b445a9fbb13dd9a92cf09 | defer | accept | False | True | 0.40 | 0.85 | 0.45 |
| disaster_response_weather | Houston Ship Channel | trace_31ae7bb2ac304fad9c91e6497bfb31c5 | defer | skip | False | False | 0.40 | 0.20 | 0.20 |
| maritime_chokepoints | Port of Singapore | trace_08ae57efb8a44376a74a3e1a2b91a037 | skip | skip | False | False | 0.57 | 0.20 | 0.37 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_e603eaf267c448b1a4ed059e6ac3e638 | skip | accept | False | True | 0.60 | 0.85 | 0.25 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_63d2f16600264d93b7443459166cfde7 | refine | refine | True | True | 0.62 | 0.55 | 0.07 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_ad3a863eae974301b903539c40a708b2 | refine | refine | True | True | 0.63 | 0.55 | 0.08 |
| disaster_response_weather | New Orleans Delta | trace_a46b65c213aa4493a0ed34aaed741a77 | refine | refine | True | True | 0.65 | 0.55 | 0.10 |
| maritime_chokepoints | Panama Canal | trace_aaa144b5ec6146cca1f2cd743a9cefe0 | refine | refine | True | True | 0.64 | 0.55 | 0.09 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_c1de50e3492b42bf912d8bc688a360cc | skip | skip | False | False | 0.56 | 0.20 | 0.36 |
| disaster_response_weather | Fort Myers Coast | trace_f3b408a550ee4031b35ab7d452c21d71 | skip | skip | False | False | 0.57 | 0.20 | 0.37 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_c3e2a26eeab540a3b34a2b5185ce6613 | refine | refine | True | True | 0.63 | 0.55 | 0.08 |
| disaster_response_weather | Houston Ship Channel | trace_cbe073faf8554afb986b632128204650 | skip | refine | False | True | 0.58 | 0.55 | 0.03 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_11691f60aa8d47d38493aebb399f41f2 | refine | refine | True | True | 0.64 | 0.55 | 0.09 |
| maritime_chokepoints | Port of Singapore | trace_6982090f337548b2939921ea3324f55a | skip | skip | False | False | 0.56 | 0.20 | 0.36 |
| maritime_chokepoints | Port of Singapore | trace_b6aa341cd881425c89fe20bd1ee2e0d1 | skip | refine | False | True | 0.57 | 0.55 | 0.02 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_8b4d77c012664ec5a72cb14c96ae08b7 | refine | refine | True | True | 0.64 | 0.55 | 0.09 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_f760ebf72a964cab8ac6372e6e2dd820 | skip | skip | False | False | 0.61 | 0.20 | 0.41 |
| pedospheric_integrity | Mato Grosso Agricultural Frontier | trace_bfa2fad124374601b3f2884c3be2a42e | refine | defer | True | False | 0.73 | 0.00 | 0.73 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_4f65355f5c954fbf8db3fc684bb377af | refine | accept | True | True | 0.65 | 1.00 | 0.35 |
| maritime_chokepoints | Port of Singapore | trace_da37bb4310014942a24d430ea238110a | skip | refine | False | True | 0.56 | 0.55 | 0.01 |
| disaster_response_weather | Fort Myers Coast | trace_cdcc2a7023d1433e9d4e0edc6bddf10e | skip | refine | False | True | 0.57 | 0.55 | 0.02 |
| maritime_chokepoints | Port of Singapore | trace_777833d1ce0847259cfbac033b271266 | skip | refine | False | True | 0.56 | 0.55 | 0.01 |
| disaster_response_weather | Fort Myers Coast | trace_88c56a41ee64478b94257e26df7f5f77 | skip | refine | False | True | 0.57 | 0.55 | 0.02 |
| maritime_chokepoints | Suez Canal | trace_a4f965f9fce642b9ac42059907c41bd8 | refine | accept | True | True | 0.75 | 0.85 | 0.10 |
| disaster_response_weather | Houston Ship Channel | trace_ad3890a1998d4d90950201e048708cb4 | refine | accept | True | True | 0.73 | 0.85 | 0.12 |
| maritime_chokepoints | Panama Canal | trace_e0aa6488d0b541f2819fcf257006b636 | refine | refine | True | True | 0.62 | 0.55 | 0.07 |
| disaster_response_weather | New Orleans Delta | trace_b992766137e2419e8b1575e8b4ae3fc2 | refine | refine | True | True | 0.65 | 0.55 | 0.10 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_212662f658bd48028077bc23fb87c442 | skip | refine | False | True | 0.55 | 0.55 | 0.00 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_58951351d068471c81106c7d029e2c81 | skip | accept | False | True | 0.60 | 0.85 | 0.25 |
| maritime_chokepoints | Panama Canal | trace_d5b4d837f3d0444f89633ec10766d73a | refine | refine | True | True | 0.64 | 0.55 | 0.09 |
| disaster_response_weather | New Orleans Delta | trace_0725ba13314d4aec8e695ed0fe2130dd | refine | refine | True | True | 0.65 | 0.55 | 0.10 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_1f00711c8db6433f9e1f7a79691c23d3 | skip | refine | False | True | 0.61 | 0.55 | 0.06 |

## Tight Claim

- Current claim: the ObservationVLA lane is now image-model-backed and shows `0.49` exact action agreement over `37` operator-reviewed Sentinel cases.
- Conservative claim: this is an early, low-N validation of image-conditioned usefulness scoring, not a broad benchmark.
