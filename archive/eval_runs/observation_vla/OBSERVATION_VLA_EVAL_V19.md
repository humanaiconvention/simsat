# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `clip_local`
- ObservationVLA model: `clip_local:openai/clip-vit-base-patch32`
- Reviewed sample size: `37`

## Summary

- Exact operator-action agreement: `0.46`
- Bucketed action agreement: `0.49`
- Useful / not-useful agreement: `0.70`
- Usefulness score MAE: `0.24`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | Fort Myers Coast | trace_22155d6f9ee14be18d072bcda4aa4a2c | refine | skip | False | False | 0.50 | 0.20 | 0.30 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_2c2e81628756417dbb0093a1babf0ffd | refine | skip | False | False | 0.39 | 0.20 | 0.19 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_10d844ffa6a14e39b7e933375f05c9cd | defer | accept | True | True | 0.53 | 0.85 | 0.32 |
| maritime_chokepoints | Panama Canal | trace_31ec6996796d40c2b09e946f854f0d4a | refine | skip | False | False | 0.49 | 0.20 | 0.29 |
| maritime_chokepoints | Port of Singapore | trace_1d9e98b6bc7d493f8c292947938b6d4e | refine | skip | False | False | 0.41 | 0.20 | 0.21 |
| disaster_response_weather | Fort Myers Coast | trace_978e6a6a5bf8466e99aa7b38ef7e35a8 | refine | skip | False | False | 0.50 | 0.20 | 0.30 |
| disaster_response_weather | Fort Myers Coast | trace_85ddfa677bfd459d95dbed455dd38505 | refine | refine | False | True | 0.53 | 0.55 | 0.02 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_ad3a863eae974301b903539c40a708b2 | defer | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_63d2f16600264d93b7443459166cfde7 | refine | accept | False | True | 0.54 | 0.85 | 0.31 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_e603eaf267c448b1a4ed059e6ac3e638 | defer | accept | True | True | 0.51 | 0.85 | 0.34 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_f760ebf72a964cab8ac6372e6e2dd820 | refine | skip | False | False | 0.43 | 0.20 | 0.23 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_c3e2a26eeab540a3b34a2b5185ce6613 | refine | accept | False | True | 0.53 | 0.85 | 0.32 |
| maritime_chokepoints | Suez Canal | trace_73f56c878603406e868fcf95b164997c | accept | accept | True | True | 0.76 | 0.95 | 0.19 |
| maritime_chokepoints | Suez Canal | trace_f9b6e8e10cd84e32b64a62632887ebbc | accept | accept | True | True | 0.76 | 0.95 | 0.19 |
| disaster_response_weather | Houston Ship Channel | trace_efa9e41802814c429b102c2f77eb64fc | skip | skip | False | False | 0.61 | 0.20 | 0.41 |
| urban_coastal_ambiguity | San Francisco Bay | trace_b5870f83612a4a7798c4c342661efeab | skip | skip | False | False | 0.63 | 0.20 | 0.43 |
| maritime_chokepoints | Port of Singapore | trace_50c8eab02a1b445a9fbb13dd9a92cf09 | skip | accept | False | True | 0.62 | 0.85 | 0.23 |
| disaster_response_weather | Houston Ship Channel | trace_31ae7bb2ac304fad9c91e6497bfb31c5 | skip | skip | False | False | 0.62 | 0.20 | 0.42 |
| maritime_chokepoints | Port of Singapore | trace_08ae57efb8a44376a74a3e1a2b91a037 | skip | skip | False | False | 0.63 | 0.20 | 0.43 |
| disaster_response_weather | New Orleans Delta | trace_a46b65c213aa4493a0ed34aaed741a77 | refine | refine | True | True | 0.71 | 0.55 | 0.16 |
| maritime_chokepoints | Panama Canal | trace_aaa144b5ec6146cca1f2cd743a9cefe0 | refine | refine | True | True | 0.70 | 0.55 | 0.15 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_c1de50e3492b42bf912d8bc688a360cc | skip | skip | False | False | 0.61 | 0.20 | 0.41 |
| disaster_response_weather | Fort Myers Coast | trace_f3b408a550ee4031b35ab7d452c21d71 | skip | skip | False | False | 0.63 | 0.20 | 0.43 |
| disaster_response_weather | Houston Ship Channel | trace_cbe073faf8554afb986b632128204650 | skip | refine | False | True | 0.63 | 0.55 | 0.08 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_11691f60aa8d47d38493aebb399f41f2 | refine | refine | True | True | 0.69 | 0.55 | 0.14 |
| maritime_chokepoints | Port of Singapore | trace_6982090f337548b2939921ea3324f55a | skip | skip | False | False | 0.62 | 0.20 | 0.42 |
| maritime_chokepoints | Port of Singapore | trace_b6aa341cd881425c89fe20bd1ee2e0d1 | skip | refine | False | True | 0.63 | 0.55 | 0.08 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_8b4d77c012664ec5a72cb14c96ae08b7 | refine | refine | True | True | 0.69 | 0.55 | 0.14 |
| pedospheric_integrity | Mato Grosso Agricultural Frontier | trace_bfa2fad124374601b3f2884c3be2a42e | accept | defer | True | False | 0.78 | 0.00 | 0.78 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_4f65355f5c954fbf8db3fc684bb377af | refine | accept | True | True | 0.71 | 1.00 | 0.29 |
| maritime_chokepoints | Port of Singapore | trace_da37bb4310014942a24d430ea238110a | skip | refine | False | True | 0.62 | 0.55 | 0.07 |
| disaster_response_weather | Fort Myers Coast | trace_cdcc2a7023d1433e9d4e0edc6bddf10e | skip | refine | False | True | 0.63 | 0.55 | 0.08 |
| maritime_chokepoints | Port of Singapore | trace_777833d1ce0847259cfbac033b271266 | skip | refine | False | True | 0.62 | 0.55 | 0.07 |
| disaster_response_weather | Fort Myers Coast | trace_88c56a41ee64478b94257e26df7f5f77 | skip | refine | False | True | 0.63 | 0.55 | 0.08 |
| maritime_chokepoints | Suez Canal | trace_a4f965f9fce642b9ac42059907c41bd8 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| disaster_response_weather | Houston Ship Channel | trace_ad3890a1998d4d90950201e048708cb4 | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Panama Canal | trace_e0aa6488d0b541f2819fcf257006b636 | refine | refine | True | True | 0.68 | 0.55 | 0.13 |

## Tight Claim

- Current claim: the ObservationVLA lane is now image-model-backed and shows `0.46` exact action agreement over `37` operator-reviewed Sentinel cases.
- Conservative claim: this is an early, low-N validation of image-conditioned usefulness scoring, not a broad benchmark.
