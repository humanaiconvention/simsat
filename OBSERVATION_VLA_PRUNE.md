# ObservationVLA Fallback Prune Report

- Generated: `2026-04-11T15:41:41Z`
- Total traces: `546`
- Runtime distribution: `clip_local=55, stub=11, stub_fallback=480`
- Protected trace references: `11`
- Keep latest fallback per target: `2`
- Safe prune candidates: `460`

## Candidate Counts By Target

| target | prune_candidates |
| --- | --- |
| Fort Myers Coast | 2 |
| Houston Ship Channel | 149 |
| New Orleans Delta | 3 |
| Panama Canal | 2 |
| Port of Los Angeles | 2 |
| Port of Rotterdam | 2 |
| Port of Singapore | 2 |
| San Francisco Bay | 148 |
| Shenzhen Bay | 4 |
| Suez Canal | 146 |

## Protected Fallback Rows

These are fallback rows that the tool refuses to prune because they are referenced or retained as latest debugging examples.

| target | trace | reason |
| --- | --- | --- |
| Fort Myers Coast | trace_6f7838564cc94b07a7f29a430a804832 | kept_latest_per_target |
| Fort Myers Coast | trace_978e6a6a5bf8466e99aa7b38ef7e35a8 | kept_latest_per_target |
| Houston Ship Channel | trace_ad3890a1998d4d90950201e048708cb4 | kept_latest_per_target |
| Houston Ship Channel | trace_efa9e41802814c429b102c2f77eb64fc | kept_latest_per_target |
| New Orleans Delta | trace_75f406cf5bff4771945fee5c975e65f1 | kept_latest_per_target |
| New Orleans Delta | trace_e0bb6c2b04514c90a8892dc9e17630b6 | kept_latest_per_target |
| Panama Canal | trace_31ec6996796d40c2b09e946f854f0d4a | kept_latest_per_target |
| Panama Canal | trace_a0aab6981bb24016bd2d36a859e5d7e9 | kept_latest_per_target |
| Port of Los Angeles | trace_10d844ffa6a14e39b7e933375f05c9cd | kept_latest_per_target |
| Port of Los Angeles | trace_e47d4364940b489e8e41e4a6091ddd3e | kept_latest_per_target |
| Port of Rotterdam | trace_fccb18798b844f68b6d0ad583d1e3caf | kept_latest_per_target |
| Port of Rotterdam | trace_fe171375483748c39828af9e2587378d | kept_latest_per_target |
| Port of Singapore | trace_1d9e98b6bc7d493f8c292947938b6d4e | kept_latest_per_target |
| Port of Singapore | trace_b98b5d0bb9f14ef1a2526f0afdc12c02 | kept_latest_per_target |
| San Francisco Bay | trace_5bc171357b3a431983b9f81646117522 | kept_latest_per_target |
| San Francisco Bay | trace_b5870f83612a4a7798c4c342661efeab | kept_latest_per_target |
| Shenzhen Bay | trace_2c2e81628756417dbb0093a1babf0ffd | kept_latest_per_target |
| Shenzhen Bay | trace_511eb967219b4a3db07e08cbcc77064a | kept_latest_per_target |
| Suez Canal | trace_a4f965f9fce642b9ac42059907c41bd8 | kept_latest_per_target |
| Suez Canal | trace_a8e23f1ff498486bbf0d60417d46f9b0 | kept_latest_per_target |

## Sample Prune Candidates

| scenario | target | trace | created_at | action | assessment_record |
| --- | --- | --- | --- | --- | --- |
| urban_coastal_ambiguity | San Francisco Bay | trace_bb98198b8cd14513ac9cd1e0c8d3314d | 2026-04-11T07:25:25Z | skip | True |
| urban_coastal_ambiguity | Port of Los Angeles | trace_6606efbf18e74407ab0383ef02565609 | 2026-04-11T07:25:26Z | skip | True |
| urban_coastal_ambiguity | Port of Rotterdam | trace_2b3551af49a0461dbfaf74e41e5790c5 | 2026-04-11T07:25:27Z | skip | True |
| urban_coastal_ambiguity | Shenzhen Bay | trace_c91668cc689848d6a7e3354708a8732e | 2026-04-11T07:25:27Z | skip | True |
| disaster_response_weather | Fort Myers Coast | trace_1b8840f8e05b435d8ba20375aed932c8 | 2026-04-11T07:25:31Z | skip | True |
| disaster_response_weather | Houston Ship Channel | trace_c72807d0ec8e4141badd9cf704cd7234 | 2026-04-11T07:25:31Z | skip | True |
| disaster_response_weather | New Orleans Delta | trace_299f7d4efac14fdb81395aa07feb3e30 | 2026-04-11T07:25:32Z | skip | True |
| maritime_chokepoints | Port of Singapore | trace_a576c9dab2224cef9d745fb2fb6f6f27 | 2026-04-11T07:25:35Z | skip | True |
| maritime_chokepoints | Suez Canal | trace_d6cfcb37cd634d50a811191ac7246e73 | 2026-04-11T07:25:35Z | skip | True |
| maritime_chokepoints | Panama Canal | trace_2187d113234e432fb50624cd7d323852 | 2026-04-11T07:25:36Z | skip | True |
| urban_coastal_ambiguity | San Francisco Bay | trace_e332ec9eb80c4088b01f988236aeddda | 2026-04-11T07:25:40Z | skip | True |
| urban_coastal_ambiguity | Port of Los Angeles | trace_b2ac997302e7482194196f9f9efd6f6e | 2026-04-11T07:25:41Z | skip | True |
| urban_coastal_ambiguity | Port of Rotterdam | trace_c36da1e06d2d41b39699b19abdffa7ec | 2026-04-11T07:25:41Z | skip | True |
| urban_coastal_ambiguity | Shenzhen Bay | trace_63404f83281246e8a809344b28199613 | 2026-04-11T07:25:42Z | skip | True |
| disaster_response_weather | Houston Ship Channel | trace_fe37fe153cec44a991a8e0ac79bde9b4 | 2026-04-11T07:25:45Z | skip | True |
| disaster_response_weather | Fort Myers Coast | trace_3924092f5d31445e8e5050d664ab30c5 | 2026-04-11T07:25:46Z | skip | True |
| disaster_response_weather | New Orleans Delta | trace_07034fd3e9e34f03956c1af339250283 | 2026-04-11T07:25:47Z | skip | True |
| maritime_chokepoints | Suez Canal | trace_ce84edb9a08f42408d2314e65ae8556a | 2026-04-11T07:25:50Z | skip | True |
| maritime_chokepoints | Panama Canal | trace_84e0f19d97004946b3bca6e337527547 | 2026-04-11T07:25:51Z | skip | True |
| maritime_chokepoints | Port of Singapore | trace_fe788d4932ab41c6804dd440efb7a182 | 2026-04-11T07:25:51Z | skip | True |

## Safe Apply Rule

- Only `stub_fallback` traces are considered.
- Any trace with an ObservationVLA outcome is protected.
- Any trace pinned as a submission case is protected.
- Any trace referenced by a mission-response action is protected.
- The newest `2` fallback traces per target are retained for debugging.
- Matching orphaned assessment records are pruned alongside deleted traces.
