# ObservationVLA Fallback Prune Report

- Generated: `2026-05-05T14:28:48Z`
- Total traces: `86`
- Runtime distribution: `clip_local=58, stub=8, stub_fallback=20`
- Protected trace references: `47`
- Keep latest fallback per target: `2`
- Safe prune candidates: `0`

## Candidate Counts By Target

| target | prune_candidates |
| --- | --- |
| none | 0 |

## Protected Fallback Rows

These are fallback rows that the tool refuses to prune because they are referenced or retained as latest debugging examples.

| target | trace | reason |
| --- | --- | --- |
| Fort Myers Coast | trace_6f7838564cc94b07a7f29a430a804832 | kept_latest_per_target |
| Fort Myers Coast | trace_978e6a6a5bf8466e99aa7b38ef7e35a8 | kept_latest_per_target |
| Houston Ship Channel | trace_ad3890a1998d4d90950201e048708cb4 | protected_reference |
| Houston Ship Channel | trace_efa9e41802814c429b102c2f77eb64fc | protected_reference |
| New Orleans Delta | trace_75f406cf5bff4771945fee5c975e65f1 | kept_latest_per_target |
| New Orleans Delta | trace_e0bb6c2b04514c90a8892dc9e17630b6 | kept_latest_per_target |
| Panama Canal | trace_31ec6996796d40c2b09e946f854f0d4a | kept_latest_per_target |
| Panama Canal | trace_a0aab6981bb24016bd2d36a859e5d7e9 | kept_latest_per_target |
| Port of Los Angeles | trace_10d844ffa6a14e39b7e933375f05c9cd | kept_latest_per_target |
| Port of Los Angeles | trace_e47d4364940b489e8e41e4a6091ddd3e | kept_latest_per_target |
| Port of Rotterdam | trace_fccb18798b844f68b6d0ad583d1e3caf | protected_reference |
| Port of Rotterdam | trace_fe171375483748c39828af9e2587378d | kept_latest_per_target |
| Port of Singapore | trace_1d9e98b6bc7d493f8c292947938b6d4e | kept_latest_per_target |
| Port of Singapore | trace_b98b5d0bb9f14ef1a2526f0afdc12c02 | kept_latest_per_target |
| San Francisco Bay | trace_5bc171357b3a431983b9f81646117522 | kept_latest_per_target |
| San Francisco Bay | trace_b5870f83612a4a7798c4c342661efeab | protected_reference |
| Shenzhen Bay | trace_2c2e81628756417dbb0093a1babf0ffd | kept_latest_per_target |
| Shenzhen Bay | trace_511eb967219b4a3db07e08cbcc77064a | kept_latest_per_target |
| Suez Canal | trace_a4f965f9fce642b9ac42059907c41bd8 | protected_reference |
| Suez Canal | trace_a8e23f1ff498486bbf0d60417d46f9b0 | kept_latest_per_target |

## Sample Prune Candidates

| scenario | target | trace | created_at | action | assessment_record |
| --- | --- | --- | --- | --- | --- |
| none | - | - | - | - | - |

## Safe Apply Rule

- Only `stub_fallback` traces are considered.
- Any trace with an ObservationVLA outcome is protected.
- Any trace pinned as a submission case is protected.
- Any trace referenced by a mission-response action is protected.
- The newest `2` fallback traces per target are retained for debugging.
- Matching orphaned assessment records are pruned alongside deleted traces.
