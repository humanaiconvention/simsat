# ObservationVLA Corpus Drift and Calibration Report

## Runtime

- ObservationVLA runtime: `clip_local`
- ObservationVLA model: `clip_local:openai/clip-vit-base-patch32`
- Total traces reassessed: `86`
- Stored runtime cohorts: `clip_local=55, stub=11, stub_fallback=20`
- Scenario coverage: `disaster_response_weather=27, maritime_chokepoints=25, urban_coastal_ambiguity=34`

## Corpus Summary

- Stored action distribution: `accept=16, defer=4, refine=52, skip=14`
- Current `clip_local` action distribution: `accept=22, defer=1, refine=35, skip=28`
- Current refine rate: `0.41`
- Stored `accept -> refine` shifts: `1` / `16`
- Current label sources: `operator_review=3, simulated=8, unlabelled=75`

## Drift Signals

- `refine->refine`: `27`
- `refine->skip`: `19`
- `accept->accept`: `15`
- `skip->skip`: `7`
- `refine->accept`: `6`
- `skip->refine`: `5`
- `defer->refine`: `2`
- `defer->skip`: `2`

## Reviewed Calibration

- Reviewed sample size: `3`
- Useful-but-refined reviewed cases: `0` / `3`
- Reviewed usefulness score MAE: `0.12`
- Exact action agreement on reviewed cases: `3/3`
- Bucketed agreement on reviewed cases: `3/3`

## Interpretation

- The worst `accept -> refine` collapse is gone. The current action policy now preserves historical high-confidence accepts while still keeping many newly added targets in `refine`.
- On the small reviewed set, current `clip_local` actions now match the operator labels, but this is still a low-N check rather than a broad validation.
- The safest current claim is that `clip_local` is functioning as a conservative image-conditioned scorer with a partially calibrated action policy. It is not yet broadly validated for autonomous `accept` decisions.

## Most Conservative Targets

| target | refine_rate | refine_count | accept_count |
| --- | --- | --- | --- |
| Port of Rotterdam | 0.78 | 7 | 0 |
| New Orleans Delta | 0.78 | 7 | 1 |
| Panama Canal | 0.67 | 6 | 2 |
| Port of Los Angeles | 0.44 | 4 | 0 |
| Fort Myers Coast | 0.36 | 4 | 3 |
| Houston Ship Channel | 0.29 | 2 | 3 |
| San Francisco Bay | 0.29 | 2 | 5 |
| Shenzhen Bay | 0.22 | 2 | 0 |
| Port of Singapore | 0.11 | 1 | 1 |
| Suez Canal | 0.00 | 0 | 7 |

## Priority Review Shortlist

Best next labels if we want to stress-test the conservative bias.

| scenario | target | trace | stored_action | current_action | current_confidence | label_source |
| --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | New Orleans Delta | trace_38571614c1464b8d994f61629eb22772 | defer | refine | 0.72 | unlabelled |
| urban_coastal_ambiguity | Shenzhen Bay | trace_ad3a863eae974301b903539c40a708b2 | defer | refine | 0.63 | unlabelled |
| urban_coastal_ambiguity | Shenzhen Bay | trace_f760ebf72a964cab8ac6372e6e2dd820 | refine | skip | 0.60 | unlabelled |
| urban_coastal_ambiguity | Shenzhen Bay | trace_1f00711c8db6433f9e1f7a79691c23d3 | refine | skip | 0.60 | unlabelled |
| disaster_response_weather | Houston Ship Channel | trace_ad3890a1998d4d90950201e048708cb4 | accept | refine | 0.60 | unlabelled |
| urban_coastal_ambiguity | Port of Los Angeles | trace_f77a115170574657ba4228ed818ec68d | refine | skip | 0.59 | unlabelled |
| urban_coastal_ambiguity | Port of Los Angeles | trace_9afbbc332a96426abd0629fc019d395e | refine | skip | 0.59 | unlabelled |
| urban_coastal_ambiguity | Port of Los Angeles | trace_58951351d068471c81106c7d029e2c81 | defer | skip | 0.59 | unlabelled |
| urban_coastal_ambiguity | Port of Rotterdam | trace_e603eaf267c448b1a4ed059e6ac3e638 | defer | skip | 0.58 | unlabelled |
| disaster_response_weather | Houston Ship Channel | trace_cbe073faf8554afb986b632128204650 | refine | skip | 0.55 | unlabelled |
| disaster_response_weather | Fort Myers Coast | trace_f3b408a550ee4031b35ab7d452c21d71 | refine | skip | 0.54 | unlabelled |
| disaster_response_weather | Fort Myers Coast | trace_cdcc2a7023d1433e9d4e0edc6bddf10e | refine | skip | 0.54 | unlabelled |

## Tighter Claim

- Current evidence supports a narrow claim: `clip_local` now preserves the historical high-confidence `accept` cases while remaining conservative on newly added targets that still need human labels.
- The next human-review pass should focus on first labels for the newly added targets and any remaining non-accept high-confidence cases, rather than adding more near-duplicate accepted scenes.
