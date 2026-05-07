# ObservationVLA Corpus Drift and Calibration Report

## Runtime

- ObservationVLA runtime: `clip_local`
- ObservationVLA model: `clip_local:openai/clip-vit-base-patch32`
- Total traces reassessed: `95`
- Stored runtime cohorts: `clip_local=67, stub=8, stub_fallback=20`
- Scenario coverage: `disaster_response_weather=29, maritime_chokepoints=29, pedospheric_integrity=1, urban_coastal_ambiguity=36`

## Corpus Summary

- Stored action distribution: `accept=19, defer=6, refine=56, skip=14`
- Current `clip_local` action distribution: `accept=28, refine=31, skip=36`
- Current refine rate: `0.33`
- Stored `accept -> refine` shifts: `0` / `19`
- Current label sources: `operator_review=66, simulated=6, unlabelled=23`

## Drift Signals

- `refine->refine`: `28`
- `refine->skip`: `21`
- `accept->accept`: `19`
- `skip->skip`: `13`
- `refine->accept`: `7`
- `defer->refine`: `3`
- `defer->skip`: `2`
- `defer->accept`: `1`

## Reviewed Calibration

- Reviewed sample size: `66`
- Useful-but-refined reviewed cases: `19` / `52`
- Reviewed usefulness score MAE: `0.20`
- Exact action agreement on reviewed cases: `42/66`
- Bucketed agreement on reviewed cases: `42/66`

## Interpretation

- The worst `accept -> refine` collapse is gone. The current action policy now preserves historical high-confidence accepts while still keeping many newly added targets in `refine`.
- On the reviewed set, the model still marks the scenes as useful while downgrading them to `refine`, which suggests the scene/evidence scores are usable but the decision policy is too cautious.
- The safest current claim is that `clip_local` is functioning as a conservative image-conditioned scorer with a partially calibrated action policy. It is not yet broadly validated for autonomous `accept` decisions.

## Most Conservative Targets

| target | refine_rate | refine_count | accept_count |
| --- | --- | --- | --- |
| New Orleans Delta | 0.80 | 8 | 1 |
| Port of Rotterdam | 0.78 | 7 | 0 |
| Panama Canal | 0.64 | 7 | 3 |
| Port of Los Angeles | 0.40 | 4 | 1 |
| Shenzhen Bay | 0.20 | 2 | 0 |
| Fort Myers Coast | 0.17 | 2 | 3 |
| San Francisco Bay | 0.14 | 1 | 5 |
| Port of Singapore | 0.00 | 0 | 1 |
| Mato Grosso Agricultural Frontier | 0.00 | 0 | 1 |
| Houston Ship Channel | 0.00 | 0 | 4 |

## Priority Review Shortlist

Best next labels if we want to stress-test the conservative bias.

| scenario | target | trace | stored_action | current_action | current_confidence | label_source |
| --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | New Orleans Delta | trace_38571614c1464b8d994f61629eb22772 | defer | refine | 0.75 | unlabelled |
| urban_coastal_ambiguity | Port of Los Angeles | trace_4bccb9cf7b864225bad705117654f277 | refine | refine | 0.78 | unlabelled |
| urban_coastal_ambiguity | San Francisco Bay | trace_5bc171357b3a431983b9f81646117522 | refine | refine | 0.77 | unlabelled |
| urban_coastal_ambiguity | Port of Rotterdam | trace_e086e5c0816d4b148ce12db9ef1684e0 | refine | refine | 0.75 | unlabelled |
| urban_coastal_ambiguity | Port of Rotterdam | trace_1113fa0223014708a27c28f6cb876e73 | refine | refine | 0.75 | unlabelled |
| urban_coastal_ambiguity | Port of Los Angeles | trace_de8ffd5863714ac5bd22ff398e4fccc9 | refine | refine | 0.75 | unlabelled |
| disaster_response_weather | New Orleans Delta | trace_75f406cf5bff4771945fee5c975e65f1 | refine | refine | 0.70 | unlabelled |
| urban_coastal_ambiguity | Shenzhen Bay | trace_511eb967219b4a3db07e08cbcc77064a | skip | skip | 0.59 | unlabelled |
| urban_coastal_ambiguity | Port of Rotterdam | trace_fe171375483748c39828af9e2587378d | skip | skip | 0.59 | unlabelled |
| urban_coastal_ambiguity | Port of Los Angeles | trace_e47d4364940b489e8e41e4a6091ddd3e | skip | skip | 0.59 | unlabelled |
| disaster_response_weather | New Orleans Delta | trace_e0bb6c2b04514c90a8892dc9e17630b6 | skip | skip | 0.54 | unlabelled |
| disaster_response_weather | Fort Myers Coast | trace_6f7838564cc94b07a7f29a430a804832 | skip | skip | 0.54 | unlabelled |

## Tighter Claim

- Current evidence supports a narrow claim: `clip_local` now preserves the historical high-confidence `accept` cases while remaining conservative on newly added targets that still need human labels.
- The next human-review pass should focus on first labels for the newly added targets and any remaining non-accept high-confidence cases, rather than adding more near-duplicate accepted scenes.
