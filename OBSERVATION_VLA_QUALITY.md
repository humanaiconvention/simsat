# ObservationVLA Corpus Quality Report

## Runtime Mix

- Total traces: `86`
- Runtime distribution: `clip_local=55, stub=11, stub_fallback=20`
- Assessment modes: `image_conditioned=79, metadata_only=7`
- Current outcomes: `11`
- Pinned submission cases: `3`

## Per-Target Coverage

| scenario | target | traces | distinct_any | distinct_model_backed | clip_local | stub_fallback | operator_reviewed | pinned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | Fort Myers Coast | 11 | 7 | 5 | 9 | 2 | 0 | False |
| disaster_response_weather | Houston Ship Channel | 7 | 7 | 2 | 2 | 2 | 1 | True |
| disaster_response_weather | New Orleans Delta | 9 | 7 | 5 | 7 | 2 | 0 | False |
| maritime_chokepoints | Panama Canal | 9 | 7 | 5 | 7 | 2 | 0 | False |
| maritime_chokepoints | Port of Singapore | 9 | 7 | 5 | 7 | 2 | 0 | False |
| maritime_chokepoints | Suez Canal | 7 | 7 | 1 | 1 | 2 | 1 | True |
| urban_coastal_ambiguity | Port of Los Angeles | 9 | 7 | 5 | 7 | 2 | 0 | False |
| urban_coastal_ambiguity | Port of Rotterdam | 9 | 7 | 5 | 7 | 2 | 0 | False |
| urban_coastal_ambiguity | San Francisco Bay | 7 | 7 | 1 | 1 | 2 | 1 | True |
| urban_coastal_ambiguity | Shenzhen Bay | 9 | 7 | 5 | 7 | 2 | 0 | False |

## Model-Backed Morning Shortlist

One pending `clip_local` case per target, avoiding the fallback-heavy tail.

### Fort Myers Coast

- Scenario: `disaster_response_weather`
- Trace: `trace_00850a706d0244eaabd9799791a914a3`
- Action: `skip`
- Confidence: `0.58`
- Label source: `unlabelled`

![Fort Myers Coast](D:\SimSat\review_queue_assets\disaster_response_weather_trace_00850a706d0244eaabd9799791a914a3.png)

- Asset: [D:\SimSat\review_queue_assets\disaster_response_weather_trace_00850a706d0244eaabd9799791a914a3.png](D:\SimSat\review_queue_assets\disaster_response_weather_trace_00850a706d0244eaabd9799791a914a3.png)

### Houston Ship Channel

- Scenario: `disaster_response_weather`
- Trace: `trace_cbe073faf8554afb986b632128204650`
- Action: `refine`
- Confidence: `0.39`
- Label source: `unlabelled`

![Houston Ship Channel](D:\SimSat\review_queue_assets\disaster_response_weather_trace_cbe073faf8554afb986b632128204650.png)

- Asset: [D:\SimSat\review_queue_assets\disaster_response_weather_trace_cbe073faf8554afb986b632128204650.png](D:\SimSat\review_queue_assets\disaster_response_weather_trace_cbe073faf8554afb986b632128204650.png)

### New Orleans Delta

- Scenario: `disaster_response_weather`
- Trace: `trace_32c41b142ecf42a5b378caa654efe6c7`
- Action: `refine`
- Confidence: `0.53`
- Label source: `unlabelled`

![New Orleans Delta](D:\SimSat\review_queue_assets\disaster_response_weather_trace_32c41b142ecf42a5b378caa654efe6c7.png)

- Asset: [D:\SimSat\review_queue_assets\disaster_response_weather_trace_32c41b142ecf42a5b378caa654efe6c7.png](D:\SimSat\review_queue_assets\disaster_response_weather_trace_32c41b142ecf42a5b378caa654efe6c7.png)

### Panama Canal

- Scenario: `maritime_chokepoints`
- Trace: `trace_35e90312fef94a0ca53839d7b3cdfe5d`
- Action: `refine`
- Confidence: `0.53`
- Label source: `unlabelled`

![Panama Canal](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_35e90312fef94a0ca53839d7b3cdfe5d.png)

- Asset: [D:\SimSat\review_queue_assets\maritime_chokepoints_trace_35e90312fef94a0ca53839d7b3cdfe5d.png](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_35e90312fef94a0ca53839d7b3cdfe5d.png)

### Port of Singapore

- Scenario: `maritime_chokepoints`
- Trace: `trace_e9e45ad652ff4e74853f0c6d0c03a827`
- Action: `refine`
- Confidence: `0.59`
- Label source: `unlabelled`

![Port of Singapore](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_e9e45ad652ff4e74853f0c6d0c03a827.png)

- Asset: [D:\SimSat\review_queue_assets\maritime_chokepoints_trace_e9e45ad652ff4e74853f0c6d0c03a827.png](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_e9e45ad652ff4e74853f0c6d0c03a827.png)

### Suez Canal

- Scenario: `maritime_chokepoints`
- Trace: `trace_3957ae2fabb34f298e9ff95ec912b6e1`
- Action: `accept`
- Confidence: `0.61`
- Label source: `unlabelled`

![Suez Canal](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_3957ae2fabb34f298e9ff95ec912b6e1.png)

- Asset: [D:\SimSat\review_queue_assets\maritime_chokepoints_trace_3957ae2fabb34f298e9ff95ec912b6e1.png](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_3957ae2fabb34f298e9ff95ec912b6e1.png)

### Port of Los Angeles

- Scenario: `urban_coastal_ambiguity`
- Trace: `trace_4bccb9cf7b864225bad705117654f277`
- Action: `refine`
- Confidence: `0.57`
- Label source: `unlabelled`

![Port of Los Angeles](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_4bccb9cf7b864225bad705117654f277.png)

- Asset: [D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_4bccb9cf7b864225bad705117654f277.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_4bccb9cf7b864225bad705117654f277.png)

### Port of Rotterdam

- Scenario: `urban_coastal_ambiguity`
- Trace: `trace_e086e5c0816d4b148ce12db9ef1684e0`
- Action: `refine`
- Confidence: `0.52`
- Label source: `unlabelled`

![Port of Rotterdam](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_e086e5c0816d4b148ce12db9ef1684e0.png)

- Asset: [D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_e086e5c0816d4b148ce12db9ef1684e0.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_e086e5c0816d4b148ce12db9ef1684e0.png)

### San Francisco Bay

- Scenario: `urban_coastal_ambiguity`
- Trace: `trace_dbbe33015d04413082df17ba2f0c8d12`
- Action: `accept`
- Confidence: `0.62`
- Label source: `unlabelled`

![San Francisco Bay](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_dbbe33015d04413082df17ba2f0c8d12.png)

- Asset: [D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_dbbe33015d04413082df17ba2f0c8d12.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_dbbe33015d04413082df17ba2f0c8d12.png)

### Shenzhen Bay

- Scenario: `urban_coastal_ambiguity`
- Trace: `trace_ad3a863eae974301b903539c40a708b2`
- Action: `defer`
- Confidence: `0.50`
- Label source: `unlabelled`

![Shenzhen Bay](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_ad3a863eae974301b903539c40a708b2.png)

- Asset: [D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_ad3a863eae974301b903539c40a708b2.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_ad3a863eae974301b903539c40a708b2.png)

## Interpretation

- The corpus is now large enough that raw trace count is no longer the bottleneck; image-backed coverage is.
- The key weak spots remain the historically pinned targets where repeated materializations mostly fall back instead of producing fresh `clip_local` cases.
- The safest next human review should prioritize the model-backed shortlist above, not the full mixed-quality trace store.
