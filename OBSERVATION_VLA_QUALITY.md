# ObservationVLA Corpus Quality Report

## Runtime Mix

- Total traces: `95`
- Runtime distribution: `clip_local=67, stub=8, stub_fallback=20`
- Assessment modes: `image_conditioned=88, metadata_only=7`
- Current outcomes: `72`
- Pinned submission cases: `7`

## Per-Target Coverage

| scenario | target | traces | distinct_any | distinct_model_backed | clip_local | stub_fallback | operator_reviewed | pinned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | Fort Myers Coast | 12 | 8 | 6 | 10 | 2 | 7 | False |
| disaster_response_weather | Houston Ship Channel | 7 | 7 | 3 | 3 | 2 | 7 | True |
| disaster_response_weather | New Orleans Delta | 10 | 8 | 6 | 8 | 2 | 5 | False |
| maritime_chokepoints | Panama Canal | 11 | 9 | 7 | 9 | 2 | 6 | False |
| maritime_chokepoints | Port of Singapore | 9 | 7 | 5 | 7 | 2 | 8 | False |
| maritime_chokepoints | Suez Canal | 9 | 9 | 4 | 4 | 2 | 7 | True |
| pedospheric_integrity | Mato Grosso Agricultural Frontier | 1 | 1 | 1 | 1 | 0 | 1 | True |
| urban_coastal_ambiguity | Port of Los Angeles | 10 | 8 | 6 | 8 | 2 | 6 | False |
| urban_coastal_ambiguity | Port of Rotterdam | 9 | 7 | 5 | 7 | 2 | 6 | True |
| urban_coastal_ambiguity | San Francisco Bay | 7 | 7 | 2 | 2 | 2 | 5 | True |
| urban_coastal_ambiguity | Shenzhen Bay | 10 | 8 | 6 | 8 | 2 | 8 | False |

## Model-Backed Morning Shortlist

One pending `clip_local` case per target, avoiding the fallback-heavy tail.

### Fort Myers Coast

- Scenario: `disaster_response_weather`
- Trace: `trace_41cc4824fd824e7796f585f9bdbd0b03`
- Action: `refine`
- Confidence: `0.61`
- Label source: `simulated`

![Fort Myers Coast](D:\SimSat\review_queue_assets\disaster_response_weather_trace_41cc4824fd824e7796f585f9bdbd0b03.png)

- Asset: [D:\SimSat\review_queue_assets\disaster_response_weather_trace_41cc4824fd824e7796f585f9bdbd0b03.png](D:\SimSat\review_queue_assets\disaster_response_weather_trace_41cc4824fd824e7796f585f9bdbd0b03.png)

### New Orleans Delta

- Scenario: `disaster_response_weather`
- Trace: `trace_bab79b553c7841558e0c42272cc87b61`
- Action: `refine`
- Confidence: `0.70`
- Label source: `simulated`

![New Orleans Delta](D:\SimSat\review_queue_assets\disaster_response_weather_trace_bab79b553c7841558e0c42272cc87b61.png)

- Asset: [D:\SimSat\review_queue_assets\disaster_response_weather_trace_bab79b553c7841558e0c42272cc87b61.png](D:\SimSat\review_queue_assets\disaster_response_weather_trace_bab79b553c7841558e0c42272cc87b61.png)

### Panama Canal

- Scenario: `maritime_chokepoints`
- Trace: `trace_864233c63eee45df8beb5bd69fb6b9ed`
- Action: `refine`
- Confidence: `0.73`
- Label source: `simulated`

![Panama Canal](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_864233c63eee45df8beb5bd69fb6b9ed.png)

- Asset: [D:\SimSat\review_queue_assets\maritime_chokepoints_trace_864233c63eee45df8beb5bd69fb6b9ed.png](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_864233c63eee45df8beb5bd69fb6b9ed.png)

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
- Trace: `trace_5acf98c2915148b78456b775372c2c81`
- Action: `accept`
- Confidence: `0.76`
- Label source: `simulated`

![Port of Los Angeles](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_5acf98c2915148b78456b775372c2c81.png)

- Asset: [D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_5acf98c2915148b78456b775372c2c81.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_5acf98c2915148b78456b775372c2c81.png)

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
- Trace: `trace_46aad3c86251428b9bbb1d5dbd4b3e04`
- Action: `defer`
- Confidence: `0.64`
- Label source: `simulated`

![Shenzhen Bay](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_46aad3c86251428b9bbb1d5dbd4b3e04.png)

- Asset: [D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_46aad3c86251428b9bbb1d5dbd4b3e04.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_46aad3c86251428b9bbb1d5dbd4b3e04.png)

## Interpretation

- The corpus is now large enough that raw trace count is no longer the bottleneck; image-backed coverage is.
- The key weak spots remain the historically pinned targets where repeated materializations mostly fall back instead of producing fresh `clip_local` cases.
- The safest next human review should prioritize the model-backed shortlist above, not the full mixed-quality trace store.
