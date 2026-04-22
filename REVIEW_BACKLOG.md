# SimSat Review Backlog

- Total stored traces: `86`
- Current outcomes: `11`
- Pinned submission traces: `3`
- Targets represented: `10`

## Corpus Coverage

| scenario | target | traces | distinct_cases | operator_reviewed | simulated | pinned |
| --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | Houston Ship Channel | 7 | 7 | 1 | 2 | True |
| disaster_response_weather | Fort Myers Coast | 11 | 7 | 0 | 0 | False |
| disaster_response_weather | New Orleans Delta | 9 | 7 | 0 | 0 | False |
| maritime_chokepoints | Suez Canal | 7 | 7 | 1 | 3 | True |
| maritime_chokepoints | Port of Singapore | 9 | 7 | 0 | 0 | False |
| maritime_chokepoints | Panama Canal | 9 | 7 | 0 | 0 | False |
| urban_coastal_ambiguity | San Francisco Bay | 7 | 7 | 1 | 3 | True |
| urban_coastal_ambiguity | Port of Los Angeles | 9 | 7 | 0 | 0 | False |
| urban_coastal_ambiguity | Port of Rotterdam | 9 | 7 | 0 | 0 | False |
| urban_coastal_ambiguity | Shenzhen Bay | 9 | 7 | 0 | 0 | False |

## Next Review Shortlist

One best pending candidate per target.

### disaster_response_weather

- `Houston Ship Channel`
  trace=`trace_cbe073faf8554afb986b632128204650` runtime=`clip_local` recommended=`refine` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\disaster_response_weather_trace_cbe073faf8554afb986b632128204650.png](D:\SimSat\review_queue_assets\disaster_response_weather_trace_cbe073faf8554afb986b632128204650.png)
- `Fort Myers Coast`
  trace=`trace_f7c1153a370b4c828274566a2714a524` runtime=`clip_local` recommended=`refine` label_source=`unlabelled`
- `New Orleans Delta`
  trace=`trace_b992766137e2419e8b1575e8b4ae3fc2` runtime=`clip_local` recommended=`refine` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\disaster_response_weather_trace_b992766137e2419e8b1575e8b4ae3fc2.png](D:\SimSat\review_queue_assets\disaster_response_weather_trace_b992766137e2419e8b1575e8b4ae3fc2.png)

### maritime_chokepoints

- `Suez Canal`
  trace=`trace_3957ae2fabb34f298e9ff95ec912b6e1` runtime=`clip_local` recommended=`accept` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\maritime_chokepoints_trace_3957ae2fabb34f298e9ff95ec912b6e1.png](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_3957ae2fabb34f298e9ff95ec912b6e1.png)
- `Port of Singapore`
  trace=`trace_e9e45ad652ff4e74853f0c6d0c03a827` runtime=`clip_local` recommended=`refine` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\maritime_chokepoints_trace_e9e45ad652ff4e74853f0c6d0c03a827.png](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_e9e45ad652ff4e74853f0c6d0c03a827.png)
- `Panama Canal`
  trace=`trace_e0aa6488d0b541f2819fcf257006b636` runtime=`clip_local` recommended=`refine` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\maritime_chokepoints_trace_e0aa6488d0b541f2819fcf257006b636.png](D:\SimSat\review_queue_assets\maritime_chokepoints_trace_e0aa6488d0b541f2819fcf257006b636.png)

### urban_coastal_ambiguity

- `San Francisco Bay`
  trace=`trace_dbbe33015d04413082df17ba2f0c8d12` runtime=`clip_local` recommended=`accept` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_dbbe33015d04413082df17ba2f0c8d12.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_dbbe33015d04413082df17ba2f0c8d12.png)
- `Port of Los Angeles`
  trace=`trace_f77a115170574657ba4228ed818ec68d` runtime=`clip_local` recommended=`refine` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_f77a115170574657ba4228ed818ec68d.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_f77a115170574657ba4228ed818ec68d.png)
- `Port of Rotterdam`
  trace=`trace_e603eaf267c448b1a4ed059e6ac3e638` runtime=`clip_local` recommended=`defer` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_e603eaf267c448b1a4ed059e6ac3e638.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_e603eaf267c448b1a4ed059e6ac3e638.png)
- `Shenzhen Bay`
  trace=`trace_f760ebf72a964cab8ac6372e6e2dd820` runtime=`clip_local` recommended=`refine` label_source=`unlabelled`
  asset=[D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_f760ebf72a964cab8ac6372e6e2dd820.png](D:\SimSat\review_queue_assets\urban_coastal_ambiguity_trace_f760ebf72a964cab8ac6372e6e2dd820.png)
