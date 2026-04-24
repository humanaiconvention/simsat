# SimSat Review Set Build

- Base start times: `2026-04-10T00:00:00Z`
- Sweep direction: `backward`
- Quota mode: `model_backed`
- Target filter: `all`
- Sweep count: `1`
- Sweep times: `2026-04-10T00:00:00Z`
- Target quota: `3`
- Max quality skips per target: `12`
- Queue limit per scenario: `8`

| scenario | target | existing_distinct | added_distinct | total_distinct | duplicate_skips | quality_skips |
| --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | Houston Ship Channel | 2 | 0 | 2 | 0 | 0 |
| disaster_response_weather | Fort Myers Coast | 5 | 0 | 5 | 0 | 0 |
| disaster_response_weather | New Orleans Delta | 5 | 0 | 5 | 0 | 0 |
| maritime_chokepoints | Suez Canal | 1 | 0 | 1 | 0 | 0 |
| maritime_chokepoints | Port of Singapore | 5 | 0 | 5 | 0 | 0 |
| maritime_chokepoints | Panama Canal | 5 | 0 | 5 | 0 | 0 |
| urban_coastal_ambiguity | San Francisco Bay | 1 | 0 | 1 | 0 | 0 |
| urban_coastal_ambiguity | Port of Los Angeles | 5 | 0 | 5 | 0 | 0 |
| urban_coastal_ambiguity | Port of Rotterdam | 5 | 0 | 5 | 0 | 0 |
| urban_coastal_ambiguity | Shenzhen Bay | 5 | 0 | 5 | 0 | 0 |

## Assess Errors

- urban_coastal_ambiguity/sf_bay @ 2026-04-10T00:00:00Z: /encounter/decision/dec_0db847f10ef94705ae428394ae6bd81f/assess failed with 503: {"detail":"Observation VLA fell back to stub_fallback; refusing to persist a low-value fallback trace"}
- disaster_response_weather/houston_ship @ 2026-04-10T00:00:00Z: /encounter/decision/dec_539ee054c3924bd88fe3b4b0c3b35e20/assess failed with 503: {"detail":"Observation VLA fell back to stub_fallback; refusing to persist a low-value fallback trace"}
- maritime_chokepoints/suez @ 2026-04-10T00:00:00Z: /encounter/decision/dec_34c7f7bff4f344b8b184f2b418342346/assess failed with 503: {"detail":"Observation VLA fell back to stub_fallback; refusing to persist a low-value fallback trace"}