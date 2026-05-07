# SimSat Challenge Test Status

## Fast Gate

- Pytest pass: `True`
- Pytest summary: `112 passed in 11.76s`

## Runtime

- Sentinel enabled: `True`
- Mapbox enabled: `False`
- Sentinel-first challenge scoring: `True`
- ObservationVLA runtime: `clip_local`

## Scenario Matrix

| scenario | windows | scaffold_accept | trust_accept | trust_refine | changed | accept_to_refine | trust_yield | reviewed_ready | top_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maritime_chokepoints (48h) | 8 | 3 | 4 | 3 | 8 | 0 | 1.00 | True | compound_risk_refine |
| disaster_response_weather (16h) | 3 | 0 | 0 | 0 | 3 | 0 | 1.00 | True | high_visibility,sentinel_available,line_of_sight,cloud_risk |
| urban_coastal_ambiguity (8h) | 3 | 0 | 0 | 2 | 3 | 0 | 1.00 | True | compound_risk_refine |
| pedospheric_integrity (24h) | 6 | 4 | 4 | 0 | 6 | 0 | 1.00 | True | high_priority,sentinel_available,low_cloud_risk,line_of_sight,edge_geometry |

## Horizon Guidance

| scenario | smoke_lane | competition_lane | rationale |
| --- | --- | --- | --- |
| maritime_chokepoints | 8h smoke check | 16h ambiguity rehearsal | compound_risk_refine |
| disaster_response_weather | 8h smoke check | 48h ambiguity rehearsal | sentinel_available,line_of_sight,cloud_risk |
| urban_coastal_ambiguity | 8h smoke check | 8h ambiguity rehearsal | compound_risk_refine |
| pedospheric_integrity | 8h smoke check | 48h ambiguity rehearsal | compound_risk_refine |

## Refine Diagnosis

| scenario | stored_traces | planner_refine | effective_refine | clip_local_traces | stub_family_traces |
| --- | --- | --- | --- | --- | --- |
| maritime_chokepoints | 29 | 0 | 22 | 20 | 9 |
| disaster_response_weather | 29 | 0 | 24 | 21 | 8 |
| urban_coastal_ambiguity | 36 | 0 | 31 | 25 | 11 |
| pedospheric_integrity | 1 | 0 | 0 | 1 | 0 |

## Review Focus

### maritime_chokepoints

- Trace: `trace_868637cfaf6343acba5dc01bc22cfbd9`
- Target: `Port of Singapore`
- Recommended action: `defer`
- Current label source: `operator_review`
- Needs operator review: `False`

### disaster_response_weather

- Trace: `trace_1926b646ee4b48478913681a33fcdfb1`
- Target: `Houston Ship Channel`
- Recommended action: `accept`
- Current label source: `operator_review`
- Needs operator review: `False`

### urban_coastal_ambiguity

- Trace: `trace_4f65355f5c954fbf8db3fc684bb377af`
- Target: `Port of Rotterdam`
- Recommended action: `refine`
- Current label source: `operator_review`
- Needs operator review: `False`

### pedospheric_integrity

- Trace: `trace_bfa2fad124374601b3f2884c3be2a42e`
- Target: `Mato Grosso Agricultural Frontier`
- Recommended action: `defer`
- Current label source: `operator_review`
- Needs operator review: `False`

## Artifact Presence

- Submission packet exists: `True`
- Submission casebook exists: `True`
- Submission readiness exists: `True`
- ObservationVLA eval exists: `True`

## Recommended Next Action

- Keep short horizons as a smoke lane, but use the scenario-specific ambiguity-rehearsal horizons below for competition-facing evaluation and operator review.
