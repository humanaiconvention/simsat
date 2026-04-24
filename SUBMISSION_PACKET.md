# SimSat Submission Packet

This packet was generated from the live challenge stack.

## Runtime

- Sentinel enabled: `True`
- Mapbox enabled: `False`
- Challenge no-Mapbox-safe: `True`
- Sentinel-first challenge scoring: `True`
- ObservationVLA runtime: `clip_local`

## Scorecard

| scenario | windows | scaffold_accept | trust_accept | trust_refine | scaffold_yield | trust_yield | labelled_cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | 3 | 1 | 1 | 0 | 1.0 | 1.0 | 4 |
| maritime_chokepoints | 2 | 1 | 1 | 0 | 1.0 | 1.0 | 5 |
| urban_coastal_ambiguity | 3 | 0 | 1 | 0 | 1.0 | 1.0 | 6 |

## disaster_response_weather

- Evaluation: `eval_3cdeedbeeb284411b5f36d0a066a41a8`
- Policy horizon: `16.0` hour(s)
- Horizon used: `16.0` hour(s)
- Transition counts: `{'accept->accept': 1, 'defer->defer': 2}`
- Top delta: `Houston Ship Channel` `defer -> defer` (reason: `sentinel_available,line_of_sight,cloud_risk`)
- Curated case: Houston Ship Channel: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`stub`, recommended=`accept`, useful=`True`, usefulness_score=`0.92`
- Mission response: action=`materialize_now`, utility_realized=`0.92`
- Trace: `trace_1926b646ee4b48478913681a33fcdfb1`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## maritime_chokepoints

- Evaluation: `eval_27293cdacc0542899af69187d90de71f`
- Policy horizon: `48.0` hour(s)
- Horizon used: `48.0` hour(s)
- Transition counts: `{'accept->accept': 1, 'defer->defer': 1}`
- Top delta: `Panama Canal` `defer -> defer` (reason: `high_priority,sentinel_available,low_cloud_risk,line_of_sight,edge_geometry`)
- Curated case: Suez Canal: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`stub`, recommended=`accept`, useful=`True`, usefulness_score=`0.95`
- Mission response: action=`materialize_now`, utility_realized=`0.95`
- Trace: `trace_2507337b7939460ebf01cbc9fcef8055`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## urban_coastal_ambiguity

- Evaluation: `eval_4baae27ac2064c7b872d1113ed295117`
- Policy horizon: `8.0` hour(s)
- Horizon used: `8.0` hour(s)
- Transition counts: `{'defer->accept': 1, 'defer->defer': 2}`
- Top delta: `Shenzhen Bay` `defer -> accept` (reason: `high_visibility,high_priority,sentinel_available,line_of_sight,cloud_risk`)
- Curated case: Port of Rotterdam: scaffold accept -> refine; assessment refine; outcome refine (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`refine`, useful=`True`, usefulness_score=`0.89`
- Mission response: action=`escalate_operator`, utility_realized=`0.89`
- Trace: `trace_4f65355f5c954fbf8db3fc684bb377af`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## Notes

- This packet is Sentinel-first and does not depend on Mapbox.
- Submission cases are generated from the live planner and stored assessment stack.
- Stored case traces may have been assessed under an earlier ObservationVLA runtime than the current configured backend.
- The current backend-specific reviewed check lives in `OBSERVATION_VLA_EVAL.md` and should be used for tight model claims.
- Operator-reviewed labels are preferred when they exist for a scenario case.
- This packet was generated in `reviewed-only` mode and fails if any scenario lacks a pinned operator-reviewed case.
