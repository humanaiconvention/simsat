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
| disaster_response_weather | 2 | 2 | 2 | 0 | 1.0 | 1.0 | 3 |
| maritime_chokepoints | 1 | 1 | 1 | 0 | 1.0 | 1.0 | 4 |
| urban_coastal_ambiguity | 3 | 3 | 3 | 0 | 1.0 | 1.0 | 4 |

## disaster_response_weather

- Evaluation: `eval_1261eaa52f9b4e97ad274675022f5315`
- Horizon used: `8.0` hour(s)
- Transition counts: `{'accept->accept': 2}`
- Top delta: `New Orleans Delta` `accept -> accept` (reason: `high_priority,sentinel_available,line_of_sight,cloud_risk`)
- Curated case: Houston Ship Channel: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`stub`, recommended=`accept`, useful=`True`, usefulness_score=`0.92`
- Mission response: action=`materialize_now`, utility_realized=`0.92`
- Trace: `trace_1926b646ee4b48478913681a33fcdfb1`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## maritime_chokepoints

- Evaluation: `eval_e61056e278974852b71003b2ebcf2236`
- Horizon used: `8.0` hour(s)
- Transition counts: `{'accept->accept': 1}`
- Top delta: `Suez Canal` `accept -> accept` (reason: `high_visibility,high_priority,sentinel_available,low_cloud_risk,line_of_sight`)
- Curated case: Suez Canal: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`stub`, recommended=`accept`, useful=`True`, usefulness_score=`0.95`
- Mission response: action=`materialize_now`, utility_realized=`0.95`
- Trace: `trace_2507337b7939460ebf01cbc9fcef8055`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## urban_coastal_ambiguity

- Evaluation: `eval_9647da2965164afa9a04f40db2d31338`
- Horizon used: `8.0` hour(s)
- Transition counts: `{'accept->accept': 3}`
- Top delta: `Port of Rotterdam` `accept -> accept` (reason: `high_visibility,high_priority,sentinel_available,line_of_sight,cloud_risk`)
- Curated case: San Francisco Bay: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`stub`, recommended=`accept`, useful=`True`, usefulness_score=`0.9`
- Mission response: action=`materialize_now`, utility_realized=`0.9`
- Trace: `trace_a4e31b4c39224d8fbdb2c4bf0f444823`
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
