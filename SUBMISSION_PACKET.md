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
| disaster_response_weather | 3 | 1 | 1 | 0 | 1.0 | 1.0 | 11 |
| maritime_chokepoints | 8 | 6 | 6 | 2 | 1.0 | 1.0 | 12 |
| urban_coastal_ambiguity | 3 | 2 | 2 | 0 | 1.0 | 1.0 | 14 |

## disaster_response_weather

- Evaluation: `eval_983e4bf78f5949f39f55153688030c06`
- Policy horizon: `16.0` hour(s)
- Horizon used: `16.0` hour(s)
- Transition counts: `{'accept->accept': 1, 'defer->defer': 2}`
- Top delta: `Houston Ship Channel` `defer -> defer` (reason: `high_visibility,sentinel_available,line_of_sight,cloud_risk`)
- Curated case: Houston Ship Channel: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.92`
- Mission response: action=`materialize_now`, utility_realized=`0.92`
- Trace: `trace_1926b646ee4b48478913681a33fcdfb1`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`

## maritime_chokepoints

- Evaluation: `eval_a3dd64d0f2084330aa459d7327f82c3d`
- Policy horizon: `48.0` hour(s)
- Horizon used: `48.0` hour(s)
- Transition counts: `{'accept->accept': 6, 'defer->refine': 2}`
- Top delta: `Suez Canal` `defer -> refine` (reason: `compound_risk_refine`)
- Curated case: Suez Canal: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.95`
- Mission response: action=`materialize_now`, utility_realized=`0.95`
- Trace: `trace_2507337b7939460ebf01cbc9fcef8055`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`

## urban_coastal_ambiguity

- Evaluation: `eval_ac02270be0474c0caa7035d4d30b03f5`
- Policy horizon: `8.0` hour(s)
- Horizon used: `8.0` hour(s)
- Transition counts: `{'accept->accept': 2, 'defer->defer': 1}`
- Top delta: `Shenzhen Bay` `accept -> accept` (reason: `high_visibility,high_priority,sentinel_available,line_of_sight,cloud_risk`)
- Curated case: San Francisco Bay: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.9`
- Mission response: action=`materialize_now`, utility_realized=`0.9`
- Trace: `trace_a4e31b4c39224d8fbdb2c4bf0f444823`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`

## Notes

- This packet is Sentinel-first and does not depend on Mapbox.
- Submission cases are generated from the live planner and stored assessment stack.
- Stored case traces may have been assessed under an earlier ObservationVLA runtime than the current configured backend.
- The current backend-specific reviewed check lives in `OBSERVATION_VLA_EVAL.md` and should be used for tight model claims.
- Operator-reviewed labels are preferred when they exist for a scenario case.
- Outcome labels fall back to `simulated_submission_case` only when no operator-reviewed label is available.
