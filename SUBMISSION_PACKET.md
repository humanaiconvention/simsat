# SimSat Submission Packet

This packet was generated from the live challenge stack.

## Runtime

- Sentinel enabled: `True`
- Mapbox enabled: `False`
- Challenge no-Mapbox-safe: `True`
- Sentinel-first challenge scoring: `True`
- ObservationVLA runtime: `transformers_vlm_local`

## Scorecard

| scenario | windows | scaffold_accept | trust_accept | trust_refine | scaffold_yield | trust_yield | labelled_cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | 5 | 3 | 3 | 0 | 1.0 | 1.0 | 13 |
| maritime_chokepoints | 8 | 4 | 6 | 2 | 1.0 | 1.0 | 14 |
| pedospheric_integrity | 5 | 3 | 3 | 1 | 1.0 | 1.0 | 1 |
| urban_coastal_ambiguity | 3 | 1 | 2 | 0 | 1.0 | 1.0 | 16 |

## disaster_response_weather

- Evaluation: `eval_d35e5cee30e047f492908dfbb5841493`
- Policy horizon: `16.0` hour(s)
- Horizon used: `16.0` hour(s)
- Transition counts: `{'accept->accept': 3, 'defer->defer': 2}`
- Top delta: `New Orleans Delta` `defer -> defer` (reason: `high_priority,sentinel_available,line_of_sight,edge_geometry,cloud_risk`)
- Curated case: Houston Ship Channel: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.92`
- Mission response: action=`materialize_now`, utility_realized=`0.92`
- Trace: `trace_1926b646ee4b48478913681a33fcdfb1`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## maritime_chokepoints

- Evaluation: `eval_5b85aec7ceee47968c06236f59586a7e`
- Policy horizon: `48.0` hour(s)
- Horizon used: `48.0` hour(s)
- Transition counts: `{'accept->accept': 4, 'defer->accept': 2, 'defer->refine': 2}`
- Top delta: `Port of Singapore` `defer -> refine` (reason: `compound_risk_refine`)
- Curated case: Suez Canal: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.95`
- Mission response: action=`materialize_now`, utility_realized=`0.95`
- Trace: `trace_2507337b7939460ebf01cbc9fcef8055`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## pedospheric_integrity

- Evaluation: `eval_35138611d15a4e07a86afdaa157068c1`
- Policy horizon: `24.0` hour(s)
- Horizon used: `24.0` hour(s)
- Transition counts: `{'accept->accept': 3, 'defer->defer': 1, 'defer->refine': 1}`
- Top delta: `Nile Delta Agricultural Zone` `defer -> refine` (reason: `trust_below_refine_threshold`)
- Curated case: Mato Grosso Agricultural Frontier: scaffold accept -> defer; assessment defer; outcome defer (not_useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`defer`, useful=`False`, usefulness_score=`0.0`
- Mission response: action=`queue_refine_review`, utility_realized=`0.0`
- Trace: `trace_bfa2fad124374601b3f2884c3be2a42e`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## urban_coastal_ambiguity

- Evaluation: `eval_7161100d9a82458aa556b1c14ddc7055`
- Policy horizon: `8.0` hour(s)
- Horizon used: `8.0` hour(s)
- Transition counts: `{'defer->accept': 1, 'accept->accept': 1, 'defer->defer': 1}`
- Top delta: `Port of Los Angeles` `defer -> accept` (reason: `high_visibility,high_priority,sentinel_available,line_of_sight,cloud_risk`)
- Curated case: Port of Rotterdam: scaffold accept -> refine; assessment refine; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`refine`, useful=`True`, usefulness_score=`1.0`
- Mission response: action=`escalate_operator`, utility_realized=`1.0`
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
