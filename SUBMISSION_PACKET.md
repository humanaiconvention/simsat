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
| disaster_response_weather | 10 | 3 | 3 | 2 | 0.7142857142857143 | 0.7142857142857143 | 19 |
| maritime_chokepoints | 10 | 4 | 6 | 2 | 1.0 | 1.0 | 21 |
| pedospheric_integrity | 11 | 7 | 7 | 1 | 1.0 | 1.0 | 1 |
| urban_coastal_ambiguity | 19 | 8 | 8 | 10 | 1.0 | 1.0 | 25 |

## disaster_response_weather

- Evaluation: `eval_55594d376cfd47a0bd19804af2a8cfdf`
- Policy horizon: `16.0` hour(s)
- Horizon used: `16.0` hour(s)
- Transition counts: `{'accept->accept': 3, 'defer->defer': 5, 'defer->refine': 2}`
- Top delta: `Houston Ship Channel` `defer -> refine` (reason: `compound_risk_refine`)
- Curated case: Houston Ship Channel: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.92`
- Mission response: action=`materialize_now`, utility_realized=`0.92`
- Trace: `trace_1926b646ee4b48478913681a33fcdfb1`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## maritime_chokepoints

- Evaluation: `eval_be861764d9b84c8aa66b2096b9296f77`
- Policy horizon: `48.0` hour(s)
- Horizon used: `48.0` hour(s)
- Transition counts: `{'accept->accept': 4, 'defer->accept': 2, 'defer->defer': 2, 'defer->refine': 2}`
- Top delta: `Port of Singapore` `defer -> refine` (reason: `compound_risk_refine`)
- Curated case: Suez Canal: scaffold accept -> refine; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.95`
- Mission response: action=`queue_refine_review`, utility_realized=`0.95`
- Trace: `trace_73f56c878603406e868fcf95b164997c`
- Label source: `operator_review`; reviewer=`regression_operator`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## pedospheric_integrity

- Evaluation: `eval_f7da819c71284a6dab4ba732f22a4b42`
- Policy horizon: `24.0` hour(s)
- Horizon used: `24.0` hour(s)
- Transition counts: `{'accept->accept': 7, 'defer->defer': 3, 'defer->refine': 1}`
- Top delta: `Nile Delta Agricultural Zone` `defer -> refine` (reason: `compound_risk_refine`)
- Curated case: Mato Grosso Agricultural Frontier: scaffold accept -> defer; assessment defer; outcome defer (not_useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`defer`, useful=`False`, usefulness_score=`0.0`
- Mission response: action=`queue_refine_review`, utility_realized=`0.0`
- Trace: `trace_bfa2fad124374601b3f2884c3be2a42e`
- Label source: `operator_review`; reviewer=`Ben Haslam`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## urban_coastal_ambiguity

- Evaluation: `eval_a0e50a8a66c648ed961f615d054680f2`
- Policy horizon: `8.0` hour(s)
- Horizon used: `8.0` hour(s)
- Transition counts: `{'accept->accept': 8, 'defer->defer': 1, 'defer->refine': 10}`
- Top delta: `Port of Rotterdam` `defer -> refine` (reason: `compound_risk_refine`)
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
