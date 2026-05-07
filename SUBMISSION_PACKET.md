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
| disaster_response_weather | 3 | 0 | 0 | 0 | 1.0 | 1.0 | 30 |
| maritime_chokepoints | 8 | 3 | 3 | 3 | 1.0 | 1.0 | 31 |
| pedospheric_integrity | 5 | 3 | 3 | 0 | 1.0 | 1.0 | 54 |
| urban_coastal_ambiguity | 3 | 0 | 0 | 2 | 1.0 | 1.0 | 37 |

## disaster_response_weather

- Evaluation: `eval_8ef0286000eb424c8c9f197cee6b90d9`
- Policy horizon: `16.0` hour(s)
- Horizon used: `16.0` hour(s)
- Transition counts: `{'defer->defer': 3}`
- Top delta: `Houston Ship Channel` `defer -> defer` (reason: `sentinel_available,line_of_sight,cloud_risk`)
- Curated case: Houston Ship Channel: scaffold accept -> accept; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`stub`, recommended=`accept`, useful=`True`, usefulness_score=`0.85`
- Mission response: action=`materialize_now`, utility_realized=`0.85`
- Trace: `trace_d886c4872e1047d8b3f50b3563448968`
- Label source: `operator_review`; reviewer=`ben`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## maritime_chokepoints

- Evaluation: `eval_9928df6176ea40a9b02614210f4b6a3e`
- Policy horizon: `48.0` hour(s)
- Horizon used: `48.0` hour(s)
- Transition counts: `{'accept->accept': 3, 'defer->defer': 2, 'defer->refine': 3}`
- Top delta: `Panama Canal` `defer -> refine` (reason: `compound_risk_refine`)
- Curated case: Suez Canal: scaffold accept -> refine; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.85`
- Mission response: action=`queue_refine_review`, utility_realized=`0.85`
- Trace: `trace_3957ae2fabb34f298e9ff95ec912b6e1`
- Label source: `operator_review`; reviewer=`ben`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## pedospheric_integrity

- Evaluation: `eval_0567d72271c440e7b2ab9c24e463ea9b`
- Policy horizon: `24.0` hour(s)
- Horizon used: `24.0` hour(s)
- Transition counts: `{'accept->accept': 3, 'defer->defer': 2}`
- Top delta: `Punjab Indo-Gangetic Plain` `accept -> accept` (reason: `high_visibility,high_priority,sentinel_available,line_of_sight,cloud_risk`)
- Curated case: Nile Delta Agricultural Zone: scaffold accept -> refine; assessment accept; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`accept`, useful=`True`, usefulness_score=`0.85`
- Mission response: action=`queue_refine_review`, utility_realized=`0.85`
- Trace: `trace_3d10cd03f80b48b88d3fcc895ff8917a`
- Label source: `operator_review`; reviewer=`ben`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## urban_coastal_ambiguity

- Evaluation: `eval_aeb9b9de742d484b9cff28561376052a`
- Policy horizon: `8.0` hour(s)
- Horizon used: `8.0` hour(s)
- Transition counts: `{'defer->defer': 1, 'defer->refine': 2}`
- Top delta: `Shenzhen Bay` `defer -> refine` (reason: `compound_risk_refine`)
- Curated case: Port of Rotterdam: scaffold accept -> refine; assessment refine; outcome accept (useful)
- Stored visual assessment: mode=`image_conditioned`, runtime=`clip_local`, recommended=`refine`, useful=`True`, usefulness_score=`0.85`
- Mission response: action=`escalate_operator`, utility_realized=`0.85`
- Trace: `trace_4f65355f5c954fbf8db3fc684bb377af`
- Label source: `operator_review`; reviewer=`ben`; status=`reviewed`
- Submission case: `pinned`
- Case source: `historical labelled trace`

## Notes

- This packet is Sentinel-first and does not depend on Mapbox.
- Submission cases are generated from the live planner and stored assessment stack.
- Stored case traces may have been assessed under an earlier ObservationVLA runtime than the current configured backend.
- The current backend-specific reviewed check lives in `OBSERVATION_VLA_EVAL.md` and should be used for tight model claims.
- Operator-reviewed labels are preferred when they exist for a scenario case.
- This packet was generated in `reviewed-only` mode and fails if any scenario lacks a pinned operator-reviewed case.
