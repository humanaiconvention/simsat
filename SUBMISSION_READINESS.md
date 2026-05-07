# SimSat Submission Readiness

## Runtime

- Sentinel enabled: `True`
- Mapbox enabled: `False`
- Sentinel-first challenge scoring: `True`
- ObservationVLA runtime: `clip_local`

## Checks

- Submission packet exists: `True`
- Submission casebook exists: `True`
- Submission assets dir exists: `True`
- ObservationVLA eval exists: `True`
- Pinned submission cases: `8`

## Scenario Status

| scenario | reviewed_ready | pinned_trace | reviewer | packet_case | image_asset |
| --- | --- | --- | --- | --- | --- |
| disaster_response_weather | True | trace_d886c4872e1047d8b3f50b3563448968 | Ben Haslam | True | True |
| maritime_chokepoints | True | trace_3957ae2fabb34f298e9ff95ec912b6e1 | Ben Haslam | True | True |
| pedospheric_integrity | True | trace_3d10cd03f80b48b88d3fcc895ff8917a | Ben Haslam | True | True |
| urban_coastal_ambiguity | True | trace_4f65355f5c954fbf8db3fc684bb377af (+1 more) | Ben Haslam | True | True |

## Honesty Boundary

- ObservationVLA is now image-model-backed via the configured runtime. The reviewed evaluation pool grew from N=8 to N=37 on 2026-04-27 via `scripts/batch_review.py`; v11 (the first SimSat Gemma-4 fine-tune to actually train language-model parameters — see `notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md`) reaches exact action agreement 0.86 and useful agreement 0.97 over those 37 cases. The local backend should still be treated as an evidence scorer rather than a trusted autonomous action policy.
- Mission-response actions are policy outputs with logged utility, not live spacecraft actuation.
- The reviewed packet is Sentinel-first and does not depend on Mapbox.
