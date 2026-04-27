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
- Pinned submission cases: `4`

## Scenario Status

| scenario | reviewed_ready | pinned_trace | reviewer | packet_case | image_asset |
| --- | --- | --- | --- | --- | --- |
| disaster_response_weather | True | trace_1926b646ee4b48478913681a33fcdfb1 | Ben Haslam | True | True |
| maritime_chokepoints | True | trace_2507337b7939460ebf01cbc9fcef8055 | Ben Haslam | True | True |
| urban_coastal_ambiguity | True | trace_4f65355f5c954fbf8db3fc684bb377af | Ben Haslam | False | True |

## Honesty Boundary

- ObservationVLA is now image-model-backed via the configured runtime. The reviewed evaluation pool grew from N=8 to N=37 on 2026-04-27 via `scripts/batch_review.py`; v11 (the first SimSat Gemma-4 fine-tune to actually train language-model parameters — see `notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md`) reaches exact action agreement 0.86 and useful agreement 0.97 over those 37 cases. The local backend should still be treated as an evidence scorer rather than a trusted autonomous action policy.
- Mission-response actions are policy outputs with logged utility, not live spacecraft actuation.
- The reviewed packet is Sentinel-first and does not depend on Mapbox.
