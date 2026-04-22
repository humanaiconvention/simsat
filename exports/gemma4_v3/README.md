# SimSat Gemma 4 v3 Export

This bundle is intended for the next SimSat-focused Gemma run on Kaggle T4 or A100 hardware.

## Files

- `simsat_format_train.jsonl`: `66` rows
- `simsat_multimodal_weak.jsonl`: `53` rows
- `simsat_multimodal_reviewed.jsonl`: `0` rows
- `simsat_eval_reviewed.jsonl`: `3` rows
- `simsat_eval_shortlist.jsonl`: `10` rows
- `images/`: `77` copied assets

## Notes

- SimSat native action space is `accept|defer|refine|skip`.
- Reviewed labels are sparse and intentionally preserved as a separate higher-quality slice.
- Weak multimodal rows come from current image-backed SimSat assessments and should be treated as weak labels.
- This export is Sentinel-first and does not depend on Mapbox.

## Runtime Guidance

- T4: start with `simsat_format_train.jsonl` plus `simsat_multimodal_reviewed.jsonl`, then add a capped subset of weak rows.
- A100: use the full bundle and higher decoding limits for structured JSON generation.

## Source Paths

- `D:\SimSat\src\sim\data\observation_vla\traces.json`
- `D:\SimSat\src\sim\data\observation_vla\outcomes.json`
- `D:\SimSat\src\sim\data\observation_vla\submission_cases.json`
- `D:\SimSat\src\sim\data\encounter\targets.json`
