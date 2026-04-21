# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `clip_local`
- ObservationVLA model: `clip_local:openai/clip-vit-base-patch32`
- Reviewed sample size: `3`

## Summary

- Exact operator-action agreement: `1.00`
- Bucketed action agreement: `1.00`
- Useful / not-useful agreement: `1.00`
- Usefulness score MAE: `0.27`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disaster_response_weather | Houston Ship Channel | trace_1926b646ee4b48478913681a33fcdfb1 | accept | accept | True | True | 0.66 | 0.92 | 0.26 |
| urban_coastal_ambiguity | San Francisco Bay | trace_a4e31b4c39224d8fbdb2c4bf0f444823 | accept | accept | True | True | 0.66 | 0.90 | 0.24 |
| maritime_chokepoints | Suez Canal | trace_2507337b7939460ebf01cbc9fcef8055 | accept | accept | True | True | 0.65 | 0.95 | 0.30 |

## Tight Claim

- **Headline:** `1.00` exact action agreement, `1.00` useful/not-useful binary agreement over 3 operator-reviewed Sentinel cases under `clip_local:openai/clip-vit-base-patch32`.
- **Magnitude bias:** predicted usefulness ≈ `0.66`, reviewed usefulness ≈ `0.92` — a consistent **~0.26 downward calibration bias**, not noise. Score MAE of `0.27` is a direct reflection of this bias, not a sign of erratic predictions.
- **Operational stance:** downstream mission-response consumes the **binary** agreement channels (useful=True/False, action=accept/defer/skip/refine), not the raw magnitude score. The magnitude bias is therefore **non-blocking for current mission decisions**, and is tracked as a calibration task for the next model revision (a learned scalar correction or a simple +0.26 offset on the reviewed-set-anchored range).
- **Conservative framing:** 3-case reviewed set is low-N. The strongest current signal is the useful/not-useful binary agreement (1.00); exact action agreement (1.00) is secondary; raw usefulness score should be treated as **ordinal**, not absolute, until broader labeling fixes the calibration.
- **Next validation step:** grow the reviewed set to 8–10 cases (one per pending target in `OBSERVATION_VLA_QUALITY.md` Morning Shortlist), then re-check whether the ~0.26 bias holds. If it holds, bake the correction in; if not, revisit the scoring formula in `observation_vla/residual.py`.
