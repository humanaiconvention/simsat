# ObservationVLA Reviewed Evaluation

## Runtime

- ObservationVLA runtime: `transformers_vlm_local`
- ObservationVLA model: `transformers_vlm:lora:simsat-gemma4-v11-real`
- Reviewed sample size: `37`

## Summary

- Exact operator-action agreement: `0.86`
- Bucketed action agreement: `0.86`
- Useful / not-useful agreement: `0.97`
- Usefulness score MAE: `0.13`

## Reviewed Cases

| scenario | target | trace | model_action | reviewer_action | useful_pred | useful_true | pred_score | true_score | abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maritime_chokepoints | Port of Singapore | trace_da37bb4310014942a24d430ea238110a | skip | refine | False | True | 0.56 | 0.55 | 0.01 |
| disaster_response_weather | Fort Myers Coast | trace_cdcc2a7023d1433e9d4e0edc6bddf10e | refine | refine | True | True | 0.41 | 0.55 | 0.14 |
| maritime_chokepoints | Port of Singapore | trace_777833d1ce0847259cfbac033b271266 | refine | refine | True | True | 0.40 | 0.55 | 0.15 |
| disaster_response_weather | Fort Myers Coast | trace_88c56a41ee64478b94257e26df7f5f77 | refine | refine | True | True | 0.40 | 0.55 | 0.15 |
| maritime_chokepoints | Suez Canal | trace_a4f965f9fce642b9ac42059907c41bd8 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| disaster_response_weather | Houston Ship Channel | trace_ad3890a1998d4d90950201e048708cb4 | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Panama Canal | trace_e0aa6488d0b541f2819fcf257006b636 | refine | refine | True | True | 0.49 | 0.55 | 0.06 |
| disaster_response_weather | New Orleans Delta | trace_b992766137e2419e8b1575e8b4ae3fc2 | refine | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_212662f658bd48028077bc23fb87c442 | refine | refine | True | True | 0.39 | 0.55 | 0.16 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_58951351d068471c81106c7d029e2c81 | refine | accept | True | True | 0.47 | 0.85 | 0.38 |
| maritime_chokepoints | Panama Canal | trace_d5b4d837f3d0444f89633ec10766d73a | refine | refine | True | True | 0.64 | 0.55 | 0.09 |
| disaster_response_weather | New Orleans Delta | trace_0725ba13314d4aec8e695ed0fe2130dd | refine | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_1f00711c8db6433f9e1f7a79691c23d3 | refine | refine | True | True | 0.43 | 0.55 | 0.12 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_03147bb1b2234f489ce01208d11bbf4a | refine | accept | True | True | 0.50 | 0.85 | 0.35 |
| maritime_chokepoints | Panama Canal | trace_599f94be68e14eccb19ce49bac32d609 | refine | refine | True | True | 0.49 | 0.55 | 0.06 |
| disaster_response_weather | New Orleans Delta | trace_ae1998677df5406093334f1fcf0abc66 | refine | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_ecb6521c521446838ea9035d9631f4ca | refine | refine | True | True | 0.39 | 0.55 | 0.16 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_f77a115170574657ba4228ed818ec68d | refine | accept | True | True | 0.47 | 0.85 | 0.38 |
| urban_coastal_ambiguity | Shenzhen Bay | trace_7edbad05378b45eb88f8fedf7a71331c | refine | refine | True | True | 0.39 | 0.55 | 0.16 |
| urban_coastal_ambiguity | Port of Los Angeles | trace_9afbbc332a96426abd0629fc019d395e | refine | accept | True | True | 0.48 | 0.85 | 0.37 |
| maritime_chokepoints | Panama Canal | trace_c6da2e5391a8418c9ec2f379436509f2 | refine | refine | True | True | 0.62 | 0.55 | 0.07 |
| disaster_response_weather | New Orleans Delta | trace_04e08f0b4c4c42869a7663c3cb8bf641 | refine | refine | True | True | 0.52 | 0.55 | 0.03 |
| urban_coastal_ambiguity | San Francisco Bay | trace_67918155fd3e47b8a207ae1bf635269c | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Suez Canal | trace_103f989c13d1437a8927fc031cfefc56 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| urban_coastal_ambiguity | San Francisco Bay | trace_26f2081882cf46ba94e10b53ca5c2707 | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Suez Canal | trace_46c71a07326648039517d0a736b3ba31 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| disaster_response_weather | Houston Ship Channel | trace_d886c4872e1047d8b3f50b3563448968 | accept | accept | True | True | 0.80 | 0.85 | 0.05 |
| urban_coastal_ambiguity | San Francisco Bay | trace_ba0e1a1a2b744cdf94f76fd990fec3f8 | accept | accept | True | True | 0.79 | 0.85 | 0.06 |
| maritime_chokepoints | Suez Canal | trace_f889dbec218e45039cbf63c5b092aa60 | accept | accept | True | True | 0.81 | 0.85 | 0.04 |
| disaster_response_weather | Houston Ship Channel | trace_e5a00b774bcf410fafe9944b8a4cc4f8 | accept | accept | True | True | 0.80 | 0.85 | 0.05 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_4f65355f5c954fbf8db3fc684bb377af | refine | refine | True | True | 0.55 | 0.89 | 0.34 |
| urban_coastal_ambiguity | Port of Rotterdam | trace_fccb18798b844f68b6d0ad583d1e3caf | refine | refine | True | True | 0.71 | 0.87 | 0.16 |
| disaster_response_weather | Fort Myers Coast | trace_f7c1153a370b4c828274566a2714a524 | refine | refine | True | True | 0.40 | 0.90 | 0.50 |
| maritime_chokepoints | Port of Singapore | trace_e9e45ad652ff4e74853f0c6d0c03a827 | refine | refine | True | True | 0.73 | 0.88 | 0.15 |
| urban_coastal_ambiguity | San Francisco Bay | trace_a4e31b4c39224d8fbdb2c4bf0f444823 | accept | accept | True | True | 0.79 | 0.90 | 0.11 |
| disaster_response_weather | Houston Ship Channel | trace_1926b646ee4b48478913681a33fcdfb1 | accept | accept | True | True | 0.80 | 0.92 | 0.12 |
| maritime_chokepoints | Suez Canal | trace_2507337b7939460ebf01cbc9fcef8055 | accept | accept | True | True | 0.81 | 0.95 | 0.14 |

## Per-Pack Breakdown

| pack | correct/total | exact agreement | operator distribution |
| --- | --- | --- | --- |
| disaster_response_weather | 11/11 | **1.00** | refine=7, accept=4 |
| maritime_chokepoints | 11/12 | **0.92** | refine=7, accept=5 |
| urban_coastal_ambiguity | 10/14 | **0.71** | refine=6, accept=8 |
| pedospheric_integrity | — | n/a | (no reviewed cases at v11 N=37 eval time; pedospheric expansion landed in the N=152 cross-distribution eval below — see "Cross-Distribution Eval") |

The urban-coastal pack is the weakest, which is consistent with the `urban_coastal_ambiguity` framing — it is the deliberately hardest geometric pack. v11 reaches **1.00** on disaster_response_weather and **0.92** on maritime_chokepoints.

## Baselines

| baseline | description | exact agreement |
| --- | --- | --- |
| **v11 (current)** | Gemma-4-E2B SimSat fine-tune, image-conditioned | **0.86 (32/37)** |
| Always-majority (`refine`) | Predict the most common operator action every time | 0.54 (20/37) |
| Random uniform (4-class) | Uniform random over {accept, refine, defer, skip} | 0.25 (~9/37) |
| CLIP baseline (clip_local) | OpenAI CLIP ViT-B/32, encoder-only similarity | 0.56 (28/50, on a different and broader pool — see note below) |

v11 outperforms the always-majority baseline by **+0.32 exact agreement** and beats random uniform by **+0.61**. The CLIP baseline is measured on a broader N=50 pool (includes defer/skip cases the v11-eval pool did not contain); a like-for-like comparison would require a re-run of CLIP on the same 37 cases (next eval cycle).

## Caveats and honest scope

- **Class coverage:** The N=37 v11 reviewed pool is `accept` + `refine` only — **no `defer` or `skip` cases**. The 0.86 figure measures v11's accept↔refine boundary, the most operationally consequential decision (commit vs. wait-for-secondary-evidence). Adding `defer/skip` cases is what the expanded N=152 pool (complete — see cross-distribution results below) measures.
- **Cross-register scope:** All 37 cases are from the three geometric/structural packs (maritime, disaster, urban-coastal). v11 was trained on the same register. The pedospheric (spectral-biochemical) register expansion eval **completed 2026-05-06** — see the cross-distribution results below, which include the 54 pedospheric reviews (intermediate eval snapshots archived under `archive/eval_runs/`).
- **Adapter integrity:** v11's saved adapter passes the dynamic LoRA tensor sanity gate at 410/410. See [`V11_AUDIT.md`](./V11_AUDIT.md).

## Cross-Distribution Eval (v11 on N=152 expanded pool, balanced 4-class)

After the operator-review pool was reset and re-labeled from scratch via `gallery_review.html` (2026-05-06), the pool grew to **N=152** with a balanced 4-class operator-action distribution (accept=38, refine=37, defer=38, skip=39) covering all four scenario packs — including 54 newly-generated `pedospheric_integrity` traces (spectral-biochemical register). v11 was re-evaluated against this broader pool on Kaggle T4:

| Metric | v11 in-distribution (N=37) | v11 cross-distribution (N=152) |
|---|---|---|
| Exact action agreement | **0.86** | **0.30** |
| Useful agreement | 0.97 | 0.54 |
| Score MAE | 0.13 | 0.18 |
| Parse rate | 100% | 90.8% (138/152) |
| Class coverage | accept + refine | balanced 4-class (a/r/d/s) |
| Register | geometric | geometric + spectral-biochemical |
| Random baseline | 0.25 | 0.25 |
| Majority baseline | 0.54 (refine) | 0.28 (skip) |

**The 56-point gap is the central distribution-shift finding.** v11 was trained on dataset v2 (713 weighted rows, ~93% accept+refine, only ~12 defer / 4 skip examples). Confronted with a balanced operator pool that includes substantial defer/skip classes — and pedospheric tiles in a different observational register — v11 class-collapses to its training distribution and never produces `defer` or `skip` predictions. Of the 0 operator-defer cases parsed, the model output for those was different enough that the JSON parser couldn't extract an action.

This is exactly the on-orbit-drift problem the architecture is built to solve via **stacked TTT + six non-compensatory viability gates**. The drop is real generalization data, not a sign the model is broken — and it directly motivates the central claim that on-orbit operations cannot rely on the static training distribution remaining valid as a satellite encounters new geographies, seasons, sensor conditions, and observational registers.

## v12 Retrain (In Flight)

Following the v11 cross-distribution finding, **v12 is retraining on dataset v4** (1638 weighted rows, balanced 4-class, includes the 152 fresh operator-reviewed labels). Eval will report two numbers honestly:

- **v12 full-pool (N=152)** — in-distribution; ~141/152 cases are in v12's training set; this is a training-set-leakage metric, useful for v11-vs-v12 delta comparison but not a generalization claim
- **v12 held-out (N=4)** — true generalization, but no statistical power at this sample size

Headline reported when training completes; current state in [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md).

## Tight Claim

- **In-distribution headline:** v11 reaches **0.86 exact action agreement** over 37 operator-reviewed Sentinel cases on the geometric register's accept↔refine boundary, with **1.00 on disaster_response_weather** and **0.92 on maritime_chokepoints**. +0.32 above always-majority, +0.61 above uniform random.
- **Cross-distribution baseline:** v11 reaches **0.30 exact agreement** on the broader N=152 balanced-4-class pool (including spectral-biochemical pedospheric register). The 56-point gap is genuine distribution-shift data — it is not a number to hide; it is the motivation for the on-orbit TTT + viability-gate architecture.
- **Useful agreement / MAE:** 0.97 / 0.13 in-distribution; 0.54 / 0.18 cross-distribution.
