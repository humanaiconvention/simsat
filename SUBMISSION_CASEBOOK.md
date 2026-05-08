# SimSat Submission Casebook

> **Submitted by HumanAI Convention** · [humanaiconvention.com](https://humanaiconvention.com)
>
> Each case below shows the *operator-labelled JSON within uplink bandwidth
> parameters* loop in action: scaffold proposes, trust layer disagrees or
> agrees, the operator confirms, and the disagreement (or confirmation)
> becomes signal that the six viability gates filter into gradient updates.

Generated on `2026-05-07T04:51:12Z` from pinned operator-reviewed submission cases. One primary case per scenario pack; reviewer=`ben` throughout. Additional pinned traces per pack visible in `gallery_review.html`. Traces marked `stub` were assessed before the torchvision reinstall; the Gemma-4 v11 reviewed eval runs separately against the same Sentinel imagery.

---

## disaster_response_weather

- Target: `Houston Ship Channel`
- Trace: `trace_d886c4872e1047d8b3f50b3563448968`
- Reviewer: `ben`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.85`
- Observation runtime: `stub`
- Sentinel source: `sentinel-2c`
- Cloud cover: `4.050763%`
- Mission response: action=`materialize_now`, utility_realized=`0.85`

**Register: geometric/structural.** At 4.05% cloud the ship channel is effectively clear. The signal of interest is vessel position (berth occupancy, channel clearance, vessel density), industrial infrastructure condition, and potential flood-damage extent along the channel banks. Sub-5% cloud provides reliable structural reads on all of these: vessel silhouettes are unambiguous, berth geometry is legible, and infrastructure edges are intact. Disaster response decisions — emergency routing, berth priority, damage triage — benefit directly from this level of confidence. The `accept → materialize_now` path is unambiguous at this cloud fraction.

![Houston Ship Channel](submission_assets/disaster_response_weather_houston_ship_channel.png)

- Image asset: [submission_assets/disaster_response_weather_houston_ship_channel.png](submission_assets/disaster_response_weather_houston_ship_channel.png)

---

## maritime_chokepoints

- Target: `Suez Canal`
- Trace: `trace_3957ae2fabb34f298e9ff95ec912b6e1`
- Reviewer: `ben`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.85`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2a`
- Cloud cover: `0.130026%`
- Mission response: action=`queue_refine_review`, utility_realized=`0.85`

**Register: geometric/structural.** Near-zero cloud (0.13%) is near-ideal for a maritime chokepoint read. The Suez corridor is linear geometry: transit vessels are well-separated, convoy spacing is measurable, and any blockage or grounding event produces an unambiguous positional anomaly against the canal's known geometry. At this cloud fraction there is no competing hypothesis — vessel positions are accurate, transit density is measurable, and even small positional deviations from the expected transit lane would be detectable.

![Suez Canal](submission_assets/maritime_chokepoints_suez_canal.png)

- Image asset: [submission_assets/maritime_chokepoints_suez_canal.png](submission_assets/maritime_chokepoints_suez_canal.png)

---

## pedospheric_integrity

- Target: `Nile Delta Agricultural Zone`
- Trace: `trace_3d10cd03f80b48b88d3fcc895ff8917a`
- Reviewer: `ben`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.85`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2c`
- Cloud cover: `0.003304%`
- Mission response: action=`queue_refine_review`, utility_realized=`0.85`

**Register: spectral-biochemical.** This case operates in a fundamentally different signal domain from the three geometric packs. The signal of interest is **soil salinization** in irrigated farmland at the Nile Delta — expressed through Sentinel-2 spectral indices rather than structural geometry:

- **NDVI** `(B08 − B04) / (B08 + B04)` — the primary vegetation-stress signal. Salinization suppresses NDVI in irrigated cropland *before* visibly bare soil emerges, producing a measurable downward trend across consecutive overflights.
- **SWIR ratio B11/B12** — soil moisture and clay-content proxy. Salt-affected soils retain moisture differently and produce a characteristic SWIR ratio shift; combined with NDVI, this distinguishes drought stress from salinization.
- **EVI** — canopy structure for cropland. Tracks early canopy thinning ahead of full crop loss.

At **0.00% cloud** this is the ideal spectral-biochemical observation. The operator action `accept` confirms the window is suitable for committing NDVI/SWIR/EVI baselines for inter-pass trend analysis. This case demonstrates the architectural claim that the **same encounter planner, trust layer, viability gates, and TTT loop** operate identically across observational registers — geometric structure (Suez, Houston) and spectral biochemistry (Nile Delta) — without architectural modification.

![Nile Delta Agricultural Zone](submission_assets/pedospheric_integrity_nile_delta_agricultural_zone.png)

- Image asset: [submission_assets/pedospheric_integrity_nile_delta_agricultural_zone.png](submission_assets/pedospheric_integrity_nile_delta_agricultural_zone.png)

---

## urban_coastal_ambiguity

- Target: `Port of Rotterdam`
- Trace: `trace_4f65355f5c954fbf8db3fc684bb377af`
- Reviewer: `ben`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.85`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2c`
- Cloud cover: `48.782516%`
- Mission response: action=`escalate_operator`, utility_realized=`0.85`

**Register: geometric/structural — trust calibration demo.** This trace is pinned specifically because it demonstrates the scaffold↔trust↔operator three-way relationship that is the system's primary architectural claim. Scaffold scored the window and returned `accept`; the WCLI trust layer overrode to `refine` because of compound cloud risk; the operator reviewed the imagery and labeled it `accept` because the visible portion was operationally sufficient.

At 48.78% cloud the port basin is partially occluded. Rotterdam is a dense mixed-use port: berth occupancy, quay crane positions, and vessel orientations are the structural signals of interest. The trust layer's `refine` flag is conservative on purpose — the operator's `accept` override is what tunes its threshold. This is the `accept → refine → accept` calibration loop the architecture is built around: the scaffold commits on geometry alone; the trust layer raises the bar on compound risk; the operator's override becomes signal for the trust-layer TTT loop to re-tune its thresholds.

![Port of Rotterdam](submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png)

- Image asset: [submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png](submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png)
