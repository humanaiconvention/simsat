# SimSat Submission Casebook

Generated on `2026-05-05T02:30:40Z` from pinned operator-reviewed submission cases.

## disaster_response_weather

- Target: `Houston Ship Channel`
- Trace: `trace_1926b646ee4b48478913681a33fcdfb1`
- Reviewer: `Ben Haslam`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.92`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2c`
- Cloud cover: `4.050763`
- Review notes: No observable cloud cover, user wouldn't know Houston but appears correct and useful.
- Mission response: action=`materialize_now`, utility_realized=`0.92`

**Register: geometric/structural.** At 4.05% cloud cover the ship channel is effectively clear. The signal of interest is vessel position (berth occupancy, channel clearance, vessel density), industrial infrastructure condition, and potential flood-damage extent along the channel banks. Sub-5% cloud provides reliable structural reads on all of these: vessel silhouettes are unambiguous, berth geometry is legible, and infrastructure edges are intact. Disaster response decisions — emergency routing, berth priority, damage triage — benefit directly from this level of confidence. The `accept → materialize_now` path is unambiguous at this cloud fraction.

![Houston Ship Channel](submission_assets/disaster_response_weather_houston_ship_channel.png)

- Image asset: [submission_assets/disaster_response_weather_houston_ship_channel.png](submission_assets/disaster_response_weather_houston_ship_channel.png)

---

## maritime_chokepoints

- Target: `Suez Canal`
- Trace: `trace_2507337b7939460ebf01cbc9fcef8055`
- Reviewer: `Ben Haslam`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.95`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2a`
- Cloud cover: `0.130026`
- Review notes: No observable cloud cover, user wouldn't know Suez canal but appears correct and useful.
- Mission response: action=`materialize_now`, utility_realized=`0.95`

**Register: geometric/structural.** Near-zero cloud (0.13%) is near-ideal for a maritime chokepoint read. The Suez corridor is linear geometry: transit vessels are well-separated, convoy spacing is measurable, and any blockage or grounding event produces an unambiguous positional anomaly against the canal's known geometry. At this cloud fraction there is no competing hypothesis — vessel positions are accurate, transit density is measurable, and even small positional deviations from the expected transit lane would be detectable. The high usefulness score (0.95) reflects that nothing about this window degraded the geometric read.

![Suez Canal](submission_assets/maritime_chokepoints_suez_canal.png)

- Image asset: [submission_assets/maritime_chokepoints_suez_canal.png](submission_assets/maritime_chokepoints_suez_canal.png)

---

## pedospheric_integrity

- Target: `Mato Grosso Agricultural Frontier`
- Trace: `trace_bfa2fad124374601b3f2884c3be2a42e`
- Reviewer: `Ben Haslam`
- Operator action: `defer`
- Useful: `False`
- Usefulness score: `0.0`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2a`
- Cloud cover: `12.191363`
- Mission response: action=`queue_refine_review`, utility_realized=`0.0`

**Register: spectral-biochemical.** This case operates in a fundamentally different signal domain from the three geometric packs. The signal of interest is soil health at the deforestation frontier, expressed through Sentinel-2 spectral indices rather than structural geometry:

- **NDVI** `(B08 − B04) / (B08 + B04)` — the primary deforestation signal. Advancing clearing produces a sharp NDVI edge that moves between passes; comparing NDVI maps across consecutive overflights shows the rate and direction of clearing. Cloud shadows are radiometrically indistinguishable from actual low-NDVI bare soil, making even modest cloud cover catastrophic for this measurement.
- **EVI** — canopy structure at the forest margin. Less prone to NDVI's saturation in dense high-biomass forest, EVI is sensitive to early canopy thinning before full clearing is visible. Cloud shadows suppress EVI in the same way they suppress NDVI.
- **SWIR ratio B11/B12** — organic carbon and moisture proxy. As forest converts to cleared and cultivated land, soil organic carbon falls and moisture retention drops, raising the B11/B12 ratio. This signal is weaker and requires a clean spectral read; even a few percent cloud cover introduces enough noise to make inter-pass trend analysis unreliable.

At 12.19% cloud the trust layer correctly chose defer. Cloud shadow patches at this coverage fraction would corrupt NDVI and EVI readings at the exact boundary locations most important to track — the deforestation edge itself is often in partial shadow before the full wet-season cloud build-up. Accepting a corrupted spectral baseline and treating it as a data point would poison downstream trend analysis across all future passes over this target. The right call is to wait for a window where the frontier is fully clear.

This is the honest result: the system deferred a pass that was not good enough for the spectral task at hand. A lower-scoring window that leads to a correct `defer` is more valuable to the learning loop than an inflated score from a bad read.

![Mato Grosso Agricultural Frontier](submission_assets/pedospheric_integrity_mato_grosso_agricultural_frontier.png)

- Image asset: [submission_assets/pedospheric_integrity_mato_grosso_agricultural_frontier.png](submission_assets/pedospheric_integrity_mato_grosso_agricultural_frontier.png)

---

## urban_coastal_ambiguity

- Target: `Port of Rotterdam`
- Trace: `trace_4f65355f5c954fbf8db3fc684bb377af`
- Reviewer: `Ben Haslam`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `1.0`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2c`
- Cloud cover: `48.782516`
- Mission response: action=`escalate_operator`, utility_realized=`1.0`

**Register: geometric/structural — trust architecture demo.** This trace is pinned specifically because it demonstrates the scaffold-vs-trust delta that is the system's primary architectural claim. Scaffold scored the window and returned `accept`; the WCLI trust layer overrode to `refine`, and the operator confirmed `refine` was correct. The reason is geometric ambiguity driven by cloud coverage.

At 48.78% cloud the port basin geometry is partially occluded. Rotterdam is a dense mixed-use port: berth occupancy, quay crane positions, and vessel orientations are the structural signals of interest. Cloud shadows over the basin at this coverage fraction create dark patches that are locally indistinguishable from empty berths or open water — the trust layer correctly identified that accepting a 50% cloud port geometry read risks materializing incorrect berth-occupancy intelligence. A refine trigger — requesting a second pass or spectral disambiguation — is the appropriate response.

The operator's usefulness score of 1.0 reflects that the trust layer's refinement decision was precisely what was needed. This is the `accept → refine` transition the architecture is built around: the scaffold commits on geometry alone; the trust layer catches the cloud-driven risk and downgrades before expensive materialization.

![Port of Rotterdam](submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png)

- Image asset: [submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png](submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png)

---

## urban_coastal_ambiguity — San Francisco Bay

- Target: `San Francisco Bay`
- Trace: `trace_a4e31b4c39224d8fbdb2c4bf0f444823`
- Reviewer: `Ben Haslam`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.9`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2b`
- Cloud cover: `6.890936`
- Review notes: No observable cloud cover, user wouldn't know SF Bay at all but accepted as correct and useful.
- Mission response: action=`materialize_now`, utility_realized=`0.9`

**Register: geometric/structural.** A contrast case to Rotterdam within the same scenario pack: same urban coastal scene type, but 6.89% vs 48.78% cloud cover produces a qualitatively different decision. At sub-7% cloud the bay geometry is fully legible — the estuary outline, bridge infrastructure, and port facilities near Oakland provide clear structural anchors. Vessel traffic in the bay and at port facilities is readable without the shadow-confusion risk that affects Rotterdam at 50% cloud. The `accept → materialize_now` path is appropriate: the structural read is reliable and there is no competing cloud-shadow hypothesis that could corrupt vessel or infrastructure intelligence. The usefulness score of 0.9 (vs 0.95 for Suez at near-zero cloud) reflects a marginal residual from the 6.89% cloud fraction — some corner occlusion without full basin interference.

![San Francisco Bay](submission_assets/urban_coastal_ambiguity_san_francisco_bay.png)

- Image asset: [submission_assets/urban_coastal_ambiguity_san_francisco_bay.png](submission_assets/urban_coastal_ambiguity_san_francisco_bay.png)
