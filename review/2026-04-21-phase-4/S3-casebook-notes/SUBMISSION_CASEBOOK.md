# SimSat Submission Casebook

Generated on `2026-04-10T22:24:14Z` from pinned operator-reviewed submission cases.

**Note:** Pinned traces in this casebook carry `Observation runtime: stub` because they were labeled before the ObservationVLA backend was swapped to `clip_local`. The current backend's recommendation on the same imagery is tracked in `OBSERVATION_VLA_EVAL.md`; re-materialization under `clip_local` is shipped in `review/2026-04-21-phase-2/S3-rematerialize/`.

## disaster_response_weather

- Target: `Houston Ship Channel`
- Trace: `trace_1926b646ee4b48478913681a33fcdfb1`
- Reviewer: `Ben Haslam`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.92`
- Observation runtime: `stub`
- Sentinel source: `sentinel-2c`
- Cloud cover: `4.050763`
- Review notes: Low cloud cover (4.05%) from sentinel-2c at the labeled overpass; the tile shows the linear industrial-corridor geometry between Galveston Bay and downtown Houston with adjacent port infrastructure. Usefulness 0.92 reflects a high-priority maritime-industrial target under adequate visibility. Reviewer verified the tile captures the channel geometry and is not an adjacent stretch of the Gulf coast.
- Mission response: action=`materialize_now`, utility_realized=`0.92`

![Houston Ship Channel](D:\SimSat\submission_assets\disaster_response_weather_houston_ship_channel.png)

- Image asset: [D:\SimSat\submission_assets\disaster_response_weather_houston_ship_channel.png](D:\SimSat\submission_assets\disaster_response_weather_houston_ship_channel.png)

## maritime_chokepoints

- Target: `Suez Canal`
- Trace: `trace_2507337b7939460ebf01cbc9fcef8055`
- Reviewer: `Ben Haslam`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.95`
- Observation runtime: `stub`
- Sentinel source: `sentinel-2a`
- Cloud cover: `0.130026`
- Review notes: Essentially cloud-free (0.13%) from sentinel-2a at the labeled overpass; the tile shows the canal corridor between the Mediterranean and Red Sea with high-contrast desert framing. Usefulness 0.95 reflects a high-priority maritime-chokepoint target under near-ideal viewing conditions. Reviewer verified the tile captures the canal alignment rather than an adjacent stretch of Sinai or the Nile Delta.
- Mission response: action=`materialize_now`, utility_realized=`0.95`

![Suez Canal](D:\SimSat\submission_assets\maritime_chokepoints_suez_canal.png)

- Image asset: [D:\SimSat\submission_assets\maritime_chokepoints_suez_canal.png](D:\SimSat\submission_assets\maritime_chokepoints_suez_canal.png)

## urban_coastal_ambiguity

- Target: `San Francisco Bay`
- Trace: `trace_a4e31b4c39224d8fbdb2c4bf0f444823`
- Reviewer: `Ben Haslam`
- Operator action: `accept`
- Useful: `True`
- Usefulness score: `0.9`
- Observation runtime: `stub`
- Sentinel source: `sentinel-2b`
- Cloud cover: `6.890936`
- Review notes: Moderate cloud cover (6.89%) from sentinel-2b at the labeled overpass; the tile shows the characteristic bay geometry with surrounding urban shoreline. Usefulness 0.90 reflects a high-priority urban-coastal target under usable but not ideal visibility. Reviewer verified the tile captures the bay shoreline rather than open Pacific or inland East Bay.
- Mission response: action=`materialize_now`, utility_realized=`0.9`

![San Francisco Bay](D:\SimSat\submission_assets\urban_coastal_ambiguity_san_francisco_bay.png)

- Image asset: [D:\SimSat\submission_assets\urban_coastal_ambiguity_san_francisco_bay.png](D:\SimSat\submission_assets\urban_coastal_ambiguity_san_francisco_bay.png)
