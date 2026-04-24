# SimSat Submission Casebook

Generated on `2026-04-23T21:42:37Z` from pinned operator-reviewed submission cases.

**Note:** The `disaster_response_weather` and `maritime_chokepoints` pinned traces carry `Observation runtime: stub` because they were labeled before the ObservationVLA backend was swapped to `clip_local`. The current backend's recommendation on the same imagery is tracked in `OBSERVATION_VLA_EVAL.md`; re-materialization under `clip_local` is shipped in `review/2026-04-21-phase-2/S3-rematerialize/`. The `urban_coastal_ambiguity` case (Rotterdam) was reviewed under `clip_local` and serves as the accept→refine demonstration case.

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
- Review notes: No observable cloud cover, user wouldn't know Houston but appears correct and useful.
- Mission response: action=`materialize_now`, utility_realized=`0.92`

![Houston Ship Channel](submission_assets/disaster_response_weather_houston_ship_channel.png)

- Image asset: [submission_assets/disaster_response_weather_houston_ship_channel.png](submission_assets/disaster_response_weather_houston_ship_channel.png)

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
- Review notes: No observable cloud cover, user wouldn't know Suez canal but appears correct and useful.
- Mission response: action=`materialize_now`, utility_realized=`0.95`

![Suez Canal](submission_assets/maritime_chokepoints_suez_canal.png)

- Image asset: [submission_assets/maritime_chokepoints_suez_canal.png](submission_assets/maritime_chokepoints_suez_canal.png)

## urban_coastal_ambiguity

- Target: `Port of Rotterdam`
- Trace: `trace_4f65355f5c954fbf8db3fc684bb377af`
- Reviewer: `Ben Haslam`
- Operator action: `refine`
- Useful: `True`
- Usefulness score: `0.89`
- Observation runtime: `clip_local`
- Sentinel source: `sentinel-2c`
- Cloud cover: `48.782516`
- Review notes: Human reviewed confirmed Rotterdam challenge case.
- **Planner delta: scaffold `accept` → trust `refine` (reason: `cloud_risk`, 48.78% cover)**
- Mission response: action=`escalate_operator`, utility_realized=`0.89`

![Port of Rotterdam](submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png)

- Image asset: [submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png](submission_assets/urban_coastal_ambiguity_port_of_rotterdam.png)
