# SimSat

<img src="banner.jpeg" alt="DPhi Space × Liquid AI — Beyond the Clouds. AI in Space." width="800">

[![tests](https://github.com/humanaiconvention/simsat/actions/workflows/tests.yml/badge.svg)](./.github/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue)](./requirements.txt)
[![track](https://img.shields.io/badge/AI%20in%20Space-Liquid%20%2B%20General%20AI-orange)](./CHALLENGE_ENTRY.md)
[![status](https://img.shields.io/badge/status-all%20tiers%20complete%20%7C%20Gemma--4%20v11%20canonical-brightgreen)](./KNOWN_ISSUES.md)
[![kaggle](https://img.shields.io/badge/Kaggle-simsat--gemma4--v1-20BEFF)](https://www.kaggle.com/code/benhaslam/simsat-gemma4-v1-training)
[![license](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)](./LICENSE)

## TL;DR for judges

**SimSat is a governed on-orbit continual-learning loop.** It treats mission operations as a sequence of encounter windows: a deterministic scaffold ranks them cheaply, a WCLI-style trust layer decides accept/defer/skip/refine before expensive imagery materialization, and a Sentinel-first ObservationVLA lane performs image-conditioned reassessment. **Test-time training runs simultaneously at the VLA-weight and trust-threshold layers**, with both adaptation streams gated by **six non-compensatory viability checks** before any update persists. Scenarios span two observational registers — geometric/structural (maritime, disaster, urban-coastal) and spectral-biochemical (pedospheric integrity via NDVI / SWIR / EVI). Identical pipeline across both.

**Headline evidence (operator-reviewed eval):**
- **v11 in-distribution N=37** (accept↔refine binary boundary): exact **0.86** (32/37) · useful **0.97** · MAE **0.13** · per-pack disaster=1.00, maritime=0.92, urban-coastal=0.71
- **v11 cross-distribution N=152** (balanced 4-class, includes 54 spectral-biochemical pedospheric cases): exact **0.30** — distribution-shift evidence motivating the on-orbit TTT + viability-gate architecture
- **v12 retrain on dataset v4 (1638 rows, balanced 4-class)**: eval complete — **not promoted**. 65% parse failure (53/152 parsed); effective exact ≈0.18 vs v11's 0.30. v11 remains canonical. See [`OBSERVATION_VLA_EVAL.md`](./OBSERVATION_VLA_EVAL.md) and `KNOWN_ISSUES.md` #28
- **Trust-layer TTT (10 seeds)**: 22.5% MAE improvement ± 0.1%; **distribution shift (coastal→polar)** demo: polar MAE 0.161 → 0.150 (−6.9%) in **1 cycle** — 94% of polar-regime improvement in a single pass (`ttt_stability_analysis.md`)
- **MuZero Liquid Track BC (10 seeds, LFM2.5-VL-450M)**: best val_acc **0.898 ± 0.049** range [0.825, 0.975] (`MUZERO_SEED_SWEEP.md`)
- **LFM2.5-VL-450M v3 LoRA fine-tune (canonical, Liquid Track)**: matched-pair holdout 32 samples (8 per class) — `exact_action_agreement` 0.156 → **0.844** (+68.8 pp), `score_mae` 0.365 → **0.055** (−31.0 pp), per-class accept **1.000** / refine **1.000** / defer 0.625 / skip 0.750. Adapter live at [`HumanAIConvention/simsat-lfm25vl-450m-v3`](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3). v4 trained on +20 more rows but regressed (kept v3); v5 with class-balanced augmentation (defer 34 → 51) in flight overnight on BEAST. Recipe: `LFM_FINETUNE_METHODOLOGY.md`.
- **VLA-layer TTT receipt**: `OnlineLoRAStepper` exercised on real LFM2.5-VL with v1 adapter, 5/5 attempted steps applied, 0 viability gates triggered, post-MAE held perfect on stratified probe (`LFM_TTT_POC.md` §"Live receipt — 2026-05-07")
- **Viability gates exercise**: 1100 updates × 3 synthetic streams; gates selective and fire on designed conditions (`VIABILITY_GATES_EXERCISE.md`)
- **Encounter planner (248 windows, 5×48h, 4 packs)**: scaffold produces zero `refine`; trust adds 72; 92/248 changed action
- **v11 audit**: 410/410 LoRA tensor sanity gate **PASS** ([`V11_AUDIT.md`](./V11_AUDIT.md))

**Judges, start here:** [SUBMISSION_TLDR.md](./SUBMISSION_TLDR.md) (1 page, 30 sec) → [SUBMISSION_BRIEF.md](./SUBMISSION_BRIEF.md) (rubric mapping) → [SUBMISSION_CASEBOOK.md](./SUBMISSION_CASEBOOK.md) (4 pinned cases) → [CHALLENGE_ENTRY.md](./CHALLENGE_ENTRY.md) (full thesis) → [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md) (auto-generated state).

**Why this must run in orbit:** distribution shift without a ground-truth validator + a 5 MB uplink budget that can't ship weight updates + the next encounter window arrives in minutes. See [SUBMISSION_BRIEF.md](./SUBMISSION_BRIEF.md) for the three-constraint argument.

---

## Repository Layout — Top-Level Docs

| Category | File | Purpose |
| --- | --- | --- |
| **Submission front matter** | [`SUBMISSION_BRIEF.md`](./SUBMISSION_BRIEF.md) | Short pitch — claim + tracks + headline numbers |
| | [`SUBMISSION_PACKET.md`](./SUBMISSION_PACKET.md) | Scorecard + per-pack cases + reviewer attribution |
| | [`SUBMISSION_CASEBOOK.md`](./SUBMISSION_CASEBOOK.md) | Pinned case-by-case visual narrative (one per scenario pack) |
| | [`SUBMISSION_READINESS.md`](./SUBMISSION_READINESS.md) | Pre-submission checklist |
| | [`CHALLENGE_ENTRY.md`](./CHALLENGE_ENTRY.md) | Long-form thesis — TTT mechanics, viability gates, two-track architecture |
| | [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) | Issue tracker + session log |
| **Evidence** | [`OBSERVATION_VLA_EVAL.md`](./OBSERVATION_VLA_EVAL.md) | v11 VLA eval (in-distribution N=37 + cross-distribution N=152) |
| | [`V11_AUDIT.md`](./V11_AUDIT.md) | Canonical model 410/410 sanity-gate audit |
| | [`ENCOUNTER_EVAL.md`](./ENCOUNTER_EVAL.md) | Scaffold-vs-trust planner eval over 248 windows × 4 packs |
| | [`MUZERO_LFM_EVAL.md`](./MUZERO_LFM_EVAL.md) | Liquid Track BC eval (LFM2.5-VL-450M + SigLIP-base) |
| | [`MUZERO_SEED_SWEEP.md`](./MUZERO_SEED_SWEEP.md) | 10-seed BC variance bands on canonical encoder |
| | [`VIABILITY_GATES_EXERCISE.md`](./VIABILITY_GATES_EXERCISE.md) | Six-gate fire-rate exercise under three failure conditions |
| | [`ttt_stability_analysis.md`](./ttt_stability_analysis.md) | Trust-layer TTT 10-seed stability (22.5% MAE improvement) |
| | [`LFM_FINETUNE_METHODOLOGY.md`](./LFM_FINETUNE_METHODOLOGY.md) | LFM2.5-VL-450M LoRA fine-tune methodology + eval protocol |
| | [`LFM_TTT_POC.md`](./LFM_TTT_POC.md) | VLA-layer TTT (online LoRA updates per encounter) — implementation receipt |
| **Repo standard** | [`COLLABORATOR_GUIDE.md`](./COLLABORATOR_GUIDE.md) | Eight-key VLA contract + adapter integration |
| | [`CONTRIBUTING.md`](./CONTRIBUTING.md) | PR / branching / CI conventions |
| | [`README.md`](./README.md) | This file (includes "Plugging in another VLM" section, formerly `START_HERE.md`) |

Intermediate eval runs, per-seed dumps, and superseded snapshots live under [`archive/`](./archive/) (cleaned up 2026-05-07; not part of the submission review).

---

## What this simulator does

An orbit propagator calculates the satellite position over time and an API serves as an interface to on-board users, sharing the current position, timestamp, and Sentinel-2 imagery from that location. A web-based dashboard controls and visualizes the simulation. The encounter planner, WCLI trust/refine flow, ObservationVLA traces, and viability gates run on top of this base layer.

---

## Plugging in another VLM

The ObservationVLA backend is model-agnostic. Any model that emits the eight-key JSON contract (schema at `src/sim/observation_vla/observation_payload.schema.json`) can serve as the backend without code changes. Find the path below that matches your model and follow the steps; everything else in the codebase runs on its own.

### Smoke path (first 30 seconds)

```bash
git clone <repo>
cd SimSat
cp .env.example .env
pip install -r requirements.txt
python scripts/quickstart.py
```
Runs the test suite + an eval against the pinned reviewed cases using the `clip_local` baseline. If it exits 0, your clone is wired correctly.

### If your model emits a JSON assessment (like the SimSat Gemma-4 fine-tune)

Your model needs to emit the 8-key payload defined in `observation_payload.schema.json` given the prompt shape documented in [`COLLABORATOR_GUIDE.md`](./COLLABORATOR_GUIDE.md). For a HuggingFace-style VLM, no code change is needed:

```bash
export OBSERVATION_VLA_BACKEND=transformers_vlm
export OBSERVATION_VLM_BASE_MODEL=your-org/your-model
export OBSERVATION_VLM_LORA_PATH=/path/to/your/adapter   # if you have one
export OBSERVATION_VLM_MODE=lora
python scripts/observation_vla_eval.py --inprocess
```

For a custom load path (non-HF or non-PEFT), add a sibling file to `src/sim/observation_vla/transformers_vlm_local.py` exposing the same `assess()` interface, then route it via `backend_factory.py`.

### If your model emits a per-pixel heatmap

(Alarm / anomaly / salience / change-detection — output is a 2-D `(H, W)` float map, higher = stronger signal.)

Fill in two `NotImplementedError` stubs near the top of `src/sim/observation_vla/heatmap_local.py`:
- `_load_model(weights_path, device)` — load your model
- `_predict_heatmap(model, tile, device)` — return `(H', W')` float32 numpy

The stats pooling, action mapping, and 8-key payload assembly are concrete and tested (`pytest tests/test_heatmap_adapter.py` — 11 tests).

```bash
export OBSERVATION_VLA_BACKEND=heatmap
export HEATMAP_WEIGHTS_PATH=/path/to/your/weights
export HEATMAP_DEVICE=cuda
export HEATMAP_BACKEND_TAG=your_model_name   # appears in rationale_tags
python scripts/observation_vla_eval.py --inprocess
```

Optional knobs to tune the action ladder:
```bash
export HEATMAP_PEAK_ACCEPT=0.80   # peak >= this AND scene_match >= 0.5 → accept
export HEATMAP_PEAK_REFINE=0.45   # peak >= this → refine
export HEATMAP_PEAK_DEFER=0.20    # peak >= this → defer (else skip)
```

### What you'll see when it works

`scripts/observation_vla_eval.py` prints exact / bucketed action agreement, useful / not-useful agreement, usefulness-score MAE, and a per-case breakdown.

**Reference numbers to beat** (Gemma-4 v11 SimSat fine-tune over N=37 operator-reviewed cases on the geometric register): exact **0.86**, bucketed **0.86**, useful **0.97**, MAE **0.13**. Full eval at [`OBSERVATION_VLA_EVAL.md`](./OBSERVATION_VLA_EVAL.md).

For PRs, branching, and CI: see [`CONTRIBUTING.md`](./CONTRIBUTING.md). For known limitations: [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md).

---

## Challenge Entry

This repo now also contains a challenge-focused encounter planner that treats observation opportunities as structured mission events instead of only continuous propagation. The challenge entry compares:
- a deterministic scaffold planner
- a WCLI-style trust-gated planner with an explicit `refine` action

The compact submission note and reproducible demo flow are in [CHALLENGE_ENTRY.md](./CHALLENGE_ENTRY.md).
The shortest judge-facing handoff is in [SUBMISSION_BRIEF.md](./SUBMISSION_BRIEF.md).
The current reviewed ObservationVLA evaluation is in [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md).

This repo is submitted to both tracks of the AI in Space hackathon — the Liquid Track (LFM2.5-VL encoder → MuZero) and the General AI Track (Gemma-4-E2B fine-tune). See [CHALLENGE_ENTRY.md](./CHALLENGE_ENTRY.md) for per-track details and [COLLABORATOR_GUIDE.md](./COLLABORATOR_GUIDE.md) for plugging in a new model backend.

The challenge path is explicitly `no-Mapbox-safe`: Sentinel imagery plus orbital geometry are sufficient for the encounter planner, WCLI trust/refine flow, ObservationVLA traces, and evaluation scripts. Mapbox remains an optional high-resolution perspective source, not a requirement.

### Quick Challenge Demo

For a low-compute, judge-friendly comparison run:

```bash
python scripts/encounter_eval.py --base-url http://127.0.0.1:8000 --scenario-sweep --top-k 8 --materialize-top-k 2 --markdown
```

That prints a compact scorecard across all four scenario packs:
- `maritime_chokepoints`
- `disaster_response_weather`
- `urban_coastal_ambiguity`
- `pedospheric_integrity`

This works even when `MAPBOX_ACCESS_TOKEN` is unset.

For a narrated single-scenario walkthrough:

```bash
python scripts/challenge_demo.py --base-url http://127.0.0.1:8000 --scenario-pack maritime_chokepoints
```

For a markdown-ready submission evidence report built from stored evaluations and labelled traces:

```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000
```

That generates [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md) with one curated case per scenario pack. By default the packet uses Sentinel-first evidence, prefers operator-reviewed labels when they exist, and otherwise falls back to `simulated_submission_case`.

For the current ObservationVLA backend evaluation against operator-reviewed traces:

```bash
python scripts/observation_vla_eval.py --inprocess
```

That generates [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md). The canonical backend is the Gemma-4-E2B SimSat fine-tune **v11** (`OBSERVATION_VLA_BACKEND=gemma4`), available on Hugging Face at [`HumanAIConvention/simsat-gemma4-v11`](https://huggingface.co/HumanAIConvention/simsat-gemma4-v11). On 37 operator-reviewed Sentinel cases v11 achieves usefulness MAE **0.13**, exact action agreement **0.86**, useful/not-useful agreement **0.97**. v11's adapter passes the dynamic LoRA tensor sanity gate at **410/410** — the correct count for Gemma-4-E2B's GQA architecture (15 canonical k/v modules × 2 + 35 layers × 5 non-k/v modules × 2 = 410); see [`V11_AUDIT.md`](./V11_AUDIT.md). v17–v19 retrained on an expanded auto-defer dataset and regressed to ~0.46 — that distribution-shift discovery is itself a documented finding (see [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) #27). The `clip_local` CLIP baseline remains available as a zero-weight-download reference. See [`notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md`](./notebooks/GEMMA4_LORA_NULL_TRAINING_AUDIT.md) for the v1–v10 null-training audit.

To replace a simulated label with a real operator-reviewed outcome:

```bash
python scripts/review_queue_casebook.py --inprocess
python scripts/operator_review.py --base-url http://127.0.0.1:8000 --scenario-pack maritime_chokepoints
```

The review queue writes per-case images so the next human-review pass can compare the stored trace assessment with the current backend recommendation on the same imagery.

For a specific trace, you can inspect the full review bundle first:

```bash
python scripts/operator_review.py --base-url http://127.0.0.1:8000 --trace-id <trace_id> --show-bundle-only
```

Then replace the current label and pin it as the canonical submission case for that scenario:

```bash
python scripts/operator_review.py --base-url http://127.0.0.1:8000 --trace-id <trace_id> --reviewer "Your Name" --operator-action accept --useful true --usefulness-score 0.95 --pin-submission-case --pinned-by "Your Name"
```

Once one reviewed case is pinned per scenario pack, generate a strict reviewed-only packet:

```bash
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000 --reviewed-only
```

To generate a visual companion with the pinned case images:

```bash
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000
```

To generate a low-compute readiness checklist for the current submission artifacts:

```bash
python scripts/submission_readiness.py --base-url http://127.0.0.1:8000
```

### Multi-Model VLA Backends

The ObservationVLA lane is model-agnostic by design. Select a backend via `OBSERVATION_VLA_BACKEND`:

| Value | Model |
|---|---|
| `clip_local` | CLIP ViT-Base/32 (default baseline) |
| `gemma4` | Gemma-4-E2B SimSat fine-tune (canonical) |
| `transformers_vlm` | Any HuggingFace VLM that emits the eight-key contract |

All backends share the same eight-key JSON contract (`src/sim/observation_vla/observation_payload.schema.json`) and pass through the same WCLI trust layer, viability gates, and TTT loop. See `COLLABORATOR_GUIDE.md` for the integration contract.

### Known Issues

See `KNOWN_ISSUES.md` at the repo root for the consolidated issue tracker with current status per item. All Tier-1 correctness items are resolved.

## Hackathon Context

Upstream simulator for the [AI in Space Hackathon](https://luma.com/n9cw58h0) (DPhi Space × Liquid AI). This SimSat fork extends the base simulator with the encounter planner, WCLI trust gate, ObservationVLA lane, viability gates, and stacked TTT loop documented in [`SUBMISSION_BRIEF.md`](./SUBMISSION_BRIEF.md).

## Table of Contents
- [Getting Started](#getting-started)
- [Simulation Control](#simulation-control)
- [APIs](#apis)
  - [/data/current/position](#get-datacurrentposition)
  - [/data/current/image/sentinel](#get-datacurrentimagesentinel)
  - [/data/current/image/mapbox](#get-datacurrentimagemapbox)
  - [/data/image/sentinel](#get-dataimagesentinel)
  - [/data/image/mapbox](#get-dataimagemapbox)
- [Datasets](#datasets)
  - [Sentinel-2](#sentinel-2)
  - [Mapbox](#mapbox)
- [Test Examples](#test-examples)

---
# Getting Started
To start the environment, start the Docker containers with

```bash
docker compose up
```

After the startup, the dashboard is accessible at [http://localhost:8000](http://localhost:8000). To start the simulation, press the start button. Then you can follow the satellite by clicking on the blue dot followed by clicking on the camera icon in the popup window. 

The API to fetch images is accessible at [http://localhost:9005](http://localhost:9005). In order to test this, you can run the provided script `scripts/api_test.py` from your host machine. This will fetch an image from the API and display it using matplotlib.
```bash
python scripts/api_test.py
```

Note: if no image is displayed, the satellite might be over the ocean. More details about image availability are reported in the [Datasets Section](#datasets).

For the challenge entry specifically, Mapbox is optional. If you do not want to create a billed Mapbox account, leave `MAPBOX_ACCESS_TOKEN` unset and use the encounter planner, evaluation scripts, and Sentinel-backed ObservationVLA flow as-is.

---
# Simulation Control
The simulation can be controlled through the web dashboard by setting the following parameters:
- **Start time:** it must be in the ISO-8601 UTC format `YYYY-MM-DDThh:mm:ssZ`. For example `2026-01-01T16:00:00Z`. 
- **Step size:** simulated time increment (in seconds) between each simulation update.
- **Replay speed:** how fast simulation time runs compared to real time (`1` = real time, `2` = twice as fast).

The changes are applied when the start button is pressed. If the simulation is not able to run as fast as the settings require, the system will throttle itself down. We recommend to set the step size and replay speed such that `replay_speed / step_size <= 2`.

---
# APIs

The satellite can be accessed through the provided APIs. The base URL for the APIs is [http://localhost:9005](http://localhost:9005).

### GET /data/current/position

This endpoint returns the current position of the satellite in latitude (degrees), longitude (degrees), and altitude (kilometers), as well as the current simulation timestamp.

**Response Example:**
```bash
{
  'lon-lat-alt':[130.39051988505403,17.87271962388168,791.3415172015517],
  'timestamp':'2026-01-01T16:00:00Z'
}
```

### GET /data/current/image/sentinel
This endpoint returns an image from the Sentinel-2 dataset for the current satellite position. More information about Sentinel-2 images can be found in the [Datasets Section](#datasets)

**Query Parameters:**
- `spectral_bands`: Comma-separated list of spectral bands to include in the image (default: "red","green","blue")
- `size_km`: Size of the image in kilometers (default: 5.0)
- `return_type`: Format of the returned image, either "png" or "array" (default: "png")
- `window_seconds`: Length of the time window (in seconds) used to search Sentinel images before the current simulation timestamp (default: 864000, i.e. 10 days)


**Response Example:**
If `return_type="png"` an image file is returned as the response, while if `return_type="array"` base64-encoded raw array bytes are returned. Also, the following metadata are returned:
```bash
{
  'image_available': True, 
  'source': 'sentinel-2a', 
  'spectral_bands': ['red', 'green', 'blue'], 
  'footprint': [-31.744769220901965, 67.84555724102906, -31.62541380860759, 67.890523321325],'size_km': 5.0, 
  'cloud_cover': 86.716813, 
  'datetime': '2026-03-16T13:53:37Z', 
  'satellite_position': [-31.685091514754777, 67.86804028117703, 800.0824433233049],
  'timestamp': '2026-03-17T14:03:19Z'
}
```
`image_available` is False when a Sentinel image for the current location does not exist (usually over the ocean or near the poles), or when it's not available in the specified time window.

`source` can be either "sentinel-2a", "sentinel-2b" or "sentinel-2c" depending on which of the three Sentinel satellites captured the image.

`footprint` specifies the ground area included in the image in the format [lon_min, lat_min, lon_max, lat_max].

`cloud_cover` represents the percentage of the image covered by clouds.

`datetime` indicates the timestamp at which the image was captured by the Sentinel satellite, which in general does not coincide with the timestamp of the simulated satellite. However, the retrieved image is the latest one relative to the simulation timestamp.

### GET /data/current/image/mapbox
This endpoint returns an image from the Mapbox dataset for the current satellite position pointing to a specified target location. The bearing (direction) and pitch (angle) are calculated based on the satellite position and the target location. If the elevation angle is smaller than 30° the target location is considered not visible from the satellite. More information about Mapbox images can be found in the [Datasets Section](#datasets)

**Query Parameters:**
- `lon`: Longitude of the target location (default: current satellite longitude)
- `lat`: Latitude of the target location (default: current satellite latitude)

**Response Example:**
A PNG image file is returned as the response. Also, the following metadata are returned:
```bash
{
  'target_visible': True, 
  'image_available': True, 
  'elevation_degrees': 90.0, 
  'zoom_factor': 13.406814757215615, 
  'bearing': 0.0, 
  'pitch': 0.0, 
  'satellite_position': [-64.45602720609857, -27.086293979486268, 799.2307382513263]
  'timestamp': '2026-03-17T14:30:20Z'
}
```

An API key for Mapbox (free tier available) is required to use this endpoint. Set the environment variable `MAPBOX_ACCESS_TOKEN` to your access token before starting the simulation. More details can be found in the [Datasets Section](#datasets).

This endpoint is optional and returns `503` when Mapbox is disabled. The challenge entry does not depend on it.

### GET /data/image/sentinel
This endpoint returns an image from the Sentinel-2 dataset for a given position and timestamp (not from the current satellite simulation). The metadata returned are the same as the `/data/current/image/sentinel` endpoint except `satellite_position` and `timestamp`.

**Query Parameters:**
- `lon`: Longitude of the requested location (float)
- `lat`: Latitude of the requested location (float)
- `timestamp`: Timestamp of the request (ISO-8601 UTC format) 
- `spectral_bands`: Comma-separated list of spectral bands to include in the image (default: "red","green","blue")
- `size_km`: Size of the image in kilometers (default: 5.0)
- `return_type`: Format of the returned image, either "png" or "array" (default: "png")
- `window_seconds`: Length of the time window (in seconds) used to search Sentinel images before the requested timestamp (default: 864000, i.e. 10 days)

### GET /data/image/mapbox
This endpoint returns an image from the Mapbox dataset for a given satellite position (not from the current satellite simulation) and a given target location. The metadata returned are the same as the `/data/current/image/mapbox` endpoint except `satellite_position` and `timestamp`.

**Query Parameters:**
- `lon_target`: Longitude of the target location (float)
- `lat_target`: Latitude of the target location (float)
- `lon_satellite`: Longitude of the satellite (float)
- `lat_satellite`: Latitude of the satellite (float)
- `alt_satellite`: Altitude of the satellite in kilometers (float)

---
# Datasets
We provide access to two different datasets. This section describes their features and limitations.

## Sentinel-2
[Sentinel-2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-data/sentinel-2) is a European multispectral Earth observation mission. The data is freely available in high (3-5 days interval) temporal resolution and medium (10m) spatial resolution.

Multispectral images allow the observation of spectral bands outside the commonly used red-green-blue (RGB) color space. This allows the analysis of features not visible in RGB images. The example below shows a landscape in RGB (left) and false color infrared (right). It is almost impossible to see the river in the RGB image while it is clearly visible in the false color infrared image.

<img src="fig/rgb_and_multispectral_example.png" alt="Sentinel image example" width="800">

### Image Availability
Sentinel-2 consists of three satellites (Sentinel-2A, Sentinel-2B, and Sentinel-2C) that together provide a revisit frequency of about 5 days at the equator, and higher frequency (typically 2–3 days) at mid-latitudes. This means that a new image of the same location, with a similar viewing geometry, is usually available every 2 to 5 days. The mission follows a predefined acquisition plan that determines when the sensor is active and when images are recorded. As a result, not every satellite overpass produces usable imagery, even if the satellite passes over the requested area. In particular, acquisitions over the ocean are often not recorded unless they are close to the coastline. More detailed information about the mission, coverage and acquisition can be found here: [Sentinel-2 Mission](https://sentiwiki.copernicus.eu/web/s2-mission)

### Spectral Bands
Sentinel-2 provides 13 spectral bands spanning from the visible and near infrared to the shortwave infrared. Detailed information about the available bands, how they can be combined to obtain useful insights, and popular remote sensing indices can be found here: [Sentinel-2 Multispectral](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel/sentinel-2/) 

### Uses and Limitations
Sentinel-2 images should be used for applications where temporal information and multispectral analysis are required, but high resolution is not.

It is important to note that, due to the Sentinel-2 acquisition plan, at a given time the most recent images of nearby areas around a given location may have been captured at different times and possibly by different satellites. Therefore, for some applications additional checks may be required to ensure temporal consistency.

In addition, Sentinel-2 data is divided in tiles. As a result, some requested regions may lie close to tile boundaries, causing images to appear partially cut off or filled with black areas where no data is available. If an image appears completely white, it is most likely due to cloud coverage at the time of acquisition.

The Sentinel-2 API is quite slow, so take it into account when developing your applications.

## Mapbox

The [Mapbox static images API](https://docs.mapbox.com/api/maps/static-images/) is used to generate satellite imagery of a given location, bearing, and pitch. The images have high spatial resolution (10-30cm) but are static, meaning they don't have a timestamp associated and they are not updated regularly. Only RGB images are provided, with no other bands available. The data is cloud-free and available globally in an uniform manner.

To use Mapbox images, go to [mapbox.com](https://www.mapbox.com/) and create an account to get an access token. Set the environment variable `MAPBOX_ACCESS_TOKEN` to your access token.

Mapbox is not required for the encounter-planning challenge flow. It is only needed if you specifically want the high-resolution perspective-image endpoints.

### Uses and Limitations
Mapbox images should be used for applications that are not time-dependent, where high resolution is necessary and radiometric accuracy is not required. 

Mapbox uses a 2D map on a 3D globe to create perspectives. This looks ok when observing regions where the 2D approximation holds. However, when we observe skyscrapers for instance, we have completely wrong perspectives. Also, Mapbox does not use real images to map the ocean which can lead to unrealistic monocolor images or sometimes even sampling bugs with white regions. Some examples are shown below:

<img src="fig/country_example.png" alt="Image of the country region" width="500">

*Figure 1: Mapbox static image of a nature region.*

<img src="fig/wrong_perspective_example.png" alt="Image of New York City with incorrect perspective" width="500">

*Figure 2: Mapbox static image of New York City with incorrect perspective as the image was taken at a different angle than the current perspective.*



<img src="fig/ocean_example.png" alt="Image of an ocean region" width="500">

*Figure 3: Mapbox static image of an ocean region.*

<img src="fig/bug_example.png" alt="Image with sampling bug in the ocean region" width="500">

*Figure 4: Mapbox static image with sampling bug in the ocean region.*

---
# Test Examples

`scripts/api_test.py` contains test functions for the available API endpoints and can be used as a reference for how to interact with the API. Several tests are provided, each demonstrating a different endpoint.

You can run a specific test by passing an argument to the script:
```bash
python scripts/api_test.py ARG
```

Available arguments:
- `sentinel`: Retrieves and displays a Sentinel-2 RGB image for a predefined location and timestamp
- `sentinel_current`: Retrieves and displays a Sentinel-2 RGB image for the current simulated satellite position and timestamp
- `sentinel_multispectral`: Retrieves and displays Sentinel-2 images using five different band combinations for the current simulated satellite position and timestamp
- `mapbox`: Retrieves and displays a Mapbox image for a predefined location
- `mapbox_current`: Retrieves and displays a Mapbox image for the current simulated satellite's position

If no argument is provided, the script runs the `sentinel_current` test by default.

