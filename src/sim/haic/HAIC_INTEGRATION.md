# HAIC Convention Integration — SimSat

## Overview

SimSat is extended with a full **HumanAI Convention (HAIC)** protocol layer that transforms each satellite pass into a live grounding session: real Earth imagery is served as a *stimulus*, a human participant describes what they observe, and the AI model's internal geometry is measured before and after to quantify entropy reduction.

---

## Architecture

```
SimSat Simulator (FastAPI :9005)
├── /data/current/position        ← existing
├── /data/current/image/sentinel  ← existing
├── /data/current/image/mapbox    ← existing
└── /haic/                        ← NEW convention layer
    ├── POST /haic/stimulus       — build multi-perspective GroundingStimulus
    ├── GET  /haic/stimulus/{id}  — retrieve stimulus
    ├── GET  /haic/stimulus/{id}/image/{idx} — serve PNG
    ├── POST /haic/session        — create ConventionSession
    ├── GET  /haic/session/{id}   — get session state
    ├── GET  /haic/sessions       — list sessions
    ├── POST /haic/session/{id}/turn  — submit interview turn + PoG telemetry
    ├── POST /haic/session/{id}/close — close, run PRISM + viability
    ├── GET  /haic/session/{id}/receipt — get Merkle receipt
    ├── GET  /haic/windows        — upcoming observation windows
    └── GET  /haic/health         — subsystem health

Django Dashboard (:8000)
└── /sim/*  → transparent proxy to SimSat
```

---

## Session Lifecycle

```
1. BUILD STIMULUS
   StimulusBridge fetches:
   - Sentinel-2 multispectral imagery (scientific perspective)
   - Mapbox high-res perspective imagery (visual perspective)
   → GroundingStimulus (multi-perspective, content-hashed)

2. CREATE SESSION
   ConventionSession created, PRISM entropy snapshot taken (before)

3. INTERVIEW (grounding dialogue)
   Interviewer drives Socratic dialogue grounded in the satellite image
   PoG telemetry (typing cadence, keystroke entropy) captured per turn

4. CLOSE SESSION
   - PoG verification: provenance_score ≥ 0.90 required
   - PRISM snapshot after: entropy delta computed
   - 6 viability gates evaluated (non-compensatory)
   - Merkle receipt generated

5. RECEIPT
   SHA-256 Merkle tree over 7 leaf commitments:
   session_identity | stimulus_commitment | participation_evidence |
   pog_attestation | entropy_delta_proof | viability_gates | settlement_outcome
```

---

## The Prism Principle

Every stimulus contains **two complementary imaging perspectives**:

| Perspective | Source | Characteristics |
|---|---|---|
| `sentinel_rgb` / `sentinel_multispectral` | Sentinel-2 (Copernicus) | Temporal, 10m, multispectral, cloud cover |
| `mapbox_perspective` | Mapbox Static API | High-res, cloud-free, geometric bearing/pitch |

Like a physical prism separating light, these dual perspectives reveal aspects of the observed location that neither source alone can provide — exactly the principle that grounding requires: multiple, complementary views of the same reality.

---

## PRISM Measurement

| Metric | Meaning | Grounding Effect |
|---|---|---|
| `spectral_entropy` | Disorder in eigenvalue distribution | Should *decrease* after grounding |
| `effective_dimension` | Degrees of freedom model uses | Should *increase* after grounding |
| `phase_coherence` | Cross-layer synchronisation | Should *increase* after grounding |
| `geometric_health_score` | Composite demand signal | Higher = more entropy crisis |

**Settlement requires:** `ΔS < −ε` (entropy reduced by at least epsilon)

---

## Viability Gates (non-compensatory)

1. **entropy_reduction** — ΔS < −ε
2. **extraction_risk** — extraction_risk_score ≤ 0.15
3. **prism_consistency** — claimed delta matches measured delta
4. **participation_covenant** — valid stimulus, PoG ≥ 0.90, ≥ 2 turns, ≥ 10 words
5. **federated_exchange** — no raw image data in session record
6. **epistemic_alignment** — diverse vocabulary, non-repetitive interviewer

All 6 must pass. Failure on any single gate blocks settlement regardless of scores on others.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | Enables Claude-powered interviewer |
| `HAIC_PRISM_MODE` | `synthetic` | `synthetic` or `full` |
| `HAIC_PRISM_MODEL` | (none) | HuggingFace model ID for full PRISM |
| `HAIC_SESSION_STORE` | (none) | Path for session persistence JSON |
| `HAIC_EPSILON` | `0.01` | Minimum entropy reduction threshold |
| `MAPBOX_ACCESS_TOKEN` | (required) | For Mapbox perspective imagery |

---

## Quickstart

```bash
# Start the full stack
docker-compose up

# Build a stimulus from current satellite position
curl -X POST http://localhost:9005/haic/stimulus \
  -H 'Content-Type: application/json' \
  -d '{"include_sentinel": true, "include_mapbox": true}'

# Create a session (auto-builds stimulus)
curl -X POST http://localhost:9005/haic/session \
  -H 'Content-Type: application/json' \
  -d '{"auto_stimulus": true}'

# Submit a turn
curl -X POST http://localhost:9005/haic/session/{SESSION_ID}/turn \
  -H 'Content-Type: application/json' \
  -d '{"content": "I can see a river delta with several branching channels..."}'

# Close and analyse
curl -X POST http://localhost:9005/haic/session/{SESSION_ID}/close

# Get receipt
curl http://localhost:9005/haic/session/{SESSION_ID}/receipt
```
