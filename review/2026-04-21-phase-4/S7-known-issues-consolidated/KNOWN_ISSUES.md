# SimSat Known Issues

Consolidated status as of 2026-04-21, end of full-repo review session. Drop this file at the repo root (`D:\SimSat\KNOWN_ISSUES.md`) so judges see a single audit trail.

## Tier 1 — Correctness (submission blockers)

| # | Issue | Location | Status | Fix location |
|---|---|---|---|---|
| 1 | Naive `datetime.fromtimestamp()` emits local-time ISO strings → phase-shifted encounter windows in any non-UTC environment | `src/sim/simulator.py:111` | Patch shipped, awaiting local apply | `review/2026-04-21-phase-1/` |
| 2 | 6 stray `tmp*_simsat_api.py` files at `src/sim/` root | `src/sim/` | Patch shipped, awaiting local apply | `review/2026-04-21-phase-1/` |
| 3 | Duplicate standalone `gui.py` with hardcoded LAN IP and missing imports | repo root | Patch shipped, awaiting local apply | `review/2026-04-21-phase-1/` |
| 4 | Pinned casebook reviewer notes undermine the reviewer ("user wouldn't know X") | `SUBMISSION_CASEBOOK.md` | Patch shipped (S3), awaiting operator_review re-run | `review/2026-04-21-phase-4/S3-casebook-notes/` |

## Tier 2 — Thesis + AI story (should-do before submission)

| # | Issue | Location | Status | Fix location |
|---|---|---|---|---|
| 5 | All 3 pinned cases are `accept→accept`; thesis `accept→refine` is rhetorical on demo | `SUBMISSION_CASEBOOK.md` | Patch shipped, awaiting local apply | `review/2026-04-21-phase-2/S1-accept-refine-delta/` |
| 6 | ObservationVLA usefulness-score MAE 0.27 — model under-confident vs. operator | `OBSERVATION_VLA_EVAL.md` | Patch shipped (turn weakness into rigor note), awaiting local apply. Honesty line already baked into refreshed `SUBMISSION_BRIEF.md` and `CHALLENGE_ENTRY.md`. | `review/2026-04-21-phase-2/S2-mae-calibration/` + `review/2026-04-21-phase-4/S1-submission-brief/` + `review/2026-04-21-phase-4/S2-challenge-entry/` |
| 7 | Pinned traces show `Observation runtime: stub`; current backend is `clip_local` | `SUBMISSION_CASEBOOK.md` | Patch shipped (re-materialize script), awaiting local apply | `review/2026-04-21-phase-2/S3-rematerialize-clip-local/` |
| 8 | HAIC narrative stance | `SUBMISSION_BRIEF.md` + `CHALLENGE_ENTRY.md` + `dashboard/frontend/src/App.tsx` + `dist/arch/*.html` | **Resolved (mechanism-only stance)**. HAIC brand stays out of pitch and dashboard; the viability convention (six non-compensatory gates) is featured in pitch prose described mechanically. Dashboard badge and Architecture Explorer link hidden. Repo source keeps internal HAIC naming for rigor-minded judges who clone and grep. | `review/2026-04-21-phase-4/S1-submission-brief/` + `review/2026-04-21-phase-4/S2-challenge-entry/` + `review/2026-04-21-phase-4/S5-frontend-haic-hide/` |
| 9 | Stacked TTT thesis — VLA + trust-layer test-time training gated by six viability gates | `SUBMISSION_BRIEF.md` + `CHALLENGE_ENTRY.md` | **Thesis locked in**. Described architecturally in refreshed pitch docs. Live on-orbit demonstration requires the hackathon prize hardware (NVIDIA Orin 16GB in-space compute); local demonstration shows wiring and gating logic but not a full on-orbit drift trajectory. | `review/2026-04-21-phase-4/S1-submission-brief/` + `review/2026-04-21-phase-4/S2-challenge-entry/` |

## Tier 3 — Rigor + polish (nice to have)

| # | Issue | Location | Status | Fix location |
|---|---|---|---|---|
| 10 | Sweep generation reports `quality_skips=60, added_distinct=0` on all 3 targets | `REVIEW_SET_BUILD.md` | Not yet investigated — either a dedup hashing bug or genuinely deterministic SSO windows | (open — needs debug session) |
| 11 | 460 of 546 traces are safe-prune `stub_fallback` cruft | ObservationVLA corpus | Documented, prune script ready | `OBSERVATION_VLA_PRUNE.md` (in repo) |
| 12 | `DJANGO_DEBUG` defaults to `1`; `ALLOWED_HOSTS=["*"]`; `CORS_ALLOW_ALL_ORIGINS=True` — leak risk in public deploy | `dashboard/api/settings.py` | Not yet patched | (open — one-line hardening each) |
| 13 | No propagator unit test against a known TLE-epoch expected output | `src/sim/` | Not yet added | (open — could be a 20-line pytest) |
| 14 | `flatted.py` has no clear import — likely dead code | `dashboard/api/flatted.py` | Not yet verified | (open — grep + delete if confirmed dead) |
| 15 | `WindowDetector` uses spherical Earth (R=6371 km) while propagator is WGS84 | `src/sim/encounter/` | Documented, ≤0.2° elevation error at 30° threshold | (accepted, low impact) |
| 16 | Only 3 pinned operator-reviewed cases; N=3 is noisy for model claims | `SUBMISSION_CASEBOOK.md` | Shortlist of additional candidates in `OBSERVATION_VLA_QUALITY.md` | (open — 2-3 more labels would move N to 5-6) |

## Tier 4 — Entry-specific items

| # | Issue | Location | Status | Fix location |
|---|---|---|---|---|
| 17 | Per-track one-sentence thesis statement for the pitch video | `CHALLENGE_ENTRY.md` Two-Track section | Track framing locked in ("Liquid Track (LFM2.5)" / "General AI Track (Gemma-4)"); per-track one-liner video copy still pending | (open — drafted at demo-video time) |
| 18 | Entry B fine-tune dataset design not yet specified | (no file yet) | Deferred pending team discussion | (open) |
| 19 | Entry B VLA backend adapter scaffolded and ready for a concrete model plug-in | `src/sim/observation_vla/transformers_vlm_local.py` (on `entry-b/backend` branch) | Scaffold shipped, model-agnostic, awaiting fine-tuned Gemma-4 weights | `review/2026-04-21-phase-3/B1-gemma4-haic-backend/` (pre-refactor name; now model-agnostic) |
| 20 | Google Drive OAuth scope too narrow (`drive.file` only) — blocks LFM2.5 Colab notebook recon | Platform integration `ac_GcVL6QjxCcpE` | Documented blocker, not on critical path | `review/2026-04-21-phase-3/B2-drive-recon/` |
| 21 | VLA-layer TTT wiring — Entry A (LFM2.5) | ObservationVLA adapter | Architectural claim in pitch docs; concrete adapter-update path and evaluation harness needed for the measurable-improvement-over-base deliverable | (open — pre-submission work) |
| 22 | Trust-layer TTT wiring — WCLI realized-utility feedback loop | `src/sim/encounter/trust_model.py` | Design described in pitch; online update loop needs implementation and verification | (open — pre-submission work) |
| 23 | Viability gate integration with both TTT adaptation streams | `src/sim/haic/viability.py` + ObservationVLA + WCLI trust layer | Viability gates exist as a standalone module; TTT integration points (adapter-update gate, trust-threshold-update gate) need wiring | (open — pre-submission work) |

## Conventions

- **"Patch shipped"** = code/doc change exists in `review/2026-04-21-phase-N/` on `main` in `github.com/humanaiconvention/simsat`.
- **"Awaiting local apply"** = Ben needs to copy the patch into `D:\SimSat\` and run any required scripts.
- **"Resolved"** = handled across current patches; no further action required for submission.
- **"Open"** = not yet addressed; may or may not be shipped before the May 8 deadline.
- **"Locked in"** = thesis decision is final; implementation depth may vary by submission time.

## Hackathon deadline

Friday **May 8, 2026, 5:00 PM PDT** — ~17 days from today.

Critical path to submission:
1. Apply Phase 1 (Tier 1 correctness fixes). ETA: 30 min local work.
2. Apply Phase 4 S1, S2, S4 (doc rewrites). ETA: 5 min.
3. Apply Phase 4 S5 (frontend HAIC brand hide). ETA: 20 min.
4. Apply Phase 2 S1 (accept→refine case), S2 (MAE reframe), S3 (rematerialize). ETA: 2-3 hrs.
5. Apply Phase 4 S3 (casebook re-labels) + run Phase 4 S6 regeneration pipeline. ETA: 1 hr.
6. Wire TTT streams at VLA and trust layers (items #21, #22, #23). ETA: non-trivial; may require prize-hardware access for full on-orbit demonstration — architectural claim plus local verification is the submission-time floor.
7. Finalize Entry B fine-tune dataset + pre-train → Ben × Guilherme. ETA: blocker, not time-boxed.
8. Record the 60-90 sec demo video (20% of grade per rubric) with per-track one-sentence thesis statements.
9. Final readiness gate via `scripts/submission_readiness.py`.
