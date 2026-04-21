# S3 — Re-materialize the 3 pinned cases under `clip_local`

## The problem

`SUBMISSION_CASEBOOK.md` (including the Phase 1 M4 rewrite) shows:

```
- Observation runtime: `stub`
```

for all three pinned cases. The current backend is `clip_local` (see `SUBMISSION_READINESS.md`). The traces were pinned when the backend was still `stub`, and the casebook faithfully reports that stored value.

This creates a small but real discrepancy: judges will see **"Observation runtime: stub"** in the casebook next to **"ObservationVLA runtime: clip_local"** in the readiness checklist. Reasonable question from a careful judge: "Which one is it?"

## The fix

Re-assess each of the 3 pinned traces under the current `clip_local` backend, then regenerate the casebook.

The simulator already exposes the reassessment endpoint used by `observation_vla_eval.py`:

```
POST /observation-vla/reassess/{trace_id}
```

This runs the current backend against the stored stimulus and returns an `AssessmentRecord` with `runtime_mode = "clip_local"`. Persisting it updates the trace's `assessment.runtime_mode`, which is what `submission_casebook.py` reads.

## How to apply

Requires: `docker compose up` running, or `--inprocess` mode.

### Step 1 — Re-assess each pinned trace

Run `rematerialize_pinned.sh` (POSIX) or `rematerialize_pinned.ps1` (Windows). Both call `/observation-vla/reassess/{trace_id}` with `persist=true` for each pinned trace_id.

**POSIX:**
```bash
cd D:\SimSat
bash review/2026-04-21-phase-2/S3-rematerialize-clip-local/rematerialize_pinned.sh
```

**Windows:**
```powershell
cd D:\SimSat
pwsh .\review\2026-04-21-phase-2\S3-rematerialize-clip-local\rematerialize_pinned.ps1
```

Expected output: three `200 OK` responses, each reporting `runtime_mode: clip_local`.

### Step 2 — Regenerate the casebook

```bash
python scripts/submission_casebook.py --base-url http://127.0.0.1:8000/sim
```

This rewrites `SUBMISSION_CASEBOOK.md` with the updated runtime_mode. After it runs, grep to confirm:

```bash
grep -c "Observation runtime: \`clip_local\`" SUBMISSION_CASEBOOK.md
# Should print 3
grep -c "Observation runtime: \`stub\`" SUBMISSION_CASEBOOK.md
# Should print 0
```

### Step 3 — Merge with the Phase 1 M4 prose rewrite

The Phase 1 M4 rewrite updated the human review notes but kept `Observation runtime: stub` accurate for that moment. After Step 2, apply the M4 review notes **on top** of the freshly regenerated casebook:

```bash
# Rough recipe: keep the freshly regenerated runtime_mode/metadata fields,
# but swap in the rewritten human review notes from Phase 1 M4.
# Easiest approach: open both files side-by-side and manually port the
# "Review notes:" lines from Phase 1 M4 into the Step-2 regenerated file.
```

(If you already shipped the Phase 1 M4 version to main, this step is just re-pasting the 3 review-note paragraphs into the regenerated file.)

## Fallback: if the reassess endpoint is not wired

If `POST /observation-vla/reassess/{trace_id}` with `persist=true` does not exist in your current API layer (check `src/sim/observation_vla/api_router.py`), the alternative is:

1. Call `POST /encounter/decision/{decision_id}/observation-assess` for each pinned decision (this is what `EncounterService.assess_materialized_decision` exposes).
2. This creates a **new** trace under `clip_local` with a new `trace_id`.
3. Unpin the old trace and pin the new one:
   ```bash
   python scripts/operator_review.py --trace-id <new_trace_id> --pin-submission-case --pinned-by "Ben Haslam" --scenario-pack <pack>
   ```
4. Re-register the operator review (same reviewer, same usefulness_score) on the new trace so the label persists.

This is more surgical but gets the same end state: `SUBMISSION_CASEBOOK.md` reports `clip_local` for all 3 pinned cases.

## Why this is "Should-Do" not "Must-Do"

The discrepancy is small. Most judges will accept "we re-assessed under the new backend, here's the updated eval" and not re-read the casebook metadata line-by-line. The current state is self-consistent if you read the readiness checklist's **Honesty Boundary** section. But it is cleaner to make the two docs agree, and it is cheap to do so.
