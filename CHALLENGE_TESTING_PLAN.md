# SimSat Challenge Testing Plan

This runbook organizes practical testing around the actual competition claim:

> The encounter planner should reduce wasteful materialization, preserve high-value windows through `refine`, and present honest evidence for reviewed challenge cases.

## Lanes

### 1. Fast Gate

Purpose:
Catch broken local behavior before spending time on challenge-specific evidence.

Command:

```powershell
python -m pytest
```

Pass condition:
- test suite is green

### 2. Challenge Matrix

Purpose:
Check the three seeded scenario packs against the current runtime and see whether the challenge story still holds.

Command:

```powershell
python scripts/challenge_test_matrix.py --inprocess
```

Outputs:
- [CHALLENGE_TEST_STATUS.md](./CHALLENGE_TEST_STATUS.md)

What to inspect:
- scenario window counts are non-zero
- trust planner shows meaningful `refine` behavior where expected
- reviewed readiness is present for all three scenario packs
- shortlist traces are visible for the next operator review pass

### 3. Observation Reality Check

Purpose:
Sanity check the image-conditioned evidence lane against reviewed labels.

Command:

```powershell
python scripts/observation_vla_eval.py --inprocess
```

Outputs:
- [OBSERVATION_VLA_EVAL.md](./OBSERVATION_VLA_EVAL.md)

What to inspect:
- current runtime mode
- reviewed sample size
- exact and bucketed agreement
- whether claims should stay narrow

### 4. Review Queue Triage

Purpose:
Prepare the next operator-review pass around disagreement-heavy or high-value traces.

Command:

```powershell
python scripts/review_queue_casebook.py --inprocess
```

Outputs:
- [REVIEW_QUEUE.md](./REVIEW_QUEUE.md)
- assets under [review_queue_assets](./review_queue_assets)

What to inspect:
- `clip_local` traces with disagreement against stored backend outcomes
- one best pending candidate per target
- whether each scenario still has a good next review case

### 5. Submission Evidence Rehearsal

Purpose:
Check that the challenge-facing artifacts still generate cleanly from the current stack.

Commands:

```powershell
python scripts/submission_evidence.py --inprocess
python scripts/submission_casebook.py --inprocess
python scripts/submission_readiness.py --inprocess
```

Outputs:
- [SUBMISSION_PACKET.md](./SUBMISSION_PACKET.md)
- [SUBMISSION_CASEBOOK.md](./SUBMISSION_CASEBOOK.md)
- [SUBMISSION_READINESS.md](./SUBMISSION_READINESS.md)

Pass condition:
- all three files generate
- readiness shows all three scenario packs as reviewed-ready

## Recommended Cadence

- Before code changes:
  run Lane 1
- After planner or observation changes:
  run Lanes 1, 2, and 3
- Before review sessions:
  run Lane 4
- Before demo or submission work:
  run all five lanes

## Horizon Policy

Use two different horizon styles on purpose:

- `Smoke lane`
  Keep the shortest horizon that produces non-zero windows for each scenario pack.
  This is the quick operational check that the planner, probes, and review hooks are still alive.
- `Competition lane`
  Use the first longer horizon where planner-side ambiguity actually appears through `refine`, or the strongest longer horizon if `refine` is still absent.
  This is the lane to use for challenge rehearsal, operator review prep, and submission-facing evidence.

Current recommendation from the live matrix:

- `maritime_chokepoints`
  Smoke lane: `8h`
  Competition lane: `48h`
- `disaster_response_weather`
  Smoke lane: `16h`
  Competition lane: `16h`
- `urban_coastal_ambiguity`
  Smoke lane: `8h`
  Competition lane: `8h`

## Shared Policy Runner

For repeatable local batch execution, the shared scenario-hour policy now lives in [scripts/challenge_run_config.py](./scripts/challenge_run_config.py).

To run the main local challenge lanes with one policy source of truth:

```powershell
python scripts/challenge_scale_runner.py --inprocess --policy competition
```

This writes [CHALLENGE_SCALE_RUN.md](./CHALLENGE_SCALE_RUN.md) with per-lane command output and keeps `challenge_test_matrix.py` plus `submission_evidence.py` aligned on the same scenario-hour mapping.

For a direct offline batch run without API transport, use:

```powershell
python scripts/challenge_offline_batch.py --policy competition
```

That records a run manifest under `src/sim/data/challenge_runs/` and refreshes the matrix, review, submission, readiness, and backlog artifacts from the local service stack.

Deferred medium-term scale follow-ups are tracked in [SCALE_TODO.md](./SCALE_TODO.md).

Interpretation:

- Short horizons are still worth keeping because they quickly catch runtime regressions and missing windows.
- The competition claim about preserving ambiguous high-value opportunities should be judged on the competition lane, not only on the shortest horizon.
- If the smoke lane and competition lane differ, use the competition lane for review queue generation, packet rehearsal, and any claim about planner-side `refine`.

## Suggested Human Workflow

1. Run `python scripts/challenge_test_matrix.py --inprocess`
2. Read [CHALLENGE_TEST_STATUS.md](./CHALLENGE_TEST_STATUS.md)
3. Use the `Horizon Guidance` table to decide whether to stay on the smoke lane or switch to the competition lane for the next scenario-specific pass
4. If reviewed readiness or disagreement looks weak, regenerate [REVIEW_QUEUE.md](./REVIEW_QUEUE.md)
5. Review one or two top traces with [scripts/operator_review.py](./scripts/operator_review.py:1)
6. Rebuild packet/casebook/readiness

## Operator Review Shortcuts

Show candidates:

```powershell
python scripts/operator_review.py --inprocess --scenario-pack maritime_chokepoints --list-only
```

Show one bundle:

```powershell
python scripts/operator_review.py --inprocess --trace-id trace_e9e45ad652ff4e74853f0c6d0c03a827 --show-bundle-only
```

Submit a reviewed label and pin it:

```powershell
python scripts/operator_review.py --inprocess --trace-id trace_e9e45ad652ff4e74853f0c6d0c03a827 --reviewer "Ben Haslam" --operator-action accept --useful true --usefulness-score 0.95 --pin-submission-case --pinned-by "Ben Haslam"
```
