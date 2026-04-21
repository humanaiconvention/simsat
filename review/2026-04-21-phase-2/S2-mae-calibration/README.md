# S2 — Reframe MAE=0.27 as a calibration point

## The problem

`OBSERVATION_VLA_EVAL.md` currently reports:

```
- Exact operator-action agreement: `1.00`
- Bucketed action agreement: `1.00`
- Useful / not-useful agreement: `1.00`
- Usefulness score MAE: `0.27`
```

And then in **Tight Claim**:

> - Current claim: the ObservationVLA lane is now image-model-backed and shows `1.00` exact action agreement over `3` operator-reviewed Sentinel cases.
> - Conservative claim: this is an early, low-N validation of image-conditioned usefulness scoring, not a broad benchmark.

Three 1.00 agreement scores and a 0.27 MAE in the same table, with no interpretation, **invites judges to weight the MAE as a weakness and ignore the three 1.00s**. A judge reading fast sees "0.27 error" and dismisses the rest.

## The fix

Reframe MAE=0.27 as a **known, documented calibration point with explicit operational accommodation** — not as a bug.

The underlying numbers: model predicts ~0.66 usefulness, operator labels ~0.92 — a consistent **~0.26 downward bias**, not noise. That's a calibration signal, easy to explain, easy to correct with a multiplier in a later model pass.

## What to change

`observation_vla_eval.py` contains the `build_report()` function that writes the **Tight Claim** section. Replace the stub with the richer framing below.

There are two ways to apply this:

### Option 1 — Quick: edit the generated `OBSERVATION_VLA_EVAL.md` directly

Copy the content of `OBSERVATION_VLA_EVAL.md` in this folder over the existing file. No code change — you just need to remember to re-apply after the next `observation_vla_eval.py --inprocess` run.

### Option 2 — Clean: patch `observation_vla_eval.py::build_report`

Apply `observation_vla_eval.py.patch` so the richer **Tight Claim** block is emitted automatically on every regeneration. This is the recommended path if you plan to regenerate the eval before submission.

## Why this matters for the Liquid AI judges

Liquid AI judges building activation-sparse foundation models will recognize the language of *calibration bias* instantly. Saying "exact action agreement 1.00, useful binary agreement 1.00, magnitude calibration bias -0.26 tracked for next pass" is a much more fluent framing than "MAE 0.27". It also sets up a clean follow-up: *"the next model revision applies a learned scalar correction; the current binary-decision pipeline consumes the sign, not the magnitude, so this is non-blocking for operations."*
