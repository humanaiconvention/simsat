# VLA-Layer TTT on LFM2.5-VL — Implementation Receipt

**Status (2026-05-07):** the architectural claim in
`SUBMISSION_BRIEF.md` and `CHALLENGE_ENTRY.md` previously described the
VLA-layer TTT lane as "wired, requires live encounter stream." With
`src/sim/observation_vla/lfm_ttt.py` and `tests/test_lfm_ttt.py`, the
lane is **implemented and unit-tested**. The remaining gap is the
live-stream measurement — that's what the prize hardware (NVIDIA Orin
16 GB) unlocks.

This file documents what's now in-tree.

## What's implemented

`src/sim/observation_vla/lfm_ttt.py` exposes:

- **`OnlineLoRAStepper`** — wraps a `peft.PeftModel`-wrapped VLM plus
  its processor and optimizer. The `online_step(messages, target_text)`
  method does forward + backward + step on a single operator-labelled
  encounter, gated by the same six-gate viability filter that already
  governs the trust-layer TTT.
- **`_action_error_sign`** — maps the categorical operator action onto
  a signed error scalar over the usefulness ladder
  (`skip < defer < refine < accept`), so the BLOCKING `error_bias`
  gate has a directional signal to track. Predictions that don't parse
  return 0 (don't bias the window in either direction).
- **`build_lfm_snapshot`** — emits a snapshot dict in the same shape
  `WCLITrustModel.get_weight_snapshot()` produces, so we can call the
  existing `haic.viability.evaluate_ttt_viability` directly. The same
  three TTT gates apply: `error_bias` (BLOCKING), `weight_drift`
  (post-step warning, on LoRA-delta L2 vs initial state), `update_rate`
  (post-step warning, cumulative count ceiling).

## Architectural symmetry with the trust-layer TTT

The two TTT layers now share the same architectural skeleton:

| Concept | Trust-layer | VLA-layer (LFM) |
|---|---|---|
| Step entry point | `WCLITrustModel.online_update` | `OnlineLoRAStepper.online_step` |
| Error signal | `realized_utility − learned_score` (scalar ±) | cross-entropy on assistant tokens + signed action mismatch |
| BLOCKING gate | `error_bias` ≥ 70% same-sign in last 10 | same — `error_bias` evaluated pre-update |
| Window advance on block | `record_skipped_observation` | history.append with `blocked=True` |
| Drift gate | `weight_drift` on 5 WCLI weights vs policy defaults | `weight_drift` on LoRA-delta L2 vs initial-zero baseline |
| Rate gate | `update_count > 1000` | same |
| L2 regularization | `reg=0.002 * (default − current)` | LoRA inherits the standard PEFT gradient path |

This symmetry is the architectural contribution: a *single* viability
filter governs adaptation at both layers, not two separate ones.
Operator review never needs to specify which layer to gate — the same
six-gate decision is asked of every adaptation request.

## Unit tests (16 passing)

`tests/test_lfm_ttt.py` proves the implementation works without
requiring a real LFM model loaded — every test injects a
`forward_loss_fn` stub. Coverage:

- **Action-error sign ladder** (8 parametrized cases): every pair of
  predicted/target action across the 4-class space, plus None and
  garbage inputs.
- **Snapshot shape compat:** the snapshot from `build_lfm_snapshot` is
  accepted by the existing `evaluate_ttt_viability` without modification.
- **Step fires when predictions match:** matching predictions ⇒
  action_error=0 ⇒ gate vacuously passes ⇒ `did_step=True`.
- **Gate blocks after full window same-sign:** 10 same-sign errors ⇒
  the 11th step is blocked with `blocked_by="error_bias"`.
- **Window advances on block ⇒ gate clears:** end-to-end trace showing
  the gate clears at exactly the right step (P2.5 in the trace) when
  opposite-sign feedback diversifies the window below the 70%
  threshold.
- **Blocked entries appear in history with `blocked: True`:**
  bookkeeping invariant.
- **Optimizer step counts match `did_step` count:** the optimizer is
  called exactly when the gate doesn't fire.
- **Unparseable predictions yield zero error sign:** missing/garbled
  predictions don't bias the window.
- **`snapshot_after` is `evaluate_ttt_viability`-compatible:**
  contract guarantee for downstream consumers.

Run: `pytest tests/test_lfm_ttt.py -v`.

## What's NOT yet measured

Honest gap, called out so a judge doesn't have to infer it:

- **No live offline-replay number.** I have not yet run
  `OnlineLoRAStepper` against a real LFM2.5-VL checkpoint with the v1
  LoRA loaded on a sequence of operator-labelled encounters and
  measured how the model's action-prediction accuracy moves over the
  stream. That experiment requires a GPU + v1 LoRA + ~30-60 min of
  T4 time, and it's the natural next step once the v1 fine-tune
  (`notebooks/kaggle-simsat-lfm-v1/`) lands cleanly. The kernel
  scaffold for this is straightforward: load `model + v1 LoRA + an
  AdamW(lr=1e-5) optimizer over the LoRA params`, then iterate
  `simsat_lfm_holdout.jsonl` and call `online_step` on each row.
- **No demonstration of long-horizon stability.** The trust-layer TTT
  has 100-cycle stability evidence (`ttt_stability_analysis.md`); the
  VLA-layer TTT does not yet have an equivalent.
- **No TTT vs static fine-tune comparison.** The interesting question
  is whether per-encounter TTT updates beat a one-shot fine-tune on
  the same data. We have the static fine-tune (v1) and the TTT
  scaffold; running them head-to-head on the holdout is the
  experiment that produces the comparison numbers.

These are the experiments the prize hardware (NVIDIA Orin 16 GB +
ground-compute days) is allocated to run.

## Reference

- Implementation: [`src/sim/observation_vla/lfm_ttt.py`](./src/sim/observation_vla/lfm_ttt.py)
- Tests: [`tests/test_lfm_ttt.py`](./tests/test_lfm_ttt.py)
- Trust-layer counterpart: [`src/sim/encounter/trust_model.py`](./src/sim/encounter/trust_model.py) — `WCLITrustModel.online_update` + `record_skipped_observation`
- Shared viability filter: [`src/sim/haic/viability.py`](./src/sim/haic/viability.py) — `evaluate_ttt_viability`
- v1 fine-tune kernel that produces the LoRA this stepper would adapt: [`notebooks/kaggle-simsat-lfm-v1/`](./notebooks/kaggle-simsat-lfm-v1/)
