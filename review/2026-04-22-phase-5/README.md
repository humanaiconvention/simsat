# review/2026-04-22-phase-5 — TTT foundation refactors

Generated 2026-04-22. First phase focused on **implementation work** (not doc refresh) after the Phase 4 TTT thesis was locked in.

## What this phase is

Phase 4 locked in the architectural claim: stacked test-time training at the VLA layer (LFM2.5 / Gemma-4) and the trust layer (WCLI), both gated by six non-compensatory viability checks. The pitch describes the architecture; Phase 5 starts building the pieces.

## Why foundation refactors first

The three TTT wiring items in `KNOWN_ISSUES.md` (#21 VLA-layer TTT, #22 trust-layer TTT, #23 viability-gate integration) cannot be implemented concretely until two preconditions hold:

1. **Trust-layer logic is a single, auditable module.** Today it's scattered across `planner.py`, `service.py`, and configs. Spec #22 needs a consolidated `WCLITrustModel` class to hang online-update logic on.
2. **The six-gate observation filter is implemented as callable Python.** The project doc describes it; the geometric `GeometricViability` module is part of it; the metadata-level gates (PRISM consistency, participation covenant, federated exchange, extraction risk) are not yet coded.

Phase 5 addresses the first precondition. T1 ships a starter `trust_model.py` plus a parity harness so the refactor is verifiably a no-op. Once that lands and the parity harness passes, Phase 6 (or a T2 in this phase) will ship the viability-gate module, at which point Specs #22 and #23 have a real foundation to build on.

## Contents

| Sub-bundle | What it ships | Kind |
|---|---|---|
| `T1-trust-model-starter/` | `trust_model.py` (starter), `trust_model_parity.py` (harness), `README.md` (refactor plan) | Code + refactor plan |

## Order Ben should apply

1. Copy `T1-trust-model-starter/trust_model.py` into `D:\SimSat\src\sim\encounter\trust_model.py`.
2. Copy `T1-trust-model-starter/trust_model_parity.py` into `D:\SimSat\scripts\trust_model_parity.py`.
3. Smoke-test: `python -c "from sim.encounter.trust_model import WCLITrustModel; print(WCLITrustModel.from_defaults().snapshot_thresholds())"`.
4. Port scattered WCLI logic from `planner.py` / `service.py` / configs into `WCLITrustModel.decide()` per the migration checklist in `T1/README.md`.
5. Update call-sites to use the new module.
6. Run the parity harness against the 86-trace corpus — should be **zero decision mismatches** (pure refactor).
7. If parity passes, commit locally. If it fails, the refactor dropped a reason code or mis-mapped an action — fix and re-run.

## Preconditions this does **not** establish

- **Does not** push the refactored `planner.py` / `service.py` back to GitHub. Those files still live only at `D:\SimSat` until Ben pushes them.
- **Does not** define the six viability gates. That's Phase 5 T2 or Phase 6.
- **Does not** begin Spec #22 or Spec #23 implementation. Those need T1 and T2 landed first.

## What this unblocks

- Spec #22 (trust-layer TTT) has a clean `WCLITrustModel._online_update()` hook waiting for implementation.
- Spec #23 (viability-gate integration) has a clean call site in `WCLITrustModel._online_update()` for gating threshold updates.
- Phase 4 pitch language about "the WCLI trust layer" now corresponds to one file judges can cite.

## Open questions (context-dependent)

These will sharpen once Ben either pastes the local sources or pushes them to a branch:

- What does `planner.py` currently pass as the context dict into the trust decision? The starter assumes the payload shape documented in `README_ENTRY_B.md` — if the actual context has additional / different fields, `TrustContext` needs widening.
- Are there configs (YAML / JSON) with threshold values today? If so, the defaults in `_DEFAULT_THRESHOLDS` should match them or load from the config file instead.
- Does any test harness exist today that asserts decision outcomes (e.g., "Suez Canal → accept" under specific context)? If yes, those tests are the parity harness' ground truth.
