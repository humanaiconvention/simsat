# T1 — `trust_model.py` as a real module (refactor task)

## Why

The WCLI trust layer is the **thesis-critical component** of SimSat. Every submission doc now cites it as the "WCLI-style trust layer that decides accept / defer / skip / refine." Judges will want to read this layer as a single, self-contained file. Today it's scattered across `planner.py`, `service.py`, and configs. Consolidating it:

1. Gives judges one file to audit the trust claim.
2. Gives Spec #22 (trust-layer TTT) a clean injection point for online updates.
3. Gives Spec #23 (viability gate) a single choke-point for gating candidate threshold updates.
4. Lets us write a parity harness that guarantees the refactor doesn't change decisions — so we can ship the consolidation without risking the existing 86-trace eval results.

## What this bundle ships

- **`trust_model.py`** — starter module with `TrustContext`, `TrustDecision`, `WCLITrustModel`.
  - Default thresholds encode reasonable guesses from the reviewed submission cases.
  - `decide()` body encodes the canonical action vocabulary and reason-code format but is a **skeleton** — Ben ports the scattered logic in.
  - TTT hook (`_online_update()`) is a no-op with clear TODOs for Spec #22 and Spec #23.
- **`trust_model_parity.py`** — parity harness that replays the 86-trace corpus through both old scattered logic and new module, comparing decision action + reason-code set per trace. Exits non-zero on any mismatch.
- **This README** — the refactor plan.

## Known contract (already nailed down from the existing code)

- **Action vocabulary**: `accept | defer | skip | refine` — from `SUBMISSION_PACKET.md` transition counts.
- **Reason-code format**: comma-joined, e.g. `"high_priority,sentinel_available,line_of_sight,cloud_risk"` — directly from `SUBMISSION_PACKET.md` "Top delta" lines.
- **Context signals in scope**:
  - Target: `target_id`, `target_priority`, `target_tags`, `scenario_pack`
  - Sentinel probe: `sentinel_available`, `sentinel_cloud_cover`, `sentinel_source`, `sentinel_datetime`
  - Geometry: `target_visible`, `elevation_degrees`, `off_nadir_degrees`, `line_of_sight`
  - ObservationVLA payload (from `Gemma4HAICAdapter` / `ObservationVLMAdapter`): `usable_observation`, `scene_match_score`, `salience_score`, `change_or_event_score`, `occlusion_or_cloud_risk`, `confidence`, `recommended_action`
- **Current reviewed cases** (all `accept -> accept` transitions): Suez 0.95, Houston 0.92, SF Bay 0.90. The refactor MUST preserve those.

## Unknown contract (what Ben fills in during refactor)

- What configs today set the actual thresholds? (Starter defaults are placeholders.)
- Are there additional context signals the scattered logic uses beyond what's listed above? (Widen `TrustContext` if so.)
- Do reason codes today include any not in `REASON_CODES`? (Add them to the vocabulary if so.)
- What's the mapping from old scattered functions to the new `decide()` branches?

## Migration workflow

### Step 1 — Drop the starter into place

```bash
cp T1-trust-model-starter/trust_model.py D:\SimSat\src\sim\encounter\trust_model.py
cp T1-trust-model-starter/trust_model_parity.py D:\SimSat\scripts\trust_model_parity.py
```

Smoke-test the module loads:

```bash
python -c "from sim.encounter.trust_model import WCLITrustModel; print(WCLITrustModel.from_defaults().snapshot_thresholds())"
```

Expected: prints the threshold dict without errors.

### Step 2 — Locate scattered logic

Grep the existing code for where trust decisions are made today:

```bash
# In D:\SimSat
grep -n "refine\|defer\|skip" src/sim/encounter/planner.py
grep -n "refine\|defer\|skip" src/sim/encounter/service.py

# Find the reason-code assembly
grep -rn "high_priority\|sentinel_available\|line_of_sight\|cloud_risk" src/sim/

# Find config thresholds
grep -rn "threshold\|cloud_cover\|confidence" src/sim/data/ configs/ 2>/dev/null
```

### Step 3 — Port scattered logic into `decide()`

For each scattered decision function found in Step 2:

1. Copy the function body into `WCLITrustModel.decide()` — or a helper method if it's complex.
2. Map its inputs onto `TrustContext` fields (widen the dataclass if needed — and if you widen it, also update `trace_to_inputs` in the parity harness).
3. Map its outputs onto `TrustDecision` fields (action, reason).
4. Delete the old function from `planner.py` / `service.py` — or leave as a deprecated pass-through that calls the new module.

### Step 4 — Port config thresholds

If configs today hold threshold values, either:

- Update `_DEFAULT_THRESHOLDS` to match them exactly, OR
- Load via `WCLITrustModel.from_config(your_config_dict)` at planner startup.

The parity harness uses `from_defaults()`; if the real thresholds differ from the defaults, the harness won't match. Pick one and be consistent.

### Step 5 — Wire the parity harness

In `trust_model_parity.py`, fill in `old_scattered_decide()` to call the existing scattered logic:

```python
def old_scattered_decide(scaffold_action, context):
    # Example — actual imports depend on what planner.py looks like today
    from sim.encounter.planner import decide_trust_action_legacy
    result = decide_trust_action_legacy(scaffold_action, context)
    return {"action": result.action, "reason": result.reason}
```

Also fill in `load_traces()` to point at the actual local trace storage path, and adjust `trace_to_inputs()` if the stored trace JSON uses different key names than assumed.

### Step 6 — Run parity

```bash
python scripts/trust_model_parity.py --traces-dir <path-to-traces>
```

**Expected outcome**: `PASS: all N decisions match. Refactor is a pure no-op; safe to commit.`

If mismatches appear, DO NOT commit. Each mismatch is one of:
- A reason code that the old logic emits but the new module doesn't (add it to `decide()`).
- A reason code the new module emits that old didn't (add a pass-through or suppress).
- An action flip (old says `accept`, new says `refine`) — a real logic bug in the port. Fix it.

### Step 7 — Replace call sites

After parity is green, replace the scattered call-sites with the new module:

```python
# Old (in planner.py):
action, reason = decide_trust_action_legacy(scaffold_action, ctx)

# New:
from sim.encounter.trust_model import WCLITrustModel, TrustContext
trust_model = WCLITrustModel.from_config(load_config())  # or pass from service.py
decision = trust_model.decide(scaffold_action, build_trust_context(ctx))
action, reason = decision.action, decision.reason
```

### Step 8 — Re-run the existing evals

```bash
python scripts/encounter_eval.py --base-url http://127.0.0.1:8000/sim --scenario-sweep --top-k 8 --materialize-top-k 2 --markdown
python scripts/observation_vla_eval.py --inprocess
python scripts/submission_evidence.py --base-url http://127.0.0.1:8000/sim --reviewed-only
```

All three should produce **identical** output to pre-refactor. If any metric shifts, the refactor changed semantics — go back to Step 6 and find the missed case.

### Step 9 — Commit

Once evals are identical, commit the refactor. Leave `enable_online_updates=False` (the default). Spec #22 is a separate phase that turns on the TTT path — the refactor itself should ship as a pure no-op.

## What this unblocks

- **Spec #22** (trust-layer TTT): implement `_online_update()` to compute threshold deltas from `self._feedback_buffer` using reward-weighted regression or equivalent.
- **Spec #23** (viability gate integration): insert the six-gate filter inside `_online_update()` between delta computation and threshold application.
- **Submission pitch**: the Phase 4 CHALLENGE_ENTRY language about "the WCLI trust layer" now corresponds to one file a judge can cite.
- **KNOWN_ISSUES**: items #22 and (partly) #23 now have a concrete foundation to build on.

## Caveats / honesty notes

- The starter `decide()` body is a **guess** at the decision logic. It will not produce the exact same output as the scattered logic on day one — that's why the parity harness exists and why `old_scattered_decide()` must be wired up before running anything real.
- I don't have access to `planner.py` / `service.py` / configs today. If this starter misses fields or signals that actually exist in your planner, widen `TrustContext` and `_DEFAULT_THRESHOLDS` accordingly — the starter is scaffolding, not spec.
- The online-update path is DELIBERATELY no-op in this refactor. Turning it on before Spec #22 lands would introduce real behaviour change without a verification gate.
