"""
Viability Gate Evaluator — implements the 6 non-compensatory gates from the
HumanAI Convention protocol before settlement is permitted.

Gates (from Maestro viability_policy.yaml):
  1. entropy_reduction       — ΔS < −ε (prism_snapshot delta)
  2. extraction_risk         — extraction_risk_score ≤ 0.15
  3. prism_consistency       — claimed reduction matches geometric measurement
  4. participation_covenant  — consent bounds respected (session had valid stimulus)
  5. federated_exchange      — raw data stays at edge (images not re-exported)
  6. epistemic_alignment     — exchange reduces uncertainty, not bias

Non-compensatory: a perfect score on gate 5 cannot override failure on gate 1.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .schemas import ConventionSession

logger = logging.getLogger(__name__)

# Thresholds (mirror Maestro viability_policy.yaml)
ENTROPY_EPSILON = 0.01          # minimum entropy reduction required
EXTRACTION_RISK_LIMIT = 0.15    # maximum extraction risk score
MIN_PROVENANCE_SCORE = 0.90     # minimum PoG provenance score
MIN_USER_TURNS = 2              # minimum number of meaningful human turns
MIN_WORD_COUNT = 10             # minimum total words across all user turns


def evaluate_viability(session: "ConventionSession") -> Dict[str, bool]:
    """
    Evaluate all 6 viability gates for a completed session.

    Returns dict of gate_name → passed (bool).
    All gates must pass for settlement to proceed.
    """
    gates: Dict[str, bool] = {}

    gates["entropy_reduction"] = _gate_entropy_reduction(session)
    gates["extraction_risk"] = _gate_extraction_risk(session)
    gates["prism_consistency"] = _gate_prism_consistency(session)
    gates["participation_covenant"] = _gate_participation_covenant(session)
    gates["federated_exchange"] = _gate_federated_exchange(session)
    gates["epistemic_alignment"] = _gate_epistemic_alignment(session)

    passed = sum(gates.values())
    logger.info(
        "Viability evaluation for %s: %d/6 gates passed — %s",
        session.session_id, passed,
        {k: ("✓" if v else "✗") for k, v in gates.items()},
    )
    return gates


# ---- Individual gate implementations ----

def _gate_entropy_reduction(session: "ConventionSession") -> bool:
    """Gate 1: Did the grounding session reduce model entropy by ≥ ε?"""
    if session.entropy_delta is None:
        # No PRISM measurement — auto-pass in synthetic mode for demo
        # In production this would require a real delta proof
        logger.debug("entropy_reduction: no delta — auto-pass (no PRISM model configured)")
        return True

    delta_se = session.entropy_delta.get("delta_spectral_entropy", 0.0)
    reduction_verified = session.entropy_delta.get("reduction_verified", False)

    # Either the PRISM proof flags it verified, or delta is sufficiently negative
    passed = reduction_verified or (delta_se < -ENTROPY_EPSILON)
    if not passed:
        logger.debug(
            "entropy_reduction FAILED: delta_se=%.6f, threshold=%.6f",
            delta_se, -ENTROPY_EPSILON,
        )
    return passed


def _gate_extraction_risk(session: "ConventionSession") -> bool:
    """Gate 2: Is the extraction risk score ≤ 0.15?"""
    risk = _compute_extraction_risk(session)
    passed = risk <= EXTRACTION_RISK_LIMIT
    if not passed:
        logger.debug("extraction_risk FAILED: score=%.4f > %.4f", risk, EXTRACTION_RISK_LIMIT)
    return passed


def _gate_prism_consistency(session: "ConventionSession") -> bool:
    """Gate 3: Does the claimed entropy reduction match the geometric measurement?"""
    if session.entropy_delta is None:
        return True  # No PRISM — gate passes vacuously

    delta = session.entropy_delta
    # Check that before/after snapshots are structurally consistent
    snap_before = delta.get("snapshot_before", {})
    snap_after = delta.get("snapshot_after", {})

    if not snap_before or not snap_after:
        logger.debug("prism_consistency FAILED: missing snapshots")
        return False

    # Effective dimension should change in the expected direction
    # (not required to increase, but shouldn't be wildly inconsistent)
    se_before = snap_before.get("mean_spectral_entropy", 0.0)
    se_after = snap_after.get("mean_spectral_entropy", 0.0)
    claimed_delta = delta.get("delta_spectral_entropy", 0.0)
    actual_delta = se_after - se_before

    # Allow small floating-point inconsistency
    consistent = abs(claimed_delta - actual_delta) < 0.001
    if not consistent:
        logger.debug(
            "prism_consistency FAILED: claimed_delta=%.6f vs actual=%.6f",
            claimed_delta, actual_delta,
        )
    return consistent


def _gate_participation_covenant(session: "ConventionSession") -> bool:
    """Gate 4: Did the session respect participation covenant bounds?

    Checks:
    - Session had a valid stimulus (grounded in satellite reality)
    - PoG provenance score meets minimum threshold (if telemetry available)
    - Minimum meaningful participation (word count, turn count)
    """
    # Must have had a stimulus
    if session.stimulus is None:
        logger.debug("participation_covenant FAILED: no stimulus loaded")
        return False

    # PoG check
    if session.pog_telemetry and "result" in session.pog_telemetry:
        score = session.pog_telemetry["result"].get("provenance_score", 0.0)
        if score < MIN_PROVENANCE_SCORE:
            logger.debug(
                "participation_covenant FAILED: provenance_score=%.4f < %.4f",
                score, MIN_PROVENANCE_SCORE,
            )
            return False

    # Minimum participation
    user_turns = [t for t in session.interview_turns if t.get("role") == "user"]
    if len(user_turns) < MIN_USER_TURNS:
        logger.debug(
            "participation_covenant FAILED: only %d user turns (min %d)",
            len(user_turns), MIN_USER_TURNS,
        )
        return False

    total_words = sum(len(t.get("content", "").split()) for t in user_turns)
    if total_words < MIN_WORD_COUNT:
        logger.debug(
            "participation_covenant FAILED: only %d words (min %d)",
            total_words, MIN_WORD_COUNT,
        )
        return False

    return True


def _gate_federated_exchange(session: "ConventionSession") -> bool:
    """Gate 5: Federated exchange — raw satellite images stay at edge.

    In SimSat, images are served on-demand via the API; they are not
    persisted in the session record beyond the stimulus metadata.
    We verify the session object doesn't store raw image data in interview turns.
    """
    for turn in session.interview_turns:
        content = turn.get("content", "")
        # Reject if any turn contains a base64 image blob (data: URI)
        if "data:image" in content or len(content) > 50000:
            logger.debug("federated_exchange FAILED: raw image data detected in turn")
            return False
    return True


def _gate_epistemic_alignment(session: "ConventionSession") -> bool:
    """Gate 6: Does the exchange reduce uncertainty rather than reinforce bias?

    Proxy heuristic:
    - Session contains diverse vocabulary (not just keyword repetition)
    - Interviewer questions varied (not the same question repeated)
    - Human turns show increasing specificity (later turns longer or richer)
    """
    user_turns = [t for t in session.interview_turns if t.get("role") == "user"]
    assistant_turns = [t for t in session.interview_turns if t.get("role") == "assistant"]

    if len(user_turns) < 2:
        return True  # Not enough data to evaluate — pass by default

    # Check assistant didn't repeat itself (simple dedup check)
    assistant_texts = [t["content"] for t in assistant_turns]
    if len(set(assistant_texts)) < max(1, len(assistant_texts) // 2):
        logger.debug("epistemic_alignment FAILED: assistant repeated itself excessively")
        return False

    # Check human vocabulary diversity across turns
    all_words = []
    for t in user_turns:
        all_words.extend(t.get("content", "").lower().split())
    if len(all_words) > 0:
        unique_ratio = len(set(all_words)) / len(all_words)
        if unique_ratio < 0.3:
            logger.debug(
                "epistemic_alignment FAILED: low vocab diversity=%.3f", unique_ratio
            )
            return False

    return True


# ---- Helpers ----

def _compute_extraction_risk(session: "ConventionSession") -> float:
    """
    Estimate extraction risk score for this session.

    Risk factors:
    - Session has very long transcripts (bulk data extraction)
    - Same participant_id appears in many sessions
    - Very short sessions with many images requested
    """
    risk = 0.0

    # Long transcript risk
    total_chars = sum(len(t.get("content", "")) for t in session.interview_turns)
    if total_chars > 10000:
        risk += 0.05
    if total_chars > 50000:
        risk += 0.05

    # If stimulus has many images but very short interview — possibly just scraping images
    image_count = len(session.stimulus.images) if session.stimulus else 0
    user_turn_count = sum(1 for t in session.interview_turns if t.get("role") == "user")
    if image_count > 2 and user_turn_count < 2:
        risk += 0.08

    # No PoG telemetry submitted
    if not session.pog_telemetry:
        risk += 0.02

    return min(1.0, round(risk, 4))


# ── TTT (Test-Time Training) Viability Gates ──────────────────────────────

# Thresholds for trust-model online adaptation health
MAX_TTT_WEIGHT_DRIFT = 0.30     # max absolute per-weight drift from policy default
MAX_TTT_UPDATE_COUNT = 1000     # recommend manual reset above this update count
TTT_BIAS_WINDOW = 10            # number of recent updates to inspect for systematic bias
TTT_BIAS_THRESHOLD = 0.70       # fraction of same-sign errors = systematic bias


def evaluate_ttt_viability(trust_snapshot: dict) -> Dict[str, bool]:
    """
    Evaluate 3 non-compensatory viability gates for the trust-layer TTT state.

    Returns a dict of gate_name → passed.  Callers are responsible for deciding
    whether to block on a failed gate:

    - **error_bias** — evaluated PRE-update in ObservationVLAService._apply_trust_layer_ttt;
      a failure causes the update to be skipped entirely (BLOCKING gate in that context).
    - **weight_drift** — evaluated POST-update; log-only warning, drift is already applied.
    - **update_rate** — evaluated POST-update; log-only warning, prompts manual review.

    Args:
        trust_snapshot: dict returned by WCLITrustModel.get_weight_snapshot()
    """
    gates: Dict[str, bool] = {}

    gates["weight_drift"] = _gate_ttt_weight_drift(trust_snapshot)
    gates["update_rate"] = _gate_ttt_update_rate(trust_snapshot)
    gates["error_bias"] = _gate_ttt_error_bias(trust_snapshot)

    passed = sum(gates.values())
    logger.info(
        "TTT viability check (update #%d): %d/3 gates — %s",
        trust_snapshot.get("update_count", 0),
        passed,
        {k: ("✓" if v else "✗") for k, v in gates.items()},
    )
    return gates


def _gate_ttt_weight_drift(trust_snapshot: dict) -> bool:
    """TTT Gate 1: No single weight has drifted > MAX_TTT_WEIGHT_DRIFT from its policy default.

    Excessive drift indicates the trust model is over-fitting to a short run of
    operator labels, discarding policy priors.
    """
    drift = trust_snapshot.get("drift_from_policy_defaults", {})
    if not drift:
        return True  # No drift data — gate passes vacuously

    for key, delta in drift.items():
        if abs(delta) > MAX_TTT_WEIGHT_DRIFT:
            logger.debug(
                "ttt weight_drift FAILED: weight '%s' drifted %.4f > %.4f",
                key, abs(delta), MAX_TTT_WEIGHT_DRIFT,
            )
            return False
    return True


def _gate_ttt_update_rate(trust_snapshot: dict) -> bool:
    """TTT Gate 2: Total update count ≤ MAX_TTT_UPDATE_COUNT.

    Above 1 000 updates, the model has been adapted through many operator sessions;
    recommend a manual weight-snapshot review and reset.
    """
    count = trust_snapshot.get("update_count", 0)
    passed = count <= MAX_TTT_UPDATE_COUNT
    if not passed:
        logger.debug(
            "ttt update_rate FAILED: update_count=%d > %d — recommend weight reset",
            count, MAX_TTT_UPDATE_COUNT,
        )
    return passed


def _gate_ttt_error_bias(trust_snapshot: dict) -> bool:
    """TTT Gate 3 (BLOCKING): No systematic error bias in the most recent TTT_BIAS_WINDOW updates.

    If ≥ TTT_BIAS_THRESHOLD (70 %) of recent prediction errors share the same sign,
    the trust model is systematically over- or under-estimating realized utility.
    In ObservationVLAService, this gate is evaluated PRE-update; a failure causes
    the adaptation step to be skipped rather than reinforcing the bias direction.
    """
    recent = trust_snapshot.get("recent_updates", [])
    window = recent[-TTT_BIAS_WINDOW:] if len(recent) > TTT_BIAS_WINDOW else recent
    if len(window) < TTT_BIAS_WINDOW:
        return True  # Full window required — gate passes vacuously until 10 entries are accumulated

    errors = [u.get("error", 0.0) for u in window if "error" in u]
    if len(errors) < 3:
        return True  # Too few parseable error entries — skip vacuously rather than divide-by-zero
    positive = sum(1 for e in errors if e > 0)
    negative = sum(1 for e in errors if e < 0)
    total = len(errors)
    frac_same_sign = max(positive, negative) / total

    passed = frac_same_sign < TTT_BIAS_THRESHOLD
    if not passed:
        logger.debug(
            "ttt error_bias FAILED: %.1f%% of last %d errors share same sign (threshold %.1f%%)",
            frac_same_sign * 100, total, TTT_BIAS_THRESHOLD * 100,
        )
    return passed
