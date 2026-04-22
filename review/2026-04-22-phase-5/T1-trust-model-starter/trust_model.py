"""WCLI Trust Model — consolidates scattered trust-decision logic into one module.

Before this refactor, WCLI trust logic (accept / defer / skip / refine) lives in:
- src/sim/encounter/planner.py    (decision functions on raw scaffold output)
- src/sim/encounter/service.py    (decision orchestration + reason string assembly)
- configs                         (thresholds, reason-code lookups)

This module consolidates all three into a single callable unit so:
  1. Judges can audit the trust claim in one file.
  2. Test-Time Training (Spec #22) has a clean place to inject online updates.
  3. The viability gate (Spec #23) has a clean choke-point for filtering candidate
     threshold updates.

REFACTOR CONTRACT
-----------------
This is a REFACTOR, not a rewrite. The default `decide()` body below encodes the
known action / reason vocabulary and obvious threshold checks. Before committing
this module as-is, Ben should:

  1. Port the scattered WCLI decision logic from planner.py / service.py / configs
     into `WCLITrustModel.decide()`.
  2. Preserve the exact (scaffold_action, context) -> (action, reason) mapping
     that today's submission pipeline produces.
  3. Run `scripts/trust_model_parity.py` against the 86-trace corpus to verify
     ZERO decision mismatches between old scattered logic and new module.

Only after parity is green should the refactor land.

PUBLIC API
----------
    from sim.encounter.trust_model import WCLITrustModel, TrustContext, TrustDecision

    model = WCLITrustModel.from_defaults()
    decision = model.decide(scaffold_action="accept", context=ctx)
    # ... planner uses decision.action, logs decision.reason ...
    model.ingest_feedback(decision.decision_id, utility_realized=0.85, outcome="useful")
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

TrustAction = Literal["accept", "defer", "skip", "refine"]


# --------------------------------------------------------------------------- #
# Context & decision payloads                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class TrustContext:
    """Signals flowing into the trust gate from scaffold + ObservationVLA + geometry.

    Populate from the planner's existing context assembly. This is a structure
    wrapper around what scattered code already computes today.
    """

    # From target registry / scenario pack
    target_id: str
    target_priority: str = "normal"          # "high_priority" | "high_visibility" | "normal"
    target_tags: list[str] = field(default_factory=list)

    # From Sentinel probe
    sentinel_available: Optional[bool] = None
    sentinel_cloud_cover: Optional[float] = None  # 0..100 percent
    sentinel_source: Optional[str] = None         # "sentinel-2a" | "sentinel-2b" | "sentinel-2c"
    sentinel_datetime: Optional[str] = None       # ISO UTC (post-Phase-1 tzinfo fix)

    # From geometry
    target_visible: Optional[bool] = None
    elevation_degrees: Optional[float] = None
    off_nadir_degrees: Optional[float] = None
    line_of_sight: Optional[bool] = None

    # From ObservationVLA.assess() — the payload shape documented in
    # src/sim/observation_vla/README_ENTRY_B.md and produced by
    # Gemma4HAICAdapter / ObservationVLMAdapter. Names match that schema.
    vla_usable: Optional[bool] = None
    vla_scene_match: Optional[float] = None          # 0..1
    vla_salience: Optional[float] = None             # 0..1
    vla_change_or_event: Optional[float] = None      # 0..1
    vla_occlusion_or_cloud_risk: Optional[float] = None  # 0..1
    vla_confidence: Optional[float] = None           # 0..1
    vla_recommended_action: Optional[TrustAction] = None

    # From recency / pass cadence (optional depending on planner state)
    hours_since_last_materialize: Optional[float] = None

    # Observability — not part of the signal, used only for replay / logs
    scenario_pack: str = "all"
    planner_window_id: Optional[str] = None

    def stable_hash(self) -> str:
        """Stable hash over signal fields (not observability fields).

        Used for parity replay and dedup. If two contexts have the same hash, the
        trust decision must also be identical.
        """
        signal_payload = {
            k: v
            for k, v in asdict(self).items()
            if k not in {"planner_window_id"}
        }
        serialized = json.dumps(signal_payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


@dataclass
class TrustDecision:
    """Output of WCLITrustModel.decide(). Immutable record for logging and TTT replay."""

    decision_id: str
    action: TrustAction
    reason: str                    # comma-joined reason codes
    confidence: float              # 0..1
    scaffold_action: str
    context_hash: str
    thresholds_snapshot: dict[str, float]
    timestamp: str                 # ISO UTC


# --------------------------------------------------------------------------- #
# Thresholds, reason-code vocabulary                                          #
# --------------------------------------------------------------------------- #

_DEFAULT_THRESHOLDS: dict[str, float] = {
    # Decision boundaries — tune to match current planner behavior during refactor.
    "cloud_refine_threshold":     20.0,   # cloud_cover above this + accept -> refine
    "cloud_skip_threshold":       60.0,   # cloud_cover above this -> skip regardless
    "vla_confidence_refine_below": 0.55,  # VLA confidence below this -> refine
    "vla_confidence_skip_below":   0.30,  # VLA confidence below this -> skip
    "elevation_skip_below":       30.0,   # elevation below 30° -> skip (matches WindowDetector)
    "salience_refine_below":       0.50,  # salience below this -> refine
    "usefulness_accept_above":     0.70,  # proxy for realized-utility feedback (Spec #22)
}

# Canonical reason codes. The comma-joined form ("high_priority,sentinel_available,line_of_sight,cloud_risk")
# matches the format already used in SUBMISSION_PACKET.md "Top delta" lines.
REASON_CODES = (
    # Target-side signals
    "high_priority", "high_visibility",
    # Sentinel availability
    "sentinel_available", "sentinel_unavailable",
    # Cloud signals
    "low_cloud_risk", "moderate_cloud_risk", "cloud_risk",
    # Geometry signals
    "line_of_sight", "no_line_of_sight",
    # VLA signals
    "vla_confident", "vla_uncertain", "vla_rejects",
    # Recency signals
    "recent_coverage", "stale_coverage",
    # Timing signals
    "edge_of_window", "prime_window",
    # Meta
    "refine_threshold_met",
)


# --------------------------------------------------------------------------- #
# The trust model                                                             #
# --------------------------------------------------------------------------- #

class WCLITrustModel:
    """WCLI-style trust gate. Consumes scaffold action + signals, emits a TrustDecision.

    Refactor note: the default `decide()` body is a STARTER skeleton. Port the
    scattered logic from planner.py / service.py here, preserving the exact
    action/reason mapping. Run scripts/trust_model_parity.py to verify no drift.
    """

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
        *,
        enable_online_updates: bool = False,
        feedback_buffer_size: int = 32,
    ) -> None:
        self.thresholds: dict[str, float] = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.enable_online_updates = enable_online_updates
        self.feedback_buffer_size = feedback_buffer_size
        self._feedback_buffer: list[dict[str, Any]] = []
        self._decision_log: list[TrustDecision] = []

    # ---- Construction -----------------------------------------------------

    @classmethod
    def from_defaults(cls) -> "WCLITrustModel":
        return cls()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WCLITrustModel":
        """Load thresholds from a config mapping.

        The config key `wcli_trust_thresholds` (or equivalent) is expected to mirror
        `_DEFAULT_THRESHOLDS`. Port whatever key the existing scattered configs use.
        """
        thresholds = config.get("wcli_trust_thresholds", {})
        return cls(thresholds=thresholds)

    # ---- Core decision ----------------------------------------------------

    def decide(self, scaffold_action: str, context: TrustContext) -> TrustDecision:
        """Produce a WCLI trust decision given a scaffold action and context signals.

        The body below is a STARTER SKELETON encoding obvious threshold checks and
        the canonical reason-code vocabulary. During refactor, port the existing
        scattered logic here — this function should become the SINGLE SOURCE OF
        TRUTH for how (scaffold_action, context) -> (trust_action, reason, confidence).
        """

        reasons: list[str] = []

        # --- Assemble reason codes from signals ---

        # Target-side
        if context.target_priority == "high_priority":
            reasons.append("high_priority")
        elif context.target_priority == "high_visibility":
            reasons.append("high_visibility")

        # Sentinel availability
        if context.sentinel_available is True:
            reasons.append("sentinel_available")
        elif context.sentinel_available is False:
            reasons.append("sentinel_unavailable")

        # Geometry
        if context.line_of_sight is True:
            reasons.append("line_of_sight")
        elif context.line_of_sight is False:
            reasons.append("no_line_of_sight")

        # Cloud-cover classification
        if context.sentinel_cloud_cover is not None:
            if context.sentinel_cloud_cover < 10.0:
                reasons.append("low_cloud_risk")
            elif context.sentinel_cloud_cover > self.thresholds["cloud_refine_threshold"]:
                reasons.append("cloud_risk")
            else:
                reasons.append("moderate_cloud_risk")

        # VLA signals
        if context.vla_confidence is not None:
            if context.vla_confidence < self.thresholds["vla_confidence_skip_below"]:
                reasons.append("vla_rejects")
            elif context.vla_confidence < self.thresholds["vla_confidence_refine_below"]:
                reasons.append("vla_uncertain")
            else:
                reasons.append("vla_confident")

        # --- Decision logic (starter defaults; port scattered logic here) ---

        # Skip conditions (non-compensatory — applied first)
        action: TrustAction
        if context.line_of_sight is False:
            action = "skip"
        elif (
            context.elevation_degrees is not None
            and context.elevation_degrees < self.thresholds["elevation_skip_below"]
        ):
            action = "skip"
        elif (
            context.sentinel_cloud_cover is not None
            and context.sentinel_cloud_cover > self.thresholds["cloud_skip_threshold"]
        ):
            action = "skip"
        elif (
            context.vla_confidence is not None
            and context.vla_confidence < self.thresholds["vla_confidence_skip_below"]
        ):
            action = "skip"

        # Refine conditions (override accept under uncertainty)
        elif scaffold_action == "accept" and self._should_refine(context):
            action = "refine"
            reasons.append("refine_threshold_met")

        # Default: pass through scaffold action
        else:
            action = scaffold_action  # type: ignore[assignment]

        # Confidence — simple weighted blend of VLA confidence and action certainty
        base_conf = context.vla_confidence if context.vla_confidence is not None else 0.65
        if action in ("skip", "refine"):
            base_conf = max(0.50, base_conf - 0.10)
        confidence = float(max(0.0, min(1.0, base_conf)))

        decision = TrustDecision(
            decision_id=str(uuid.uuid4()),
            action=action,
            reason=",".join(reasons) if reasons else "no_explicit_reasons",
            confidence=confidence,
            scaffold_action=scaffold_action,
            context_hash=context.stable_hash(),
            thresholds_snapshot=dict(self.thresholds),
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self._decision_log.append(decision)
        logger.debug(
            "WCLITrustModel.decide: target=%s scaffold=%s -> action=%s (%s)",
            context.target_id, scaffold_action, action, decision.reason,
        )
        return decision

    def _should_refine(self, ctx: TrustContext) -> bool:
        """Helper for the accept -> refine decision. Extracted so it's easy to override."""
        cloud_triggers_refine = (
            ctx.sentinel_cloud_cover is not None
            and ctx.sentinel_cloud_cover > self.thresholds["cloud_refine_threshold"]
        )
        vla_confidence_triggers_refine = (
            ctx.vla_confidence is not None
            and ctx.vla_confidence < self.thresholds["vla_confidence_refine_below"]
        )
        salience_triggers_refine = (
            ctx.vla_salience is not None
            and ctx.vla_salience < self.thresholds["salience_refine_below"]
        )
        return cloud_triggers_refine or vla_confidence_triggers_refine or salience_triggers_refine

    # ---- Feedback & TTT hooks ---------------------------------------------

    def ingest_feedback(
        self,
        decision_id: str,
        utility_realized: float,
        outcome: str,  # "useful" | "not_useful"
    ) -> None:
        """Record realized utility for a past decision. TTT hook for Spec #22.

        The default implementation just buffers feedback — it does NOT update
        thresholds. Enabling online updates requires `enable_online_updates=True`
        AND the Spec #22 implementation of `_online_update()` landing. That way
        this module ships as a pure refactor with zero behaviour change until the
        TTT work is explicitly turned on.
        """
        if outcome not in ("useful", "not_useful"):
            logger.warning("ingest_feedback: unknown outcome %r for decision %s", outcome, decision_id)
            return

        self._feedback_buffer.append({
            "decision_id": decision_id,
            "utility_realized": float(utility_realized),
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })

        if (
            self.enable_online_updates
            and len(self._feedback_buffer) >= self.feedback_buffer_size
        ):
            self._online_update()

    def _online_update(self) -> None:
        """Threshold update step. Spec #22 hook — currently a no-op with a clear TODO.

        When Spec #22 lands:
          1. Compute threshold deltas from self._feedback_buffer using reward-weighted
             regression (or equivalent). Keep deltas bounded: no threshold moves by
             more than 10% per update.
          2. Build an AdaptationCandidate describing the proposed threshold changes.
          3. Call the viability gate (Spec #23) to filter the proposed deltas.
          4. Apply only the gated deltas to self.thresholds.
          5. Record the update event for observability.

        Default behaviour: drain the buffer and log a warning, so disabling
        `enable_online_updates` is the safe path until Spec #22 implementation lands.
        """
        logger.info(
            "WCLITrustModel._online_update called with %d buffered feedbacks; "
            "Spec #22 implementation pending — draining buffer, no threshold change",
            len(self._feedback_buffer),
        )
        # TODO(Spec #22): compute threshold deltas from self._feedback_buffer
        # TODO(Spec #23): gate the computed deltas through the viability filter
        # TODO: apply the gated deltas to self.thresholds
        self._feedback_buffer.clear()

    # ---- Observability ----------------------------------------------------

    def snapshot_thresholds(self) -> dict[str, float]:
        """Read-only snapshot of current threshold values. For logging / eval."""
        return dict(self.thresholds)

    def decision_log(self) -> list[TrustDecision]:
        """Read-only snapshot of the decision log. For parity harness and replay."""
        return list(self._decision_log)

    def feedback_buffer_size_now(self) -> int:
        return len(self._feedback_buffer)
