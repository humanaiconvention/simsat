"""
Receipt Builder — generates Merkle-rooted participation receipts for settled sessions.

Mirrors Maestro's apps/gateway/participation.py receipt structure.
A receipt is an immutable, auditable record proving:
  - Which satellite stimulus was used
  - That the participant completed the session
  - That PoG verification passed
  - That viability gates were evaluated
  - The entropy delta proof (if available)

The Merkle root commits to all of the above without exposing raw content.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import ConventionSession


def build_receipt(session: "ConventionSession") -> Dict[str, Any]:
    """
    Build a participation receipt for a completed/settled session.

    Returns a dict with:
      - receipt_id
      - session_id
      - merkle_root  (SHA-256 over all leaves)
      - leaves       (ordered list of hashed components)
      - issued_at
      - summary      (human-readable settlement summary)
    """
    leaves: List[str] = []

    # Leaf 1: session identity
    leaves.append(_leaf("session_identity", {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "participant_id": session.participant_id,
    }))

    # Leaf 2: stimulus commitment
    if session.stimulus:
        leaves.append(_leaf("stimulus_commitment", {
            "stimulus_id": session.stimulus.stimulus_id,
            "content_hash": session.stimulus.content_hash,
            "satellite_position": session.stimulus.satellite_position,
            "simulation_timestamp": session.stimulus.simulation_timestamp,
            "image_types": [img.stimulus_type.value for img in session.stimulus.images],
        }))
    else:
        leaves.append(_leaf("stimulus_commitment", {"stimulus": None}))

    # Leaf 3: participation evidence
    user_turns = [t for t in session.interview_turns if t.get("role") == "user"]
    participation_hash = _hash_text(
        " ".join(t["content"] for t in user_turns)
    )
    leaves.append(_leaf("participation_evidence", {
        "turn_count": len(user_turns),
        "content_hash": participation_hash,
        "pog_verified": session.pog_verified,
    }))

    # Leaf 4: PoG result
    pog_score = None
    if session.pog_telemetry and "result" in session.pog_telemetry:
        pog_score = session.pog_telemetry["result"].get("provenance_score")
    leaves.append(_leaf("pog_attestation", {
        "pog_verified": session.pog_verified,
        "provenance_score": pog_score,
    }))

    # Leaf 5: entropy delta proof
    if session.entropy_delta:
        leaves.append(_leaf("entropy_delta_proof", {
            "delta_spectral_entropy": session.entropy_delta.get("delta_spectral_entropy"),
            "reduction_verified": session.entropy_delta.get("reduction_verified"),
            "epsilon_threshold": session.entropy_delta.get("epsilon_threshold"),
        }))
    else:
        leaves.append(_leaf("entropy_delta_proof", {"prism": "not_configured"}))

    # Leaf 6: viability gate outcomes
    leaves.append(_leaf("viability_gates", session.viability_gates))

    # Leaf 7: settlement outcome
    all_gates_passed = all(session.viability_gates.values()) if session.viability_gates else False
    leaves.append(_leaf("settlement_outcome", {
        "status": session.status.value,
        "all_gates_passed": all_gates_passed,
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }))

    # Build Merkle root
    merkle_root = _merkle_root(leaves)

    # Human-readable summary
    summary = _build_summary(session, all_gates_passed, pog_score)

    return {
        "receipt_id": str(uuid.uuid4()),
        "session_id": session.session_id,
        "merkle_root": merkle_root,
        "leaves": leaves,
        "leaf_count": len(leaves),
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "all_gates_passed": all_gates_passed,
        "pog_verified": session.pog_verified,
    }


# ---- Helpers ----

def _leaf(label: str, data: Any) -> str:
    """Hash a labelled data element into a Merkle leaf."""
    serialised = json.dumps({"label": label, "data": data}, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _merkle_root(leaves: List[str]) -> str:
    """Compute a simple binary Merkle root from a list of hex-string leaves."""
    if not leaves:
        return hashlib.sha256(b"empty").hexdigest()

    layer = list(leaves)
    while len(layer) > 1:
        next_layer: List[str] = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left  # duplicate last if odd
            combined = hashlib.sha256((left + right).encode()).hexdigest()
            next_layer.append(combined)
        layer = next_layer

    return layer[0]


def _build_summary(
    session: "ConventionSession",
    all_gates_passed: bool,
    pog_score: Optional[float],
) -> str:
    user_turns = [t for t in session.interview_turns if t.get("role") == "user"]
    words = sum(len(t["content"].split()) for t in user_turns)

    parts = [
        f"Session {session.session_id[:8]}…",
        f"{len(user_turns)} participant turns ({words} words).",
    ]

    if session.stimulus:
        pos = session.stimulus.satellite_position
        parts.append(
            f"Satellite at ({pos[1]:.2f}°, {pos[0]:.2f}°) alt {pos[2]:.0f}km."
        )
        image_types = [img.stimulus_type.value for img in session.stimulus.images
                       if img.metadata and img.metadata.image_available]
        if image_types:
            parts.append(f"Perspectives: {', '.join(image_types)}.")

    if pog_score is not None:
        parts.append(f"PoG provenance score: {pog_score:.3f}.")

    if session.entropy_delta:
        delta = session.entropy_delta.get("delta_spectral_entropy", 0.0)
        parts.append(f"Entropy delta: {delta:+.4f} ({'reduced ✓' if delta < 0 else 'no reduction ✗'}).")

    if all_gates_passed:
        parts.append("All 6 viability gates passed. Settlement authorised.")
    else:
        failed = [k for k, v in (session.viability_gates or {}).items() if not v]
        parts.append(f"Failed gates: {', '.join(failed) if failed else 'unknown'}.")

    return " ".join(parts)
