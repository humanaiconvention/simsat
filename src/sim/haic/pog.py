"""
Proof-of-Grounding (PoG) — lightweight behavioural telemetry verifier.

Full HAIC spec requires TLSNotary + SGX attestation (Maestro's attestation.py).
For the hackathon we implement the behavioural telemetry lane: typing cadence,
session duration, and entropy of the human response text.

Rules (from Maestro viability_policy.yaml):
  - min_human_reaction_ms: 100.0     (response must arrive after 100ms)
  - min_plausible_session_ms: 5000.0 (whole session ≥ 5 seconds)
  - automation_regularity_threshold: 0.05 (metronomic typing penalised)
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class PoGTelemetry:
    """Behavioural telemetry submitted by the client during an interview turn."""
    turn_index: int = 0
    response_text: str = ""
    client_start_ms: int = 0   # epoch-ms when user started typing
    client_end_ms: int = 0     # epoch-ms when user submitted
    keystroke_intervals: List[int] = field(default_factory=list)  # ms between keystrokes


@dataclass
class PoGResult:
    """Result of a PoG verification pass."""
    verified: bool = False
    provenance_score: float = 0.0   # 0.0–1.0; must be ≥ 0.90 for settlement
    penalties: List[str] = field(default_factory=list)
    bonuses: List[str] = field(default_factory=list)
    reaction_ms: Optional[float] = None
    session_duration_ms: Optional[float] = None
    text_entropy: Optional[float] = None
    regularity_score: Optional[float] = None  # 0 = random, 1 = metronomic

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify(telemetry_records: List[PoGTelemetry], session_start_ms: Optional[int] = None) -> PoGResult:
    """
    Run PoG verification over a list of turn telemetry records.

    Returns a PoGResult with provenance_score ∈ [0, 1].
    Score ≥ 0.90 is required for settlement gate.
    """
    result = PoGResult()
    score = 1.0
    MIN_REACTION_MS = 100.0
    MIN_SESSION_MS = 5000.0
    REGULARITY_THRESHOLD = 0.05

    if not telemetry_records:
        result.penalties.append("no_telemetry_records")
        result.provenance_score = 0.0
        return result

    # ---- Reaction time check (first turn) ----
    first = telemetry_records[0]
    reaction = float(first.client_end_ms - first.client_start_ms)
    result.reaction_ms = reaction

    if reaction < MIN_REACTION_MS:
        score -= 0.15
        result.penalties.append(f"sub_human_reaction_ms:{reaction:.0f}")
    else:
        result.bonuses.append("reaction_time_ok")

    # ---- Session duration check ----
    if session_start_ms is not None:
        last = telemetry_records[-1]
        duration = float(last.client_end_ms - session_start_ms)
        result.session_duration_ms = duration
        if duration < MIN_SESSION_MS:
            score -= 0.10
            result.penalties.append(f"session_too_short_ms:{duration:.0f}")
        else:
            result.bonuses.append("session_duration_ok")

    # ---- Metronomic typing check ----
    all_intervals: List[int] = []
    for rec in telemetry_records:
        all_intervals.extend(rec.keystroke_intervals)

    if len(all_intervals) >= 5:
        mean_i = sum(all_intervals) / len(all_intervals)
        variance = sum((x - mean_i) ** 2 for x in all_intervals) / len(all_intervals)
        std_i = math.sqrt(variance) if variance > 0 else 0.0
        cv = std_i / mean_i if mean_i > 0 else 0.0  # coefficient of variation
        # Low CV = metronomic (automation red flag)
        regularity = max(0.0, 1.0 - cv)
        result.regularity_score = regularity
        if regularity > (1.0 - REGULARITY_THRESHOLD):
            score -= 0.20
            result.penalties.append(f"metronomic_typing:cv={cv:.3f}")
        else:
            result.bonuses.append(f"natural_typing_variance:cv={cv:.3f}")
    else:
        result.bonuses.append("keystroke_telemetry_not_required")

    # ---- Text entropy check (Shannon entropy of characters) ----
    combined_text = " ".join(r.response_text for r in telemetry_records)
    if combined_text.strip():
        text_entropy = _shannon_entropy(combined_text)
        result.text_entropy = text_entropy
        # Very low entropy (< 2.0 bits) suggests templated/repeated output
        if text_entropy < 2.0:
            score -= 0.10
            result.penalties.append(f"low_text_entropy:{text_entropy:.3f}")
        elif text_entropy > 3.5:
            result.bonuses.append(f"rich_text_entropy:{text_entropy:.3f}")
    else:
        score -= 0.20
        result.penalties.append("empty_response_text")

    # ---- Minimum word count ----
    word_count = len(combined_text.split())
    if word_count < 5:
        score -= 0.15
        result.penalties.append(f"too_short:{word_count}_words")
    else:
        result.bonuses.append(f"word_count:{word_count}")

    result.provenance_score = round(max(0.0, min(1.0, score)), 4)
    result.verified = result.provenance_score >= 0.90
    return result


def compute_session_hash(session_id: str, turns: List[Dict[str, str]]) -> str:
    """Deterministic hash of a session's conversation content."""
    hasher = hashlib.sha256()
    hasher.update(session_id.encode())
    for turn in turns:
        hasher.update(turn.get("role", "").encode())
        hasher.update(turn.get("content", "").encode())
    return hasher.hexdigest()


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of character distribution in bits."""
    if not text:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy
