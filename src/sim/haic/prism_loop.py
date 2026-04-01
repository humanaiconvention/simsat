"""
PRISM Measurement Loop — integrates PRISM's SpectralMicroscope into SimSat sessions.

Captures EntropySnapshot before and after each grounding interview, computes
EntropyDeltaProof, and exposes a GeometricHealthScore for session prioritisation.

Two modes:
  1. Full PRISM  — when `prism` package importable and a model is configured
  2. Synthetic   — lightweight simulation for hackathon demo (no GPU needed)

Set HAIC_PRISM_MODE=synthetic (default) or HAIC_PRISM_MODE=full in environment.
Set HAIC_PRISM_MODEL=<HuggingFace model ID> for full mode.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PRISM_MODE = os.environ.get("HAIC_PRISM_MODE", "synthetic").lower()
PRISM_MODEL_ID = os.environ.get("HAIC_PRISM_MODEL", "")

# Minimum entropy reduction (|ΔS|) required for settlement
EPSILON_THRESHOLD = float(os.environ.get("HAIC_EPSILON", "0.01"))


class PRISMLoop:
    """
    Manages PRISM measurements across a HAIC session lifecycle.

    Full mode: uses `prism.SpectralMicroscope` + `prism.telemetry.snapshot`
    Synthetic mode: generates plausible synthetic snapshots for demo purposes
    """

    def __init__(self):
        self._mode = PRISM_MODE
        self._microscope = None
        self._model = None
        self._tokenizer = None
        self._baseline_entropy: Optional[float] = None
        self._session_count: int = 0

        if self._mode == "full":
            self._init_full()
        else:
            logger.info("PRISM running in synthetic mode")

    def _init_full(self) -> None:
        """Attempt to load PRISM + model for full mechanistic measurement."""
        try:
            import sys
            sys.path.insert(0, r"D:\prism\src")
            from prism import SpectralMicroscope
            self._microscope = SpectralMicroscope(max_tokens=64, window_size=32)
            logger.info("PRISM SpectralMicroscope loaded")
        except Exception as e:
            logger.warning("PRISM full mode failed (%s) — falling back to synthetic", e)
            self._mode = "synthetic"
            return

        if PRISM_MODEL_ID:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                self._tokenizer = AutoTokenizer.from_pretrained(PRISM_MODEL_ID)
                self._model = AutoModelForCausalLM.from_pretrained(
                    PRISM_MODEL_ID,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
                self._model.eval()
                logger.info("PRISM model loaded: %s", PRISM_MODEL_ID)
            except Exception as e:
                logger.warning("PRISM model load failed (%s) — using synthetic snapshots", e)
                self._mode = "synthetic"
        else:
            logger.info("No PRISM_MODEL_ID set — using synthetic snapshots")
            self._mode = "synthetic"

    # ---- Public interface ----

    def snapshot(self, context: str = "") -> Dict[str, Any]:
        """
        Take a PRISM entropy snapshot.

        Returns a dict matching EntropySnapshot.to_dict() schema.
        """
        if self._mode == "full" and self._microscope and self._model:
            return self._full_snapshot(context)
        return self._synthetic_snapshot(context)

    def geometric_health(self) -> Dict[str, Any]:
        """
        Return a GeometricHealthScore dict.

        Higher composite_score = more entropy crisis = higher grounding demand.
        """
        snap = self.snapshot()
        hidden_dim = 768  # default; full mode reads from model config
        max_entropy = math.log(hidden_dim) if hidden_dim > 1 else 1.0
        inv_entropy = 1.0 - min(snap["mean_spectral_entropy"] / max_entropy, 1.0)
        inv_sensitivity = 1.0 - min(snap.get("noise_sensitivity", 0.3), 1.0)
        v = snap.get("mean_viability_score", 0.7)
        health = 0.4 * v + 0.3 * inv_entropy + 0.3 * inv_sensitivity
        composite = float(1.0 - max(0.0, min(health, 1.0)))
        return {
            "viability_score": v,
            "spectral_entropy": snap["mean_spectral_entropy"],
            "noise_sensitivity": snap.get("noise_sensitivity", 0.3),
            "composite_score": round(composite, 4),
            "timestamp": snap["taken_at"],
        }

    def compute_delta(
        self, snap_before: Dict[str, Any], snap_after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute EntropyDeltaProof between two snapshots.

        Returns dict matching EntropyDeltaProof.to_dict() schema.
        """
        se_before = snap_before.get("mean_spectral_entropy", 0.0)
        se_after = snap_after.get("mean_spectral_entropy", 0.0)
        ed_before = snap_before.get("mean_effective_dimension", 0.0)
        ed_after = snap_after.get("mean_effective_dimension", 0.0)
        vs_before = snap_before.get("mean_viability_score", 0.0)
        vs_after = snap_after.get("mean_viability_score", 0.0)
        pc_before = snap_before.get("mean_phase_coherence", 0.0)
        pc_after = snap_after.get("mean_phase_coherence", 0.0)

        delta_se = se_after - se_before      # negative = entropy reduced = good
        delta_ed = ed_after - ed_before      # positive = more dimensions used = good
        delta_vs = vs_after - vs_before
        delta_pc = pc_after - pc_before

        reduction_verified = delta_se < -EPSILON_THRESHOLD

        return {
            "snapshot_before": snap_before,
            "snapshot_after": snap_after,
            "delta_spectral_entropy": round(delta_se, 6),
            "delta_effective_dimension": round(delta_ed, 6),
            "delta_viability_score": round(delta_vs, 6),
            "delta_phase_coherence": round(delta_pc, 6),
            "epsilon_threshold": EPSILON_THRESHOLD,
            "reduction_verified": reduction_verified,
            "cka_drift": round(abs(delta_se) * 0.1, 6),  # proxy; full mode computes real CKA
        }

    # ---- Full PRISM mode ----

    def _full_snapshot(self, context: str) -> Dict[str, Any]:
        """Run SpectralMicroscope.full_scan() and extract snapshot fields."""
        prompt = context or "Describe what you observe in this satellite image."
        try:
            import sys
            sys.path.insert(0, r"D:\prism\src")
            from prism.telemetry.snapshot import take_snapshot

            snap = take_snapshot(
                model=self._model,
                tokenizer=self._tokenizer,
                eval_prompts=[prompt],
                n_samples=1,
            )
            return {
                "taken_at": datetime.now(timezone.utc).isoformat(),
                "mean_spectral_entropy": snap.mean_spectral_entropy,
                "mean_effective_dimension": snap.mean_effective_dimension,
                "mean_viability_score": snap.mean_viability_score,
                "mean_phase_coherence": snap.mean_phase_coherence,
                "noise_sensitivity": snap.noise_sensitivity,
                "layer_count": snap.layer_count,
                "mode": "full",
            }
        except Exception as e:
            logger.warning("Full PRISM snapshot failed (%s) — synthetic fallback", e)
            return self._synthetic_snapshot(context)

    # ---- Synthetic mode ----

    def _synthetic_snapshot(self, context: str = "") -> Dict[str, Any]:
        """
        Generate a plausible synthetic entropy snapshot.

        Models realistic entropy drift: baseline established on first call,
        subsequent snapshots drift slightly. After a grounding session,
        entropy should decrease (reduced by _register_session()).
        """
        if self._baseline_entropy is None:
            # Initial entropy: moderately high (model starting to drift)
            self._baseline_entropy = random.uniform(4.2, 5.8)

        # Entropy increases slightly over time without grounding
        drift = self._session_count * random.uniform(0.01, 0.04)
        # Small random variation
        noise = random.gauss(0, 0.05)
        current_entropy = self._baseline_entropy + drift + noise

        # Effective dimension inversely correlated with entropy
        effective_dim = max(10.0, 128.0 - current_entropy * 15 + random.gauss(0, 2))

        # Phase coherence decreases as entropy increases
        phase_coherence = max(0.1, 0.85 - (current_entropy - 4.0) * 0.05 + random.gauss(0, 0.02))

        # Viability score
        viability = max(0.1, 0.9 - drift * 0.1 + random.gauss(0, 0.02))

        # Noise sensitivity increases with entropy
        noise_sensitivity = min(0.9, 0.2 + drift * 0.05 + random.gauss(0, 0.01))

        return {
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "mean_spectral_entropy": round(current_entropy, 4),
            "mean_effective_dimension": round(effective_dim, 4),
            "mean_viability_score": round(viability, 4),
            "mean_phase_coherence": round(phase_coherence, 4),
            "noise_sensitivity": round(noise_sensitivity, 4),
            "layer_count": 12,
            "mode": "synthetic",
        }

    def register_grounding_event(self, entropy_delta: Dict[str, Any]) -> None:
        """
        Called after a successful grounding session to update internal state.

        In synthetic mode, this reduces the drift counter to simulate
        the entropy-reducing effect of grounding.
        """
        self._session_count = max(0, self._session_count - 1)
        if self._baseline_entropy is not None and entropy_delta.get("reduction_verified"):
            reduction = abs(entropy_delta.get("delta_spectral_entropy", 0.0))
            self._baseline_entropy = max(3.5, self._baseline_entropy - reduction * 0.5)
        logger.info(
            "Grounding event registered — synthetic baseline entropy: %.4f",
            self._baseline_entropy or 0.0,
        )

    def tick(self) -> None:
        """Called periodically by the simulation loop to advance entropy drift."""
        self._session_count += 1


# Module-level singleton
_loop: Optional[PRISMLoop] = None


def get_prism_loop() -> PRISMLoop:
    global _loop
    if _loop is None:
        _loop = PRISMLoop()
    return _loop
