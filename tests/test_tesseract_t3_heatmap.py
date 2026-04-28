"""Tests for the Tesseract T³ heatmap adapter — pure-Python paths.

Covers the heatmap-stats extractor and the stats→ObservationVLA payload
mapping. Skips the actual model load (which requires Garrett's weights).
"""
from __future__ import annotations

import numpy as np
import pytest

from observation_vla.backend_factory import build_vla_adapter
from observation_vla.tesseract_t3_heatmap import (
    TesseractT3HeatmapAdapter,
    _extract_stats,
    _stats_to_payload,
)


# --------------------------------------------------------------------- #
# _extract_stats                                                        #
# --------------------------------------------------------------------- #


def test_extract_stats_uniform_zero():
    h = np.zeros((16, 16), dtype=np.float32)
    s = _extract_stats(h)
    assert s["peak"] == 0.0
    assert s["mean"] == 0.0
    assert s["hotspot_density"] == 0.0


def test_extract_stats_uniform_one():
    h = np.ones((16, 16), dtype=np.float32)
    s = _extract_stats(h)
    assert s["peak"] == 1.0
    assert s["mean"] == 1.0
    # Every pixel is > 0.5 → density = 1.0
    assert s["hotspot_density"] == 1.0


def test_extract_stats_concentrated_hotspot():
    """Single bright spot in the center should have high concentration + low offset."""
    h = np.zeros((32, 32), dtype=np.float32)
    h[14:18, 14:18] = 1.0  # 4x4 hotspot dead center
    s = _extract_stats(h)
    assert s["peak"] == 1.0
    assert s["hotspot_density"] == pytest.approx(16 / 1024, abs=1e-6)
    assert s["center_offset"] < 0.1  # centered
    assert s["hotspot_concentration"] > 0  # there IS a concentrated hotspot


def test_extract_stats_off_center_hotspot():
    """Hotspot in a corner should have high center_offset."""
    h = np.zeros((32, 32), dtype=np.float32)
    h[0:4, 0:4] = 1.0  # top-left corner
    s = _extract_stats(h)
    # Centroid of top-10% is near (0,0), so offset = 0.5 + 0.5 = 1.0
    assert s["center_offset"] > 0.7


# --------------------------------------------------------------------- #
# _stats_to_payload action ladder                                       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "peak,scene_concentration,expected_action",
    [
        (0.95, 0.9, "accept"),  # high peak + concentrated → accept
        (0.60, 0.5, "refine"),  # mid peak → refine
        (0.30, 0.3, "defer"),   # low peak → defer
        (0.05, 0.0, "skip"),    # near-zero peak → skip
    ],
)
def test_stats_to_payload_action_ladder(peak, scene_concentration, expected_action):
    stats = {
        "peak": peak,
        "mean": peak * 0.5,
        "std": 0.1,
        "hotspot_density": scene_concentration,
        "hotspot_concentration": scene_concentration,
        "center_offset": 0.1,
    }
    payload = _stats_to_payload(
        stats,
        sample={},
        probe={"sentinel_cloud_cover": 5.0},
        geometry={"target_visible": True},
        peak_accept=0.80,
        peak_refine=0.45,
        peak_defer=0.20,
    )
    assert payload["recommended_action"] == expected_action


def test_stats_to_payload_target_invisible_forces_skip():
    stats = {
        "peak": 0.99, "mean": 0.5, "std": 0.1,
        "hotspot_density": 0.9, "hotspot_concentration": 0.9, "center_offset": 0.0,
    }
    payload = _stats_to_payload(
        stats, sample={}, probe={}, geometry={"target_visible": False},
        peak_accept=0.80, peak_refine=0.45, peak_defer=0.20,
    )
    assert payload["recommended_action"] == "skip"
    assert "target_not_visible" in payload["rationale_tags"]


def test_stats_to_payload_payload_keys_match_observation_vla_contract():
    """Payload must carry all 8 keys the assessor downstream consumes."""
    stats = {
        "peak": 0.5, "mean": 0.2, "std": 0.1,
        "hotspot_density": 0.3, "hotspot_concentration": 0.4, "center_offset": 0.2,
    }
    payload = _stats_to_payload(
        stats, sample={}, probe={}, geometry={"target_visible": True},
        peak_accept=0.80, peak_refine=0.45, peak_defer=0.20,
    )
    required_keys = {
        "usable_observation", "scene_match_score", "salience_score",
        "change_or_event_score", "occlusion_or_cloud_risk", "confidence",
        "recommended_action", "rationale_tags",
    }
    assert required_keys.issubset(payload.keys())
    assert isinstance(payload["usable_observation"], bool)
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["recommended_action"] in {"accept", "defer", "refine", "skip"}


def test_stats_to_payload_cloud_risk_from_probe():
    stats = {
        "peak": 0.5, "mean": 0.2, "std": 0.1,
        "hotspot_density": 0.3, "hotspot_concentration": 0.4, "center_offset": 0.2,
    }
    payload = _stats_to_payload(
        stats, sample={}, probe={"sentinel_cloud_cover": 87.5}, geometry={"target_visible": True},
        peak_accept=0.80, peak_refine=0.45, peak_defer=0.20,
    )
    assert payload["occlusion_or_cloud_risk"] == pytest.approx(0.875)
    assert "cloud_risk" in payload["rationale_tags"]


# --------------------------------------------------------------------- #
# Adapter — stub-mode integration                                       #
# --------------------------------------------------------------------- #


def test_adapter_falls_back_to_stub_when_weights_missing(monkeypatch):
    monkeypatch.setenv("TESSERACT_T3_WEIGHTS_PATH", "/nonexistent/weights.bin")
    adapter = TesseractT3HeatmapAdapter()
    assert adapter.runtime_mode == "stub_fallback"
    payload = adapter.assess("ignored", [{"sample": {}, "probe": {}, "geometry": {"target_visible": True}}], {})
    # Stub payload uses peak=0 so action ladder picks skip.
    assert payload["recommended_action"] == "skip"
    assert "tesseract_t3_heatmap" in payload["rationale_tags"]


def test_factory_resolves_tesseract_t3_heatmap(monkeypatch):
    monkeypatch.setenv("TESSERACT_T3_WEIGHTS_PATH", "")
    a = build_vla_adapter("tesseract_t3_heatmap")
    assert a.__class__.__name__ == "TesseractT3HeatmapAdapter"
    # Aliases also resolve
    a2 = build_vla_adapter("t3_heatmap")
    assert a2.__class__.__name__ == "TesseractT3HeatmapAdapter"
