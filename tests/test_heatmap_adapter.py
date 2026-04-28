"""Tests for the generic heatmap-output ObservationVLA adapter."""
from __future__ import annotations

import numpy as np
import pytest

from observation_vla.backend_factory import build_vla_adapter
from observation_vla.heatmap_local import (
    HeatmapAdapter,
    _extract_stats,
    _stats_to_payload,
)


# --------------------------------------------------------------------- #
# _extract_stats                                                        #
# --------------------------------------------------------------------- #


def test_extract_stats_uniform_zero():
    s = _extract_stats(np.zeros((16, 16), dtype=np.float32))
    assert s["peak"] == 0.0
    assert s["mean"] == 0.0
    assert s["hotspot_density"] == 0.0


def test_extract_stats_uniform_one():
    s = _extract_stats(np.ones((16, 16), dtype=np.float32))
    assert s["peak"] == 1.0
    assert s["mean"] == 1.0
    assert s["hotspot_density"] == 1.0


def test_extract_stats_concentrated_centered_hotspot():
    h = np.zeros((32, 32), dtype=np.float32)
    h[14:18, 14:18] = 1.0
    s = _extract_stats(h)
    assert s["peak"] == 1.0
    assert s["hotspot_density"] == pytest.approx(16 / 1024, abs=1e-6)
    assert s["center_offset"] < 0.1
    assert s["hotspot_concentration"] > 0


def test_extract_stats_off_center_hotspot():
    h = np.zeros((32, 32), dtype=np.float32)
    h[0:4, 0:4] = 1.0
    s = _extract_stats(h)
    assert s["center_offset"] > 0.7


# --------------------------------------------------------------------- #
# _stats_to_payload action ladder                                       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "peak,concentration,expected",
    [
        (0.95, 0.9, "accept"),
        (0.60, 0.5, "refine"),
        (0.30, 0.3, "defer"),
        (0.05, 0.0, "skip"),
    ],
)
def test_stats_to_payload_action_ladder(peak, concentration, expected):
    stats = {
        "peak": peak, "mean": peak * 0.5, "std": 0.1,
        "hotspot_density": concentration, "hotspot_concentration": concentration,
        "center_offset": 0.1,
    }
    payload = _stats_to_payload(
        stats, sample={}, probe={"sentinel_cloud_cover": 5.0},
        geometry={"target_visible": True},
        peak_accept=0.80, peak_refine=0.45, peak_defer=0.20,
    )
    assert payload["recommended_action"] == expected


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
    stats = {
        "peak": 0.5, "mean": 0.2, "std": 0.1,
        "hotspot_density": 0.3, "hotspot_concentration": 0.4, "center_offset": 0.2,
    }
    payload = _stats_to_payload(
        stats, sample={}, probe={}, geometry={"target_visible": True},
        peak_accept=0.80, peak_refine=0.45, peak_defer=0.20,
    )
    required = {
        "usable_observation", "scene_match_score", "salience_score",
        "change_or_event_score", "occlusion_or_cloud_risk", "confidence",
        "recommended_action", "rationale_tags",
    }
    assert required.issubset(payload.keys())
    assert isinstance(payload["usable_observation"], bool)
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["recommended_action"] in {"accept", "defer", "refine", "skip"}


def test_stats_to_payload_cloud_risk_from_probe():
    stats = {
        "peak": 0.5, "mean": 0.2, "std": 0.1,
        "hotspot_density": 0.3, "hotspot_concentration": 0.4, "center_offset": 0.2,
    }
    payload = _stats_to_payload(
        stats, sample={}, probe={"sentinel_cloud_cover": 87.5},
        geometry={"target_visible": True},
        peak_accept=0.80, peak_refine=0.45, peak_defer=0.20,
    )
    assert payload["occlusion_or_cloud_risk"] == pytest.approx(0.875)
    assert "cloud_risk" in payload["rationale_tags"]


def test_stats_to_payload_custom_backend_tag_propagates():
    stats = {
        "peak": 0.5, "mean": 0.2, "std": 0.1,
        "hotspot_density": 0.3, "hotspot_concentration": 0.4, "center_offset": 0.2,
    }
    payload = _stats_to_payload(
        stats, sample={}, probe={}, geometry={"target_visible": True},
        peak_accept=0.80, peak_refine=0.45, peak_defer=0.20,
        backend_tag="my_custom_model",
    )
    assert "my_custom_model" in payload["rationale_tags"]
    assert payload["_model_id"] == "my_custom_model"


# --------------------------------------------------------------------- #
# Adapter — stub-mode integration                                       #
# --------------------------------------------------------------------- #


def test_adapter_falls_back_to_stub_when_weights_missing(monkeypatch):
    monkeypatch.setenv("HEATMAP_WEIGHTS_PATH", "/nonexistent/weights.bin")
    adapter = HeatmapAdapter()
    assert adapter.runtime_mode == "stub_fallback"
    payload = adapter.assess(
        "ignored",
        [{"sample": {}, "probe": {}, "geometry": {"target_visible": True}}],
        {},
    )
    assert payload["recommended_action"] == "skip"
    assert "heatmap" in payload["rationale_tags"]


def test_factory_resolves_heatmap_aliases(monkeypatch):
    monkeypatch.setenv("HEATMAP_WEIGHTS_PATH", "")
    for alias in ("heatmap", "heatmap_local"):
        a = build_vla_adapter(alias)
        assert a.__class__.__name__ == "HeatmapAdapter"
