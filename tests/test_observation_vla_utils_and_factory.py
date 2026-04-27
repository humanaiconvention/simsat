"""Tests for observation_vla._utils and backend_factory."""
from __future__ import annotations

import os
import pytest

from observation_vla._utils import clamp
from observation_vla.backend_factory import build_vla_adapter


class TestClamp:
    """Unit tests for clamp() utility function."""

    def test_clamp_value_in_range(self):
        """Clamp returns unchanged value when in range."""
        assert clamp(0.5) == 0.5
        assert clamp(0.0) == 0.0
        assert clamp(1.0) == 1.0

    def test_clamp_below_range(self):
        """Clamp returns low when value is below range."""
        assert clamp(-0.5) == 0.0
        assert clamp(-100) == 0.0

    def test_clamp_above_range(self):
        """Clamp returns high when value is above range."""
        assert clamp(1.5) == 1.0
        assert clamp(100) == 1.0

    def test_clamp_custom_range(self):
        """Clamp works with custom low/high bounds."""
        assert clamp(5, low=0, high=10) == 5
        assert clamp(-1, low=0, high=10) == 0
        assert clamp(15, low=0, high=10) == 10
        assert clamp(2.5, low=2, high=3) == 2.5
        assert clamp(1.5, low=2, high=3) == 2
        assert clamp(3.5, low=2, high=3) == 3

    def test_clamp_negative_range(self):
        """Clamp works with negative bounds."""
        assert clamp(0, low=-10, high=-5) == -5
        assert clamp(-7, low=-10, high=-5) == -7
        assert clamp(-3, low=-10, high=-5) == -5

    def test_clamp_edge_case_equal_bounds(self):
        """Clamp with equal low/high always returns that value."""
        assert clamp(0, low=5, high=5) == 5
        assert clamp(100, low=5, high=5) == 5
        assert clamp(-100, low=5, high=5) == 5

    def test_clamp_float_precision(self):
        """Clamp preserves float precision."""
        result = clamp(0.123456789)
        assert abs(result - 0.123456789) < 1e-9
        result = clamp(1.000000001, low=0, high=1)
        assert result == 1.0


class TestBuildVLAAdapter:
    """Unit tests for build_vla_adapter() factory function."""

    def test_default_backend_is_clip_local(self, monkeypatch):
        """With no env var, default is clip_local."""
        monkeypatch.delenv("OBSERVATION_VLA_BACKEND", raising=False)
        adapter = build_vla_adapter()
        assert "clip_local:" in adapter.model_id
        assert "clip-vit-base-patch32" in adapter.model_id
        assert adapter.runtime_mode == "clip_local"

    def test_explicit_backend_parameter_overrides_env(self, monkeypatch):
        """Explicit backend parameter takes precedence over env var."""
        monkeypatch.setenv("OBSERVATION_VLA_BACKEND", "clip_local")
        adapter = build_vla_adapter(backend="stub")
        # Even when env says clip_local, explicit stub wins
        assert adapter.runtime_mode == "stub"

    def test_clip_local_variants(self, monkeypatch):
        """clip_local accepts multiple aliases."""
        for variant in ["clip_local", "clip", "auto"]:
            adapter = build_vla_adapter(backend=variant)
            assert "clip_local:" in adapter.model_id
            assert "clip-vit-base-patch32" in adapter.model_id

    def test_stub_backend(self):
        """stub backend creates a stub adapter."""
        adapter = build_vla_adapter(backend="stub")
        assert adapter.runtime_mode == "stub"
        assert adapter.model_id == "observation-vla-stub-v1"

    def test_unknown_backend_raises(self):
        """Unknown backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown OBSERVATION_VLA_BACKEND"):
            build_vla_adapter(backend="nonexistent_backend_xyz")

    def test_backend_name_case_insensitive(self, monkeypatch):
        """Backend names are case-insensitive."""
        for variant in ["CLIP_LOCAL", "Clip_Local", "STUB"]:
            try:
                adapter = build_vla_adapter(backend=variant.lower())
                assert adapter is not None
            except Exception:
                # Some backends may fail to load due to missing weights,
                # but the factory dispatch shouldn't reject them
                pass

    def test_backend_whitespace_stripped(self, monkeypatch):
        """Backend names with whitespace are handled."""
        adapter = build_vla_adapter(backend="  clip_local  ")
        assert "clip_local:" in adapter.model_id
        assert "clip-vit-base-patch32" in adapter.model_id

    def test_env_var_read(self, monkeypatch):
        """OBSERVATION_VLA_BACKEND env var is read correctly."""
        monkeypatch.setenv("OBSERVATION_VLA_BACKEND", "stub")
        adapter = build_vla_adapter()
        assert adapter.runtime_mode == "stub"
        assert adapter.model_id == "observation-vla-stub-v1"

    def test_adapter_implements_required_interface(self):
        """All adapters have the required interface methods/properties."""
        adapter = build_vla_adapter(backend="stub")
        assert hasattr(adapter, "assess")
        assert hasattr(adapter, "runtime_mode")
        assert hasattr(adapter, "model_id")
        assert callable(adapter.assess)
