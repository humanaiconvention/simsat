"""
Tests for src/sim/muzero/tile_encoder.py.

Two tiers:

  * **Always-run** (CI-safe, no downloads): build_encoder factory, IdentityEncoder
    pass-through, DummyEncoder shape/determinism, factory rejection of unknown
    kinds, deferred-import guard (importing the module without torch installed
    must not crash).

  * **Skipped unless `RUN_HF_TILE_ENCODER_TESTS=1`**: real SigLIP load + encode
    round-trip. Skipped by default because it pulls a ~370 MB model from
    HuggingFace and needs network access. Run locally with the env var set
    after a one-time `pip install transformers torch Pillow`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_SRC = REPO_ROOT / "src" / "sim"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from muzero.tile_encoder import (
    DummyEncoder,
    HFVisionTowerEncoder,
    IdentityEncoder,
    LFM2VLEncoderStub,
    build_encoder,
)


# --------------------------------------------------------------------------- #
# Always-run: protocol contract                                                #
# --------------------------------------------------------------------------- #

class TestEncoderFactory:
    def test_identity_passthrough(self):
        enc = build_encoder("identity")
        assert isinstance(enc, IdentityEncoder)
        assert enc.embed_dim is None
        tile = np.random.rand(1, 64, 64).astype(np.float32)
        out = enc.encode(tile)
        assert out is tile  # identity returns the same array

    def test_dummy_shape_and_determinism(self):
        enc = build_encoder("dummy", embed_dim=32)
        assert enc.embed_dim == 32
        tile = np.full((1, 64, 64), 0.5, dtype=np.float32)
        a = enc.encode(tile)
        b = enc.encode(tile)
        assert a.shape == (32,)
        assert a.dtype == np.float32
        np.testing.assert_array_equal(a, b)

    def test_dummy_distinguishes_different_tiles(self):
        enc = build_encoder("dummy", embed_dim=64)
        tile_a = np.full((1, 64, 64), 0.2, dtype=np.float32)
        tile_b = np.full((1, 64, 64), 0.8, dtype=np.float32)
        a = enc.encode(tile_a)
        b = enc.encode(tile_b)
        # First 4 dims encode summary stats — must differ for different tiles
        assert not np.allclose(a[:4], b[:4])

    def test_factory_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown encoder kind"):
            build_encoder("not_a_real_encoder")

    def test_factory_accepts_lfm_aliases(self):
        # Without instantiating (which would download SigLIP), just verify
        # the factory routes the aliases to HFVisionTowerEncoder. We do this
        # by patching the class to a no-op.
        from unittest.mock import patch

        sentinel = object()
        with patch(
            "muzero.tile_encoder.HFVisionTowerEncoder",
            return_value=sentinel,
        ):
            for alias in ("lfm2vl", "lfm25vl", "lfm2.5vl", "lfm2.5-vl",
                          "hf_vision_tower", "vision_tower", "hf_vlm"):
                got = build_encoder(alias)
                assert got is sentinel, f"alias {alias!r} did not route to HFVisionTowerEncoder"

    def test_lfm2vl_stub_subclasses_real_encoder(self):
        # Back-compat: LFM2VLEncoderStub must remain importable AND must be
        # a subclass of HFVisionTowerEncoder (so callers that isinstance-check
        # the parent still pass).
        assert issubclass(LFM2VLEncoderStub, HFVisionTowerEncoder)


# --------------------------------------------------------------------------- #
# Opt-in: real model load (network + torch + transformers required)            #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    os.environ.get("RUN_HF_TILE_ENCODER_TESTS") != "1",
    reason="Set RUN_HF_TILE_ENCODER_TESTS=1 to download SigLIP-base (~370 MB) and run.",
)
class TestHFVisionTowerEncoderReal:
    def test_siglip_base_load_and_encode(self):
        enc = HFVisionTowerEncoder(
            model_id="google/siglip-base-patch16-224",
            device="cpu",
        )
        assert enc.embed_dim > 0
        tile = np.random.rand(1, 64, 64).astype(np.float32)
        emb = enc.encode(tile)
        assert emb.shape == (enc.embed_dim,)
        assert emb.dtype == np.float32
        # Sanity: not all zeros, not all NaN
        assert np.isfinite(emb).all()
        assert (emb != 0).any()

    def test_siglip_base_distinguishes_different_tiles(self):
        enc = HFVisionTowerEncoder(
            model_id="google/siglip-base-patch16-224",
            device="cpu",
        )
        tile_dark = np.full((1, 64, 64), 0.05, dtype=np.float32)
        tile_bright = np.full((1, 64, 64), 0.95, dtype=np.float32)
        emb_dark = enc.encode(tile_dark)
        emb_bright = enc.encode(tile_bright)
        # Different inputs should produce different embeddings
        assert not np.allclose(emb_dark, emb_bright, atol=1e-3)

    def test_lfm2vl_450m_load_and_encode(self):
        """LFM2.5-VL-450M is the default of build_encoder('lfm2vl').

        Exercises the wrapper-aware code path (model_type=lfm2_vl,
        AutoModelForImageTextToText loader, image_processor sub-attr,
        vision_tower direct call, bf16 -> fp32 cast). Adds ~900 MB to
        the cache on first run.
        """
        enc = HFVisionTowerEncoder(
            model_id="LiquidAI/LFM2.5-VL-450M",
            device="cpu",
        )
        assert enc.embed_dim == 768
        tile = np.random.rand(1, 64, 64).astype(np.float32)
        emb = enc.encode(tile)
        assert emb.shape == (enc.embed_dim,)
        assert emb.dtype == np.float32
        assert np.isfinite(emb).all()
        assert (emb != 0).any()
