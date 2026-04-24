"""
Observation-tile encoders for SimSatEnv.

Implements the `TileEncoder` protocol plus three concrete encoders:

  * `IdentityEncoder`       — returns the raw tile unchanged (default behavior,
                              equivalent to SimSatEnv with no encoder).
  * `DummyEncoder`          — deterministic fake embedding used for unit tests
                              when LFM2.5-VL isn't available. Returns a fixed
                              low-dim vector whose values are derived from the
                              tile's mean/std, so the shape-and-range contract
                              works without any model load.
  * `LFM2VLEncoderStub`     — documented scaffold for the real LFM2.5-VL
                              encoder. Does NOT load weights; raises
                              NotImplementedError on encode(). Ship this in T4,
                              replace the body during the LFM2.5 integration
                              phase (T4c or later).

DESIGN GOAL
-----------
An encoder returns a 1-D numpy embedding of fixed dimensionality. The observation
shape produced by SimSatEnv shifts from `(1, OBS_H, OBS_W)` pixel tiles to
`(embed_dim,)` vectors when an encoder is plugged in. MuZero's representation
network adapts by switching from IMPALA CNN to a fully-connected encoder
(already present in muzero-general via `network="fullyconnected"`).

WIRING INTO SimSatEnv
---------------------
Phase 5 T3's `simsat_env.py` doesn't carry an encoder parameter yet; this T4
bundle documents the minimal patch required in `simsat_env_encoder.patch.md`.
Two-line change: accept `tile_encoder` kwarg in __init__, call it from
`_tile_to_obs()`.

Until that patch lands, this module is consumable as a standalone library: the
smoke test drives encoders directly against synthetic tiles to validate the
interface.

LFM2.5-VL INTEGRATION NOTES
---------------------------
The real LFM2.5-VL path (to be filled in at T4c or equivalent):

    1. Load LFM2.5-VL (e.g. via transformers + Liquid's model hub) in __init__.
       Frozen base weights live once on disk; LoRA adapter loads on top.
    2. Per call, run the tile through LFM2.5's vision tower, extract the
       pooled embedding (~768 or 1024 dim depending on variant).
    3. Return as numpy float32.

Per-tile latency should be sub-100ms on Orin 16GB once the model is cached in
VRAM. Stage-2 fine-tuning runs LoRA on the vision tower + a small projector.
Stage-3 TTT adapts the LoRA + MuZero heads jointly (gated by Phase 5 T2's
viability check once that ships).
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Protocol                                                                     #
# --------------------------------------------------------------------------- #

class TileEncoder(Protocol):
    """Minimum surface an encoder must expose.

    `embed_dim` is the shape[0] of the returned vector. None means the encoder
    passes through the raw tile (e.g. IdentityEncoder).
    """
    embed_dim: int | None

    def encode(self, tile: np.ndarray) -> np.ndarray:
        """Convert an input tile (1, H, W) or (C, H, W) float32 into an embedding."""
        ...


# --------------------------------------------------------------------------- #
# Concrete encoders                                                            #
# --------------------------------------------------------------------------- #

class IdentityEncoder:
    """Passes the tile through unchanged. SimSatEnv default.

    `embed_dim=None` signals to downstream consumers that the observation is
    still a pixel tile, not a flat embedding.
    """

    embed_dim: int | None = None

    def encode(self, tile: np.ndarray) -> np.ndarray:
        return tile


class DummyEncoder:
    """Deterministic fake embedding for interface testing.

    Computes a `embed_dim`-vector from simple tile statistics so tests exercise
    the same shape contract the real encoder will. No randomness — same tile in
    yields same embedding out.

    Useful when:
      * Running the smoke test on systems without torch / transformers.
      * Validating the SimSatEnv encoder hook without loading LFM2.5 weights.
      * Writing unit tests for downstream components that consume embeddings.
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim

    def encode(self, tile: np.ndarray) -> np.ndarray:
        arr = np.asarray(tile, dtype=np.float32).reshape(-1)
        # Derive an embed_dim-vector from mean, std, min, max, plus a hashed
        # fingerprint of the first N bytes. Stable + interpretable.
        mean = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std()) if arr.size else 0.0
        mn = float(arr.min()) if arr.size else 0.0
        mx = float(arr.max()) if arr.size else 0.0
        # Hash fingerprint — deterministic but spreads across embed_dim
        seed = int(abs(hash(arr.tobytes()[:1024])) % (2**32 - 1))
        rng = np.random.default_rng(seed)
        base = rng.normal(size=self.embed_dim).astype(np.float32)
        # Scale by statistics so different tiles produce distinguishable embeddings
        base[0] = mean
        base[1] = std
        base[2] = mn
        base[3] = mx
        return base


class LFM2VLEncoderStub:
    """Scaffold for the real LFM2.5-VL observation encoder.

    This stub does NOT load model weights. Calling `encode()` raises
    NotImplementedError with guidance. Replace the body during the LFM2.5
    integration phase (T4c / Phase 6).

    The constructor signature previews the config the real implementation will
    need; it intentionally accepts overrides via kwargs so the integration can
    plug in new params without breaking callers.
    """

    def __init__(
        self,
        model_id: str = "liquid/lfm2.5-vl",
        embed_dim: int = 768,
        device: str = "cuda",
        lora_adapter_path: str | None = None,
        **extra_kwargs,
    ):
        self.model_id = model_id
        self.embed_dim = embed_dim
        self.device = device
        self.lora_adapter_path = lora_adapter_path
        self._extra_kwargs = extra_kwargs
        logger.warning(
            "LFM2VLEncoderStub instantiated for model %s (device=%s, embed_dim=%d). "
            "This is a scaffold; encode() raises NotImplementedError. Replace the body "
            "when LFM2.5-VL is being integrated.",
            model_id, device, embed_dim,
        )

    def encode(self, tile: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LFM2VLEncoderStub.encode() is not yet wired. Integration tasks:\n"
            "  1. Load LFM2.5-VL base weights + LoRA adapter via transformers.\n"
            "  2. Pass tile through vision tower (expects 3-channel or preprocessed).\n"
            "  3. Extract pooled embedding (shape (embed_dim,)).\n"
            "  4. Return as numpy float32.\n"
            "See the module docstring for integration notes."
        )


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def build_encoder(kind: str = "identity", **kwargs) -> TileEncoder:
    """Build an encoder by kind string.

    Valid kinds: "identity", "dummy", "lfm2vl" (or "lfm25vl").
    kwargs are forwarded to the concrete encoder's __init__.
    """
    kind_lower = kind.lower().strip()
    if kind_lower == "identity":
        return IdentityEncoder()
    if kind_lower == "dummy":
        return DummyEncoder(**kwargs)
    if kind_lower in ("lfm2vl", "lfm25vl", "lfm2.5vl", "lfm2.5-vl"):
        return LFM2VLEncoderStub(**kwargs)
    raise ValueError(f"Unknown encoder kind {kind!r}; expected identity | dummy | lfm2vl")
