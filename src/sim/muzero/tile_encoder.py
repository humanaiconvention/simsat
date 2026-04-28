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


class HFVisionTowerEncoder:
    """Real-weights tile encoder backed by any HuggingFace vision-language model.

    Default is SigLIP-base (768-dim, ~370 MB). Now also handles
    `LiquidAI/LFM2.5-VL-450M` and `LiquidAI/LFM2.5-VL-1.6B` directly —
    the loader detects the `lfm2_vl` model type, falls back to
    `AutoModelForImageTextToText`, and routes the encode pass through
    `model.model.vision_tower` (a `Siglip2VisionModel` under the LFM
    wrapper) instead of `model.vision_model`. Same `(embed_dim,)` numpy
    output, same MuZero env hook.

    Liquid Track integration: `LiquidAI/LFM2.5-VL-450M` is now public
    (released 2026-04-11) and is the recommended choice for the on-orbit
    framing — 450M params, sub-250ms edge inference per Liquid AI's
    benchmarks, vision tower is SigLIP-2 NaFlex shape-optimized 86M.

    Network access required at first use to download the model. After the
    initial download HuggingFace caches it under ~/.cache/huggingface/.
    Offline operation works once the cache is populated.

    Performance notes (BEAST RTX 2080):
        * SigLIP-base CPU: ~150 ms / tile (offline replay budget).
        * SigLIP-base CUDA: ~10 ms / tile (verified end-to-end).
        * LFM2.5-VL-450M CUDA: TBD — measure after first eval.
        * Output is float32 numpy of shape (embed_dim,).
    """

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
        embed_dim: int | None = None,
        lora_adapter_path: str | None = None,
        **extra_kwargs,
    ):
        # Imports are deferred so that simply importing this module
        # doesn't require torch/transformers/PIL to be installed (CI
        # smoke tests use Identity/Dummy encoders).
        from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText, AutoProcessor
        import torch

        self.model_id = model_id
        self.device = device
        self.lora_adapter_path = lora_adapter_path
        self._extra_kwargs = extra_kwargs
        self._torch = torch

        # Decide loader by model_type. Standalone vision encoders (clip,
        # siglip, siglip2) load via AutoModel; full VL wrappers (lfm2_vl,
        # gemma3, etc.) need AutoModelForImageTextToText so the wrapper's
        # vision_tower submodule is reachable.
        cfg = AutoConfig.from_pretrained(model_id)
        self._model_type = getattr(cfg, "model_type", "")
        wrapper_types = {"lfm2_vl", "gemma3", "llava", "paligemma"}
        if self._model_type in wrapper_types:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id).to(device).eval()
        else:
            self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        if lora_adapter_path:
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
                logger.info("Loaded LoRA adapter %s onto %s", lora_adapter_path, model_id)
            except ImportError:
                logger.warning(
                    "lora_adapter_path=%s ignored — `peft` not installed.",
                    lora_adapter_path,
                )

        if embed_dim is None:
            # Infer from model config — fields differ per architecture.
            cfg = self.model.config
            inferred = (
                getattr(cfg, "projection_dim", None)
                or getattr(getattr(cfg, "vision_config", None), "hidden_size", None)
                or getattr(cfg, "hidden_size", None)
            )
            if inferred is None:
                raise ValueError(
                    f"Could not infer embed_dim from {model_id!r} config. "
                    "Pass embed_dim= explicitly."
                )
            embed_dim = int(inferred)
        self.embed_dim = embed_dim

        logger.info(
            "HFVisionTowerEncoder ready: model=%s device=%s embed_dim=%d lora=%s",
            model_id, device, embed_dim, lora_adapter_path or "<none>",
        )

    def encode(self, tile: np.ndarray) -> np.ndarray:
        """Convert a (1, H, W) or (C, H, W) float32-in-[0,1] tile into an embedding."""
        from PIL import Image

        arr = np.asarray(tile, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            # Grayscale → 3-channel by replication
            arr = np.repeat(arr, 3, axis=0)
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(
                f"Expected tile shape (1,H,W) or (3,H,W); got {arr.shape}"
            )
        # CHW float32 [0,1] → HWC uint8 for PIL
        hwc = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(hwc)

        # Some VL processors (Lfm2VlProcessor, others) raise when called
        # with images-only because the chat-template expects a text prompt
        # too. For pure embedding use, prefer the inner image_processor
        # if present so we don't need to fabricate a placeholder prompt.
        image_only_processor = getattr(self.processor, "image_processor", None) or self.processor
        inputs = image_only_processor(images=img, return_tensors="pt").to(self.device)

        # Wrapper VL models (lfm2_vl, gemma3, llava, paligemma) expose
        # `get_image_features` but it returns a list (one per image, NaFlex
        # style) and is meant for downstream multimodal generation, not
        # standalone embedding. For wrappers we always call the vision
        # tower directly. For standalone CLIP/SigLIP, get_image_features
        # is the canonical pooled-embedding path and we use it.
        wrapper_types = {"lfm2_vl", "gemma3", "llava", "paligemma"}
        with self._torch.no_grad():
            if self._model_type not in wrapper_types and hasattr(self.model, "get_image_features"):
                # CLIP / SigLIP path
                out = self.model.get_image_features(**inputs)
            else:
                # Generic VLM path — run the vision tower directly. Wrapper
                # classes (Lfm2VlForConditionalGeneration etc.) expose the
                # vision tower at .model.vision_tower; standalone vision
                # towers expose it as .vision_model on the top object.
                vision_tower = (
                    getattr(self.model, "vision_model", None)
                    or getattr(getattr(self.model, "model", None), "vision_tower", None)
                    or getattr(self.model, "vision_tower", None)
                )
                if vision_tower is None:
                    raise RuntimeError(
                        f"HFVisionTowerEncoder could not locate a vision tower on "
                        f"{type(self.model).__name__} (model_id={self.model_id!r}, "
                        f"model_type={self._model_type!r})"
                    )
                # Some VL wrappers' processors return text + vision keys; keep
                # only the vision inputs (pixel_values / pixel_attention_mask /
                # spatial_shapes for SigLIP-2 NaFlex).
                vision_keys = {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
                vision_inputs = {k: v for k, v in inputs.items() if k in vision_keys}
                out = vision_tower(**vision_inputs)

        # Normalize to a flat tensor across (transformers version, model type).
        # Different transformers releases return either a raw tensor or a
        # ModelOutput dataclass; cover both.
        if self._torch.is_tensor(out):
            features = out
        elif hasattr(out, "image_embeds") and out.image_embeds is not None:
            features = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            features = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            features = out.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError(
                f"HFVisionTowerEncoder could not extract an embedding from "
                f"{type(out).__name__} for model {self.model_id!r}"
            )
        # SigLIP-2 NaFlex (the LFM2.5-VL vision tower) splits a single
        # image into multiple shape-optimized sub-tiles, so the output's
        # leading dim can be > 1 even for a single input image. Pool
        # across that dim to keep the (embed_dim,) output contract.
        features = features.detach().to(self._torch.float32)
        while features.dim() > 1:
            features = features.mean(dim=0)
        # Cast to float32 BEFORE numpy() — numpy does not natively support
        # bfloat16, and modern VL wrappers default to bf16 on CUDA.
        return features.cpu().numpy().astype(np.float32)


# Back-compat alias — `LFM2VLEncoderStub` was the documented integration seat
# before SigLIP substitution. Keeping the name preserves any callers that
# imported it directly. The stub is now a thin shim over the real encoder.
class LFM2VLEncoderStub(HFVisionTowerEncoder):
    """Deprecated alias retained for back-compat. Use HFVisionTowerEncoder."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("model_id", "google/siglip-base-patch16-224")
        super().__init__(*args, **kwargs)
        logger.info(
            "LFM2VLEncoderStub is deprecated; backed by HFVisionTowerEncoder "
            "(model=%s). Update callers to instantiate HFVisionTowerEncoder "
            "directly.", self.model_id,
        )


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def build_encoder(kind: str = "identity", **kwargs) -> TileEncoder:
    """Build an encoder by kind string.

    Valid kinds:
      * "identity"        — pass-through (no encoder)
      * "dummy"           — deterministic fake (no model load; for tests)
      * "hf_vision_tower" — any HuggingFace VLM via HFVisionTowerEncoder
                            (default model: SigLIP-base)
      * "lfm2vl" (or "lfm25vl", "lfm2.5vl", "lfm2.5-vl") — LFM2.5-VL slot.
                            Defaults to `LiquidAI/LFM2.5-VL-450M` (public
                            since 2026-04-11). HFVisionTowerEncoder
                            auto-detects the `lfm2_vl` model_type and routes
                            the encode pass through `model.model.vision_tower`
                            (Siglip2VisionModel, 768-dim). Pass model_id=
                            "LiquidAI/LFM2.5-VL-1.6B" for the larger variant.

    kwargs are forwarded to the concrete encoder's __init__.
    """
    kind_lower = kind.lower().strip()
    if kind_lower == "identity":
        return IdentityEncoder()
    if kind_lower == "dummy":
        return DummyEncoder(**kwargs)
    if kind_lower in ("hf_vision_tower", "vision_tower", "hf_vlm"):
        return HFVisionTowerEncoder(**kwargs)
    if kind_lower in ("lfm2vl", "lfm25vl", "lfm2.5vl", "lfm2.5-vl"):
        kwargs.setdefault("model_id", "LiquidAI/LFM2.5-VL-450M")
        return HFVisionTowerEncoder(**kwargs)
    raise ValueError(
        f"Unknown encoder kind {kind!r}; expected "
        "identity | dummy | hf_vision_tower | lfm2vl"
    )
