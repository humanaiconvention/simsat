"""Heatmap-output ObservationVLA backend — collaborator insertion seat.

Most ObservationVLA backends emit a JSON assessment directly (the
8-key payload the assessor consumes). Some Earth-observation models
instead emit a per-pixel ALARM / ANOMALY / SALIENCE HEATMAP — a
2-D float map where higher values mean stronger signal.

This adapter bridges that contract:

  Sentinel tile (3, H, W)  →  your model  →  heatmap (H', W')
                                                  ↓
                                         _extract_stats(heatmap)
                                                  ↓
                            (peak, mean, std, hotspot_density,
                             hotspot_concentration, center_offset)
                                                  ↓
                       _stats_to_payload(stats, sample, probe, geometry)
                                                  ↓
                        ObservationVLA 8-key JSON assessment payload
                        — same schema the rest of the pipeline consumes.

WHY HEATMAP-STATS, NOT "TRAIN A HEAD ON YOUR EMBEDDINGS"

  - If your model already emits a heatmap, you don't need to train and
    ship an additional JSON-output head.
  - Alarm/anomaly heatmaps map SEMANTICALLY to ObservationVLA fields
    (peak intensity ≈ change_or_event_score, hotspot density ≈ scene
    salience, etc.) — the rule is interpretable.
  - No retraining required when you ship updated weights — drop them
    in at the path below and the mapping handles them.

INTEGRATION CONTRACT — TWO STUBS TO IMPLEMENT

  1. _load_model(weights_path, device)
     Load your model and return whatever object your forward pass
     expects.

  2. _predict_heatmap(model, tile, device)
     Take a (3, H, W) float32 [0, 1] tile and return a (H', W')
     float32 numpy array (higher = stronger signal). Output spatial
     dims need not match input.

Everything downstream is concrete and tested:

  _extract_stats(heatmap) → 6-key stats dict
  _stats_to_payload(stats, sample, probe, geometry, ...) → 8-key payload

You can override either if you want different pooling or different
stats→action thresholds, but the defaults are designed to be sane for
satellite-imagery anomaly/change-detection models.

SETUP

  export OBSERVATION_VLA_BACKEND=heatmap
  export HEATMAP_WEIGHTS_PATH=/path/to/your/weights.<format>
  export HEATMAP_DEVICE=cuda
  # Optional thresholds on the action ladder (peak ≥ accept → accept,
  # peak ≥ refine → refine, peak ≥ defer → defer, else → skip):
  # export HEATMAP_PEAK_ACCEPT=0.80
  # export HEATMAP_PEAK_REFINE=0.45
  # export HEATMAP_PEAK_DEFER=0.20

Until your loader is plugged in, the adapter falls back to a stub
payload (action=skip, runtime_mode=stub_fallback) so the service
stays alive during integration. Stub mode logs the underlying load
failure so you can diagnose without grepping.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from ._utils import clamp as _clamp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Contract stubs — implement these two for your model.                  #
# --------------------------------------------------------------------- #

def _load_model(weights_path: Path, device: str) -> Any:
    """STUB — replace with your loader.

    Load your model from `weights_path` onto `device` and return a
    callable / model object that `_predict_heatmap` understands.

    Common patterns:

        # PyTorch checkpoint
        import torch
        model = YourModel(...).to(device).eval()
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return model

        # safetensors / HF
        from transformers import AutoModel
        return AutoModel.from_pretrained(weights_path).to(device).eval()
    """
    raise NotImplementedError(
        "Heatmap loader not implemented. Edit "
        "src/sim/observation_vla/heatmap_local.py:_load_model "
        "to load your model from weights_path."
    )


def _predict_heatmap(model: Any, tile: np.ndarray, device: str) -> np.ndarray:
    """STUB — replace with your forward pass.

    Run `model` on a (3, H, W) float32 [0, 1] RGB tile and return a
    (H', W') float32 numpy alarm/anomaly/salience heatmap (higher =
    stronger signal). Output spatial dims (H', W') can differ from
    input (H, W) — the stats extractor doesn't care.

    Common pattern:

        import torch
        x = torch.from_numpy(tile).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        # If out shape is (1, 1, H', W'):
        return out.squeeze().detach().cpu().numpy().astype(np.float32)
    """
    raise NotImplementedError(
        "Heatmap forward not implemented. Edit "
        "src/sim/observation_vla/heatmap_local.py:_predict_heatmap "
        "to run your model on the tile."
    )


# --------------------------------------------------------------------- #
# Concrete: heatmap → stats → ObservationVLA payload                    #
# --------------------------------------------------------------------- #

def _extract_stats(heatmap: np.ndarray) -> dict[str, float]:
    """Pool a (H, W) heatmap into a small dict of summary stats.

    Stats:
      peak                    max value across the heatmap
      mean                    mean value
      std                     std deviation
      hotspot_density         fraction of pixels with value > 0.5
      hotspot_concentration   density of >0.5 pixels in the top-quartile
                              region (proxy for "is the alarm focused or diffuse?")
      center_offset           L1 distance of the centroid of the top
                              alarm pixels from image center, in [0, 1]
                              (0 = perfectly centered)
    """
    h = np.asarray(heatmap, dtype=np.float32)
    if h.ndim > 2:
        h = h.reshape(h.shape[-2], h.shape[-1])

    flat = h.flatten()
    peak = float(flat.max()) if flat.size else 0.0
    mean = float(flat.mean()) if flat.size else 0.0
    std = float(flat.std()) if flat.size else 0.0

    above_half = (h > 0.5)
    hotspot_density = float(above_half.mean()) if h.size else 0.0

    if h.size > 4:
        threshold = np.quantile(flat, 0.75)
        top_quartile_mask = h >= threshold
        if top_quartile_mask.sum() > 0:
            hotspot_concentration = float(above_half[top_quartile_mask].mean())
        else:
            hotspot_concentration = 0.0
    else:
        hotspot_concentration = 0.0

    # Centroid of top alarm values relative to image center.
    # Use max(90th quantile, half-peak) so a sparse but bright hotspot
    # doesn't get diluted by a quantile of mostly zeros.
    if peak > 0 and h.ndim == 2:
        threshold10 = max(float(np.quantile(flat, 0.90)), 0.5 * peak)
        ys, xs = np.where(h >= threshold10)
        if ys.size > 0:
            cy = ys.mean() / max(1, h.shape[0] - 1)  # in [0, 1]
            cx = xs.mean() / max(1, h.shape[1] - 1)  # in [0, 1]
            center_offset = abs(cy - 0.5) + abs(cx - 0.5)  # L1, [0, 1]
        else:
            center_offset = 0.5
    else:
        center_offset = 0.5

    return {
        "peak": peak,
        "mean": mean,
        "std": std,
        "hotspot_density": hotspot_density,
        "hotspot_concentration": hotspot_concentration,
        "center_offset": float(center_offset),
    }


def _stats_to_payload(
    stats: dict[str, float],
    *,
    sample: dict[str, Any],
    probe: dict[str, Any],
    geometry: dict[str, Any],
    peak_accept: float,
    peak_refine: float,
    peak_defer: float,
    backend_tag: str = "heatmap",
    raw_text_summary: str = "",
) -> dict[str, Any]:
    """Map heatmap stats into the 8-key ObservationVLA assessment payload.

    Mapping rationale:
      usable_observation       True if peak < peak_accept * 1.2 (not
                                  saturated) AND target_visible
      scene_match_score        blend of hotspot_concentration (focused?)
                                  and (1 - center_offset) (centered?)
      salience_score           peak (strongest signal in the tile)
      change_or_event_score    peak (alarm IS change/event)
      occlusion_or_cloud_risk  probe.cloud_cover if present, else
                                  inverse of (peak - mean) margin
      confidence               clamped (peak * concentration scaled)
      recommended_action       threshold ladder on peak
      rationale_tags           [backend_tag, "peak:<v>", ...] + probe tags
    """
    peak = stats["peak"]
    concentration = stats["hotspot_concentration"]
    centered = 1.0 - stats["center_offset"]
    target_visible = bool(geometry.get("target_visible", True))

    scene_match = _clamp(0.5 * concentration + 0.5 * centered)
    salience = _clamp(peak)
    change_or_event = _clamp(peak)
    confidence = _clamp(peak * (0.4 + 0.6 * concentration))

    cloud = probe.get("sentinel_cloud_cover")
    if cloud is not None:
        occlusion = _clamp(float(cloud) / 100.0)
    else:
        diffuse_factor = 1.0 - max(0.0, peak - stats["mean"])
        occlusion = _clamp(diffuse_factor * 0.6)

    if not target_visible:
        action = "skip"
    elif peak >= peak_accept and scene_match >= 0.5:
        action = "accept"
    elif peak >= peak_refine:
        action = "refine"
    elif peak >= peak_defer:
        action = "defer"
    else:
        action = "skip"

    tags = [
        backend_tag,
        f"peak:{peak:.2f}",
        f"hotspot_density:{stats['hotspot_density']:.2f}",
        f"hotspot_concentration:{concentration:.2f}",
        f"center_offset:{stats['center_offset']:.2f}",
    ]
    if cloud is not None and cloud >= 50:
        tags.append("cloud_risk")
    if not target_visible:
        tags.append("target_not_visible")
    if probe.get("sentinel_available") is False:
        tags.append("no_sentinel")

    raw = raw_text_summary or (
        f"{backend_tag}:{action}:peak={peak:.2f}:concentration={concentration:.2f}"
    )

    return {
        "usable_observation": bool(target_visible and peak < (peak_accept * 1.2)),
        "scene_match_score": scene_match,
        "salience_score": salience,
        "change_or_event_score": change_or_event,
        "occlusion_or_cloud_risk": occlusion,
        "confidence": confidence,
        "recommended_action": action,
        "rationale_tags": tags,
        "_model_id": backend_tag,
        "raw_response_text": raw,
        "_heatmap_stats": stats,
    }


# --------------------------------------------------------------------- #
# Adapter — same .assess() interface as the other VLA backends.         #
# --------------------------------------------------------------------- #

class HeatmapAdapter:
    """Heatmap-output backend with the standard ObservationVLA contract."""

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        peak_accept: float | None = None,
        peak_refine: float | None = None,
        peak_defer: float | None = None,
        backend_tag: str | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.weights_path = weights_path or os.environ.get("HEATMAP_WEIGHTS_PATH") or ""
        self.device = (
            device
            or os.environ.get("HEATMAP_DEVICE")
            or os.environ.get("OBSERVATION_VLA_DEVICE")
            or "cpu"
        )
        self.peak_accept = float(peak_accept if peak_accept is not None else os.environ.get("HEATMAP_PEAK_ACCEPT", 0.80))
        self.peak_refine = float(peak_refine if peak_refine is not None else os.environ.get("HEATMAP_PEAK_REFINE", 0.45))
        self.peak_defer = float(peak_defer if peak_defer is not None else os.environ.get("HEATMAP_PEAK_DEFER", 0.20))
        self.backend_tag = backend_tag or os.environ.get("HEATMAP_BACKEND_TAG") or "heatmap"
        self.allow_fallback = allow_fallback

        self._model: Any = None
        self._load_error: str | None = None
        if self.weights_path and Path(self.weights_path).exists():
            try:
                self._model = _load_model(Path(self.weights_path), self.device)
                logger.info("Heatmap model loaded: %s on %s", self.weights_path, self.device)
            except Exception as exc:
                self._load_error = str(exc)
                if not allow_fallback:
                    raise
                logger.warning("Heatmap model load failed — using stub fallback. Reason: %s", exc)
        else:
            self._load_error = f"weights_path not found or empty: {self.weights_path!r}"
            logger.warning("Heatmap adapter: %s — using stub fallback", self._load_error)

    @property
    def runtime_mode(self) -> str:
        return "heatmap_local" if self._model is not None else "stub_fallback"

    @property
    def model_id(self) -> str:
        if self._model is None:
            return f"{self.backend_tag}:stub:fallback"
        return f"{self.backend_tag}:{Path(self.weights_path).name}"

    def _decode_tile(self, images: list[Any]) -> np.ndarray | None:
        """Find the first base64 PNG in `images` and return as (3, H, W) float32 [0, 1]."""
        import base64
        import io
        from PIL import Image

        for item in images:
            if not isinstance(item, dict):
                continue
            b64 = item.get("image_b64")
            if not b64:
                continue
            try:
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                return arr
            except Exception as exc:
                logger.debug("Failed to decode image: %s, trying next", exc)
                continue
        return None

    def _stub_payload(self, sample: dict, probe: dict, geometry: dict) -> dict:
        return _stats_to_payload(
            {"peak": 0.0, "mean": 0.0, "std": 0.0, "hotspot_density": 0.0,
             "hotspot_concentration": 0.0, "center_offset": 0.5},
            sample=sample, probe=probe, geometry=geometry,
            peak_accept=self.peak_accept, peak_refine=self.peak_refine, peak_defer=self.peak_defer,
            backend_tag=self.backend_tag,
            raw_text_summary=f"{self.backend_tag}:stub_fallback:reason={self._load_error}",
        )

    def assess(self, prompt: str, images: list[Any], response_schema: dict[str, Any]) -> dict[str, Any]:
        """Public entry point — same contract as the other VLA adapters."""
        sample, probe, geometry = {}, {}, {}
        for item in images:
            if isinstance(item, dict):
                if "sample" in item:
                    sample = item["sample"] or {}
                if "probe" in item:
                    probe = item["probe"] or {}
                if "geometry" in item:
                    geometry = item["geometry"] or {}

        if self._model is None:
            return self._stub_payload(sample, probe, geometry)

        tile = self._decode_tile(images)
        if tile is None:
            logger.warning("Heatmap adapter: no tile in images; falling back to stub payload")
            return self._stub_payload(sample, probe, geometry)

        try:
            heatmap = _predict_heatmap(self._model, tile, self.device)
            stats = _extract_stats(heatmap)
            return _stats_to_payload(
                stats,
                sample=sample, probe=probe, geometry=geometry,
                peak_accept=self.peak_accept, peak_refine=self.peak_refine, peak_defer=self.peak_defer,
                backend_tag=self.backend_tag,
            )
        except Exception as exc:
            logger.warning("Heatmap inference failed: %s — stub fallback", exc)
            if not self.allow_fallback:
                raise
            return self._stub_payload(sample, probe, geometry)
