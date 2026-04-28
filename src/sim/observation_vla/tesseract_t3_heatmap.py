"""Tesseract T³ heatmap-output backend for ObservationVLA.

Garrett's Tesseract T³ model (2026-04-27 status: 89M params, DOFA backbone
transfer, ~40 ms inference on his RTX) outputs a per-pixel ALARM HEATMAP
(H, W) over each Sentinel tile rather than a JSON assessment. This adapter
bridges that contract:

  Sentinel tile (3, H, W)  →  Tesseract T³ encoder  →  alarm heatmap (H', W')
                                                         ↓
                                         _extract_stats(heatmap)
                                                         ↓
                            (peak, mean, std, hotspot_count, spatial moments)
                                                         ↓
                       _stats_to_payload(stats, sample, probe, geometry)
                                                         ↓
                        ObservationVLA 8-key JSON assessment payload
                        — same schema the rest of the pipeline consumes.

Why heatmap-stats and not "train a head on T³ embeddings":

  - Garrett's model already emits the heatmap; no extra training step.
  - Alarm/anomaly heatmaps map SEMANTICALLY to the ObservationVLA
    fields: peak intensity ≈ change_or_event_score, hotspot count in
    target area ≈ scene_match_score, etc.
  - Rule-based stats→fields is interpretable; the operator can read
    `rationale_tags` and trace each numeric back to a specific stat.
  - No retraining required when Garrett ships an updated T³ — just drop
    in the new weights at the path below.

INTEGRATION CONTRACT — what Garrett needs to plug in
----------------------------------------------------

This module ships as a stub. Three functions need real implementations:

  1. _load_tesseract_t3(weights_path, device) — load Garrett's actual
     model into a callable object. Returns whatever object Garrett's
     forward path expects.

  2. _predict_heatmap(model, tile) — call the model on a (3, H, W)
     float32 [0, 1] tile and return a (H', W') float32 numpy array
     where higher values = stronger alarm/anomaly signal.

  3. (optional) _extract_stats and _stats_to_payload — the defaults
     here are reasonable; override only if Garrett wants different
     pooling or a different stats→action rule set.

Setup (env vars):

  OBSERVATION_VLA_BACKEND=tesseract_t3_heatmap
  TESSERACT_T3_WEIGHTS_PATH=/path/to/tesseract-t3-weights.<format>
  TESSERACT_T3_DEVICE=cuda
  # Optional tuning of the stats→action thresholds:
  TESSERACT_T3_PEAK_ACCEPT=0.80     # peak >= this AND target match → accept
  TESSERACT_T3_PEAK_REFINE=0.45     # peak in [refine, accept) → refine
  TESSERACT_T3_PEAK_DEFER=0.20      # peak in [defer, refine) → defer
                                     # peak < defer → skip

Falls back to a stub payload (action=defer, confidence=0.10, rationale_tag
"tesseract_t3_heatmap_stub") when the weights aren't present, so the
service stays alive during integration.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from ._utils import clamp as _clamp

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- #
# Contract stubs — Garrett: replace these two with real loader + predict. #
# ----------------------------------------------------------------------- #

def _load_tesseract_t3(weights_path: Path, device: str) -> Any:
    """Stub: load the Tesseract T³ model from `weights_path` onto `device`.

    Replace with the actual loader. Should return a callable / model object
    that `_predict_heatmap` understands. Allowed to raise any exception
    on failure — the adapter catches and falls back to stub mode.

    Garrett: this is the entry point. Drop your loader here. Common
    patterns:

        # PyTorch checkpoint
        import torch
        model = TesseractT3(...).to(device).eval()
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return model

        # safetensors / HF
        from transformers import AutoModel
        return AutoModel.from_pretrained(weights_path).to(device).eval()
    """
    raise NotImplementedError(
        "Tesseract T³ loader not implemented. Garrett: edit "
        "src/sim/observation_vla/tesseract_t3_heatmap.py:_load_tesseract_t3 "
        "to load your model from weights_path."
    )


def _predict_heatmap(model: Any, tile: np.ndarray, device: str) -> np.ndarray:
    """Stub: run the model on a (3, H, W) float32 [0, 1] tile and return
    a (H', W') float32 numpy alarm heatmap (higher = stronger signal).

    Replace with the actual forward path. Output spatial dims (H', W')
    can differ from input (H, W) — the stats extractor doesn't care.

    Garrett: this is the inference call. Drop your forward path here.
    Common pattern:

        import torch
        x = torch.from_numpy(tile).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        # Assuming out shape is (1, 1, H', W'):
        heatmap = out.squeeze().detach().cpu().numpy().astype(np.float32)
        return heatmap
    """
    raise NotImplementedError(
        "Tesseract T³ forward not implemented. Garrett: edit "
        "src/sim/observation_vla/tesseract_t3_heatmap.py:_predict_heatmap "
        "to run your model on the tile."
    )


# ----------------------------------------------------------------------- #
# Concrete: heatmap → stats → ObservationVLA payload                      #
# ----------------------------------------------------------------------- #

def _extract_stats(heatmap: np.ndarray) -> dict[str, float]:
    """Pool a (H, W) alarm heatmap into a small dict of summary stats.

    Stats:
      peak                 max alarm value across the heatmap
      mean                 mean alarm value
      std                  std deviation of alarm values
      hotspot_density      fraction of pixels with value > 0.5
      hotspot_concentration  hotspot_density of the top-quartile region
                              (proxy for "the alarm is concentrated, not diffuse")
      center_offset        L1 distance of the centroid of the top 10% of
                              alarm values from image center, normalized
                              to [0, 1] (0 = perfectly centered)
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

    # Top-quartile concentration: density of >0.5 pixels in the top-quartile
    # region (i.e. "is the alarm clustered or scattered?"). Higher = more
    # focused alarm.
    if h.size > 4:
        threshold = np.quantile(flat, 0.75)
        top_quartile_mask = h >= threshold
        if top_quartile_mask.sum() > 0:
            hotspot_concentration = float(above_half[top_quartile_mask].mean())
        else:
            hotspot_concentration = 0.0
    else:
        hotspot_concentration = 0.0

    # Centroid of top-10% alarm values relative to image center.
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
    raw_text_summary: str = "",
) -> dict[str, Any]:
    """Map heatmap stats into the 8-key ObservationVLA assessment payload.

    Mapping rationale:
      usable_observation       True if peak < peak_accept * 1.2 (not overwhelmed
                                  by alarm — "we can actually see something here")
                                  AND not_visible_target → False
      scene_match_score        hotspot_concentration (focused alarm = clear target)
                                  blended with (1 - center_offset) (centered = matched)
      salience_score           peak (the strongest signal in the tile)
      change_or_event_score    peak (alarm IS change/event detection)
      occlusion_or_cloud_risk  derived from probe metadata if available, else
                                  inverse of (mean - peak) margin
      confidence               clamped (peak * hotspot_concentration)
      recommended_action       threshold ladder on peak (see env vars)
      rationale_tags           ["tesseract_t3_heatmap", "peak:<v>",
                                  "hotspots:<n>", + probe tags]
    """
    peak = stats["peak"]
    concentration = stats["hotspot_concentration"]
    centered = 1.0 - stats["center_offset"]
    target_visible = bool(geometry.get("target_visible", True))

    # scene_match: combo of "alarm is concentrated" and "alarm is centered on target"
    scene_match = _clamp(0.5 * concentration + 0.5 * centered)
    salience = _clamp(peak)
    change_or_event = _clamp(peak)
    confidence = _clamp(peak * (0.4 + 0.6 * concentration))

    # Occlusion: prefer probe metadata; fall back to "low alarm everywhere"
    cloud = probe.get("sentinel_cloud_cover")
    if cloud is not None:
        occlusion = _clamp(float(cloud) / 100.0)
    else:
        # If mean is high relative to peak, signal is diffuse → likely occluded
        diffuse_factor = 1.0 - max(0.0, peak - stats["mean"])
        occlusion = _clamp(diffuse_factor * 0.6)

    # Action ladder
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
        "tesseract_t3_heatmap",
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
        f"tesseract_t3_heatmap:{action}:peak={peak:.2f}:concentration={concentration:.2f}"
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
        "_model_id": "tesseract_t3_heatmap",
        "raw_response_text": raw,
        "_heatmap_stats": stats,
    }


# ----------------------------------------------------------------------- #
# Adapter — same interface as ObservationVLMAdapter / TransformersVLMAdapter
# ----------------------------------------------------------------------- #

class TesseractT3HeatmapAdapter:
    """Heatmap-output backend with the same .assess(prompt, images, schema) contract."""

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        peak_accept: float | None = None,
        peak_refine: float | None = None,
        peak_defer: float | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.weights_path = weights_path or os.environ.get("TESSERACT_T3_WEIGHTS_PATH") or ""
        self.device = device or os.environ.get("TESSERACT_T3_DEVICE") or os.environ.get("OBSERVATION_VLA_DEVICE") or "cpu"
        self.peak_accept = float(peak_accept if peak_accept is not None else os.environ.get("TESSERACT_T3_PEAK_ACCEPT", 0.80))
        self.peak_refine = float(peak_refine if peak_refine is not None else os.environ.get("TESSERACT_T3_PEAK_REFINE", 0.45))
        self.peak_defer = float(peak_defer if peak_defer is not None else os.environ.get("TESSERACT_T3_PEAK_DEFER", 0.20))
        self.allow_fallback = allow_fallback

        self._model = None
        self._load_error: str | None = None
        if self.weights_path and Path(self.weights_path).exists():
            try:
                self._model = _load_tesseract_t3(Path(self.weights_path), self.device)
                logger.info("Tesseract T³ heatmap model loaded: %s on %s", self.weights_path, self.device)
            except Exception as exc:
                self._load_error = str(exc)
                if not allow_fallback:
                    raise
                logger.warning("Tesseract T³ heatmap load failed — using stub fallback. Reason: %s", exc)
        else:
            self._load_error = f"weights_path not found or empty: {self.weights_path!r}"
            logger.warning("Tesseract T³ heatmap: %s — using stub fallback", self._load_error)

    @property
    def runtime_mode(self) -> str:
        return "tesseract_t3_heatmap" if self._model is not None else "stub_fallback"

    @property
    def model_id(self) -> str:
        if self._model is None:
            return "tesseract_t3_heatmap:stub:fallback"
        return f"tesseract_t3_heatmap:{Path(self.weights_path).name}"

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
        """Used when the model isn't loaded yet. Conservative defer + low confidence."""
        return _stats_to_payload(
            {"peak": 0.0, "mean": 0.0, "std": 0.0, "hotspot_density": 0.0,
             "hotspot_concentration": 0.0, "center_offset": 0.5},
            sample=sample, probe=probe, geometry=geometry,
            peak_accept=self.peak_accept, peak_refine=self.peak_refine, peak_defer=self.peak_defer,
            raw_text_summary=f"tesseract_t3_heatmap:stub_fallback:reason={self._load_error}",
        )

    def assess(self, prompt: str, images: list[Any], response_schema: dict[str, Any]) -> dict[str, Any]:
        """Public entry point — same contract as the other VLA adapters."""
        # Pull context out of the images blob (the bridge stuffs sample/probe/geometry there)
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
            logger.warning("Tesseract T³ heatmap: no tile in images; falling back to stub payload")
            return self._stub_payload(sample, probe, geometry)

        try:
            heatmap = _predict_heatmap(self._model, tile, self.device)
            stats = _extract_stats(heatmap)
            return _stats_to_payload(
                stats,
                sample=sample, probe=probe, geometry=geometry,
                peak_accept=self.peak_accept, peak_refine=self.peak_refine, peak_defer=self.peak_defer,
            )
        except Exception as exc:
            logger.warning("Tesseract T³ heatmap inference failed: %s — stub fallback", exc)
            if not self.allow_fallback:
                raise
            return self._stub_payload(sample, probe, geometry)
