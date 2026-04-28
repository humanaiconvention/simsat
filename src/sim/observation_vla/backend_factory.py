"""VLA backend factory — select the ObservationVLA backend via env var.

Usage in service init:
    from observation_vla.backend_factory import build_vla_adapter
    adapter = build_vla_adapter()

OBSERVATION_VLA_BACKEND values:
    clip_local          ObservationVLMAdapter  (default — CLIP-based baseline)
    gemma4              TransformersVLMAdapter  (SimSat Gemma-4-E2B fine-tune)
    genesis             GenesisAdapter          (Guilherme's Genesis model)
    tesseract_t3        TesseractT3Adapter      (Garrett's Tesseract T3 model)
    transformers_vlm    TransformersVLMAdapter  (generic — use OBSERVATION_VLM_* vars)
    gemma4_haic         Gemma4HAICAdapter       (HAIC v35-gov, legacy reference)
    stub                ObservationVLMAdapter   (stub-only, no model load)
"""
from __future__ import annotations

import os
from typing import Any


def build_vla_adapter(backend: str | None = None) -> Any:
    """Return the appropriate VLA adapter for the configured backend.

    Reads OBSERVATION_VLA_BACKEND if backend is not explicitly provided.
    Every returned adapter implements the same interface:
        .assess(prompt, images, response_schema) -> dict
        .runtime_mode -> str
        .model_id -> str
    """
    key = (backend or os.environ.get("OBSERVATION_VLA_BACKEND") or "clip_local").strip().lower()

    if key in {"clip_local", "clip", "auto"}:
        from .adapter import ObservationVLMAdapter
        return ObservationVLMAdapter()

    if key == "stub":
        from .adapter import ObservationVLMAdapter
        return ObservationVLMAdapter(backend="stub")

    if key in {"gemma4", "simsat_gemma4", "gemma4_simsat"}:
        # SimSat Gemma-4-E2B fine-tune (benhaslam/simsat-gemma4-v1).
        # Uses TransformersVLMAdapter with OBSERVATION_VLM_* env vars;
        # defaults model_label to "gemma4-simsat" for model_id clarity.
        from .transformers_vlm_local import TransformersVLMAdapter
        return TransformersVLMAdapter(
            model_label=os.environ.get("OBSERVATION_VLM_MODEL_LABEL") or "gemma4-simsat",
        )

    if key in {"transformers_vlm", "transformers_vlm_local"}:
        from .transformers_vlm_local import TransformersVLMAdapter
        return TransformersVLMAdapter()

    if key in {"gemma4_haic", "gemma4_haic_local"}:
        # HAIC v35-gov Gemma-4 (convention/governance fine-tune — kept for reference).
        from .gemma4_haic_local import Gemma4HAICAdapter
        return Gemma4HAICAdapter()

    if key in {"genesis", "genesis_local"}:
        from .genesis_local import GenesisAdapter
        return GenesisAdapter()

    if key in {"tesseract_t3", "tesseract-t3", "tesseract_t3_local", "tesseract"}:
        from .tesseract_t3_local import TesseractT3Adapter
        return TesseractT3Adapter()

    if key in {"heatmap", "heatmap_local"}:
        # Generic insertion seat for any model that emits a per-pixel
        # heatmap rather than a JSON assessment. See
        # heatmap_local.py module docstring for the integration contract
        # (two stubs to implement: _load_model + _predict_heatmap).
        from .heatmap_local import HeatmapAdapter
        return HeatmapAdapter()

    if key in {"heatmap", "heatmap_local"}:
        # Generic insertion seat for any model that emits a per-pixel
        # heatmap rather than a JSON assessment. See
        # heatmap_local.py module docstring for the integration contract
        # (two stubs to implement: _load_model + _predict_heatmap).
        from .heatmap_local import HeatmapAdapter
        return HeatmapAdapter()

    raise ValueError(
        f"Unknown OBSERVATION_VLA_BACKEND: {key!r}. "
        "Valid values: clip_local, gemma4, genesis, tesseract_t3, "
        "heatmap, transformers_vlm, gemma4_haic, stub"
    )
