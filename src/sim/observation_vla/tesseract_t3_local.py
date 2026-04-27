"""Tesseract T3 VLA backend — General AI Track collaborator seat (Garrett Sutherland).

Wraps TransformersVLMAdapter and points it at the Tesseract T3 model.
Enable with: OBSERVATION_VLA_BACKEND=tesseract_t3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLLABORATOR SETUP — Garrett Sutherland (Tesseract T3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Point to your model weights:

    # Option A: HuggingFace repo (merged weights, no LoRA)
    TESSERACT_T3_MODE=merged
    TESSERACT_T3_MERGED_PATH=your-hf-username/tesseract-t3    # or local /path/to/merged/

    # Option B: Base + LoRA adapter
    TESSERACT_T3_MODE=lora
    TESSERACT_T3_BASE_MODEL=your-hf-username/tesseract-t3-base
    TESSERACT_T3_LORA_PATH=/path/to/tesseract-t3-simsat-adapter/

    # Option C: Quantized GGUF (best for Orin on-orbit demo)
    TESSERACT_T3_MODE=gguf
    TESSERACT_T3_GGUF_PATH=/path/to/tesseract-t3.gguf

Step 2 — Set the backend selector:
    OBSERVATION_VLA_BACKEND=tesseract_t3
    OBSERVATION_VLA_DEVICE=cuda   # or cpu

Step 3 — Verify compatibility:
    python scripts/observation_vla_eval.py --inprocess
    # You should see: runtime_mode=tesseract_t3_local in the output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your model's response (before JSON parsing) must contain a single JSON object:

{
  "usable_observation": true,              // bool
  "scene_match_score": 0.82,              // float 0-1: does the tile match the target?
  "salience_score": 0.75,                 // float 0-1: how much detail / signal?
  "change_or_event_score": 0.60,          // float 0-1: notable change / event?
  "occlusion_or_cloud_risk": 0.15,        // float 0-1: cloud / occlusion fraction
  "confidence": 0.78,                     // float 0-1: overall assessment confidence
  "recommended_action": "accept",         // "accept" | "defer" | "refine" | "skip"
  "rationale_tags": ["tesseract_t3_local", "sentinel_support"]
}

The triage prompt you receive describes the target, scenario pack, Sentinel
metadata, geometry, and asks you to emit this JSON. No markdown fences needed.

Full prompt format and worked examples: see COLLABORATOR_GUIDE.md at repo root.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your model sits in the VLA lane of a governed on-orbit continual-learning loop:

  Sentinel tile → Tesseract T3 (you) → ObservationVLA JSON
    → WCLI trust layer → viability gates (6 non-compensatory checks)
    → accept / defer / refine / skip decision
    → mission-response layer (logged utility)
    → TTT feedback loop (trust-layer weights updated from realized utility)

The six viability gates will reject any adaptation that:
  1. Does not reduce entropy (PRISM consistency check)
  2. Looks like bulk data extraction
  3. Has inconsistent claimed vs. measured metadata
  4. Lacks a human-in-the-loop stimulus
  5. Sends raw imagery off-satellite (only derived updates leave)
  6. Reinforces prior bias without adding new information

Your model's output feeds into gate 1 (confidence must track actual utility)
and gate 6 (epistemic alignment). The TTT loop will tune the trust layer's
thresholds and priors over successive passes via online_update() calls
driven by the realized-utility signal from the mission-response layer.

See CHALLENGE_ENTRY.md § "Two-Track Submission" and COLLABORATOR_GUIDE.md
for the full submission framing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _tesseract_weights_root() -> Path:
    return Path(os.environ.get("TESSERACT_T3_WEIGHTS_DIR", "./weights/tesseract-t3")).expanduser()


class TesseractT3Adapter:
    """Tesseract T3 adapter for ObservationVLA.

    Delegates to TransformersVLMAdapter with Tesseract-T3-specific env vars and
    patches the returned payload to carry tesseract_t3_local branding in
    runtime_mode, model_id, and rationale_tags.
    """

    def __init__(
        self,
        base_model: str | None = None,
        lora_path: str | None = None,
        merged_path: str | None = None,
        gguf_path: str | None = None,
        mode: str | None = None,
        device: str | None = None,
        allow_fallback: bool = True,
        max_new_tokens: int = 256,
        model_label: str | None = None,
    ) -> None:
        root = _tesseract_weights_root()
        resolved_mode = (mode or os.environ.get("TESSERACT_T3_MODE") or "lora").strip().lower()
        resolved_base = base_model or os.environ.get("TESSERACT_T3_BASE_MODEL") or ""
        resolved_lora = lora_path or os.environ.get("TESSERACT_T3_LORA_PATH") or str(root / "adapter")
        resolved_merged = merged_path or os.environ.get("TESSERACT_T3_MERGED_PATH") or str(root / "base")
        resolved_gguf = gguf_path or os.environ.get("TESSERACT_T3_GGUF_PATH") or str(root / "gguf" / "tesseract-t3.gguf")
        resolved_device = device or os.environ.get("OBSERVATION_VLA_DEVICE") or "cpu"
        resolved_label = model_label or os.environ.get("TESSERACT_T3_MODEL_LABEL") or "tesseract-t3"

        from .transformers_vlm_local import TransformersVLMAdapter
        self._delegate = TransformersVLMAdapter(
            base_model=resolved_base,
            lora_path=resolved_lora,
            merged_path=resolved_merged,
            gguf_path=resolved_gguf,
            mode=resolved_mode,
            device=resolved_device,
            allow_fallback=allow_fallback,
            max_new_tokens=max_new_tokens,
            model_label=resolved_label,
        )

    @property
    def runtime_mode(self) -> str:
        delegate_mode = self._delegate.runtime_mode
        if delegate_mode == "stub_fallback":
            return "stub_fallback"
        return "tesseract_t3_local"

    @property
    def model_id(self) -> str:
        delegate_id = self._delegate.model_id
        # "transformers_vlm:{mode}:{label}" → "tesseract_t3:{mode}:{label}"
        return delegate_id.replace("transformers_vlm:", "tesseract_t3:", 1)

    def assess(
        self,
        prompt: str,
        images: list[Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self._delegate.assess(prompt, images, response_schema)
        # Patch branding so the service layer sees tesseract_t3_local identity.
        payload["_model_id"] = self.model_id
        tags: list[str] = list(payload.get("rationale_tags", []))
        if "tesseract_t3_local" not in tags:
            tags = ["tesseract_t3_local"] + [t for t in tags if t != "transformers_vlm_local"]
            payload["rationale_tags"] = list(dict.fromkeys(tags))
        # Rewrite delegate-leaked OBSERVATION_VLM_* env-var names in error tags
        # so users see the TESSERACT_T3_* equivalent they actually configure.
        payload["rationale_tags"] = [
            t.replace("OBSERVATION_VLM_", "TESSERACT_T3_") for t in payload["rationale_tags"]
        ]
        raw = payload.get("raw_response_text", "")
        if raw.startswith("transformers_vlm_local:"):
            payload["raw_response_text"] = "tesseract_t3_local:" + raw[len("transformers_vlm_local:"):]
        return payload
