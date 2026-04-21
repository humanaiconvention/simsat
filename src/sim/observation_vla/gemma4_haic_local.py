"""Gemma-4 + HAIC LoRA backend for ObservationVLA.

This is the Entry B backend — a drop-in replacement for the `clip_local`
path in `adapter.ObservationVLMAdapter`. It loads a LoRA-adapted Gemma-4
fine-tuned on the HAIC v35-gov convention-grounded training corpus and
uses it to produce structured observation assessments that match the
same payload schema as the CLIP backend (so the assessor / residual /
service layers don't need to change).

Three load modes, selected via `HAIC_GEMMA4_MODE`:
    - "lora"   (default) — transformers + peft, base model + LoRA adapter
    - "merged"           — transformers, pre-merged full safetensors model
    - "gguf"             — llama-cpp-python, quantized GGUF (for Orin on-orbit)

Weights are expected under `./weights/haic-v35-gov/` by default, in the
layout produced by `scripts/download_haic_v35_gov.sh`. Override via
`HAIC_GEMMA4_BASE_MODEL`, `HAIC_GEMMA4_LORA_PATH`, `HAIC_GEMMA4_MERGED_PATH`,
`HAIC_GEMMA4_GGUF_PATH`.

This adapter exposes the same public interface as `ObservationVLMAdapter`
(`.assess(prompt, images, response_schema)`, `.runtime_mode`, `.model_id`)
so the service layer can swap adapters via env var selection:

    OBSERVATION_VLA_BACKEND=gemma4_haic_local  -> Gemma4HAICAdapter
    OBSERVATION_VLA_BACKEND=clip_local         -> ObservationVLMAdapter (existing)

See `README_ENTRY_B.md` in this folder for integration details.

SCAFFOLD NOTICE:
    This file is a scaffold shipped to the `entry-b/backend` feature branch
    for local testing. It is NOT yet wired into the production service
    layer. Ben should:
      1. Run `scripts/download_haic_v35_gov.sh` to populate weights/
      2. Set OBSERVATION_VLA_BACKEND=gemma4_haic_local in docker-compose.yaml
      3. Add the factory dispatch described in README_ENTRY_B.md
      4. Run `haic_test.py` and `observation_vla_eval.py --inprocess`
         against the new backend to verify payload compatibility.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _weights_base_dir() -> Path:
    return Path(os.environ.get("HAIC_GEMMA4_WEIGHTS_DIR", "./weights/haic-v35-gov")).expanduser()


class Gemma4HAICAdapter:
    """Gemma-4 + HAIC LoRA adapter with the same interface as ObservationVLMAdapter."""

    # Keys the downstream parser expects — MUST match the clip_local payload shape.
    REQUIRED_KEYS = (
        "usable_observation",
        "scene_match_score",
        "salience_score",
        "change_or_event_score",
        "occlusion_or_cloud_risk",
        "confidence",
        "recommended_action",
        "rationale_tags",
    )

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
    ) -> None:
        self.mode = (mode or os.environ.get("HAIC_GEMMA4_MODE") or "lora").strip().lower()
        self.base_model = base_model or os.environ.get("HAIC_GEMMA4_BASE_MODEL") or "google/gemma-2-2b-it"
        self.lora_path = lora_path or os.environ.get("HAIC_GEMMA4_LORA_PATH") or str(_weights_base_dir() / "adapter")
        self.merged_path = merged_path or os.environ.get("HAIC_GEMMA4_MERGED_PATH") or str(_weights_base_dir() / "base")
        self.gguf_path = gguf_path or os.environ.get("HAIC_GEMMA4_GGUF_PATH") or str(_weights_base_dir() / "gguf" / "1.F16.gguf")
        self.device = (device or os.environ.get("OBSERVATION_VLA_DEVICE") or "cpu").strip().lower()
        self.allow_fallback = allow_fallback
        self.max_new_tokens = max_new_tokens

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._llama = None
        self._load_error: str | None = None

    # ---- Public interface: matches ObservationVLMAdapter -----------------

    @property
    def model_id(self) -> str:
        if self._load_error and self.allow_fallback:
            return "gemma4-haic-stub-fallback"
        return f"gemma4-haic-v35-gov:{self.mode}"

    @property
    def runtime_mode(self) -> str:
        if self._load_error and self.allow_fallback:
            return "stub_fallback"
        return "gemma4_haic_local"

    def assess(
        self,
        prompt: str,
        images: list[Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Main entry point. Runs Gemma-4 + HAIC inference, parses JSON, returns payload."""
        sample, probe, geometry = self._extract_context(images)
        image = self._decode_image(images)

        try:
            structured_prompt = self._build_structured_prompt(
                user_prompt=prompt,
                sample=sample,
                probe=probe,
                geometry=geometry,
                has_image=image is not None,
                response_schema=response_schema,
            )
            raw_text = self._generate(structured_prompt)
            payload = self._parse_structured_output(raw_text)
            self._enrich_payload(payload, probe=probe, geometry=geometry, raw_text=raw_text, structured_prompt=structured_prompt, response_schema=response_schema)
            return payload
        except Exception as exc:
            logger.warning("Gemma4HAICAdapter assess failed: %s", exc)
            self._load_error = str(exc)
            if not self.allow_fallback:
                raise
            return self._fallback_payload(prompt, probe, geometry, response_schema, error=str(exc))

    # ---- Loading --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self.mode == "gguf":
            self._ensure_llama_cpp()
        else:
            self._ensure_transformers()

    def _ensure_transformers(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(f"transformers/torch required for Gemma4HAICAdapter mode={self.mode!r}: {exc}") from exc

        resolved_device = self.device
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = resolved_device

        if self.mode == "lora":
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(f"peft required for LoRA mode: {exc}") from exc
            logger.info("Loading Gemma-4 base: %s", self.base_model)
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map={"": self.device},
            )
            tokenizer_src = self.base_model
            if Path(self.lora_path).exists() and (Path(self.lora_path) / "tokenizer.json").exists():
                tokenizer_src = self.lora_path
            logger.info("Loading HAIC LoRA adapter: %s", self.lora_path)
            self._model = PeftModel.from_pretrained(base, self.lora_path)
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
        elif self.mode == "merged":
            logger.info("Loading HAIC merged model: %s", self.merged_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.merged_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map={"": self.device},
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.merged_path)
        else:
            raise RuntimeError(f"Unsupported transformers-mode: {self.mode!r}")

        self._model.eval()
        self._torch = torch

    def _ensure_llama_cpp(self) -> None:
        if self._llama is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(f"llama-cpp-python required for GGUF mode: {exc}") from exc
        if not Path(self.gguf_path).exists():
            raise RuntimeError(f"GGUF weights not found at {self.gguf_path}. Run scripts/download_haic_v35_gov.sh first.")
        logger.info("Loading HAIC GGUF: %s", self.gguf_path)
        self._llama = Llama(
            model_path=self.gguf_path,
            n_ctx=4096,
            n_gpu_layers=-1 if self.device.startswith("cuda") else 0,
            verbose=False,
        )

    # ---- Prompt construction --------------------------------------------

    def _build_structured_prompt(
        self,
        *,
        user_prompt: str,
        sample: dict[str, Any],
        probe: dict[str, Any],
        geometry: dict[str, Any],
        has_image: bool,
        response_schema: dict[str, Any],
    ) -> str:
        """Build a HAIC-convention-shaped prompt for v35-gov."""
        target_label = sample.get("target_label") or sample.get("target_id") or "unknown target"
        scenario_pack = sample.get("scenario_pack", "all")
        tags = sample.get("target_tags", [])

        cloud = probe.get("sentinel_cloud_cover")
        sentinel_src = probe.get("sentinel_source")
        sentinel_dt = probe.get("sentinel_datetime")
        sentinel_ok = probe.get("sentinel_available")

        geo_vis = geometry.get("target_visible")
        geo_el = geometry.get("elevation_degrees")
        geo_off = geometry.get("off_nadir_degrees")

        # HAIC v35-gov was fine-tuned on stimulus/interview/viability prompts, so we
        # mirror that shape here. The convention is: provide probe + geometry + target
        # as stimulus metadata, and ask the model to emit a triage JSON.
        ctx_lines = [
            f"Target: {target_label}",
            f"Scenario pack: {scenario_pack}",
        ]
        if tags:
            ctx_lines.append(f"Target tags: {', '.join(str(t) for t in tags[:6])}")
        if sentinel_ok is not None:
            ctx_lines.append(f"Sentinel available: {sentinel_ok}")
        if sentinel_src:
            ctx_lines.append(f"Sentinel source: {sentinel_src}")
        if sentinel_dt:
            ctx_lines.append(f"Sentinel datetime: {sentinel_dt}")
        if cloud is not None:
            ctx_lines.append(f"Cloud cover: {cloud:.2f}%")
        if geo_vis is not None:
            ctx_lines.append(f"Target visible: {geo_vis}")
        if geo_el is not None:
            ctx_lines.append(f"Elevation deg: {geo_el:.2f}")
        if geo_off is not None:
            ctx_lines.append(f"Off-nadir deg: {geo_off:.2f}")
        ctx_lines.append(f"Image attached: {has_image}")

        schema_hint = (
            "Respond ONLY with a single JSON object matching this schema. "
            "No commentary, no markdown fences.\n"
            "{\n"
            '  "usable_observation": <bool>,\n'
            '  "scene_match_score": <float 0-1>,\n'
            '  "salience_score": <float 0-1>,\n'
            '  "change_or_event_score": <float 0-1>,\n'
            '  "occlusion_or_cloud_risk": <float 0-1>,\n'
            '  "confidence": <float 0-1>,\n'
            '  "recommended_action": "accept"|"defer"|"refine"|"skip",\n'
            '  "rationale_tags": [<string>, ...]\n'
            "}"
        )

        return (
            "You are HAIC v35-gov, a convention-grounded Earth-observation triage model. "
            "Assess whether this encounter window is useful for the mission target.\n\n"
            "<<STIMULUS>>\n"
            + "\n".join(ctx_lines)
            + "\n<<END STIMULUS>>\n\n"
            + f"Triage question: {user_prompt}\n\n"
            + schema_hint
        )

    # ---- Generation -----------------------------------------------------

    def _generate(self, prompt: str) -> str:
        self._ensure_loaded()
        if self.mode == "gguf":
            return self._generate_llama_cpp(prompt)
        return self._generate_transformers(prompt)

    def _generate_transformers(self, prompt: str) -> str:
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_prompt = prompt

        inputs = self._tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        full_text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        return full_text[len(chat_prompt):] if full_text.startswith(chat_prompt) else full_text

    def _generate_llama_cpp(self, prompt: str) -> str:
        assert self._llama is not None
        response = self._llama(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            stop=["</s>", "<|end|>", "<|eot_id|>"],
            echo=False,
        )
        return response["choices"][0]["text"]

    # ---- Parsing --------------------------------------------------------

    def _parse_structured_output(self, raw_text: str) -> dict[str, Any]:
        """Extract the JSON object from the model output, tolerating prose around it."""
        # Find the first {...} that looks like the target JSON.
        stripped = raw_text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        json_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON object found in model output: {raw_text[:200]!r}")
        payload = json.loads(json_match.group(0))

        missing = [k for k in self.REQUIRED_KEYS if k not in payload]
        if missing:
            raise ValueError(f"Gemma4 output missing required keys: {missing}")

        # Coerce types and clamp numeric scores
        payload["usable_observation"] = bool(payload["usable_observation"])
        for k in ("scene_match_score", "salience_score", "change_or_event_score", "occlusion_or_cloud_risk", "confidence"):
            payload[k] = _clamp(float(payload[k]))
        payload["recommended_action"] = str(payload["recommended_action"]).strip().lower()
        if payload["recommended_action"] not in {"accept", "defer", "refine", "skip"}:
            payload["recommended_action"] = "refine"
        payload["rationale_tags"] = [str(t) for t in payload.get("rationale_tags", [])]
        return payload

    # ---- Post-processing ------------------------------------------------

    def _enrich_payload(
        self,
        payload: dict[str, Any],
        *,
        probe: dict[str, Any],
        geometry: dict[str, Any],
        raw_text: str,
        structured_prompt: str,
        response_schema: dict[str, Any],
    ) -> None:
        # Ensure rationale_tags include the runtime-backend tag
        tags = list(payload.get("rationale_tags", []))
        for required_tag in ("gemma4_haic_local", f"haic_mode:{self.mode}"):
            if required_tag not in tags:
                tags.append(required_tag)
        if probe.get("sentinel_available"):
            tags.append("sentinel_support")
        payload["rationale_tags"] = list(dict.fromkeys(tags))

        # Convention-required payload metadata (matches what the assessor consumes)
        payload["raw_response_text"] = (
            f"gemma4_haic_local:{payload['recommended_action']}:"
            f"conf={payload['confidence']:.2f}|"
            + raw_text.replace("\n", " ")[:600]
        )
        payload["_prompt_used"] = structured_prompt
        payload["_schema_used"] = response_schema
        payload["_model_id"] = self.model_id

    def _fallback_payload(
        self,
        prompt: str,
        probe: dict[str, Any],
        geometry: dict[str, Any],
        response_schema: dict[str, Any],
        error: str,
    ) -> dict[str, Any]:
        """Metadata-only fallback — matches the clip_local stub shape when the model fails."""
        cloud = probe.get("sentinel_cloud_cover")
        cloud_fraction = _clamp(float(cloud) / 100.0) if cloud is not None else 0.35
        sentinel_ok = bool(probe.get("sentinel_available"))
        geo_vis = bool(geometry.get("target_visible", False))
        scene_match = 0.75 if sentinel_ok else (0.50 if geo_vis else 0.40)
        salience = 0.65 if sentinel_ok else (0.45 if geo_vis else 0.35)
        confidence = _clamp((scene_match + salience + (1.0 - cloud_fraction)) / 3.0)
        usable = confidence >= 0.55 and cloud_fraction <= 0.7
        action = "defer" if usable and salience < 0.55 else ("refine" if usable else "skip")

        return {
            "usable_observation": usable,
            "scene_match_score": scene_match,
            "salience_score": salience,
            "change_or_event_score": 0.5,
            "occlusion_or_cloud_risk": cloud_fraction,
            "confidence": confidence,
            "recommended_action": action,
            "rationale_tags": ["gemma4_haic_local_fallback", f"error:{error[:60]}"],
            "raw_response_text": f"gemma4_haic_fallback:{action}:{confidence:.2f}|error={error[:200]}",
            "_prompt_used": prompt,
            "_schema_used": response_schema,
            "_model_id": self.model_id,
        }

    # ---- Image helpers (kept optional for scaffold; reused from adapter.py style) -------

    def _extract_context(self, images: list[Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        sample: dict[str, Any] = {}
        probe: dict[str, Any] = {}
        geometry: dict[str, Any] = {}
        for item in images:
            if not isinstance(item, dict):
                continue
            sample = dict(item.get("sample", sample))
            probe = dict(item.get("probe", probe))
            geometry = dict(item.get("geometry", geometry))
            if sample:
                break
        return sample, probe, geometry

    def _decode_image(self, images: list[Any]) -> Image.Image | None:
        for item in images:
            if not isinstance(item, dict):
                continue
            image_b64 = item.get("image_b64")
            if not image_b64:
                continue
            try:
                image_bytes = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_bytes))
                return image.convert("RGB")
            except Exception:
                continue
        return None
