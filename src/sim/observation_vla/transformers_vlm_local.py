"""TransformersVLMAdapter — model-agnostic backend for ObservationVLA.

A drop-in replacement for `ObservationVLMAdapter` that loads ANY
transformers-compatible model (vision-language or text-only) and produces
observation triage payloads with the exact same shape as the existing
`clip_local` backend.

Originally shipped as `Gemma4HAICAdapter` pointed at HAIC v35-gov weights;
refactored to be generic so the same scaffold works for:

    Entry A (Liquid Track):     LFM2-VL  / LFM2.5-VL  (fine-tuned on Sentinel)
    Entry B (General AI Track): Gemma-4 fine-tune     (SimSat-specific)
    Testing / baselines:        any public transformers or GGUF model

Three load modes (set via `OBSERVATION_VLM_MODE`):

    - "lora"   : base model (HF id or local path) + LoRA adapter via peft
    - "merged" : pre-merged safetensors (e.g. result of `push_to_hub_merged`)
    - "gguf"   : quantized GGUF via llama-cpp-python (for Orin / on-orbit demo)

Vision-language capability is auto-detected — the adapter probes for a
`processor_config.json` / `AutoProcessor`-compatible model, and falls back
to text-only `AutoTokenizer` if the model doesn't expose one. VL models get
the sample's decoded image passed into the chat template; text-only models
get metadata-only reasoning.

Env vars (generic — no model-family-specific naming):

    OBSERVATION_VLM_MODE          lora | merged | gguf               (default: lora)
    OBSERVATION_VLM_BASE_MODEL    base model path or HF repo id
    OBSERVATION_VLM_LORA_PATH     path to LoRA adapter dir
    OBSERVATION_VLM_MERGED_PATH   path to merged safetensors dir
    OBSERVATION_VLM_GGUF_PATH     path to .gguf file
    OBSERVATION_VLM_WEIGHTS_DIR   convenience: root dir (base/, adapter/, gguf/ layout)
    OBSERVATION_VLM_DEVICE        cpu | cuda | cuda:0                (default: cpu)
    OBSERVATION_VLM_MODEL_LABEL   cosmetic tag in model_id / tags    (optional)

Public interface matches `ObservationVLMAdapter`:

    .assess(prompt, images, response_schema) -> dict   # payload
    .runtime_mode                             -> str
    .model_id                                 -> str

Payload schema matches `_assess_local_clip` so the assessor / residual /
service layers don't need to change.
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

from PIL import Image

logger = logging.getLogger(__name__)


from ._utils import clamp as _clamp


def _weights_root() -> Path:
    return Path(os.environ.get("OBSERVATION_VLM_WEIGHTS_DIR", "./weights/observation-vlm")).expanduser()


class TransformersVLMAdapter:
    """Model-agnostic vision-language (or text-only) backend.

    Loads a transformers model (LoRA+base, merged, or GGUF) and produces
    the same observation-assessment payload as the `clip_local` path.
    """

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

    _SUPPORTED_ACTIONS = {"accept", "defer", "refine", "skip"}

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
        self.mode = (mode or os.environ.get("OBSERVATION_VLM_MODE") or "lora").strip().lower()
        self.base_model = base_model or os.environ.get("OBSERVATION_VLM_BASE_MODEL") or ""
        self.lora_path = lora_path or os.environ.get("OBSERVATION_VLM_LORA_PATH") or str(_weights_root() / "adapter")
        self.merged_path = merged_path or os.environ.get("OBSERVATION_VLM_MERGED_PATH") or str(_weights_root() / "base")
        self.gguf_path = gguf_path or os.environ.get("OBSERVATION_VLM_GGUF_PATH") or str(_weights_root() / "gguf" / "model.gguf")
        self.device = (device or os.environ.get("OBSERVATION_VLM_DEVICE") or "cpu").strip().lower()
        self.allow_fallback = allow_fallback
        self.max_new_tokens = int(os.environ.get("OBSERVATION_VLM_MAX_NEW_TOKENS") or max_new_tokens)
        self.model_label = model_label or os.environ.get("OBSERVATION_VLM_MODEL_LABEL") or ""

        self._torch = None
        self._tokenizer = None
        self._processor = None
        self._model = None
        self._llama = None
        self._is_vision_language = False
        self._load_error: str | None = None

    # ---- Public interface (matches ObservationVLMAdapter) ----

    @property
    def runtime_mode(self) -> str:
        if self._load_error and self.allow_fallback:
            return "stub_fallback"
        return "transformers_vlm_local"

    @property
    def model_id(self) -> str:
        if self._load_error and self.allow_fallback:
            return "transformers-vlm-stub-fallback"
        label = self.model_label or self._inferred_label()
        return f"transformers_vlm:{self.mode}:{label}"

    def assess(
        self,
        prompt: str,
        images: list[Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Main entry point. Shape matches ObservationVLMAdapter.assess()."""
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
            raw_text = self._generate(structured_prompt, image=image)
            payload = self._parse_structured_output(raw_text)
            self._enrich_payload(
                payload,
                probe=probe,
                geometry=geometry,
                raw_text=raw_text,
                structured_prompt=structured_prompt,
                response_schema=response_schema,
                had_image=image is not None,
            )
            return payload
        except Exception as exc:
            logger.warning("TransformersVLMAdapter.assess failed: %s", exc)
            self._load_error = str(exc)
            if not self.allow_fallback:
                raise
            return self._fallback_payload(prompt, probe, geometry, response_schema, error=str(exc))

    # ---- Loading ----

    def _inferred_label(self) -> str:
        # Prefer the source that matches the active mode so the label reflects
        # what was actually loaded (avoids "base" from the merged_path default).
        if self.mode == "lora":
            ordered = (self.lora_path, self.base_model, self.merged_path, self.gguf_path)
        elif self.mode == "merged":
            ordered = (self.merged_path, self.base_model, self.lora_path, self.gguf_path)
        elif self.mode == "gguf":
            ordered = (self.gguf_path, self.base_model, self.merged_path, self.lora_path)
        else:
            ordered = (self.merged_path, self.lora_path, self.base_model, self.gguf_path)
        for source in ordered:
            if not source:
                continue
            tail = Path(source).name
            if tail:
                return tail
        return "unknown"

    def _ensure_loaded(self) -> None:
        if self.mode == "gguf":
            self._ensure_llama_cpp()
        else:
            self._ensure_transformers()

    def _ensure_transformers(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                f"transformers/torch required for mode={self.mode!r}: {exc}"
            ) from exc

        resolved_device = self.device
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = resolved_device
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        if self.mode == "lora":
            if not self.base_model:
                raise RuntimeError("OBSERVATION_VLM_BASE_MODEL is required for mode=lora")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(f"peft required for LoRA mode: {exc}") from exc

            logger.info("Loading base model: %s", self.base_model)
            base = self._load_causal_lm(AutoModelForCausalLM, self.base_model, dtype)

            logger.info("Loading LoRA adapter: %s", self.lora_path)
            self._model = PeftModel.from_pretrained(base, self.lora_path)

            # Tokenizer / processor: prefer the adapter dir if present, else base.
            tokenizer_src = self.lora_path if Path(self.lora_path).exists() else self.base_model
            self._load_tokenizer_or_processor(tokenizer_src)

        elif self.mode == "merged":
            logger.info("Loading merged model: %s", self.merged_path)
            self._model = self._load_causal_lm(AutoModelForCausalLM, self.merged_path, dtype)
            self._load_tokenizer_or_processor(self.merged_path)

        else:
            raise RuntimeError(f"Unsupported transformers mode: {self.mode!r}")

        self._model.eval()
        self._torch = torch

    def _load_causal_lm(self, AutoClass, src: str, dtype):
        """Load a causal LM with device_map set to the configured device.

        On CUDA, uses 4-bit NF4 quantization by default (set
        OBSERVATION_VLM_LOAD_IN_4BIT=0 to disable). This keeps Gemma-4-E2B
        (~2B LM + vision + audio towers) under 2 GB on the GPU rather than
        ~6 GB in float16, which exceeds an 8 GB card during inference.
        """
        import os
        _use_4bit = (
            self.device != "cpu"
            and os.environ.get("OBSERVATION_VLM_LOAD_IN_4BIT", "1") != "0"
        )
        _bnb_cfg = None
        if _use_4bit:
            try:
                import torch as _torch
                from transformers import BitsAndBytesConfig
                _bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=_torch.float16,
                )
                logger.info("_load_causal_lm: 4-bit NF4 quantization enabled")
            except Exception as _e:
                logger.warning("_load_causal_lm: 4-bit unavailable (%s), falling back to float16", _e)

        _kwargs: dict = {"device_map": {"": self.device}}
        if _bnb_cfg is not None:
            _kwargs["quantization_config"] = _bnb_cfg
        else:
            _kwargs["torch_dtype"] = dtype

        return AutoClass.from_pretrained(src, **_kwargs)

    def _load_tokenizer_or_processor(self, src: str) -> None:
        """Try AutoProcessor (vision-language) first, fall back to AutoTokenizer (text-only)."""
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(src)
            self._tokenizer = getattr(self._processor, "tokenizer", None)
            if self._tokenizer is None:
                # Some processors don't expose .tokenizer; load separately.
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(src)
            # Heuristic: if the processor has an image_processor attribute, treat as VL.
            self._is_vision_language = hasattr(self._processor, "image_processor")
            logger.info(
                "Loaded processor from %s (vision_language=%s)",
                src,
                self._is_vision_language,
            )
            return
        except Exception as exc:
            logger.debug("AutoProcessor load failed (%s) — falling back to AutoTokenizer", exc)

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        self._is_vision_language = False

    def _ensure_llama_cpp(self) -> None:
        if self._llama is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(f"llama-cpp-python required for GGUF mode: {exc}") from exc
        if not Path(self.gguf_path).exists():
            raise RuntimeError(
                f"GGUF weights not found at {self.gguf_path}. "
                f"Set OBSERVATION_VLM_GGUF_PATH or download the model first."
            )
        logger.info("Loading GGUF: %s", self.gguf_path)
        self._llama = Llama(
            model_path=self.gguf_path,
            n_ctx=4096,
            n_gpu_layers=-1 if self.device.startswith("cuda") else 0,
            verbose=False,
        )
        self._is_vision_language = False  # llama.cpp VL support varies; conservative default.

    # ---- Prompt construction ----

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
        """Build the triage prompt. Image conditioning is handled in `_generate`."""
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
            "You are an Earth-observation triage model. Assess whether this encounter "
            "window is useful for the mission target.\n\n"
            "<<CONTEXT>>\n"
            + "\n".join(ctx_lines)
            + "\n<<END CONTEXT>>\n\n"
            + f"Triage question: {user_prompt}\n\n"
            + schema_hint
        )

    # ---- Generation ----

    def _generate(self, prompt: str, *, image: Image.Image | None = None) -> str:
        self._ensure_loaded()
        if self.mode == "gguf":
            return self._generate_llama_cpp(prompt)
        if self._is_vision_language and image is not None and self._processor is not None:
            return self._generate_vlm(prompt, image)
        return self._generate_text(prompt)

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        """Prefer processor's chat template (VL), fall back to tokenizer's."""
        applier = None
        if self._processor is not None and hasattr(self._processor, "apply_chat_template"):
            applier = self._processor
        elif self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            applier = self._tokenizer
        if applier is None:
            raise RuntimeError("No chat template available on processor or tokenizer")
        return applier.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_text(self, prompt: str) -> str:
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = self._apply_chat_template(messages)
        except Exception as exc:
            logger.warning("apply_chat_template (text-only) failed, falling back to raw prompt: %s", exc)
            chat_prompt = prompt

        inputs = self._tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        print(f"[VLM] text-only — prompt_tokens={prompt_len} device={self.device} max_new_tokens={self.max_new_tokens}", flush=True)
        print(f"[VLM] generating...", flush=True)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
            )
        # Slice on token IDs (not string-startswith) — skip_special_tokens=True
        # strips <|turn> et al from decoded output but NOT from chat_prompt,
        # so the previous string-startswith check failed and the function
        # returned the entire prompt+response. The parser then saw the user
        # prompt itself instead of the model's reply.
        new_tokens = out[0, prompt_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_vlm(self, prompt: str, image: Image.Image) -> str:
        """Vision-language path — interleaves the image into the chat template."""
        assert self._model is not None and self._processor is not None and self._torch is not None

        # Modern VLMs accept a structured "content" list with image + text.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            chat_prompt = self._apply_chat_template(messages)
        except Exception as exc:
            # Fallback: some VLMs (older) want text only and consume image separately.
            logger.warning("apply_chat_template (multimodal) failed, falling back to text-only: %s", exc)
            chat_prompt = prompt

        # Resize before processor — large Sentinel tiles generate thousands of image
        # tokens and cause the processor call to hang indefinitely.
        _max_img = int(os.environ.get("OBSERVATION_VLM_MAX_IMAGE_SIZE") or 448)
        if max(image.size) > _max_img:
            print(f"[VLM] resizing image {image.size} → {_max_img}px", flush=True)
            image = image.resize((_max_img, _max_img), Image.LANCZOS)

        inputs = self._processor(
            text=chat_prompt,
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else None
        n_image_tokens = inputs.get("pixel_values", None)
        print(f"[VLM] inputs ready — prompt_tokens={prompt_len} image_tensors={'yes' if n_image_tokens is not None else 'no'} device={self.device} max_new_tokens={self.max_new_tokens}", flush=True)
        print(f"[VLM] generating...", flush=True)

        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
            )
        print(f"[VLM] generation complete — output_tokens={out.shape[-1]}", flush=True)
        decoder = self._tokenizer or self._processor
        # See _generate_text — slice on token IDs, not string-startswith.
        if prompt_len is not None:
            new_tokens = out[0, prompt_len:]
            return decoder.decode(new_tokens, skip_special_tokens=True)
        # Fallback if processor didn't expose input_ids (older API).
        full_text = decoder.decode(out[0], skip_special_tokens=True)
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

    # ---- Parsing ----

    def _parse_structured_output(self, raw_text: str) -> dict[str, Any]:
        """Extract the JSON object from the model output, tolerating prose around it."""
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
            raise ValueError(f"Model output missing required keys: {missing}")

        payload["usable_observation"] = bool(payload["usable_observation"])
        for k in (
            "scene_match_score",
            "salience_score",
            "change_or_event_score",
            "occlusion_or_cloud_risk",
            "confidence",
        ):
            payload[k] = _clamp(float(payload[k]))
        payload["recommended_action"] = str(payload["recommended_action"]).strip().lower()
        if payload["recommended_action"] not in self._SUPPORTED_ACTIONS:
            payload["recommended_action"] = "refine"
        payload["rationale_tags"] = [str(t) for t in payload.get("rationale_tags", [])]
        return payload

    # ---- Post-processing ----

    def _enrich_payload(
        self,
        payload: dict[str, Any],
        *,
        probe: dict[str, Any],
        geometry: dict[str, Any],
        raw_text: str,
        structured_prompt: str,
        response_schema: dict[str, Any],
        had_image: bool,
    ) -> None:
        tags = list(payload.get("rationale_tags", []))
        for required_tag in (
            "transformers_vlm_local",
            f"mode:{self.mode}",
        ):
            if required_tag not in tags:
                tags.append(required_tag)
        if self._is_vision_language and had_image:
            tags.append("image_conditioned")
        elif had_image:
            tags.append("image_available_text_only_model")
        else:
            tags.append("metadata_only")
        if probe.get("sentinel_available"):
            tags.append("sentinel_support")
        payload["rationale_tags"] = list(dict.fromkeys(tags))

        payload["raw_response_text"] = (
            f"transformers_vlm_local:{payload['recommended_action']}:"
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
            "rationale_tags": [
                "transformers_vlm_local_fallback",
                f"error:{error[:60]}",
            ],
            "raw_response_text": f"transformers_vlm_fallback:{action}:{confidence:.2f}|error={error[:200]}",
            "_prompt_used": prompt,
            "_schema_used": response_schema,
            "_model_id": self.model_id,
        }

    # ---- Image helpers ----

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
            except Exception as exc:
                logger.debug("Failed to decode image (trying next candidate): %s", exc)
                continue
        return None
