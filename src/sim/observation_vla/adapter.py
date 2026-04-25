from __future__ import annotations

import base64
import importlib.util
import io
import os
from typing import Any

import numpy as np
import requests
from PIL import Image


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_cosine(similarity: float) -> float:
    return _clamp((similarity + 1.0) / 2.0)


def _scenario_positive_prompts(sample: dict[str, Any]) -> list[str]:
    target_label = str(sample.get("target_label") or sample.get("target_id") or "the target location")
    target_tags = [str(tag).replace("_", " ") for tag in sample.get("target_tags", [])]
    target_metadata = dict(sample.get("target_metadata", {}))
    scenario_pack = str(sample.get("scenario_pack", "all"))
    scenario_role = str(target_metadata.get("scenario_role", "")).replace("_", " ")
    challenge_note = str(target_metadata.get("challenge_note", "")).strip()

    prompts = [
        f"a clear satellite image of {target_label}",
        f"a detailed overhead satellite image of {target_label}",
        f"an Earth observation image that matches {target_label}",
    ]
    if target_tags:
        prompts.append(f"a satellite image of {' and '.join(target_tags[:3])}")
    if scenario_role:
        prompts.append(f"a satellite image of a {scenario_role}")
    if challenge_note:
        prompts.append(f"a satellite image matching this note: {challenge_note}")

    if scenario_pack == "maritime_chokepoints":
        prompts.extend(
            [
                "a satellite image of a canal, shipping lane, or maritime chokepoint",
                "a satellite image of water with visible canal or port infrastructure",
            ]
        )
    elif scenario_pack == "disaster_response_weather":
        prompts.extend(
            [
                "a satellite image of an industrial coastal corridor",
                "a satellite image of a storm-sensitive coastline, delta, or floodplain",
            ]
        )
    elif scenario_pack == "urban_coastal_ambiguity":
        prompts.extend(
            [
                "a satellite image of a dense urban port and shoreline",
                "a satellite image of a mixed-use coastal industrial area",
            ]
        )
    return list(dict.fromkeys(prompts))


def _scenario_negative_prompts(sample: dict[str, Any]) -> list[str]:
    target_label = str(sample.get("target_label") or sample.get("target_id") or "the target")
    return [
        "a cloud-covered satellite image with little visible detail",
        "a featureless open ocean satellite image",
        "a blurry or low-detail satellite image",
        f"a satellite image that does not match {target_label}",
    ]


def _event_prompts(sample: dict[str, Any]) -> list[str]:
    scenario_pack = str(sample.get("scenario_pack", "all"))
    if scenario_pack == "maritime_chokepoints":
        positives = [
            "a satellite image showing ships, traffic, or canal activity",
            "a satellite image showing a strategic maritime transit point",
        ]
    elif scenario_pack == "disaster_response_weather":
        positives = [
            "a satellite image showing flooding, storm impact, or coastal disruption",
            "a satellite image showing weather-related change or damage",
        ]
    elif scenario_pack == "urban_coastal_ambiguity":
        positives = [
            "a satellite image showing dense port structures and shoreline clutter",
            "a satellite image where multiple coastal structures compete for attention",
        ]
    else:
        positives = [
            "a satellite image showing a notable event or change",
        ]
    return positives + ["a routine satellite image without a notable event or change"]


class ObservationVLMAdapter:
    def __init__(
        self,
        model_name_or_path: str | None = None,
        endpoint_url: str | None = None,
        request_timeout: float = 45.0,
        allow_fallback: bool = True,
        backend: str | None = None,
        device: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url or os.environ.get("OBSERVATION_VLA_ENDPOINT")
        self.request_timeout = request_timeout
        self.allow_fallback = allow_fallback
        self.backend = (backend or os.environ.get("OBSERVATION_VLA_BACKEND") or "auto").strip().lower()
        self.model_name_or_path = (
            model_name_or_path
            or os.environ.get("OBSERVATION_VLA_MODEL")
            or "openai/clip-vit-base-patch32"
        )
        self.device = (device or os.environ.get("OBSERVATION_VLA_DEVICE") or "cpu").strip().lower()
        self._processor = None
        self._model = None
        self._torch = None
        self._load_error: str | None = None
        # VLA-layer TTT: learnable confidence blend weights (sum to 1.0)
        self._score_weights: dict[str, float] = {
            "scene_match": 0.35,
            "salience": 0.20,
            "change_or_event": 0.15,
            "clarity": 0.20,
            "pos_neg_margin": 0.10,
        }
        self._ttt_update_count: int = 0
        self._ttt_log: list[dict] = []

    @property
    def model_id(self) -> str:
        if self.runtime_mode == "http_endpoint":
            return self.model_name_or_path or "observation-vla-endpoint"
        if self.runtime_mode == "clip_local":
            return f"clip_local:{self.model_name_or_path}"
        return "observation-vla-stub-v1"

    @property
    def runtime_mode(self) -> str:
        if self._load_error and self.allow_fallback:
            return "stub_fallback"
        if self.endpoint_url and self.backend in {"auto", "http_endpoint"}:
            return "http_endpoint"
        if self.backend == "stub":
            return "stub"
        if self._local_backend_available():
            return "clip_local"
        return "stub"

    def _local_backend_available(self) -> bool:
        return all(
            importlib.util.find_spec(module_name) is not None
            for module_name in ("torch", "transformers", "PIL")
        )

    def _ensure_local_model(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return
        if not self._local_backend_available():
            raise RuntimeError("Local CLIP backend dependencies are not installed")

        import torch
        from transformers import CLIPModel, CLIPProcessor

        resolved_device = self.device
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = resolved_device

        self._processor = CLIPProcessor.from_pretrained(self.model_name_or_path)
        self._model = CLIPModel.from_pretrained(self.model_name_or_path)
        self._model.eval()
        self._model.to(self.device)
        self._torch = torch
        self._load_error = None

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

    def _image_detail_score(self, image: Image.Image) -> float:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        contrast = float(rgb.std())
        edge_h = float(np.abs(np.diff(grayscale, axis=0)).mean()) if grayscale.shape[0] > 1 else 0.0
        edge_w = float(np.abs(np.diff(grayscale, axis=1)).mean()) if grayscale.shape[1] > 1 else 0.0
        edge = (edge_h + edge_w) / 2.0
        return _clamp((contrast * 1.8) + (edge * 4.5))

    def _clip_similarities(self, image: Image.Image, texts: list[str]) -> list[float]:
        self._ensure_local_model()
        if self._processor is None or self._model is None or self._torch is None:
            raise RuntimeError("Local CLIP backend did not initialise correctly")

        inputs = self._processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T)[0].detach().cpu().tolist()
        return [float(similarity) for similarity in similarities]

    def _assess_stub(
        self,
        prompt: str,
        images: list[Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        probe = {}
        geometry = {}
        if images:
            first = images[0]
            if isinstance(first, dict):
                probe = dict(first.get("probe", {}))
                geometry = dict(first.get("geometry", {}))

        cloud_cover = probe.get("sentinel_cloud_cover")
        cloud_fraction = _clamp(float(cloud_cover) / 100.0) if cloud_cover is not None else 0.35
        sentinel_available = bool(probe.get("sentinel_available"))
        geometry_visible = bool(geometry.get("target_visible", probe.get("mapbox_feasible", False)))
        scene_match_score = 0.82 if sentinel_available else (0.54 if geometry_visible else 0.42)
        salience_score = 0.75 if sentinel_available else (0.55 if geometry_visible else 0.4)
        change_or_event_score = 0.5
        occlusion_or_cloud_risk = cloud_fraction if sentinel_available else (0.50 if geometry_visible else 0.6)
        confidence = _clamp((scene_match_score + salience_score + (1.0 - occlusion_or_cloud_risk)) / 3.0)
        usable_observation = confidence >= 0.55 and occlusion_or_cloud_risk <= 0.7

        if usable_observation and confidence >= 0.75 and scene_match_score >= 0.75:
            recommended_action = "accept"
        elif usable_observation and salience_score < 0.55:
            recommended_action = "defer"
        elif usable_observation:
            recommended_action = "refine"
        else:
            recommended_action = "skip"

        rationale_tags: list[str] = []
        if sentinel_available:
            rationale_tags.append("sentinel_support")
        elif geometry_visible:
            rationale_tags.append("geometry_only_support")
        if cloud_cover is not None and float(cloud_cover) > 40.0:
            rationale_tags.append("cloud_risk")
        if not usable_observation:
            rationale_tags.append("low_visual_utility")
        if recommended_action == "refine":
            rationale_tags.append("needs_visual_review")

        return {
            "usable_observation": usable_observation,
            "scene_match_score": scene_match_score,
            "salience_score": salience_score,
            "change_or_event_score": change_or_event_score,
            "occlusion_or_cloud_risk": occlusion_or_cloud_risk,
            "confidence": confidence,
            "recommended_action": recommended_action,
            "rationale_tags": rationale_tags,
            "raw_response_text": f"stub:{recommended_action}:{confidence:.2f}",
            "_prompt_used": prompt,
            "_schema_used": response_schema,
            "_model_id": "observation-vla-stub-v1",
        }

    def _assess_local_clip(
        self,
        prompt: str,
        images: list[Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        self._load_error = None
        image = self._decode_image(images)
        if image is None:
            raise RuntimeError("No renderable image payload available for local CLIP assessment")

        sample, probe, geometry = self._extract_context(images)
        cloud_cover = probe.get("sentinel_cloud_cover")
        metadata_cloud_fraction = _clamp(float(cloud_cover) / 100.0) if cloud_cover is not None else 0.35

        positive_prompts = _scenario_positive_prompts(sample)
        negative_prompts = _scenario_negative_prompts(sample)
        event_prompts = _event_prompts(sample)
        cloud_prompts = [
            "a cloud-covered satellite image",
            "a hazy or partially obscured satellite image",
            "a clear satellite image with visible ground detail",
        ]

        positive_sims = self._clip_similarities(image, positive_prompts)
        negative_sims = self._clip_similarities(image, negative_prompts)
        event_sims = self._clip_similarities(image, event_prompts)
        cloud_sims = self._clip_similarities(image, cloud_prompts)

        positive_best = max(_normalize_cosine(sim) for sim in positive_sims)
        negative_best = max(_normalize_cosine(sim) for sim in negative_sims)
        event_best = max(_normalize_cosine(sim) for sim in event_sims[:-1]) if len(event_sims) > 1 else 0.5
        event_routine = _normalize_cosine(event_sims[-1]) if event_sims else 0.5
        cloud_like = max(_normalize_cosine(sim) for sim in cloud_sims[:2])
        clear_like = _normalize_cosine(cloud_sims[-1]) if cloud_sims else 0.5
        detail_score = self._image_detail_score(image)

        scene_match_score = _clamp((0.70 * positive_best) + (0.30 * (1.0 - negative_best)))
        salience_score = _clamp((0.55 * detail_score) + (0.45 * positive_best))
        change_or_event_score = _clamp((0.60 * event_best) + (0.40 * (1.0 - event_routine)))
        clip_cloud_risk = _clamp((0.60 * cloud_like) + (0.40 * (1.0 - clear_like)))
        occlusion_or_cloud_risk = _clamp((0.75 * metadata_cloud_fraction) + (0.25 * clip_cloud_risk))
        w = self._score_weights
        pos_neg_margin = _clamp((positive_best - negative_best + 1.0) / 2.0)
        confidence = _clamp(
            w["scene_match"] * scene_match_score
            + w["salience"] * salience_score
            + w["change_or_event"] * change_or_event_score
            + w["clarity"] * (1.0 - occlusion_or_cloud_risk)
            + w["pos_neg_margin"] * pos_neg_margin
        )
        action_score = _clamp(
            (0.38 * scene_match_score)
            + (0.18 * salience_score)
            + (0.12 * change_or_event_score)
            + (0.18 * (1.0 - occlusion_or_cloud_risk))
            + (0.14 * confidence)
        )

        geometry_visible = bool(geometry.get("target_visible", probe.get("mapbox_feasible", False)))
        usable_observation = (
            geometry_visible
            and scene_match_score >= 0.56
            and salience_score >= 0.32
            and occlusion_or_cloud_risk <= 0.72
            and confidence >= 0.50
        )

        if (
            usable_observation
            and confidence >= 0.58
            and scene_match_score >= 0.56
            and action_score >= 0.60
            and occlusion_or_cloud_risk <= 0.38
        ):
            recommended_action = "accept"
        elif not usable_observation and confidence >= 0.60:
            recommended_action = "skip"
        elif usable_observation and (salience_score < 0.42 or confidence < 0.54):
            recommended_action = "defer"
        else:
            recommended_action = "refine"

        rationale_tags: list[str] = ["image_model_backed", "clip_local"]
        if positive_prompts:
            best_prompt = positive_prompts[max(range(len(positive_sims)), key=lambda idx: positive_sims[idx])]
            if sample.get("target_label") and sample.get("target_label") in best_prompt:
                rationale_tags.append("target_match")
        if scene_match_score >= 0.7:
            rationale_tags.append("high_scene_match")
        if action_score >= 0.60:
            rationale_tags.append("high_action_score")
        if occlusion_or_cloud_risk >= 0.45:
            rationale_tags.append("cloud_risk")
        if detail_score < 0.28:
            rationale_tags.append("low_detail")
        if recommended_action == "accept":
            rationale_tags.append("conservative_accept")
        if recommended_action == "refine":
            rationale_tags.append("needs_visual_review")
        if probe.get("sentinel_available"):
            rationale_tags.append("sentinel_support")

        return {
            "usable_observation": usable_observation,
            "scene_match_score": scene_match_score,
            "salience_score": salience_score,
            "change_or_event_score": change_or_event_score,
            "occlusion_or_cloud_risk": occlusion_or_cloud_risk,
            "confidence": confidence,
            "recommended_action": recommended_action,
            "rationale_tags": list(dict.fromkeys(rationale_tags)),
            "raw_response_text": (
                f"clip_local:{recommended_action}:scene={scene_match_score:.2f}:"
                f"salience={salience_score:.2f}:event={change_or_event_score:.2f}:"
                f"risk={occlusion_or_cloud_risk:.2f}:confidence={confidence:.2f}:"
                f"action_score={action_score:.2f}"
            ),
            "_prompt_used": prompt,
            "_schema_used": response_schema,
            "_model_id": f"clip_local:{self.model_name_or_path}",
        }

    def _assess_endpoint(
        self,
        prompt: str,
        images: list[Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.endpoint_url:
            raise RuntimeError("Observation VLA endpoint not configured")
        response = requests.post(
            self.endpoint_url,
            json={
                "model": self.model_name_or_path,
                "prompt": prompt,
                "images": images,
                "response_schema": response_schema,
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Observation VLA endpoint returned a non-dict payload")
        payload.setdefault("_model_id", self.model_name_or_path or "observation-vla-endpoint")
        return payload

    def vla_online_update(
        self,
        evidence: dict[str, float],
        confidence: float,
        realized_utility: float,
        lr: float = 0.015,
    ) -> dict[str, float]:
        """VLA-layer TTT: gradient step on confidence blend weights.

        Uses operator usefulness_score as the supervision signal and the stored
        intermediate evidence scores as features, reducing confidence calibration
        error (the MAE 0.27 gap between model confidence and operator scores).
        Both the update and the renormalisation step mirror trust_model.online_update().
        """
        feature_map = {
            "scene_match": float(evidence.get("scene_match_score", 0.5)),
            "salience": float(evidence.get("salience_score", 0.5)),
            "change_or_event": float(evidence.get("change_or_event_score", 0.5)),
            "clarity": 1.0 - float(evidence.get("occlusion_or_cloud_risk", 0.5)),
            "pos_neg_margin": 0.5,  # not stored in evidence; use neutral prior
        }
        error = realized_utility - confidence
        for k in list(self._score_weights):
            self._score_weights[k] += lr * error * feature_map[k]
            self._score_weights[k] = max(0.001, self._score_weights[k])
        total = sum(self._score_weights.values())
        self._score_weights = {k: v / total for k, v in self._score_weights.items()}
        self._ttt_update_count += 1
        self._ttt_log.append(
            {
                "update": self._ttt_update_count,
                "confidence": confidence,
                "realized_utility": realized_utility,
                "error": error,
            }
        )
        if len(self._ttt_log) > 500:
            self._ttt_log = self._ttt_log[-500:]
        return dict(self._score_weights)

    def get_ttt_snapshot(self) -> dict:
        """Return current VLA TTT state: weights, drift, recent updates."""
        default_weights = {
            "scene_match": 0.35,
            "salience": 0.20,
            "change_or_event": 0.15,
            "clarity": 0.20,
            "pos_neg_margin": 0.10,
        }
        drift = {k: round(self._score_weights[k] - default_weights[k], 6) for k in default_weights}
        return {
            "runtime_mode": self.runtime_mode,
            "update_count": self._ttt_update_count,
            "current_weights": dict(self._score_weights),
            "default_weights": default_weights,
            "drift": drift,
            "recent_updates": self._ttt_log[-10:],
        }

    def assess(
        self,
        prompt: str,
        images: list[Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        if self.runtime_mode == "http_endpoint":
            try:
                return self._assess_endpoint(prompt, images, response_schema)
            except Exception:
                if not self.allow_fallback:
                    raise

        if self.runtime_mode == "clip_local":
            try:
                return self._assess_local_clip(prompt, images, response_schema)
            except Exception as exc:
                self._load_error = str(exc)
                if not self.allow_fallback:
                    raise

        payload = self._assess_stub(prompt, images, response_schema)
        if self._load_error and payload.get("raw_response_text"):
            payload["raw_response_text"] = f"{payload['raw_response_text']}|fallback={self._load_error}"
        return payload
