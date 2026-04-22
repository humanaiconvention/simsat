from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from encounter.schemas import EncounterRecord, TargetSpec
from haic.schemas import GroundingStimulus

from .adapter import ObservationVLMAdapter
from .prompts import build_triage_prompt, expected_response_schema
from .schemas import (
    ObservationAssessment,
    ObservationEvidence,
    ObservationImageRef,
    ObservationSample,
)


class ObservationAssessor:
    def __init__(self, adapter: ObservationVLMAdapter, model_id: str = "observation-vla-stub-v1") -> None:
        self.adapter = adapter
        self.model_id = model_id

    def build_sample(
        self,
        record: EncounterRecord,
        target: TargetSpec,
        stimulus: GroundingStimulus | None,
        scenario_pack: str = "all",
    ) -> ObservationSample:
        images: list[ObservationImageRef] = []
        if stimulus is not None:
            for image in stimulus.images:
                metadata = image.metadata.to_dict() if image.metadata is not None else {}
                images.append(
                    ObservationImageRef(
                        source=metadata.get("source", image.stimulus_type.value),
                        image_type=image.stimulus_type.value,
                        timestamp=metadata.get("timestamp") or metadata.get("datetime"),
                        metadata=metadata,
                    )
                )

        return ObservationSample(
            decision_id=record.decision.decision_id,
            window_id=record.window.window_id,
            target_id=target.target_id,
            scenario_pack=scenario_pack,
            target_label=target.label,
            target_tags=list(target.tags),
            target_metadata=dict(target.metadata),
            geometry=record.window.geometry.to_dict(),
            probe=record.probe.to_dict(),
            images=images,
        )

    def extract_renderable_images(
        self,
        stimulus: GroundingStimulus | None,
    ) -> list[Any]:
        if stimulus is None:
            return []
        rendered: list[Any] = []
        for image in stimulus.images:
            rendered.append(
                {
                    "image_b64": image.image_b64,
                    "metadata": image.metadata.to_dict() if image.metadata is not None else {},
                }
            )
        return rendered

    def parse_assessment_payload(
        self,
        payload: dict[str, Any],
        sample_id: str,
        assessment_mode: str,
        image_count: int,
        runtime_mode: str,
        model_id: str,
    ) -> ObservationAssessment:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        evidence = ObservationEvidence(
            usable_observation=bool(payload.get("usable_observation", False)),
            scene_match_score=float(payload.get("scene_match_score", 0.0)),
            salience_score=float(payload.get("salience_score", 0.0)),
            change_or_event_score=float(payload.get("change_or_event_score", 0.0)),
            occlusion_or_cloud_risk=float(payload.get("occlusion_or_cloud_risk", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            rationale_tags=[str(v) for v in payload.get("rationale_tags", [])],
        )
        return ObservationAssessment(
            sample_id=sample_id,
            model_id=model_id,
            created_at=created_at,
            assessment_mode=assessment_mode,
            runtime_mode=runtime_mode,
            image_count=image_count,
            recommended_action=str(payload.get("recommended_action", "refine")),
            evidence=evidence,
            raw_response_text=str(payload["raw_response_text"]) if payload.get("raw_response_text") is not None else None,
        )

    def assess(
        self,
        sample: ObservationSample,
        rendered_images: list[Any] | None = None,
    ) -> ObservationAssessment:
        prompt = build_triage_prompt(sample)
        schema = expected_response_schema()
        image_payloads = list(rendered_images or [])
        if not image_payloads:
            image_payloads = [{"probe": sample.probe, "geometry": sample.geometry, "sample": sample.to_dict()}]
        elif image_payloads and isinstance(image_payloads[0], dict):
            image_payloads[0].setdefault("probe", sample.probe)
            image_payloads[0].setdefault("geometry", sample.geometry)
            image_payloads[0].setdefault("sample", sample.to_dict())
        has_image_conditioning = any(
            isinstance(item, dict)
            and (
                bool(item.get("image_b64"))
                or bool(item.get("metadata", {}).get("image_available"))
            )
            for item in image_payloads
        )
        assessment_mode = "image_conditioned" if has_image_conditioning else "metadata_only"
        payload = self.adapter.assess(prompt, image_payloads, schema)
        model_id = str(payload.get("_model_id", self.model_id))
        return self.parse_assessment_payload(
            payload,
            sample.sample_id,
            assessment_mode=assessment_mode,
            image_count=len(image_payloads),
            runtime_mode=self.adapter.runtime_mode,
            model_id=model_id,
        )
