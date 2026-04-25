from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ImagingProviders.mapbox_provider import MapboxlProvider
from ImagingProviders.sentinel_provider import SentinelProvider
from encounter.ephemeris import EphemerisService
from encounter.features import FeatureBuilder
from encounter.materialize import StimulusMaterializer
from encounter.planner import AnalyticPlanner
from encounter.policy import build_default_policy
from encounter.probes import ImagingProbeService
from encounter.service import EncounterService
from encounter.store import get_encounter_store
from encounter.targets import get_target_repository
from encounter.trust_model import WCLITrustModel
from encounter.windows import WindowDetector
from haic.bridge import StimulusBridge
from haic.stimulus_store import get_stimulus_store
from mission_response.planner import MissionResponsePlanner
from mission_response.policy import build_default_policy as build_mission_response_policy
from mission_response.service import MissionResponseService
from mission_response.store import MissionResponseStore
from mission_response.utility import MissionUtilityModel
from observation_vla.assessor import ObservationAssessor
from observation_vla.backend_factory import build_vla_adapter
from observation_vla.dataset import ObservationDatasetBuilder
from observation_vla.memory import ObservationTTTMemory
from observation_vla.residual import ObservationResidualBuilder
from observation_vla.service import ObservationVLAService
from observation_vla.store import ObservationStore
from orbit_config import DEFAULT_TLE, SATELLITE_NAME


@dataclass
class RuntimeBundle:
    shared_data: dict[str, Any]
    sentinel_provider: SentinelProvider
    mapbox_provider: MapboxlProvider | None
    encounter_service: EncounterService
    observation_vla_service: ObservationVLAService
    mission_response_service: MissionResponseService


def build_runtime_capabilities(bundle: RuntimeBundle) -> dict[str, Any]:
    runtime_mode = bundle.observation_vla_service.assessor.adapter.runtime_mode
    model_id = bundle.observation_vla_service.assessor.adapter.model_id
    return {
        "sentinel_enabled": bundle.sentinel_provider is not None,
        "mapbox_enabled": bundle.mapbox_provider is not None,
        "challenge_no_mapbox_safe": True,
        "sentinel_first_challenge_scoring": True,
        "high_res_perspective_optional": True,
        "observation_vla_runtime_mode": runtime_mode,
        "observation_vla_model_id": model_id,
    }


def build_runtime_bundle(shared_data: Mapping[str, Any] | None = None) -> RuntimeBundle:
    import logging as _logging

    shared = dict(shared_data or {})
    sentinel = SentinelProvider()
    try:
        mapbox = MapboxlProvider()
    except ValueError:
        _logging.getLogger(__name__).warning(
            "MAPBOX_ACCESS_TOKEN not set — Mapbox imagery disabled"
        )
        mapbox = None

    bridge = StimulusBridge(
        shared_data=shared,
        sentinel_provider=sentinel,
        mapbox_provider=mapbox,
    )
    observation_adapter = build_vla_adapter()
    observation_store = ObservationStore()
    observation_service = ObservationVLAService(
        assessor=ObservationAssessor(adapter=observation_adapter, model_id=observation_adapter.model_id),
        residual_builder=ObservationResidualBuilder(),
        memory=ObservationTTTMemory(observation_store),
        dataset_builder=ObservationDatasetBuilder(),
        store=observation_store,
    )
    mission_policy = build_mission_response_policy()
    mission_service = MissionResponseService(
        planner=MissionResponsePlanner(mission_policy),
        utility_model=MissionUtilityModel(),
        store=MissionResponseStore(),
        policy=mission_policy,
        observation_vla=observation_service,
    )
    encounter_policy = build_default_policy()
    encounter_service = EncounterService(
        shared_data=shared,
        target_repo=get_target_repository(),
        ephemeris=EphemerisService(SATELLITE_NAME, DEFAULT_TLE),
        window_detector=WindowDetector(
            min_elevation_degrees=encounter_policy.min_elevation_degrees,
            min_window_seconds=encounter_policy.min_window_seconds,
        ),
        probe_service=ImagingProbeService(
            sentinel_provider=sentinel,
            mapbox_provider=mapbox,
        ),
        feature_builder=FeatureBuilder(),
        planner=AnalyticPlanner(encounter_policy, trust_model=WCLITrustModel(encounter_policy)),
        store=get_encounter_store(),
        materializer=StimulusMaterializer(bridge, get_stimulus_store()),
        policy=encounter_policy,
        observation_vla=observation_service,
    )
    observation_service.attach_context(
        encounter_service=encounter_service,
        mission_response_service=mission_service,
    )
    return RuntimeBundle(
        shared_data=shared,
        sentinel_provider=sentinel,
        mapbox_provider=mapbox,
        encounter_service=encounter_service,
        observation_vla_service=observation_service,
        mission_response_service=mission_service,
    )
