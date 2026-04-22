from __future__ import annotations

from haic.bridge import StimulusBridge
from haic.schemas import GroundingStimulus
from haic.stimulus_store import StimulusStore

from .schemas import EncounterRecord, TargetSpec


class StimulusMaterializer:
    def __init__(self, bridge: StimulusBridge, stimulus_store: StimulusStore) -> None:
        self.bridge = bridge
        self.stimulus_store = stimulus_store

    def materialize(self, record: EncounterRecord, target: TargetSpec, persist: bool = True) -> GroundingStimulus:
        sat_lon, sat_lat, sat_alt = record.window.satellite_position_peak
        stimulus = self.bridge.build_stimulus_at(
            lon=target.lon,
            lat=target.lat,
            timestamp=record.window.peak_time,
            sat_lon=sat_lon,
            sat_lat=sat_lat,
            sat_alt=sat_alt,
            size_km=target.size_km,
        )
        if persist:
            self.stimulus_store.put(stimulus)
        return stimulus
