from __future__ import annotations

from ImagingProviders.mapbox_provider import MapboxlProvider
from ImagingProviders.sentinel_provider import SentinelProvider

from .schemas import EncounterProbeResult, EncounterWindow, TargetSpec


class ImagingProbeService:
    def __init__(self, sentinel_provider: SentinelProvider | None, mapbox_provider: MapboxlProvider | None) -> None:
        self.sentinel_provider = sentinel_provider
        self.mapbox_provider = mapbox_provider

    def probe(self, window: EncounterWindow, target: TargetSpec) -> EncounterProbeResult:
        sat_lon, sat_lat, sat_alt = window.satellite_position_peak
        mapbox_feasible = bool(window.geometry.target_visible)
        if self.mapbox_provider is not None:
            mapbox_probe = self.mapbox_provider.probe_target_geometry(
                sat_lon,
                sat_lat,
                sat_alt,
                target.lon,
                target.lat,
            )
            mapbox_feasible = bool(mapbox_probe.get("target_visible", False))

        sentinel_available = None
        sentinel_cloud_cover = None
        sentinel_source = None
        sentinel_datetime = None
        if self.sentinel_provider is not None:
            sentinel_probe = self.sentinel_provider.probe_latest_scene_lon_lat(
                target.lon,
                target.lat,
                window.peak_time,
                size_km=target.size_km,
            )
            sentinel_available = bool(sentinel_probe.get("image_available", False))
            sentinel_cloud_cover = sentinel_probe.get("cloud_cover")
            sentinel_source = sentinel_probe.get("source")
            sentinel_datetime = sentinel_probe.get("datetime")

        return EncounterProbeResult(
            window_id=window.window_id,
            target_id=target.target_id,
            mapbox_feasible=mapbox_feasible,
            sentinel_available=sentinel_available,
            sentinel_cloud_cover=float(sentinel_cloud_cover) if sentinel_cloud_cover is not None else None,
            sentinel_source=str(sentinel_source) if sentinel_source is not None else None,
            sentinel_datetime=str(sentinel_datetime) if sentinel_datetime is not None else None,
        )
