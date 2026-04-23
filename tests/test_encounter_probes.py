from __future__ import annotations

from encounter.probes import ImagingProbeService
from encounter.schemas import EncounterGeometry, EncounterWindow, TargetSpec


class FakeMapboxProvider:
    def __init__(self, *, target_visible: bool) -> None:
        self.target_visible = target_visible
        self.calls: list[tuple[float, float, float, float, float]] = []

    def probe_target_geometry(self, sat_lon, sat_lat, sat_alt, target_lon, target_lat):
        self.calls.append((sat_lon, sat_lat, sat_alt, target_lon, target_lat))
        return {"target_visible": self.target_visible}


class FakeSentinelProvider:
    def __init__(self, *, image_available: bool, cloud_cover: float | None, source: str | None, datetime: str | None) -> None:
        self.image_available = image_available
        self.cloud_cover = cloud_cover
        self.source = source
        self.datetime = datetime
        self.calls: list[tuple[float, float, str, float]] = []

    def probe_latest_scene_lon_lat(self, lon, lat, timestamp, *, size_km=5, **_kwargs):
        self.calls.append((lon, lat, timestamp, size_km))
        return {
            "image_available": self.image_available,
            "cloud_cover": self.cloud_cover,
            "source": self.source,
            "datetime": self.datetime,
        }


def _window() -> EncounterWindow:
    return EncounterWindow(
        window_id="window-1",
        target_id="target-1",
        encounter_type="imaging_window",
        start_time="2026-03-10T12:00:00Z",
        end_time="2026-03-10T12:00:20Z",
        peak_time="2026-03-10T12:00:10Z",
        satellite_position_peak=[10.0, 20.0, 500.0],
        target_position=[11.0, 21.0],
        geometry=EncounterGeometry(
            elevation_degrees=55.0,
            off_nadir_degrees=15.0,
            slant_range_km=600.0,
            target_visible=True,
            bearing=180.0,
            pitch=35.0,
        ),
        target_priority=0.8,
        duration_seconds=30.0,
    )


def _target() -> TargetSpec:
    return TargetSpec(target_id="target-1", label="Target One", lon=11.0, lat=21.0, priority=0.8, size_km=7.5)


def test_probe_uses_provider_outputs_when_available():
    mapbox = FakeMapboxProvider(target_visible=False)
    sentinel = FakeSentinelProvider(
        image_available=True,
        cloud_cover=12.5,
        source="sentinel-2a",
        datetime="2026-03-10T11:50:00Z",
    )
    service = ImagingProbeService(sentinel_provider=sentinel, mapbox_provider=mapbox)

    result = service.probe(_window(), _target())

    assert result.mapbox_feasible is False
    assert result.sentinel_available is True
    assert result.sentinel_cloud_cover == 12.5
    assert result.sentinel_source == "sentinel-2a"
    assert result.sentinel_datetime == "2026-03-10T11:50:00Z"
    assert mapbox.calls == [(10.0, 20.0, 500.0, 11.0, 21.0)]
    assert sentinel.calls == [(11.0, 21.0, "2026-03-10T12:00:10Z", 7.5)]


def test_probe_falls_back_to_window_visibility_without_providers():
    service = ImagingProbeService(sentinel_provider=None, mapbox_provider=None)

    result = service.probe(_window(), _target())

    assert result.mapbox_feasible is True
    assert result.sentinel_available is None
    assert result.sentinel_cloud_cover is None
    assert result.sentinel_source is None
    assert result.sentinel_datetime is None
