from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from math import acos, cos, radians

import numpy as np

from .schemas import EncounterGeometry, EncounterWindow, EphemerisPoint, TargetSpec

EARTH_RADIUS_KM = 6371.0
EPS = 1e-12


class WindowDetector:
    def __init__(self, min_elevation_degrees: float = 30.0, min_window_seconds: float = 20.0) -> None:
        self.min_elevation_degrees = min_elevation_degrees
        self.min_window_seconds = min_window_seconds

    def _spherical_to_cartesian(self, lon: float, lat: float, radius: float) -> np.ndarray:
        lon_rad = radians(lon)
        lat_rad = radians(lat)
        x = radius * cos(lat_rad) * np.cos(lon_rad)
        y = radius * cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)
        return np.array([x, y, z], dtype=float)

    def compute_geometry(
        self,
        sat_lon: float,
        sat_lat: float,
        sat_alt_km: float,
        target_lon: float,
        target_lat: float,
    ) -> EncounterGeometry:
        cartesian_sat = self._spherical_to_cartesian(sat_lon, sat_lat, EARTH_RADIUS_KM + sat_alt_km)
        cartesian_target = self._spherical_to_cartesian(target_lon, target_lat, EARTH_RADIUS_KM)

        target_to_sat_vector = cartesian_sat - cartesian_target
        distance = float(np.linalg.norm(target_to_sat_vector))
        target_to_sat_unit_vector = target_to_sat_vector / max(distance, EPS)
        target_unit_vector = cartesian_target / max(float(np.linalg.norm(cartesian_target)), EPS)

        theta = acos(np.clip(np.dot(target_unit_vector, target_to_sat_unit_vector), -1.0, 1.0))
        elevation_degrees = float(90.0 - np.degrees(theta))
        pitch = float(np.degrees(theta))
        target_visible = elevation_degrees >= self.min_elevation_degrees

        sat_to_target_unit_vector = (cartesian_target - cartesian_sat) / max(distance, EPS)
        nadir_unit_vector = -cartesian_sat / max(float(np.linalg.norm(cartesian_sat)), EPS)
        off_nadir_degrees = float(
            np.degrees(acos(np.clip(np.dot(nadir_unit_vector, sat_to_target_unit_vector), -1.0, 1.0)))
        )

        earth_center_to_south_vector = np.array([0.0, 0.0, 1.0])
        target_surface_projection = target_to_sat_unit_vector - np.dot(target_to_sat_unit_vector, target_unit_vector) * target_unit_vector
        target_proj_norm = float(np.linalg.norm(target_surface_projection))
        bearing = 0.0
        if target_proj_norm >= EPS:
            target_surface_unit = target_surface_projection / target_proj_norm
            south_surface_projection = earth_center_to_south_vector - np.dot(earth_center_to_south_vector, target_unit_vector) * target_unit_vector
            south_proj_norm = float(np.linalg.norm(south_surface_projection))
            if south_proj_norm >= EPS:
                south_surface_unit = south_surface_projection / south_proj_norm
                bearing = float(180.0 - np.degrees(acos(np.clip(np.dot(south_surface_unit, target_surface_unit), -1.0, 1.0))))
                bearing_cross = np.cross(south_surface_unit, target_surface_unit)
                if np.dot(bearing_cross, target_unit_vector) < 0:
                    bearing = -bearing
                if bearing < 0:
                    bearing += 360.0

        return EncounterGeometry(
            elevation_degrees=elevation_degrees,
            off_nadir_degrees=off_nadir_degrees,
            slant_range_km=distance,
            target_visible=target_visible,
            bearing=bearing if target_visible else None,
            pitch=pitch if target_visible else None,
        )

    def _window_id(self, target_id: str, start_time: str, end_time: str) -> str:
        digest = hashlib.sha256(f"{target_id}:{start_time}:{end_time}".encode("utf-8")).hexdigest()[:16]
        return f"enc_{digest}"

    def _epoch_seconds(self, timestamp: str) -> int:
        normalized = timestamp.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp())

    def enumerate_windows(
        self,
        ephemeris_points: list[EphemerisPoint],
        targets: list[TargetSpec],
    ) -> list[EncounterWindow]:
        windows: list[EncounterWindow] = []
        if len(ephemeris_points) < 2:
            return windows

        step_seconds = max(
            self._epoch_seconds(ephemeris_points[1].timestamp) - self._epoch_seconds(ephemeris_points[0].timestamp),
            1,
        )

        for target in targets:
            active_start: EphemerisPoint | None = None
            peak_point: EphemerisPoint | None = None
            peak_geometry: EncounterGeometry | None = None
            last_point: EphemerisPoint | None = None

            for point in ephemeris_points:
                sat_lon, sat_lat, sat_alt = point.satellite_position
                geometry = self.compute_geometry(sat_lon, sat_lat, sat_alt, target.lon, target.lat)

                if geometry.target_visible:
                    if active_start is None:
                        active_start = point
                        peak_point = point
                        peak_geometry = geometry
                    elif peak_geometry is not None and geometry.elevation_degrees > peak_geometry.elevation_degrees:
                        peak_point = point
                        peak_geometry = geometry
                    last_point = point
                    continue

                if active_start is not None and peak_point is not None and peak_geometry is not None and last_point is not None:
                    duration_seconds = max(
                        self._epoch_seconds(last_point.timestamp) - self._epoch_seconds(active_start.timestamp) + step_seconds,
                        step_seconds,
                    )
                    if duration_seconds >= self.min_window_seconds:
                        windows.append(
                            EncounterWindow(
                                window_id=self._window_id(target.target_id, active_start.timestamp, last_point.timestamp),
                                target_id=target.target_id,
                                encounter_type="imaging_window",
                                start_time=active_start.timestamp,
                                end_time=last_point.timestamp,
                                peak_time=peak_point.timestamp,
                                satellite_position_peak=peak_point.satellite_position,
                                target_position=[target.lon, target.lat],
                                geometry=peak_geometry,
                                target_priority=target.priority,
                                duration_seconds=float(duration_seconds),
                            )
                        )
                active_start = None
                peak_point = None
                peak_geometry = None
                last_point = None

            if active_start is not None and peak_point is not None and peak_geometry is not None and last_point is not None:
                duration_seconds = max(
                    self._epoch_seconds(last_point.timestamp) - self._epoch_seconds(active_start.timestamp) + step_seconds,
                    step_seconds,
                )
                if duration_seconds >= self.min_window_seconds:
                    windows.append(
                        EncounterWindow(
                            window_id=self._window_id(target.target_id, active_start.timestamp, last_point.timestamp),
                            target_id=target.target_id,
                            encounter_type="imaging_window",
                            start_time=active_start.timestamp,
                            end_time=last_point.timestamp,
                            peak_time=peak_point.timestamp,
                            satellite_position_peak=peak_point.satellite_position,
                            target_position=[target.lon, target.lat],
                            geometry=peak_geometry,
                            target_priority=target.priority,
                            duration_seconds=float(duration_seconds),
                        )
                    )

        return windows
