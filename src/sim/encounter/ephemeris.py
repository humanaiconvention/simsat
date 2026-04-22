from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from pyorbital.orbital import Orbital

from .schemas import EphemerisPoint


class EphemerisService:
    def __init__(self, satellite_name: str, tle: tuple[str, str]) -> None:
        self.satellite_name = satellite_name
        self.tle = tle
        self.orbital = Orbital(satellite_name, line1=tle[0], line2=tle[1])

    def _coerce_datetime(self, timestamp: str | float | datetime | None) -> datetime:
        if timestamp is None:
            return datetime.now(timezone.utc)
        if isinstance(timestamp, datetime):
            dt = timestamp
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        elif isinstance(timestamp, str):
            normalized = timestamp.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            dt = datetime.fromisoformat(normalized)
        else:
            raise TypeError(f"Unsupported timestamp type: {type(timestamp)!r}")

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _to_iso(self, dt: datetime) -> str:
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def position_at(self, timestamp: str | float | datetime) -> EphemerisPoint:
        dt = self._coerce_datetime(timestamp)
        lon, lat, alt = self.orbital.get_lonlatalt(np.datetime64(dt.replace(tzinfo=None)))
        return EphemerisPoint(
            timestamp=self._to_iso(dt),
            satellite_position=[float(lon), float(lat), float(alt)],
        )

    def propagate_series(
        self,
        start_time: str | float | datetime,
        hours: float,
        step_seconds: int,
    ) -> list[EphemerisPoint]:
        start = self._coerce_datetime(start_time)
        total_steps = max(int((hours * 3600) / step_seconds), 1)
        points: list[EphemerisPoint] = []
        for idx in range(total_steps + 1):
            points.append(self.position_at(start + timedelta(seconds=idx * step_seconds)))
        return points
