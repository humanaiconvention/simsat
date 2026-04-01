"""
HAIC Stimulus Bridge — wraps SimSat's imaging pipeline into HAIC-compatible
GroundingStimulus packets for convention sessions.

This is the core integration layer between SimSat (satellite simulation)
and the HumanAI Convention protocol (Maestro/PRISM).
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schemas import (
    GroundingStimulus,
    ObservationMetadata,
    ObservationWindow,
    StimulusImage,
    StimulusType,
)

logger = logging.getLogger(__name__)


class StimulusBridge:
    """
    Bridges SimSat's satellite data into HAIC GroundingStimulus packets.

    Reads from the shared_data_dict (multiprocessing) and calls imaging
    providers to produce multi-perspective observation packets.
    """

    def __init__(self, shared_data: Dict[str, Any], sentinel_provider=None, mapbox_provider=None):
        self.shared_data = shared_data
        self.sentinel = sentinel_provider
        self.mapbox = mapbox_provider

    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """Get current satellite position from shared data."""
        pos = self.shared_data.get("satellite_position")
        ts = self.shared_data.get("last_updated")
        if pos is None:
            return None
        return {
            "lon": pos[0],
            "lat": pos[1],
            "alt": pos[2],
            "timestamp": ts,
        }

    def build_stimulus(
        self,
        include_sentinel: bool = True,
        include_mapbox: bool = True,
        sentinel_bands: Optional[List[str]] = None,
        size_km: float = 5.0,
        target_lon: Optional[float] = None,
        target_lat: Optional[float] = None,
    ) -> GroundingStimulus:
        """
        Build a GroundingStimulus packet from current satellite position.

        This is the "prism" — multiple imaging perspectives of the same location,
        each revealing different aspects of the observed area.

        Args:
            include_sentinel: Include Sentinel-2 multispectral imagery
            include_mapbox: Include Mapbox high-res perspective imagery
            sentinel_bands: Spectral bands for Sentinel (default: RGB)
            size_km: Image size in km for Sentinel
            target_lon: Override target longitude (default: satellite nadir)
            target_lat: Override target latitude (default: satellite nadir)

        Returns:
            GroundingStimulus with multi-perspective satellite imagery
        """
        pos = self.get_current_position()
        if pos is None:
            raise ValueError("No satellite position available — is the simulator running?")

        if sentinel_bands is None:
            sentinel_bands = ["red", "green", "blue"]

        stimulus = GroundingStimulus(
            satellite_position=[pos["lon"], pos["lat"], pos["alt"]],
            simulation_timestamp=self._format_timestamp(pos["timestamp"]),
        )

        t_lon = target_lon if target_lon is not None else pos["lon"]
        t_lat = target_lat if target_lat is not None else pos["lat"]

        # Sentinel-2: scientific perspective (multispectral, temporal)
        if include_sentinel and self.sentinel:
            sentinel_image = self._fetch_sentinel(
                pos["lon"], pos["lat"], pos["timestamp"],
                sentinel_bands, size_km,
            )
            if sentinel_image:
                stimulus.images.append(sentinel_image)

        # Mapbox: visual perspective (high-res, cloud-free, geometric)
        if include_mapbox and self.mapbox:
            mapbox_image = self._fetch_mapbox(
                pos["lon"], pos["lat"], pos["alt"],
                t_lon, t_lat,
            )
            if mapbox_image:
                stimulus.images.append(mapbox_image)

        # Build observation context
        stimulus.observation_context = self._build_context(stimulus, t_lon, t_lat)
        stimulus.compute_hash()

        return stimulus

    def build_stimulus_at(
        self,
        lon: float,
        lat: float,
        timestamp: str,
        sat_lon: Optional[float] = None,
        sat_lat: Optional[float] = None,
        sat_alt: float = 500.0,
        sentinel_bands: Optional[List[str]] = None,
        size_km: float = 5.0,
    ) -> GroundingStimulus:
        """Build a GroundingStimulus for an arbitrary location and time."""
        if sentinel_bands is None:
            sentinel_bands = ["red", "green", "blue"]

        s_lon = sat_lon if sat_lon is not None else lon
        s_lat = sat_lat if sat_lat is not None else lat

        stimulus = GroundingStimulus(
            satellite_position=[s_lon, s_lat, sat_alt],
            simulation_timestamp=timestamp,
        )

        if self.sentinel:
            sentinel_image = self._fetch_sentinel(lon, lat, timestamp, sentinel_bands, size_km)
            if sentinel_image:
                stimulus.images.append(sentinel_image)

        if self.mapbox:
            mapbox_image = self._fetch_mapbox(s_lon, s_lat, sat_alt, lon, lat)
            if mapbox_image:
                stimulus.images.append(mapbox_image)

        stimulus.observation_context = self._build_context(stimulus, lon, lat)
        stimulus.compute_hash()
        return stimulus

    def predict_observation_windows(self, count: int = 5, step_seconds: int = 300) -> List[ObservationWindow]:
        """
        Predict upcoming observation windows based on current orbital trajectory.

        Returns future positions the satellite will pass over, useful for
        scheduling grounding sessions in advance.
        """
        pos = self.get_current_position()
        if pos is None:
            return []

        windows: List[ObservationWindow] = []
        # Project forward using current position and approximate orbital speed
        # Real implementation would use pyorbital propagation
        current_lon = pos["lon"]
        current_lat = pos["lat"]
        current_ts = pos["timestamp"]

        for i in range(count):
            offset = (i + 1) * step_seconds
            # Approximate: satellite moves ~7.5 km/s, ~0.067 deg/s longitude at equator
            future_lon = (current_lon + 0.067 * offset) % 360
            if future_lon > 180:
                future_lon -= 360
            # Latitude oscillates based on inclination (~98.5 deg for sun-synchronous)
            import math
            future_lat = current_lat + 15 * math.sin(0.001 * offset)
            future_lat = max(-90, min(90, future_lat))

            start_ts = self._offset_timestamp(current_ts, offset)
            end_ts = self._offset_timestamp(current_ts, offset + step_seconds)

            windows.append(ObservationWindow(
                start_time=start_ts,
                end_time=end_ts,
                satellite_position_start=[future_lon, future_lat, pos["alt"]],
                satellite_position_end=[future_lon + 0.067 * step_seconds, future_lat, pos["alt"]],
                region_description=self._describe_region(future_lon, future_lat),
            ))

        return windows

    # ---- Private helpers ----

    def _fetch_sentinel(
        self, lon: float, lat: float, timestamp: Any,
        bands: List[str], size_km: float,
    ) -> Optional[StimulusImage]:
        """Fetch Sentinel-2 imagery and package as StimulusImage."""
        try:
            result = self.sentinel.get_single_image_lon_lat(
                lon, lat, timestamp,
                data_type="png",
                spectral_bands=bands,
                size_km=size_km,
            )
            meta = result["metadata"]
            image_buf = result["image"]

            is_multispectral = len(bands) > 3 or any(
                b not in ("red", "green", "blue") for b in bands
            )

            obs_meta = ObservationMetadata(
                satellite_position=[lon, lat, 0],
                timestamp=self._format_timestamp(timestamp),
                footprint=meta.get("footprint"),
                cloud_cover=meta.get("cloud_cover"),
                source=meta.get("source"),
                spectral_bands=bands,
                size_km=size_km,
                image_available=meta.get("image_available", False),
            )

            image_b64 = None
            if image_buf is not None:
                if hasattr(image_buf, "getvalue"):
                    image_b64 = base64.b64encode(image_buf.getvalue()).decode("utf-8")
                elif isinstance(image_buf, bytes):
                    image_b64 = base64.b64encode(image_buf).decode("utf-8")

            return StimulusImage(
                stimulus_type=StimulusType.SENTINEL_MULTISPECTRAL if is_multispectral else StimulusType.SENTINEL_RGB,
                image_b64=image_b64,
                metadata=obs_meta,
            )
        except Exception as e:
            logger.warning("Failed to fetch Sentinel imagery: %s", e)
            return None

    def _fetch_mapbox(
        self, sat_lon: float, sat_lat: float, sat_alt: float,
        target_lon: float, target_lat: float,
    ) -> Optional[StimulusImage]:
        """Fetch Mapbox perspective imagery and package as StimulusImage."""
        try:
            result = self.mapbox.get_target_image(
                sat_lon, sat_lat, sat_alt, target_lon, target_lat,
            )
            meta = result["metadata"]
            image_data = result["image"]

            obs_meta = ObservationMetadata(
                satellite_position=[sat_lon, sat_lat, sat_alt],
                timestamp=datetime.now(timezone.utc).isoformat(),
                elevation_degrees=meta.get("elevation_degrees"),
                bearing=meta.get("bearing"),
                pitch=meta.get("pitch"),
                target_visible=meta.get("target_visible"),
                image_available=meta.get("image_available", False),
                source="mapbox",
            )

            image_b64 = None
            if image_data is not None:
                image_b64 = base64.b64encode(image_data).decode("utf-8")

            return StimulusImage(
                stimulus_type=StimulusType.MAPBOX_PERSPECTIVE,
                image_b64=image_b64,
                metadata=obs_meta,
            )
        except Exception as e:
            logger.warning("Failed to fetch Mapbox imagery: %s", e)
            return None

    def _format_timestamp(self, ts: Any) -> str:
        """Normalize timestamp to ISO-8601 UTC string."""
        if isinstance(ts, str):
            return ts
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.isoformat().replace("+00:00", "Z")
        return str(ts)

    def _offset_timestamp(self, ts: Any, offset_seconds: int) -> str:
        """Add seconds to a timestamp and return ISO string."""
        if isinstance(ts, str):
            ts = ts.strip()
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                return ts
        elif isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        from datetime import timedelta
        dt = dt + timedelta(seconds=offset_seconds)
        return dt.isoformat().replace("+00:00", "Z")

    def _describe_region(self, lon: float, lat: float) -> str:
        """Generate a basic region description from coordinates."""
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"

        if abs(lat) > 66.5:
            zone = "polar"
        elif abs(lat) > 23.5:
            zone = "temperate"
        else:
            zone = "tropical"

        # Basic land/ocean heuristic
        terrain = "region"

        return f"{zone.title()} {terrain} at {abs(lat):.1f}{lat_dir}, {abs(lon):.1f}{lon_dir}"

    def _build_context(self, stimulus: GroundingStimulus, lon: float, lat: float) -> str:
        """Build observation context string for grounding sessions."""
        parts = [
            f"Satellite observation at ({lat:.4f}, {lon:.4f})",
            f"from altitude {stimulus.satellite_position[2]:.1f} km.",
        ]

        available = []
        for img in stimulus.images:
            if img.metadata and img.metadata.image_available:
                available.append(img.stimulus_type.value)

        if available:
            parts.append(f"Available perspectives: {', '.join(available)}.")

        sentinel_imgs = [
            img for img in stimulus.images
            if img.stimulus_type in (StimulusType.SENTINEL_RGB, StimulusType.SENTINEL_MULTISPECTRAL)
            and img.metadata
        ]
        for img in sentinel_imgs:
            if img.metadata.cloud_cover is not None:
                parts.append(f"Cloud cover: {img.metadata.cloud_cover:.1f}%.")
            if img.metadata.source:
                parts.append(f"Source: {img.metadata.source}.")

        return " ".join(parts)
