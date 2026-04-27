from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI, HTTPException, Query, Response, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from typing import List, Literal, Optional
import base64
import json
from datetime import datetime, timezone
from encounter import api_router as encounter_router
from haic.api_router import router as haic_router
import haic.api_router as _haic_api
from mission_response import api_router as mission_response_router
from observation_vla import api_router as observation_vla_router
from runtime import build_runtime_bundle, build_runtime_capabilities

logger = logging.getLogger(__name__)

# Rate limiter for external API endpoints (Sentinel/Mapbox)
# Default: 30 requests per minute per client IP. Override via SENTINEL_MAPBOX_RATE_LIMIT.
SENTINEL_MAPBOX_RATE_LIMIT = os.environ.get("SENTINEL_MAPBOX_RATE_LIMIT", "30/minute")
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inject providers into sub-routers after shared_data is available."""
    from haic.prism_loop import get_prism_loop

    shared = getattr(app.state, "shared_data", {})
    runtime_bundle = build_runtime_bundle(shared)
    app.state.runtime_bundle = runtime_bundle
    bridge = runtime_bundle.encounter_service.materializer.bridge
    app.state.haic_bridge = bridge
    prism = get_prism_loop()
    _haic_api.init(bridge, prism)
    app.state.observation_vla_service = runtime_bundle.observation_vla_service
    app.state.mission_response_service = runtime_bundle.mission_response_service
    app.state.encounter_service = runtime_bundle.encounter_service
    observation_vla_router.init(runtime_bundle.observation_vla_service)
    mission_response_router.init(runtime_bundle.mission_response_service)
    encounter_router.init(runtime_bundle.encounter_service)
    yield


api = FastAPI(title="SimSat API", description="Satellite simulation + HAIC convention layer", lifespan=lifespan)
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount HAIC router
api.include_router(haic_router)
api.include_router(encounter_router.router)
api.include_router(observation_vla_router.router)
api.include_router(mission_response_router.router)


def serialize_xarray_dataset(ds):
    if ds is None or not hasattr(ds, "to_array"):
        raise ValueError("Expected an xarray.Dataset-like object for serialization")
    da = ds.to_array()
    metadata = {
        "shape": da.shape,        # e.g., (3, 512, 512)
        "dtype": str(da.dtype),   # e.g., "uint16" or "float32"
        "bands": list(ds.data_vars) # e.g., ["red", "green", "blue"]
    }
    image_bytes = da.values.tobytes()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    return {
        "metadata": metadata,
        "image": image_b64
    }


def format_timestamp_utc(timestamp):
    if isinstance(timestamp, datetime):
        dt = timestamp
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    elif isinstance(timestamp, str):
        normalized = timestamp.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return timestamp
    else:
        return timestamp

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_runtime_capabilities() -> dict:
    runtime_bundle = getattr(api.state, "runtime_bundle", None)
    if runtime_bundle is None:
        return {
            "sentinel_enabled": False,
            "mapbox_enabled": False,
            "challenge_no_mapbox_safe": True,
            "sentinel_first_challenge_scoring": True,
            "high_res_perspective_optional": True,
            "observation_vla_runtime_mode": "uninitialised",
            "observation_vla_model_id": None,
        }
    return build_runtime_capabilities(runtime_bundle)


def _get_runtime_bundle():
    runtime_bundle = getattr(api.state, "runtime_bundle", None)
    if runtime_bundle is None:
        raise HTTPException(status_code=503, detail="Runtime not initialised")
    return runtime_bundle

@api.get("/data/current/position")
async def get_metrics():
    data = getattr(api.state, "shared_data", {})
    timestamp = format_timestamp_utc(data.get("last_updated", 0))
    return {
        "lon-lat-alt": data.get("satellite_position", [0, 0, 0]),
        "timestamp": timestamp
    }


@api.get("/data/current/image/sentinel")
@limiter.limit(SENTINEL_MAPBOX_RATE_LIMIT)
async def get_sentinel_image(
    request: Request,
    spectral_bands: List[str] = Query(default=["red", "green", "blue"]),
    size_km: float = 5.0,
    window_seconds: float = Query(default=10 * 24 * 60 * 60, gt=0),
    return_type: Literal["array", "png"] = "png"
):
    data = getattr(api.state, "shared_data", {}).get("satellite_position", None)
    timestamp = getattr(api.state, "shared_data", {}).get("last_updated", None)
    if data is None:
        raise HTTPException(status_code=500, detail="Error fetching satellite position from shared data - is the simulator running?")
    sentinel = _get_runtime_bundle().sentinel_provider
    sentinel_data = sentinel.get_single_image_lon_lat(
        data[0],
        data[1],
        timestamp,
        data_type=return_type,
        spectral_bands=spectral_bands,
        size_km=size_km,
        window_seconds=window_seconds,
    )
    image = sentinel_data["image"]
    metadata = sentinel_data["metadata"]

    if return_type == "png":
        headers = {
            "sentinel_metadata": json.dumps(
                {
                    "image_available": metadata["image_available"],
                    "source": metadata["source"],
                    "spectral_bands": metadata["spectral_bands"],
                    "footprint": metadata["footprint"],
                    "size_km": metadata["size_km"],
                    "cloud_cover": metadata["cloud_cover"],
                    "datetime": metadata["datetime"],
                    "satellite_position": data,
                    "timestamp": format_timestamp_utc(timestamp),
                }
            ),
            "Access-Control-Expose-Headers": "sentinel_metadata",
        }
        return Response(content=image.getvalue() if image is not None else "", media_type="image/png", headers=headers)
    elif return_type == "array":
        image = serialize_xarray_dataset(image) if metadata["image_available"] and image is not None else None
        return {
            "image": image,
            "sentinel_metadata": {
                "image_available": metadata["image_available"],
                "source": metadata["source"],
                "spectral_bands": metadata["spectral_bands"],
                "footprint": metadata["footprint"],
                "size_km": metadata["size_km"],
                "cloud_cover": metadata["cloud_cover"],
                "datetime": metadata["datetime"],
                "satellite_position": data,
                "timestamp": format_timestamp_utc(timestamp),
            }
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid return_type specified")

@api.get("/data/current/image/mapbox")
@limiter.limit(SENTINEL_MAPBOX_RATE_LIMIT)
async def get_mapbox_image(
    request: Request,
    lon: Optional[float] = Query(default=None, description="Target longitude (defaults to current satellite longitude)", ge=-180, le=180),
    lat: Optional[float] = Query(default=None, description="Target latitude (defaults to current satellite latitude)", ge=-90, le=90)
):
    try:
        satellite_position = getattr(api.state, "shared_data", {}).get("satellite_position", None)
        timestamp = getattr(api.state, "shared_data", {}).get("last_updated", None)

        runtime_bundle = getattr(api.state, "runtime_bundle", None)
        mapbox = runtime_bundle.mapbox_provider if runtime_bundle is not None else None
        if mapbox is None:
            raise HTTPException(status_code=503, detail="Mapbox imagery disabled — MAPBOX_ACCESS_TOKEN not set")

        if satellite_position is None:
            raise HTTPException(status_code=500, detail="Error fetching satellite position from shared data - is the simulator running?")

        target_lon = satellite_position[0] if lon is None else lon
        target_lat = satellite_position[1] if lat is None else lat

        mapbox_data = mapbox.get_target_image(
            satellite_position[0],
            satellite_position[1],
            satellite_position[2],
            target_lon,
            target_lat,
        )
        image = mapbox_data["image"]
        metadata = mapbox_data["metadata"]

        response_metadata = {
            "target_visible": metadata["target_visible"],
            "image_available": metadata["image_available"],
            "elevation_degrees": metadata["elevation_degrees"],
            "zoom_factor": metadata["zoom_factor"],
            "bearing": metadata["bearing"],
            "pitch": metadata["pitch"],
            "satellite_position": satellite_position,
            "timestamp": format_timestamp_utc(timestamp),
        }
        headers = {
            "mapbox_metadata": json.dumps(response_metadata),
            "Access-Control-Expose-Headers": "mapbox_metadata",
        }

        return Response(content=image if image is not None else b"", media_type="image/png",headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching Mapbox image")
        raise HTTPException(status_code=500, detail="Error fetching Mapbox image: " + str(e))


@api.get("/data/image/sentinel")
@limiter.limit(SENTINEL_MAPBOX_RATE_LIMIT)
async def get_sentinel_image_lon_lat(
    request: Request,
    lon: float = Query(..., description="The longitude of the location", ge=-180, le=180),
    lat: float = Query(..., description="The latitude of the location", ge=-90, le=90),
    timestamp: str = Query(..., description="The timestamp of the image", format="date-time"),
    spectral_bands: List[str] = Query(default=["red", "green", "blue"]),
    size_km: float = 5.0,
    window_seconds: float = Query(default=10 * 24 * 60 * 60, gt=0),
    return_type: Literal["array", "png"] = "png"
):
    sentinel = _get_runtime_bundle().sentinel_provider
    sentinel_data = sentinel.get_single_image_lon_lat(
        lon,
        lat,
        timestamp,
        data_type=return_type,
        spectral_bands=spectral_bands,
        size_km=size_km,
        window_seconds=window_seconds,
    )
    image = sentinel_data["image"]
    metadata = sentinel_data["metadata"]

    if return_type == "png":
        headers = {
            "sentinel_metadata": json.dumps(
                {
                    "image_available": metadata["image_available"],
                    "source": metadata["source"],
                    "spectral_bands": metadata["spectral_bands"],
                    "footprint": metadata["footprint"],
                    "size_km": metadata["size_km"],
                    "cloud_cover": metadata["cloud_cover"],
                    "datetime": metadata["datetime"]
                }
            ),
            "Access-Control-Expose-Headers": "sentinel_metadata",
        }
        return Response(content=image.getvalue() if image is not None else "", media_type="image/png", headers=headers)
    elif return_type == "array":
        image = serialize_xarray_dataset(image) if metadata["image_available"] and image is not None else None
        return {
            "image": image,
            "sentinel_metadata": {
                "image_available": metadata["image_available"],
                "source": metadata["source"],
                "spectral_bands": metadata["spectral_bands"],
                "footprint": metadata["footprint"],
                "size_km": metadata["size_km"],
                "cloud_cover": metadata["cloud_cover"],
                "datetime": metadata["datetime"]
            }
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid return_type specified")


@api.get("/data/image/mapbox")
@limiter.limit(SENTINEL_MAPBOX_RATE_LIMIT)
async def get_mapbox_image_lon_lat(
    request: Request,
    lon_target: float = Query(..., description="The longitude of the target location", ge=-180, le=180),
    lat_target: float = Query(..., description="The latitude of the target location", ge=-90, le=90),
    lon_satellite: float = Query(..., description="The longitude of the satellite", ge=-180, le=180),
    lat_satellite: float = Query(..., description="The latitude of the satellite", ge=-90, le=90),
    alt_satellite: float = Query(..., description="The altitude of the satellite", ge=0),
):
    try:
        runtime_bundle = getattr(api.state, "runtime_bundle", None)
        mapbox = runtime_bundle.mapbox_provider if runtime_bundle is not None else None
        if mapbox is None:
            raise HTTPException(status_code=503, detail="Mapbox imagery disabled — MAPBOX_ACCESS_TOKEN not set")
        mapbox_data = mapbox.get_target_image(lon_satellite, lat_satellite, alt_satellite, lon_target, lat_target)
        image = mapbox_data["image"]
        metadata = mapbox_data["metadata"]

        response_metadata = {
            "target_visible": metadata["target_visible"],
            "image_available": metadata["image_available"],
            "elevation_degrees": metadata["elevation_degrees"],
            "zoom_factor": metadata["zoom_factor"],
            "bearing": metadata["bearing"],
            "pitch": metadata["pitch"],
        }
        headers = {
            "mapbox_metadata": json.dumps(response_metadata),
            "Access-Control-Expose-Headers": "mapbox_metadata",
        }

        return Response(content=image if image is not None else b"", media_type="image/png",headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching Mapbox image")
        raise HTTPException(status_code=500, detail="Error fetching Mapbox image: " + str(e))


@api.get("/")
async def root():
    return {"message": "Simulation API is online", "haic": "enabled"}


@api.get("/capabilities")
async def capabilities():
    return get_runtime_capabilities()


