"""
HAIC Schemas for SimSat — data structures bridging satellite observations
into HumanAI Convention grounding protocols.

Aligned with Maestro's GroundingRequest schema and PRISM's EntropySnapshot.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class StimulusType(str, Enum):
    SENTINEL_RGB = "sentinel_rgb"
    SENTINEL_MULTISPECTRAL = "sentinel_multispectral"
    MAPBOX_PERSPECTIVE = "mapbox_perspective"
    COMPOSITE = "composite"


class SessionStatus(str, Enum):
    PENDING = "pending"
    STIMULUS_READY = "stimulus_ready"
    INTERVIEW_ACTIVE = "interview_active"
    INTERVIEW_COMPLETE = "interview_complete"
    PRISM_MEASURED = "prism_measured"
    SETTLEMENT_PENDING = "settlement_pending"
    SETTLED = "settled"
    FAILED = "failed"


class ViabilityGate(str, Enum):
    ENTROPY_REDUCTION = "entropy_reduction"
    EXTRACTION_RISK = "extraction_risk"
    PRISM_CONSISTENCY = "prism_consistency"
    PARTICIPATION_COVENANT = "participation_covenant"
    FEDERATED_EXCHANGE = "federated_exchange"
    EPISTEMIC_ALIGNMENT = "epistemic_alignment"


@dataclass
class ObservationMetadata:
    """Metadata about a single satellite observation."""
    satellite_position: List[float]  # [lon, lat, alt_km]
    timestamp: str  # ISO-8601 UTC
    footprint: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
    cloud_cover: Optional[float] = None
    source: Optional[str] = None  # e.g., "sentinel-2a", "mapbox"
    spectral_bands: Optional[List[str]] = None
    elevation_degrees: Optional[float] = None
    bearing: Optional[float] = None
    pitch: Optional[float] = None
    size_km: Optional[float] = None
    target_visible: Optional[bool] = None
    image_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StimulusImage:
    """A single image within a grounding stimulus packet."""
    stimulus_type: StimulusType
    image_b64: Optional[str] = None  # base64-encoded PNG
    metadata: Optional[ObservationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"stimulus_type": self.stimulus_type.value}
        if self.image_b64 is not None:
            d["image_b64"] = self.image_b64
        if self.metadata is not None:
            d["metadata"] = self.metadata.to_dict()
        return d


@dataclass
class GroundingStimulus:
    """
    A packaged satellite observation ready for HAIC convention sessions.

    This is the bridge between SimSat's imaging pipeline and Maestro's
    grounding request protocol. Contains one or more images from different
    imaging perspectives (the "prism" — multiple views of the same location).
    """
    stimulus_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Satellite context
    satellite_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    simulation_timestamp: str = ""

    # Multi-perspective images (the "prism")
    images: List[StimulusImage] = field(default_factory=list)

    # Geographic context for grounding
    location_description: str = ""  # reverse-geocoded or generated description
    observation_context: str = ""   # what makes this observation interesting for grounding

    # Integrity
    content_hash: str = ""  # SHA-256 of all image data + metadata

    def compute_hash(self) -> str:
        """Compute content hash over all images and metadata."""
        hasher = hashlib.sha256()
        hasher.update(self.stimulus_id.encode())
        hasher.update(self.simulation_timestamp.encode())
        hasher.update(json.dumps(self.satellite_position).encode())
        for img in self.images:
            if img.image_b64:
                hasher.update(img.image_b64.encode())
            hasher.update(img.stimulus_type.value.encode())
        self.content_hash = hasher.hexdigest()
        return self.content_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stimulus_id": self.stimulus_id,
            "created_at": self.created_at,
            "satellite_position": self.satellite_position,
            "simulation_timestamp": self.simulation_timestamp,
            "images": [img.to_dict() for img in self.images],
            "location_description": self.location_description,
            "observation_context": self.observation_context,
            "content_hash": self.content_hash,
        }


@dataclass
class ConventionSession:
    """
    A full HAIC convention session grounded in satellite observation.

    Tracks the lifecycle: stimulus → interview → PRISM measurement → settlement.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: SessionStatus = SessionStatus.PENDING

    # Stimulus
    stimulus: Optional[GroundingStimulus] = None

    # Interview
    interview_turns: List[Dict[str, str]] = field(default_factory=list)
    participant_id: Optional[str] = None

    # Proof-of-Grounding
    pog_verified: bool = False
    pog_telemetry: Optional[Dict[str, Any]] = None

    # PRISM measurements
    prism_snapshot_before: Optional[Dict[str, Any]] = None
    prism_snapshot_after: Optional[Dict[str, Any]] = None
    entropy_delta: Optional[Dict[str, Any]] = None
    geometric_health_before: Optional[Dict[str, Any]] = None
    geometric_health_after: Optional[Dict[str, Any]] = None

    # Settlement
    viability_gates: Dict[str, bool] = field(default_factory=dict)
    settlement_result: Optional[Dict[str, Any]] = None
    receipt_merkle_root: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": self.status.value,
            "interview_turns": self.interview_turns,
            "participant_id": self.participant_id,
            "pog_verified": self.pog_verified,
            "viability_gates": self.viability_gates,
            "receipt_merkle_root": self.receipt_merkle_root,
        }
        if self.stimulus:
            d["stimulus"] = self.stimulus.to_dict()
        if self.prism_snapshot_before:
            d["prism_snapshot_before"] = self.prism_snapshot_before
        if self.prism_snapshot_after:
            d["prism_snapshot_after"] = self.prism_snapshot_after
        if self.entropy_delta:
            d["entropy_delta"] = self.entropy_delta
        if self.geometric_health_before:
            d["geometric_health_before"] = self.geometric_health_before
        if self.geometric_health_after:
            d["geometric_health_after"] = self.geometric_health_after
        if self.pog_telemetry:
            d["pog_telemetry"] = self.pog_telemetry
        if self.settlement_result:
            d["settlement_result"] = self.settlement_result
        return d


@dataclass
class ObservationWindow:
    """A predicted future observation window for scheduling grounding sessions."""
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: str = ""
    end_time: str = ""
    satellite_position_start: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    satellite_position_end: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    estimated_cloud_cover: Optional[float] = None
    region_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
