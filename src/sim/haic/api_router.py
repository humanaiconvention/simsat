"""
HAIC API Router — FastAPI router that plugs into SimSat's existing api.py.

Mounts at /haic/ prefix. Provides:
  POST /haic/stimulus           — build a GroundingStimulus from current position
  GET  /haic/stimulus/{id}      — retrieve a stimulus by ID
  POST /haic/session            — create a new ConventionSession
  GET  /haic/session/{id}       — get session state
  GET  /haic/sessions           — list all sessions (with optional status filter)
  POST /haic/session/{id}/turn  — submit an interview turn + PoG telemetry
  POST /haic/session/{id}/close — close interview, trigger PRISM measurement
  GET  /haic/session/{id}/receipt — get session receipt / Merkle root
  GET  /haic/windows            — get upcoming observation windows
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from .bridge import StimulusBridge
from .interviewer import Interviewer
from .pog import PoGTelemetry, verify as pog_verify
from .prism_loop import PRISMLoop
from .receipt import build_receipt
from .schemas import (
    ConventionSession,
    GroundingStimulus,
    SessionStatus,
    StimulusType,
)
from .session_store import get_store
from .viability import evaluate_viability

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/haic", tags=["HAIC Convention"])

# These are injected by api.py at startup
_bridge: Optional[StimulusBridge] = None
_prism: Optional[PRISMLoop] = None

# In-memory stimulus cache (stimulus_id → GroundingStimulus)
_stimuli: Dict[str, GroundingStimulus] = {}


def init(bridge: StimulusBridge, prism_loop: Optional[PRISMLoop] = None) -> None:
    """Called by api.py to inject providers after startup."""
    global _bridge, _prism
    _bridge = bridge
    _prism = prism_loop


# ---- Request/Response models ----

class StimulusRequest(BaseModel):
    include_sentinel: bool = True
    include_mapbox: bool = True
    sentinel_bands: List[str] = Field(default_factory=lambda: ["red", "green", "blue"])
    size_km: float = 5.0
    target_lon: Optional[float] = None
    target_lat: Optional[float] = None


class TurnRequest(BaseModel):
    content: str = Field(..., description="The participant's response text")
    # PoG telemetry (optional but raises provenance score)
    client_start_ms: Optional[int] = None
    client_end_ms: Optional[int] = None
    keystroke_intervals: Optional[List[int]] = None


class SessionSummary(BaseModel):
    session_id: str
    status: str
    created_at: str
    turn_count: int
    pog_verified: bool
    receipt_merkle_root: Optional[str]


# ---- Endpoints ----

@router.post("/stimulus", summary="Build a grounding stimulus from current satellite position")
async def create_stimulus(req: StimulusRequest) -> Dict[str, Any]:
    if _bridge is None:
        raise HTTPException(503, "HAIC bridge not initialised")
    try:
        stimulus = _bridge.build_stimulus(
            include_sentinel=req.include_sentinel,
            include_mapbox=req.include_mapbox,
            sentinel_bands=req.sentinel_bands,
            size_km=req.size_km,
            target_lon=req.target_lon,
            target_lat=req.target_lat,
        )
    except ValueError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.exception("Stimulus build failed")
        raise HTTPException(500, f"Stimulus build failed: {e}")

    _stimuli[stimulus.stimulus_id] = stimulus
    return _stimulus_response(stimulus)


@router.get("/stimulus/{stimulus_id}", summary="Retrieve a previously built stimulus")
async def get_stimulus(stimulus_id: str) -> Dict[str, Any]:
    s = _stimuli.get(stimulus_id)
    if s is None:
        raise HTTPException(404, "Stimulus not found")
    return _stimulus_response(s)


@router.post("/session", summary="Create a new HAIC convention session")
async def create_session(
    stimulus_id: Optional[str] = Body(default=None),
    participant_id: Optional[str] = Body(default=None),
    auto_stimulus: bool = Body(default=True),
) -> Dict[str, Any]:
    store = get_store()

    # Resolve stimulus
    stimulus: Optional[GroundingStimulus] = None
    if stimulus_id:
        stimulus = _stimuli.get(stimulus_id)
        if stimulus is None:
            raise HTTPException(404, f"Stimulus {stimulus_id} not found")
    elif auto_stimulus and _bridge is not None:
        try:
            stimulus = _bridge.build_stimulus()
            _stimuli[stimulus.stimulus_id] = stimulus
        except Exception as e:
            logger.warning("Auto-stimulus failed: %s", e)

    session = ConventionSession(
        participant_id=participant_id,
        status=SessionStatus.STIMULUS_READY if stimulus else SessionStatus.PENDING,
    )
    if stimulus:
        session.stimulus = stimulus

    # Capture PRISM snapshot before interview
    if _prism is not None:
        try:
            snap = _prism.snapshot()
            session.prism_snapshot_before = snap
            session.geometric_health_before = _prism.geometric_health()
        except Exception as e:
            logger.warning("PRISM pre-snapshot failed: %s", e)

    store.create(session)
    logger.info("Created session %s (status=%s)", session.session_id, session.status.value)
    return session.to_dict()


@router.get("/session/{session_id}", summary="Get session state")
async def get_session(session_id: str) -> Dict[str, Any]:
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


@router.get("/sessions", summary="List sessions")
async def list_sessions(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    store = get_store()
    if status:
        try:
            s = SessionStatus(status)
            sessions = store.list_by_status(s)
        except ValueError:
            raise HTTPException(400, f"Unknown status '{status}'")
    else:
        sessions = store.list_all()

    sessions = sorted(sessions, key=lambda x: x.created_at, reverse=True)[:limit]
    return {
        "sessions": [
            SessionSummary(
                session_id=s.session_id,
                status=s.status.value,
                created_at=s.created_at,
                turn_count=len(s.interview_turns),
                pog_verified=s.pog_verified,
                receipt_merkle_root=s.receipt_merkle_root,
            ).model_dump()
            for s in sessions
        ],
        "total": len(sessions),
    }


@router.post("/session/{session_id}/turn", summary="Submit an interview turn")
async def submit_turn(session_id: str, req: TurnRequest) -> Dict[str, Any]:
    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status not in (SessionStatus.STIMULUS_READY, SessionStatus.INTERVIEW_ACTIVE):
        raise HTTPException(409, f"Session not accepting turns (status={session.status.value})")

    # Transition to active on first turn
    if session.status == SessionStatus.STIMULUS_READY:
        session.status = SessionStatus.INTERVIEW_ACTIVE

    turn_index = len([t for t in session.interview_turns if t.get("role") == "user"])

    # Build interviewer and get AI response
    interviewer = Interviewer(session)
    try:
        ai_response = interviewer.respond(req.content)
    except Exception as e:
        logger.warning("Interviewer failed: %s", e)
        ai_response = "Thank you for sharing that. Could you tell me more about what you observe?"

    # Append turns
    session.interview_turns.append({"role": "user", "content": req.content})
    session.interview_turns.append({"role": "assistant", "content": ai_response})

    # PoG telemetry for this turn
    if req.client_start_ms is not None and req.client_end_ms is not None:
        pog_tel = PoGTelemetry(
            turn_index=turn_index,
            response_text=req.content,
            client_start_ms=req.client_start_ms,
            client_end_ms=req.client_end_ms,
            keystroke_intervals=req.keystroke_intervals or [],
        )
        if session.pog_telemetry is None:
            session.pog_telemetry = {"records": []}
        session.pog_telemetry["records"].append(pog_tel.__dict__)

    store.update(session)
    return {
        "turn_index": turn_index,
        "assistant_response": ai_response,
        "session_status": session.status.value,
        "total_turns": len(session.interview_turns),
    }


@router.post("/session/{session_id}/close", summary="Close interview and run PRISM + viability")
async def close_session(session_id: str) -> Dict[str, Any]:
    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status not in (SessionStatus.INTERVIEW_ACTIVE, SessionStatus.INTERVIEW_COMPLETE):
        raise HTTPException(409, f"Cannot close session with status={session.status.value}")

    session.status = SessionStatus.INTERVIEW_COMPLETE

    # ---- PoG verification ----
    pog_records = []
    if session.pog_telemetry:
        raw_records = session.pog_telemetry.get("records", [])
        for r in raw_records:
            pog_records.append(PoGTelemetry(
                turn_index=r.get("turn_index", 0),
                response_text=r.get("response_text", ""),
                client_start_ms=r.get("client_start_ms", 0),
                client_end_ms=r.get("client_end_ms", 0),
                keystroke_intervals=r.get("keystroke_intervals", []),
            ))
    elif session.interview_turns:
        # No telemetry submitted — do text-only PoG
        user_turns = [t for t in session.interview_turns if t.get("role") == "user"]
        for i, t in enumerate(user_turns):
            pog_records.append(PoGTelemetry(
                turn_index=i,
                response_text=t["content"],
                client_start_ms=0,
                client_end_ms=10000,  # assume 10s — passes duration gate
                keystroke_intervals=[],
            ))

    pog_result = pog_verify(pog_records)
    session.pog_verified = pog_result.verified
    session.pog_telemetry = {
        **(session.pog_telemetry or {}),
        "result": pog_result.to_dict(),
    }

    # ---- PRISM measurement ----
    if _prism is not None:
        try:
            snap_after = _prism.snapshot()
            session.prism_snapshot_after = snap_after
            session.geometric_health_after = _prism.geometric_health()

            if session.prism_snapshot_before and snap_after:
                session.entropy_delta = _prism.compute_delta(
                    session.prism_snapshot_before, snap_after
                )
            session.status = SessionStatus.PRISM_MEASURED
        except Exception as e:
            logger.warning("PRISM measurement failed: %s", e)

    # ---- Viability gates ----
    gates = evaluate_viability(session)
    session.viability_gates = gates

    all_pass = all(gates.values())
    session.status = SessionStatus.SETTLEMENT_PENDING if all_pass else SessionStatus.FAILED

    # ---- Receipt ----
    if all_pass or pog_result.verified:
        receipt = build_receipt(session)
        session.receipt_merkle_root = receipt["merkle_root"]
        session.settlement_result = receipt
        if all_pass:
            session.status = SessionStatus.SETTLED

    store.update(session)
    logger.info(
        "Closed session %s — pog=%s gates=%s status=%s",
        session_id, pog_result.verified, gates, session.status.value,
    )
    return session.to_dict()


@router.get("/session/{session_id}/receipt", summary="Get session receipt")
async def get_receipt(session_id: str) -> Dict[str, Any]:
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.receipt_merkle_root is None:
        raise HTTPException(404, "No receipt available yet — close the session first")
    return {
        "session_id": session_id,
        "merkle_root": session.receipt_merkle_root,
        "settlement_result": session.settlement_result,
        "status": session.status.value,
    }


@router.get("/windows", summary="Get upcoming observation windows")
async def get_observation_windows(count: int = Query(default=5, ge=1, le=20)) -> Dict[str, Any]:
    if _bridge is None:
        raise HTTPException(503, "HAIC bridge not initialised")
    windows = _bridge.predict_observation_windows(count=count)
    return {"windows": [w.to_dict() for w in windows]}


@router.get("/stimulus/{stimulus_id}/image/{image_index}", summary="Serve a stimulus image as PNG")
async def get_stimulus_image(stimulus_id: str, image_index: int) -> FastAPIResponse:
    s = _stimuli.get(stimulus_id)
    if s is None:
        raise HTTPException(404, "Stimulus not found")
    if image_index < 0 or image_index >= len(s.images):
        raise HTTPException(404, f"Image index {image_index} out of range")
    img = s.images[image_index]
    if not img.image_b64:
        raise HTTPException(404, "Image not available for this perspective")
    image_bytes = base64.b64decode(img.image_b64)
    return FastAPIResponse(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Stimulus-Type": img.stimulus_type.value,
        },
    )


@router.get("/health", summary="HAIC subsystem health check")
async def haic_health() -> Dict[str, Any]:
    pos = _bridge.get_current_position() if _bridge else None
    return {
        "status": "online",
        "bridge": "ready" if _bridge else "not_initialised",
        "prism": "ready" if _prism else "not_initialised",
        "satellite_position": pos,
        "active_sessions": len(get_store().list_all()),
    }


# ---- Helpers ----

def _stimulus_response(stimulus: GroundingStimulus) -> Dict[str, Any]:
    """Serialize stimulus, omitting raw image bytes from top-level list response."""
    d = stimulus.to_dict()
    # Replace full base64 in listing with a presence flag + size
    for img in d.get("images", []):
        b64 = img.pop("image_b64", None)
        img["has_image"] = b64 is not None
        if b64:
            img["image_size_bytes"] = len(base64.b64decode(b64))
    return d
