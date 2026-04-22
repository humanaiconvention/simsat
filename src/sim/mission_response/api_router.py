from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .service import MissionResponseService

router = APIRouter(prefix="/mission-response", tags=["Mission Response"])

_service: MissionResponseService | None = None


def init(service: MissionResponseService) -> None:
    global _service
    _service = service


def _require_service() -> MissionResponseService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Mission response service not initialised")
    return _service


class ActionOutcomeRequest(BaseModel):
    utility_realized: float = Field(ge=0.0, le=1.0)
    execution_status: str = Field(default="simulated", max_length=32)
    notes: str | None = Field(default=None, max_length=1000)


@router.get("/actions")
async def list_actions(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    service = _require_service()
    actions = service.list_actions(limit=limit)
    return {"actions": [action.to_dict() for action in actions], "total": len(actions)}


@router.get("/outcomes")
async def list_outcomes(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    service = _require_service()
    outcomes = service.list_outcomes(limit=limit)
    return {"outcomes": [outcome.to_dict() for outcome in outcomes], "total": len(outcomes)}


@router.post("/from-trace/{trace_id}")
async def propose_from_trace(trace_id: str) -> dict[str, Any]:
    service = _require_service()
    try:
        action = service.propose_from_trace_id(trace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return action.to_dict()


@router.get("/trace/{trace_id}")
async def get_trace_response(trace_id: str) -> dict[str, Any]:
    service = _require_service()
    try:
        return service.get_trace_response(trace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/action/{action_id}/outcome")
async def register_action_outcome(action_id: str, req: ActionOutcomeRequest) -> dict[str, Any]:
    service = _require_service()
    try:
        outcome = service.register_action_outcome(
            action_id=action_id,
            utility_realized=req.utility_realized,
            execution_status=req.execution_status,
            notes=req.notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return outcome.to_dict()


@router.get("/summary")
async def summary(
    mission_id: str = Query(default="default"),
    scenario_pack: str = Query(default="all"),
) -> dict[str, Any]:
    service = _require_service()
    return service.summarize_mission(mission_id=mission_id, scenario_pack=scenario_pack).to_dict()
