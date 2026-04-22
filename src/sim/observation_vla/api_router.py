from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .service import ObservationVLAService

router = APIRouter(prefix="/observation-vla", tags=["Observation VLA"])

_service: ObservationVLAService | None = None


def init(service: ObservationVLAService) -> None:
    global _service
    _service = service


def _require_service() -> ObservationVLAService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Observation VLA service not initialised")
    return _service


class OutcomeRequest(BaseModel):
    operator_action: str = Field(min_length=1, max_length=32)
    useful: bool
    usefulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome_tags: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=1000)
    mission_id: str | None = Field(default=None, max_length=128)
    label_source: str = Field(default="simulated", min_length=1, max_length=64)
    reviewer: str | None = Field(default=None, max_length=128)
    review_status: str = Field(default="provisional", min_length=1, max_length=32)
    replace_existing: bool = Field(default=True)


class OperatorReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=128)
    operator_action: str = Field(min_length=1, max_length=32)
    useful: bool
    usefulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome_tags: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=1000)
    mission_id: str | None = Field(default=None, max_length=128)


class SubmissionCaseRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=128)
    pinned_by: str | None = Field(default=None, max_length=128)
    notes: str | None = Field(default=None, max_length=1000)


@router.get("/assessments")
async def list_assessments(limit: int = Query(default=20, ge=1, le=1000)) -> dict[str, Any]:
    service = _require_service()
    records = service.list_assessments(limit=limit)
    return {"assessments": [record.to_dict() for record in records], "total": len(records)}


@router.get("/assessment/{assessment_id}")
async def get_assessment(assessment_id: str) -> dict[str, Any]:
    service = _require_service()
    record = service.get_assessment(assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Assessment {assessment_id} not found")
    return record.to_dict()


@router.get("/traces")
async def list_traces(limit: int = Query(default=20, ge=1, le=1000)) -> dict[str, Any]:
    service = _require_service()
    traces = service.list_traces(limit=limit)
    return {"traces": [trace.to_dict() for trace in traces], "total": len(traces)}


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    service = _require_service()
    trace, outcome = service.get_trace_with_outcome(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return {
        "trace": trace.to_dict(),
        "current_outcome": outcome.to_dict() if outcome is not None else None,
    }


@router.post("/reassess/{trace_id}")
async def reassess_trace(trace_id: str, persist: bool = Query(default=False)) -> dict[str, Any]:
    service = _require_service()
    try:
        record = service.reassess_trace(trace_id, persist=persist)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return record.to_dict()


@router.post("/trace/{trace_id}/outcome")
async def register_outcome(trace_id: str, req: OutcomeRequest) -> dict[str, Any]:
    service = _require_service()
    try:
        outcome, states, trace = service.register_outcome(
            trace_id=trace_id,
            operator_action=req.operator_action,
            useful=req.useful,
            usefulness_score=req.usefulness_score,
            outcome_tags=req.outcome_tags,
            notes=req.notes,
            mission_id=req.mission_id,
            label_source=req.label_source,
            reviewer=req.reviewer,
            review_status=req.review_status,
            replace_existing=req.replace_existing,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "outcome": outcome.to_dict(),
        "memory_states": [state.to_dict() for state in states],
        "trace": trace.to_dict(),
    }


@router.post("/trace/{trace_id}/operator-review")
async def register_operator_review(trace_id: str, req: OperatorReviewRequest) -> dict[str, Any]:
    service = _require_service()
    try:
        outcome, states, trace = service.register_operator_review(
            trace_id=trace_id,
            reviewer=req.reviewer,
            operator_action=req.operator_action,
            useful=req.useful,
            usefulness_score=req.usefulness_score,
            outcome_tags=req.outcome_tags,
            notes=req.notes,
            mission_id=req.mission_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "outcome": outcome.to_dict(),
        "memory_states": [state.to_dict() for state in states],
        "trace": trace.to_dict(),
    }


@router.get("/outcomes")
async def list_outcomes(
    limit: int = Query(default=20, ge=1, le=1000),
    current_only: bool = Query(default=True),
) -> dict[str, Any]:
    service = _require_service()
    outcomes = service.list_outcomes(limit=limit, current_only=current_only)
    return {"outcomes": [outcome.to_dict() for outcome in outcomes], "total": len(outcomes)}


@router.get("/review-candidates")
async def list_review_candidates(
    scenario_pack: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=20, ge=1, le=1000),
) -> dict[str, Any]:
    service = _require_service()
    candidates = service.list_review_candidates(scenario_pack=scenario_pack, limit=limit)
    return {"candidates": candidates, "total": len(candidates)}


@router.get("/review-bundle/{trace_id}")
async def get_review_bundle(trace_id: str) -> dict[str, Any]:
    service = _require_service()
    try:
        return service.get_review_bundle(trace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/submission-cases")
async def list_submission_cases() -> dict[str, Any]:
    service = _require_service()
    cases = service.list_submission_cases()
    return {"cases": [case.to_dict() for case in cases], "total": len(cases)}


@router.post("/submission-case/{scenario_pack}")
async def pin_submission_case(scenario_pack: str, req: SubmissionCaseRequest) -> dict[str, Any]:
    service = _require_service()
    try:
        pin = service.pin_submission_case(
            scenario_pack=scenario_pack,
            trace_id=req.trace_id,
            pinned_by=req.pinned_by,
            notes=req.notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return pin.to_dict()


@router.get("/memory")
async def list_memory(
    scope: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    service = _require_service()
    states = service.list_memory_states(scope=scope, limit=limit)
    return {"states": [state.to_dict() for state in states], "total": len(states)}


@router.get("/dataset")
async def export_dataset(
    limit: int = Query(default=100, ge=1, le=1000),
    labelled_only: bool = Query(default=True),
) -> dict[str, Any]:
    service = _require_service()
    rows = service.export_dataset(limit=limit, labelled_only=labelled_only)
    return {"examples": rows, "total": len(rows), "labelled_only": labelled_only}
