from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .schemas import TargetSpec
from .service import EncounterService

router = APIRouter(prefix="/encounter", tags=["Encounter Planner"])

_service: EncounterService | None = None


def init(service: EncounterService) -> None:
    global _service
    _service = service


class PlanRequest(BaseModel):
    start_time: str | None = None
    hours: float = Field(default=6.0, gt=0, le=48)
    step_seconds: int = Field(default=30, ge=5, le=300)
    top_k: int = Field(default=20, ge=1, le=100)
    scenario_pack: str | None = Field(default=None, max_length=128)


class EvaluateRequest(BaseModel):
    start_time: str | None = None
    hours: float = Field(default=6.0, gt=0, le=48)
    step_seconds: int = Field(default=30, ge=5, le=300)
    top_k: int = Field(default=20, ge=1, le=100)
    materialize_top_k: int = Field(default=3, ge=0, le=20)
    scenario_pack: str | None = Field(default=None, max_length=128)


class TargetRequest(BaseModel):
    target_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=128)
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    priority: float = Field(default=0.5, ge=0, le=1)
    size_km: float = Field(default=5.0, gt=0, le=100)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_target(self) -> TargetSpec:
        return TargetSpec(
            target_id=self.target_id,
            label=self.label,
            lon=self.lon,
            lat=self.lat,
            priority=self.priority,
            size_km=self.size_km,
            tags=self.tags,
            metadata=self.metadata,
        )


def _require_service() -> EncounterService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Encounter service not initialised")
    return _service


@router.get("/policy")
async def get_policy() -> dict[str, Any]:
    service = _require_service()
    return service.get_policy().to_dict()


@router.get("/targets")
async def get_targets() -> dict[str, Any]:
    service = _require_service()
    targets = service.list_targets()
    return {
        "targets": [target.to_dict() for target in targets],
        "total": len(targets),
        "scenario_packs": service.list_scenario_packs(),
    }


@router.post("/targets")
async def create_target(req: TargetRequest) -> dict[str, Any]:
    service = _require_service()
    try:
        target = service.create_target(req.to_target())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return target.to_dict()


@router.put("/targets/{target_id}")
async def update_target(target_id: str, req: TargetRequest) -> dict[str, Any]:
    service = _require_service()
    try:
        target = service.update_target(target_id, req.to_target())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return target.to_dict()


@router.delete("/targets/{target_id}")
async def delete_target(target_id: str) -> dict[str, Any]:
    service = _require_service()
    deleted = service.delete_target(target_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Target {target_id} not found")
    return {"target_id": target_id, "deleted": True}


@router.get("/windows")
async def get_windows(
    start_time: str | None = Query(default=None),
    hours: float = Query(default=6.0, gt=0, le=48),
    step_seconds: int = Query(default=30, ge=5, le=300),
    top_k: int = Query(default=20, ge=1, le=100),
    scenario_pack: str | None = Query(default=None),
) -> dict[str, Any]:
    service = _require_service()
    windows = service.preview_windows(
        start_time=start_time,
        hours=hours,
        step_seconds=step_seconds,
        top_k=top_k,
        scenario_pack=scenario_pack,
    )
    return {
        "policy_version": service.get_policy().policy_id,
        "targets": len(service.list_targets_for_scenario(scenario_pack=scenario_pack)),
        "scenario_pack": scenario_pack or "all",
        "windows": [window.to_dict() for window in windows],
    }


@router.post("/plan")
async def plan_encounters(req: PlanRequest) -> dict[str, Any]:
    service = _require_service()
    records = service.plan(
        start_time=req.start_time,
        hours=req.hours,
        step_seconds=req.step_seconds,
        top_k=req.top_k,
        scenario_pack=req.scenario_pack,
    )
    return {
        "policy_version": service.get_policy().policy_id,
        "scenario_pack": req.scenario_pack or "all",
        "decisions": [record.decision.to_dict() for record in records],
    }


@router.post("/evaluate")
async def evaluate_encounters(req: EvaluateRequest) -> dict[str, Any]:
    service = _require_service()
    evaluation = service.evaluate(
        start_time=req.start_time,
        hours=req.hours,
        step_seconds=req.step_seconds,
        top_k=req.top_k,
        materialize_top_k=req.materialize_top_k,
        scenario_pack=req.scenario_pack,
    )
    return evaluation.to_dict()


@router.get("/evaluations")
async def list_evaluations(
    limit: int = Query(default=10, ge=1, le=100),
    scenario_pack: str | None = Query(default=None),
) -> dict[str, Any]:
    service = _require_service()
    evaluations = service.list_evaluations(limit=limit, scenario_pack=scenario_pack)
    return {"evaluations": [evaluation.to_dict() for evaluation in evaluations], "total": len(evaluations)}


@router.get("/evidence")
async def get_scenario_evidence(
    scenario_pack: str = Query(..., min_length=1, max_length=128),
    case_limit: int = Query(default=3, ge=1, le=10),
    trace_limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    service = _require_service()
    evidence = service.build_scenario_evidence(
        scenario_pack=scenario_pack,
        case_limit=case_limit,
        trace_limit=trace_limit,
    )
    return evidence.to_dict()


@router.get("/decisions")
async def list_decisions(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    service = _require_service()
    decisions = service.list_decisions(limit=limit)
    return {"decisions": [decision.to_dict() for decision in decisions], "total": len(decisions)}


@router.post("/decision/{decision_id}/materialize")
async def materialize_decision(decision_id: str) -> dict[str, Any]:
    service = _require_service()
    try:
        record = service.materialize_decision(decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "decision_id": record.decision.decision_id,
        "stimulus_id": record.decision.stimulus_id,
        "status": "materialized",
    }


@router.post("/decision/{decision_id}/assess")
async def assess_decision(decision_id: str) -> dict[str, Any]:
    service = _require_service()
    try:
        return service.assess_materialized_decision(decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


class TrustFeedbackRequest(BaseModel):
    trust_details: dict[str, float] = Field(
        description="trust_details dict from decision_after (must include target_priority)"
    )
    learned_score: float = Field(ge=0.0, le=1.0)
    realized_utility: float = Field(
        ge=0.0, le=1.0,
        description="Operator usefulness_score; pass 0.0 for useful=False outcomes",
    )
    lr: float = Field(default=0.02, ge=1e-4, le=0.5, description="Learning rate")


@router.post("/trust-feedback")
async def apply_trust_feedback(req: TrustFeedbackRequest) -> dict[str, Any]:
    """Feed realized operator utility back into the WCLI trust model.

    This is the trust-layer TTT loop: after an operator reviews a trace and
    assigns a usefulness_score, call this endpoint to shift the model's
    learned_score_weights toward patterns that predict utility more accurately.

    The trust_details dict comes from decision_after.trust_details in the stored
    trace.  target_priority must be present (it is included automatically in
    traces produced after the Phase 1 trust_model update).
    """
    service = _require_service()
    trust_model = getattr(service.planner, "trust_model", None)
    if trust_model is None:
        raise HTTPException(
            status_code=503,
            detail="No trust model active on the current planner (scaffold-only mode)",
        )
    updated_weights = trust_model.online_update(
        trust_details=req.trust_details,
        learned_score=req.learned_score,
        realized_utility=req.realized_utility,
        lr=req.lr,
    )
    return {
        "updated_learned_score_weights": updated_weights,
        "snapshot": trust_model.get_weight_snapshot(),
    }


@router.get("/trust-weights")
async def get_trust_weights() -> dict[str, Any]:
    """Return current adaptive weights and drift since startup."""
    service = _require_service()
    trust_model = getattr(service.planner, "trust_model", None)
    if trust_model is None:
        raise HTTPException(status_code=503, detail="No trust model active")
    return trust_model.get_weight_snapshot()
