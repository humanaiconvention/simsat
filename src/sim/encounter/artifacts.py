from __future__ import annotations

import hashlib
import json

from .policy import policy_hash
from .schemas import DecisionArtifact, EncounterPolicy, EncounterRecord


def stable_json_hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(record: EncounterRecord, policy: EncounterPolicy) -> DecisionArtifact:
    window_hash = stable_json_hash(record.window.to_dict())
    feature_hash = stable_json_hash(record.features.to_dict())
    decision_hash = stable_json_hash(record.decision.to_dict())
    hashed_policy = policy_hash(policy)
    merkle_root = hashlib.sha256(
        f"{hashed_policy}:{window_hash}:{feature_hash}:{decision_hash}".encode("utf-8")
    ).hexdigest()
    return DecisionArtifact(
        artifact_id=f"art_{record.decision.decision_id}",
        decision_id=record.decision.decision_id,
        created_at=record.decision.created_at,
        model_id=policy.model_id,
        policy_hash=hashed_policy,
        window_hash=window_hash,
        feature_hash=feature_hash,
        decision_hash=decision_hash,
        merkle_root=merkle_root,
        action=record.decision.action,
    )
