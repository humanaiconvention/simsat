from __future__ import annotations

import hashlib
import json

from .schemas import EncounterPolicy


def build_default_policy() -> EncounterPolicy:
    return EncounterPolicy()


def policy_hash(policy: EncounterPolicy) -> str:
    payload = json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
