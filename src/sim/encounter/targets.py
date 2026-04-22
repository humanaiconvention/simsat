from __future__ import annotations

import json
from pathlib import Path

from .schemas import TargetSpec

DEFAULT_TARGETS_PATH = Path(__file__).resolve().parents[1] / "data" / "encounter" / "targets.json"

_REPOSITORIES: dict[str, "TargetRepository"] = {}


class TargetRepository:
    def __init__(self, path: str | Path = DEFAULT_TARGETS_PATH) -> None:
        self.path = Path(path)

    def _load_payload(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "targets": []}
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("schema_version", 1)
        payload.setdefault("targets", [])
        return payload

    def _write_payload(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.path)

    def list_targets(self) -> list[TargetSpec]:
        payload = self._load_payload()
        return [TargetSpec.from_dict(raw) for raw in payload.get("targets", [])]

    def get_target(self, target_id: str) -> TargetSpec | None:
        for target in self.list_targets():
            if target.target_id == target_id:
                return target
        return None

    def replace_targets(self, targets: list[TargetSpec]) -> None:
        ids = [target.target_id for target in targets]
        if len(ids) != len(set(ids)):
            raise ValueError("Target IDs must be unique")
        payload = {"schema_version": 1, "targets": [target.to_dict() for target in targets]}
        self._write_payload(payload)

    def upsert_target(self, target: TargetSpec) -> TargetSpec:
        targets = self.list_targets()
        for idx, existing in enumerate(targets):
            if existing.target_id == target.target_id:
                targets[idx] = target
                self.replace_targets(targets)
                return target
        targets.append(target)
        self.replace_targets(targets)
        return target

    def delete_target(self, target_id: str) -> bool:
        targets = self.list_targets()
        filtered = [target for target in targets if target.target_id != target_id]
        if len(filtered) == len(targets):
            return False
        self.replace_targets(filtered)
        return True


def get_target_repository(path: str | Path = DEFAULT_TARGETS_PATH) -> TargetRepository:
    key = str(Path(path).resolve())
    if key not in _REPOSITORIES:
        _REPOSITORIES[key] = TargetRepository(path)
    return _REPOSITORIES[key]
