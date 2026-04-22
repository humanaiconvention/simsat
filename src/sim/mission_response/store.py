from __future__ import annotations

import json
import threading
from pathlib import Path

from .schemas import MissionResponseAction, MissionResponseOutcome, MissionUtilitySnapshot

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "mission_response"


class MissionResponseStore:
    def __init__(self, base_dir: str | Path = DEFAULT_STORE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.actions_path = self.base_dir / "actions.json"
        self.outcomes_path = self.base_dir / "outcomes.json"
        self.snapshots_path = self.base_dir / "snapshots.json"
        self._lock = threading.Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _load_payload(self, path: Path, key: str) -> dict:
        if not path.exists():
            return {"schema_version": 1, key: []}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("schema_version", 1)
        payload.setdefault(key, [])
        return payload

    def _write_payload(self, path: Path, payload: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(path)

    def save_action(self, action: MissionResponseAction) -> MissionResponseAction:
        with self._lock:
            payload = self._load_payload(self.actions_path, "actions")
            actions = payload.get("actions", [])
            updated = False
            for idx, raw in enumerate(actions):
                if raw.get("action_id") == action.action_id:
                    actions[idx] = action.to_dict()
                    updated = True
                    break
            if not updated:
                actions.append(action.to_dict())
            payload["actions"] = actions
            self._write_payload(self.actions_path, payload)
        return action

    def list_actions(self, limit: int = 50) -> list[MissionResponseAction]:
        with self._lock:
            payload = self._load_payload(self.actions_path, "actions")
        actions = [MissionResponseAction.from_dict(raw) for raw in payload.get("actions", [])]
        actions.sort(key=lambda action: action.created_at, reverse=True)
        return actions[:limit]

    def get_action(self, action_id: str) -> MissionResponseAction | None:
        with self._lock:
            payload = self._load_payload(self.actions_path, "actions")
        for raw in payload.get("actions", []):
            if raw.get("action_id") == action_id:
                return MissionResponseAction.from_dict(raw)
        return None

    def get_action_for_trace(self, trace_id: str) -> MissionResponseAction | None:
        with self._lock:
            payload = self._load_payload(self.actions_path, "actions")
        matches = [
            MissionResponseAction.from_dict(raw)
            for raw in payload.get("actions", [])
            if raw.get("trace_id") == trace_id
        ]
        if not matches:
            return None
        matches.sort(key=lambda action: action.created_at, reverse=True)
        return matches[0]

    def save_outcome(self, outcome: MissionResponseOutcome) -> MissionResponseOutcome:
        with self._lock:
            payload = self._load_payload(self.outcomes_path, "outcomes")
            outcomes = payload.get("outcomes", [])
            updated = False
            for idx, raw in enumerate(outcomes):
                if raw.get("outcome_id") == outcome.outcome_id:
                    outcomes[idx] = outcome.to_dict()
                    updated = True
                    break
            if not updated:
                outcomes.append(outcome.to_dict())
            payload["outcomes"] = outcomes
            self._write_payload(self.outcomes_path, payload)
        return outcome

    def list_outcomes(self, limit: int = 50) -> list[MissionResponseOutcome]:
        with self._lock:
            payload = self._load_payload(self.outcomes_path, "outcomes")
        outcomes = [MissionResponseOutcome.from_dict(raw) for raw in payload.get("outcomes", [])]
        outcomes.sort(key=lambda outcome: outcome.executed_at, reverse=True)
        return outcomes[:limit]

    def get_outcome_for_action(self, action_id: str) -> MissionResponseOutcome | None:
        with self._lock:
            payload = self._load_payload(self.outcomes_path, "outcomes")
        matches = [
            MissionResponseOutcome.from_dict(raw)
            for raw in payload.get("outcomes", [])
            if raw.get("action_id") == action_id
        ]
        if not matches:
            return None
        matches.sort(key=lambda outcome: outcome.executed_at, reverse=True)
        return matches[0]

    def save_snapshot(self, snapshot: MissionUtilitySnapshot) -> MissionUtilitySnapshot:
        with self._lock:
            payload = self._load_payload(self.snapshots_path, "snapshots")
            snapshots = payload.get("snapshots", [])
            updated = False
            for idx, raw in enumerate(snapshots):
                if raw.get("mission_id") == snapshot.mission_id and raw.get("scenario_pack") == snapshot.scenario_pack:
                    snapshots[idx] = snapshot.to_dict()
                    updated = True
                    break
            if not updated:
                snapshots.append(snapshot.to_dict())
            payload["snapshots"] = snapshots
            self._write_payload(self.snapshots_path, payload)
        return snapshot

    def get_snapshot(self, mission_id: str, scenario_pack: str) -> MissionUtilitySnapshot | None:
        with self._lock:
            payload = self._load_payload(self.snapshots_path, "snapshots")
        for raw in payload.get("snapshots", []):
            if raw.get("mission_id") == mission_id and raw.get("scenario_pack") == scenario_pack:
                return MissionUtilitySnapshot.from_dict(raw)
        return None
