from __future__ import annotations

import json
import threading
from pathlib import Path

from .schemas import EncounterEvaluation, EncounterRecord

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "encounter"

_STORES: dict[str, "EncounterStore"] = {}


class EncounterStore:
    def __init__(self, base_dir: str | Path = DEFAULT_STORE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / "records.json"
        self.evaluations_path = self.base_dir / "evaluations.json"
        self._lock = threading.Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _load_payload(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "records": []}
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "records" not in payload:
            return {"schema_version": 1, "records": payload.get("records", [])}
        payload.setdefault("schema_version", 1)
        payload.setdefault("records", [])
        return payload

    def _write_payload(self, payload: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.path)

    def _load_evaluations_payload(self) -> dict:
        if not self.evaluations_path.exists():
            return {"schema_version": 1, "evaluations": []}
        with self.evaluations_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "evaluations" not in payload:
            return {"schema_version": 1, "evaluations": payload.get("evaluations", [])}
        payload.setdefault("schema_version", 1)
        payload.setdefault("evaluations", [])
        return payload

    def _write_evaluations_payload(self, payload: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.evaluations_path.with_suffix(f"{self.evaluations_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.evaluations_path)

    def save_record(self, record: EncounterRecord) -> EncounterRecord:
        with self._lock:
            payload = self._load_payload()
            records = payload.get("records", [])
            updated = False
            for idx, raw in enumerate(records):
                if raw.get("decision", {}).get("decision_id") == record.decision.decision_id:
                    records[idx] = record.to_dict()
                    updated = True
                    break
            if not updated:
                records.append(record.to_dict())
            payload["records"] = records
            self._write_payload(payload)
        return record

    def get_record(self, decision_id: str) -> EncounterRecord | None:
        with self._lock:
            payload = self._load_payload()
        for raw in payload.get("records", []):
            if raw.get("decision", {}).get("decision_id") == decision_id:
                return EncounterRecord.from_dict(raw)
        return None

    def list_records(self, limit: int = 50) -> list[EncounterRecord]:
        with self._lock:
            payload = self._load_payload()
        records = [EncounterRecord.from_dict(raw) for raw in payload.get("records", [])]
        records.sort(key=lambda record: record.decision.created_at, reverse=True)
        return records[:limit]

    def save_evaluation(self, evaluation: EncounterEvaluation) -> EncounterEvaluation:
        with self._lock:
            payload = self._load_evaluations_payload()
            evaluations = payload.get("evaluations", [])
            updated = False
            for idx, raw in enumerate(evaluations):
                if raw.get("evaluation_id") == evaluation.evaluation_id:
                    evaluations[idx] = evaluation.to_dict()
                    updated = True
                    break
            if not updated:
                evaluations.append(evaluation.to_dict())
            payload["evaluations"] = evaluations
            self._write_evaluations_payload(payload)
        return evaluation

    def list_evaluations(self, limit: int | None = 20) -> list[EncounterEvaluation]:
        with self._lock:
            payload = self._load_evaluations_payload()
        evaluations = [EncounterEvaluation.from_dict(raw) for raw in payload.get("evaluations", [])]
        evaluations.sort(key=lambda evaluation: evaluation.created_at, reverse=True)
        if limit is None:
            return evaluations
        return evaluations[:limit]


def get_encounter_store(base_dir: str | Path = DEFAULT_STORE_DIR) -> EncounterStore:
    key = str(Path(base_dir).resolve())
    if key not in _STORES:
        _STORES[key] = EncounterStore(base_dir)
    return _STORES[key]
