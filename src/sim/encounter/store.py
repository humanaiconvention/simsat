from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from .schemas import EncounterEvaluation, EncounterEvaluationPrimitive, EncounterRecord

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "encounter"

_STORES: dict[str, "EncounterStore"] = {}


class EncounterStore:
    def __init__(self, base_dir: str | Path = DEFAULT_STORE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / "records.json"
        self.evaluations_path = self.base_dir / "evaluations.json"
        self.sqlite_path = self.base_dir / "encounter.sqlite3"
        self._lock = threading.RLock()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._maybe_import_legacy_json()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS encounter_records (
                    decision_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_encounter_records_created_at
                    ON encounter_records(created_at DESC);

                CREATE TABLE IF NOT EXISTS encounter_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    scenario_pack TEXT NOT NULL,
                    full_cache_key TEXT NOT NULL UNIQUE,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_encounter_evaluations_created_at
                    ON encounter_evaluations(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_encounter_evaluations_scenario_created
                    ON encounter_evaluations(scenario_pack, created_at DESC);

                CREATE TABLE IF NOT EXISTS encounter_evaluation_primitives (
                    primitive_id TEXT PRIMARY KEY,
                    cache_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    scenario_pack TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_encounter_primitives_scenario_created
                    ON encounter_evaluation_primitives(scenario_pack, created_at DESC);
                """
            )

    def _safe_load_json(self, path: Path, key: str) -> list[dict]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        items = payload.get(key, []) if isinstance(payload, dict) else []
        return list(items) if isinstance(items, list) else []

    def _table_has_rows(self, table: str) -> bool:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
        return row is not None

    def _maybe_import_legacy_json(self) -> None:
        with self._lock:
            if not self._table_has_rows("encounter_records"):
                for raw in self._safe_load_json(self.path, "records"):
                    decision_id = str(raw.get("decision", {}).get("decision_id", ""))
                    created_at = str(raw.get("decision", {}).get("created_at", ""))
                    target_id = str(raw.get("decision", {}).get("target_id", ""))
                    if not decision_id:
                        continue
                    with closing(self._connect()) as connection, connection:
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO encounter_records(decision_id, created_at, target_id, raw_json)
                            VALUES (?, ?, ?, ?)
                            """,
                            (decision_id, created_at, target_id, json.dumps(raw)),
                        )
            if not self._table_has_rows("encounter_evaluations"):
                for raw in self._safe_load_json(self.evaluations_path, "evaluations"):
                    evaluation = EncounterEvaluation.from_dict(raw)
                    with closing(self._connect()) as connection, connection:
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO encounter_evaluations(
                                evaluation_id,
                                created_at,
                                scenario_pack,
                                full_cache_key,
                                raw_json
                            )
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                evaluation.evaluation_id,
                                evaluation.created_at,
                                evaluation.scenario_pack,
                                self.build_full_cache_key(
                                    policy_version=str(evaluation.parameters.get("policy_version", "")),
                                    scenario_pack=evaluation.scenario_pack,
                                    start_time=str(evaluation.parameters.get("start_time", "")),
                                    hours=float(evaluation.parameters.get("hours", 0.0)),
                                    step_seconds=int(evaluation.parameters.get("step_seconds", 0)),
                                    top_k=int(evaluation.parameters.get("top_k", 0)),
                                    materialize_top_k=int(evaluation.parameters.get("materialize_top_k", 0)),
                                ),
                                json.dumps(evaluation.to_dict()),
                            ),
                        )

    @staticmethod
    def build_primitive_cache_key(
        *,
        policy_version: str,
        scenario_pack: str,
        start_time: str,
        hours: float,
        step_seconds: int,
        top_k: int,
    ) -> str:
        payload = {
            "policy_version": policy_version,
            "scenario_pack": scenario_pack,
            "start_time": start_time,
            "hours": hours,
            "step_seconds": step_seconds,
            "top_k": top_k,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def build_full_cache_key(
        cls,
        *,
        policy_version: str,
        scenario_pack: str,
        start_time: str,
        hours: float,
        step_seconds: int,
        top_k: int,
        materialize_top_k: int,
    ) -> str:
        payload = json.loads(
            cls.build_primitive_cache_key(
                policy_version=policy_version,
                scenario_pack=scenario_pack,
                start_time=start_time,
                hours=hours,
                step_seconds=step_seconds,
                top_k=top_k,
            )
        )
        payload["materialize_top_k"] = materialize_top_k
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def save_record(self, record: EncounterRecord) -> EncounterRecord:
        raw_json = json.dumps(record.to_dict())
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO encounter_records(decision_id, created_at, target_id, raw_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.decision.decision_id,
                    record.decision.created_at,
                    record.decision.target_id,
                    raw_json,
                ),
            )
        return record

    def get_record(self, decision_id: str) -> EncounterRecord | None:
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT raw_json FROM encounter_records WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            return None
        return EncounterRecord.from_dict(json.loads(str(row["raw_json"])))

    def list_records(self, limit: int = 50) -> list[EncounterRecord]:
        with self._lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT raw_json
                FROM encounter_records
                ORDER BY created_at DESC, decision_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [EncounterRecord.from_dict(json.loads(str(row["raw_json"]))) for row in rows]

    def save_evaluation(self, evaluation: EncounterEvaluation, *, full_cache_key: str | None = None) -> EncounterEvaluation:
        raw_json = json.dumps(evaluation.to_dict())
        cache_key = full_cache_key or self.build_full_cache_key(
            policy_version=str(evaluation.parameters.get("policy_version", "")),
            scenario_pack=evaluation.scenario_pack,
            start_time=str(evaluation.parameters.get("start_time", "")),
            hours=float(evaluation.parameters.get("hours", 0.0)),
            step_seconds=int(evaluation.parameters.get("step_seconds", 0)),
            top_k=int(evaluation.parameters.get("top_k", 0)),
            materialize_top_k=int(evaluation.parameters.get("materialize_top_k", 0)),
        )
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO encounter_evaluations(
                    evaluation_id,
                    created_at,
                    scenario_pack,
                    full_cache_key,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    evaluation.evaluation_id,
                    evaluation.created_at,
                    evaluation.scenario_pack,
                    cache_key,
                    raw_json,
                ),
            )
        return evaluation

    def get_evaluation_by_cache_key(self, full_cache_key: str) -> EncounterEvaluation | None:
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT raw_json FROM encounter_evaluations WHERE full_cache_key = ?",
                (full_cache_key,),
            ).fetchone()
        if row is None:
            return None
        return EncounterEvaluation.from_dict(json.loads(str(row["raw_json"])))

    def list_evaluations(self, limit: int | None = 20, scenario_pack: str | None = None) -> list[EncounterEvaluation]:
        params: list[object] = []
        query = "SELECT raw_json FROM encounter_evaluations"
        if scenario_pack not in {None, "", "all"}:
            query += " WHERE scenario_pack = ?"
            params.append(scenario_pack)
        query += " ORDER BY created_at DESC, evaluation_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [EncounterEvaluation.from_dict(json.loads(str(row["raw_json"]))) for row in rows]

    def save_evaluation_primitive(self, primitive: EncounterEvaluationPrimitive) -> EncounterEvaluationPrimitive:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO encounter_evaluation_primitives(
                    primitive_id,
                    cache_key,
                    created_at,
                    scenario_pack,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    primitive.primitive_id,
                    primitive.cache_key,
                    primitive.created_at,
                    primitive.scenario_pack,
                    json.dumps(primitive.to_dict()),
                ),
            )
        return primitive

    def get_evaluation_primitive(self, cache_key: str) -> EncounterEvaluationPrimitive | None:
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT raw_json FROM encounter_evaluation_primitives WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return EncounterEvaluationPrimitive.from_dict(json.loads(str(row["raw_json"])))


def get_encounter_store(base_dir: str | Path = DEFAULT_STORE_DIR) -> EncounterStore:
    key = str(Path(base_dir).resolve())
    if key not in _STORES:
        _STORES[key] = EncounterStore(base_dir)
    return _STORES[key]
