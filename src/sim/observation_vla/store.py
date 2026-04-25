from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from .schemas import ObservationAssessmentRecord, ObservationMemoryState, ObservationOutcome, ObservationTraceRecord, SubmissionCasePin

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "observation_vla"


class ObservationStore:
    def __init__(self, base_dir: str | Path = DEFAULT_STORE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / "assessments.json"
        self.traces_path = self.base_dir / "traces.json"
        self.outcomes_path = self.base_dir / "outcomes.json"
        self.memory_path = self.base_dir / "memory.json"
        self.submission_cases_path = self.base_dir / "submission_cases.json"
        self.sqlite_path = self.base_dir / "observation_vla.sqlite3"
        self._lock = threading.RLock()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._migrate_submission_cases_pk()
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
                CREATE TABLE IF NOT EXISTS observation_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    scenario_pack TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observation_assessments_created_at
                    ON observation_assessments(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_assessments_scenario_created
                    ON observation_assessments(scenario_pack, created_at DESC);

                CREATE TABLE IF NOT EXISTS observation_traces (
                    trace_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    scenario_pack TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    outcome_id TEXT,
                    runtime_mode TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observation_traces_created_at
                    ON observation_traces(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_traces_scenario_created
                    ON observation_traces(scenario_pack, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_traces_target_created
                    ON observation_traces(target_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_traces_decision_id
                    ON observation_traces(decision_id);
                CREATE INDEX IF NOT EXISTS idx_observation_traces_runtime_mode
                    ON observation_traces(runtime_mode, created_at DESC);

                CREATE TABLE IF NOT EXISTS observation_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    registered_at TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    assessment_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    mission_id TEXT NOT NULL,
                    label_source TEXT NOT NULL,
                    reviewer TEXT,
                    review_status TEXT NOT NULL,
                    is_current INTEGER NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observation_outcomes_registered_at
                    ON observation_outcomes(registered_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_outcomes_trace_current
                    ON observation_outcomes(trace_id, is_current, registered_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_outcomes_label_current
                    ON observation_outcomes(label_source, is_current, registered_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_outcomes_review_status
                    ON observation_outcomes(review_status, registered_at DESC);

                CREATE TABLE IF NOT EXISTS observation_memory_states (
                    state_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observation_memory_scope_updated
                    ON observation_memory_states(scope, last_updated DESC);

                CREATE TABLE IF NOT EXISTS observation_submission_cases (
                    trace_id TEXT PRIMARY KEY,
                    scenario_pack TEXT NOT NULL,
                    pinned_at TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    target_label TEXT NOT NULL,
                    outcome_id TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observation_submission_cases_pinned_at
                    ON observation_submission_cases(pinned_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_submission_cases_scenario_pack
                    ON observation_submission_cases(scenario_pack, pinned_at DESC);
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

    def _migrate_submission_cases_pk(self) -> None:
        """Migrate observation_submission_cases from scenario_pack PK → trace_id PK.

        Safe to call every startup — detects whether migration is needed via
        PRAGMA table_info, and no-ops if already on the new schema.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            cols = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(observation_submission_cases)"
                ).fetchall()
            }
            if not cols:
                return  # table not yet created
            # Check if scenario_pack is still the primary key (pk column = 1)
            pk_cols = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(observation_submission_cases)"
                ).fetchall()
                if row[5] == 1  # pk flag
            }
            if "trace_id" in pk_cols:
                return  # already migrated
            # Need to migrate: old PK = scenario_pack, new PK = trace_id
            connection.executescript(
                """
                ALTER TABLE observation_submission_cases
                    RENAME TO observation_submission_cases_old;
                CREATE TABLE observation_submission_cases (
                    trace_id TEXT PRIMARY KEY,
                    scenario_pack TEXT NOT NULL,
                    pinned_at TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    target_label TEXT NOT NULL,
                    outcome_id TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observation_submission_cases_pinned_at
                    ON observation_submission_cases(pinned_at DESC);
                CREATE INDEX IF NOT EXISTS idx_observation_submission_cases_scenario_pack
                    ON observation_submission_cases(scenario_pack, pinned_at DESC);
                INSERT INTO observation_submission_cases
                    SELECT trace_id, scenario_pack, pinned_at, target_id, target_label, outcome_id, raw_json
                    FROM observation_submission_cases_old;
                DROP TABLE observation_submission_cases_old;
                """
            )

    _VALID_TABLES = frozenset({
        "observation_assessments",
        "observation_traces",
        "observation_outcomes",
        "observation_memory_states",
        "observation_submission_cases",
    })

    def _table_has_rows(self, table: str) -> bool:
        if table not in self._VALID_TABLES:
            raise ValueError(f"Unknown table: {table!r}")
        with closing(self._connect()) as connection, connection:
            row = connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()  # noqa: S608
        return row is not None

    def _maybe_import_legacy_json(self) -> None:
        with self._lock:
            if not self._table_has_rows("observation_assessments"):
                for raw in self._safe_load_json(self.path, "records"):
                    record = ObservationAssessmentRecord.from_dict(raw)
                    self.save_record(record)
            if not self._table_has_rows("observation_traces"):
                for raw in self._safe_load_json(self.traces_path, "traces"):
                    trace = ObservationTraceRecord.from_dict(raw)
                    self.save_trace(trace)
            if not self._table_has_rows("observation_outcomes"):
                for raw in self._safe_load_json(self.outcomes_path, "outcomes"):
                    outcome = ObservationOutcome.from_dict(raw)
                    self.save_outcome(outcome)
            if not self._table_has_rows("observation_memory_states"):
                states = [ObservationMemoryState.from_dict(raw) for raw in self._safe_load_json(self.memory_path, "states")]
                if states:
                    self.replace_memory_states(states)
            if not self._table_has_rows("observation_submission_cases"):
                for raw in self._safe_load_json(self.submission_cases_path, "cases"):
                    pin = SubmissionCasePin.from_dict(raw)
                    self.save_submission_case(pin)

    def save_record(self, record: ObservationAssessmentRecord) -> ObservationAssessmentRecord:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO observation_assessments(
                    assessment_id,
                    created_at,
                    scenario_pack,
                    target_id,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.assessment.assessment_id,
                    record.assessment.created_at,
                    record.sample.scenario_pack,
                    record.sample.target_id,
                    json.dumps(record.to_dict()),
                ),
            )
        return record

    def get_record(self, assessment_id: str) -> ObservationAssessmentRecord | None:
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT raw_json FROM observation_assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
        if row is None:
            return None
        return ObservationAssessmentRecord.from_dict(json.loads(str(row["raw_json"])))

    def list_records(self, limit: int = 50, scenario_pack: str | None = None) -> list[ObservationAssessmentRecord]:
        query = "SELECT raw_json FROM observation_assessments"
        params: list[object] = []
        if scenario_pack not in {None, "", "all"}:
            query += " WHERE scenario_pack = ?"
            params.append(scenario_pack)
        query += " ORDER BY created_at DESC, assessment_id DESC LIMIT ?"
        params.append(limit)
        with self._lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [ObservationAssessmentRecord.from_dict(json.loads(str(row["raw_json"]))) for row in rows]

    def save_trace(self, trace: ObservationTraceRecord) -> ObservationTraceRecord:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO observation_traces(
                    trace_id,
                    created_at,
                    scenario_pack,
                    target_id,
                    decision_id,
                    outcome_id,
                    runtime_mode,
                    recommended_action,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.created_at,
                    trace.scenario_pack,
                    trace.target_id,
                    trace.decision_id,
                    trace.outcome_id,
                    trace.assessment.runtime_mode,
                    trace.assessment.recommended_action,
                    json.dumps(trace.to_dict()),
                ),
            )
        return trace

    def list_traces(self, limit: int = 50, scenario_pack: str | None = None) -> list[ObservationTraceRecord]:
        query = "SELECT raw_json FROM observation_traces"
        params: list[object] = []
        if scenario_pack not in {None, "", "all"}:
            query += " WHERE scenario_pack = ?"
            params.append(scenario_pack)
        query += " ORDER BY created_at DESC, trace_id DESC LIMIT ?"
        params.append(limit)
        with self._lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [ObservationTraceRecord.from_dict(json.loads(str(row["raw_json"]))) for row in rows]

    def get_trace(self, trace_id: str) -> ObservationTraceRecord | None:
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT raw_json FROM observation_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return ObservationTraceRecord.from_dict(json.loads(str(row["raw_json"])))

    def save_outcome(self, outcome: ObservationOutcome) -> ObservationOutcome:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO observation_outcomes(
                    outcome_id,
                    registered_at,
                    trace_id,
                    assessment_id,
                    decision_id,
                    mission_id,
                    label_source,
                    reviewer,
                    review_status,
                    is_current,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.outcome_id,
                    outcome.registered_at,
                    outcome.trace_id,
                    outcome.assessment_id,
                    outcome.decision_id,
                    outcome.mission_id,
                    outcome.label_source,
                    outcome.reviewer,
                    outcome.review_status,
                    1 if outcome.is_current else 0,
                    json.dumps(outcome.to_dict()),
                ),
            )
        return outcome

    def list_outcomes(
        self,
        limit: int = 50,
        current_only: bool = True,
        scenario_pack: str | None = None,
    ) -> list[ObservationOutcome]:
        query = """
            SELECT o.raw_json
            FROM observation_outcomes o
            LEFT JOIN observation_traces t ON t.trace_id = o.trace_id
        """
        clauses: list[str] = []
        params: list[object] = []
        if current_only:
            clauses.append("o.is_current = 1")
        if scenario_pack not in {None, "", "all"}:
            clauses.append("t.scenario_pack = ?")
            params.append(scenario_pack)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY o.registered_at DESC, o.outcome_id DESC LIMIT ?"
        params.append(limit)
        with self._lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [ObservationOutcome.from_dict(json.loads(str(row["raw_json"]))) for row in rows]

    def get_outcome(self, outcome_id: str) -> ObservationOutcome | None:
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT raw_json FROM observation_outcomes WHERE outcome_id = ?",
                (outcome_id,),
            ).fetchone()
        if row is None:
            return None
        return ObservationOutcome.from_dict(json.loads(str(row["raw_json"])))

    def get_outcome_for_trace(self, trace_id: str) -> ObservationOutcome | None:
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT raw_json
                FROM observation_outcomes
                WHERE trace_id = ?
                ORDER BY is_current DESC, registered_at DESC, outcome_id DESC
                LIMIT 1
                """,
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return ObservationOutcome.from_dict(json.loads(str(row["raw_json"])))

    def replace_memory_states(self, states: list[ObservationMemoryState]) -> None:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM observation_memory_states")
            connection.executemany(
                """
                INSERT INTO observation_memory_states(state_id, scope, scope_id, last_updated, raw_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        state.state_id,
                        state.scope,
                        state.scope_id,
                        state.last_updated,
                        json.dumps(state.to_dict()),
                    )
                    for state in states
                ],
            )

    def save_memory_state(self, state: ObservationMemoryState) -> ObservationMemoryState:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO observation_memory_states(state_id, scope, scope_id, last_updated, raw_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    state.state_id,
                    state.scope,
                    state.scope_id,
                    state.last_updated,
                    json.dumps(state.to_dict()),
                ),
            )
        return state

    def get_memory_state(self, scope: str, scope_id: str) -> ObservationMemoryState | None:
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT raw_json
                FROM observation_memory_states
                WHERE scope = ? AND scope_id = ?
                LIMIT 1
                """,
                (scope, scope_id),
            ).fetchone()
        if row is None:
            return None
        return ObservationMemoryState.from_dict(json.loads(str(row["raw_json"])))

    def list_memory_states(self, scope: str | None = None, limit: int = 50) -> list[ObservationMemoryState]:
        query = "SELECT raw_json FROM observation_memory_states"
        params: list[object] = []
        if scope:
            query += " WHERE scope = ?"
            params.append(scope)
        query += " ORDER BY last_updated DESC, state_id DESC LIMIT ?"
        params.append(limit)
        with self._lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [ObservationMemoryState.from_dict(json.loads(str(row["raw_json"]))) for row in rows]

    def save_submission_case(self, pin: SubmissionCasePin) -> SubmissionCasePin:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO observation_submission_cases(
                    trace_id,
                    scenario_pack,
                    pinned_at,
                    target_id,
                    target_label,
                    outcome_id,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pin.trace_id,
                    pin.scenario_pack,
                    pin.pinned_at,
                    pin.target_id,
                    pin.target_label,
                    pin.outcome_id,
                    json.dumps(pin.to_dict()),
                ),
            )
        return pin

    def get_submission_case_by_trace(self, trace_id: str) -> SubmissionCasePin | None:
        """Look up a pinned case by its trace_id (exact match)."""
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT raw_json FROM observation_submission_cases WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return SubmissionCasePin.from_dict(json.loads(str(row["raw_json"])))

    def get_submission_case(self, scenario_pack: str) -> SubmissionCasePin | None:
        """Return the most-recently-pinned case for a scenario_pack.

        Backward-compatible: callers that need pack-level lookup still work.
        For exact trace lookup, use get_submission_case_by_trace(trace_id).
        """
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                """SELECT raw_json FROM observation_submission_cases
                   WHERE scenario_pack = ?
                   ORDER BY pinned_at DESC LIMIT 1""",
                (scenario_pack,),
            ).fetchone()
        if row is None:
            return None
        return SubmissionCasePin.from_dict(json.loads(str(row["raw_json"])))

    def list_submission_cases(self) -> list[SubmissionCasePin]:
        with self._lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT raw_json
                FROM observation_submission_cases
                ORDER BY pinned_at DESC, scenario_pack DESC
                """
            ).fetchall()
        return [SubmissionCasePin.from_dict(json.loads(str(row["raw_json"]))) for row in rows]
