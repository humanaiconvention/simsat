from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from .schemas import MissionResponseAction, MissionResponseOutcome, MissionUtilitySnapshot

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "mission_response"


class MissionResponseStore:
    def __init__(self, base_dir: str | Path = DEFAULT_STORE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.sqlite_path = self.base_dir / "mission_response.sqlite3"
        # Legacy JSON paths — read-only, used only during one-time import
        self._actions_json = self.base_dir / "actions.json"
        self._outcomes_json = self.base_dir / "outcomes.json"
        self._snapshots_json = self.base_dir / "snapshots.json"
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

                CREATE TABLE IF NOT EXISTS mission_response_actions (
                    action_id    TEXT PRIMARY KEY,
                    created_at   TEXT NOT NULL,
                    trace_id     TEXT,
                    scenario_pack TEXT NOT NULL,
                    raw_json     TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mr_actions_created_at
                    ON mission_response_actions(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_mr_actions_trace_id
                    ON mission_response_actions(trace_id);
                CREATE INDEX IF NOT EXISTS idx_mr_actions_scenario
                    ON mission_response_actions(scenario_pack, created_at DESC);

                CREATE TABLE IF NOT EXISTS mission_response_outcomes (
                    outcome_id   TEXT PRIMARY KEY,
                    action_id    TEXT NOT NULL,
                    executed_at  TEXT NOT NULL,
                    raw_json     TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mr_outcomes_action_id
                    ON mission_response_outcomes(action_id);
                CREATE INDEX IF NOT EXISTS idx_mr_outcomes_executed_at
                    ON mission_response_outcomes(executed_at DESC);

                CREATE TABLE IF NOT EXISTS mission_response_snapshots (
                    mission_id    TEXT NOT NULL,
                    scenario_pack TEXT NOT NULL,
                    raw_json      TEXT NOT NULL,
                    PRIMARY KEY (mission_id, scenario_pack)
                );
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
            if not self._table_has_rows("mission_response_actions"):
                for raw in self._safe_load_json(self._actions_json, "actions"):
                    action = MissionResponseAction.from_dict(raw)
                    with closing(self._connect()) as connection, connection:
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO mission_response_actions
                                (action_id, created_at, trace_id, scenario_pack, raw_json)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                action.action_id,
                                action.created_at,
                                action.trace_id,
                                action.scenario_pack,
                                json.dumps(raw),
                            ),
                        )

            if not self._table_has_rows("mission_response_outcomes"):
                for raw in self._safe_load_json(self._outcomes_json, "outcomes"):
                    outcome = MissionResponseOutcome.from_dict(raw)
                    with closing(self._connect()) as connection, connection:
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO mission_response_outcomes
                                (outcome_id, action_id, executed_at, raw_json)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                outcome.outcome_id,
                                outcome.action_id,
                                outcome.executed_at,
                                json.dumps(raw),
                            ),
                        )

            if not self._table_has_rows("mission_response_snapshots"):
                for raw in self._safe_load_json(self._snapshots_json, "snapshots"):
                    snapshot = MissionUtilitySnapshot.from_dict(raw)
                    with closing(self._connect()) as connection, connection:
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO mission_response_snapshots
                                (mission_id, scenario_pack, raw_json)
                            VALUES (?, ?, ?)
                            """,
                            (snapshot.mission_id, snapshot.scenario_pack, json.dumps(raw)),
                        )

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def save_action(self, action: MissionResponseAction) -> MissionResponseAction:
        with self._lock:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO mission_response_actions
                        (action_id, created_at, trace_id, scenario_pack, raw_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        action.action_id,
                        action.created_at,
                        action.trace_id,
                        action.scenario_pack,
                        json.dumps(action.to_dict()),
                    ),
                )
        return action

    def list_actions(self, limit: int = 50) -> list[MissionResponseAction]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT raw_json FROM mission_response_actions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [MissionResponseAction.from_dict(json.loads(row["raw_json"])) for row in rows]

    def get_action(self, action_id: str) -> MissionResponseAction | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT raw_json FROM mission_response_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            return None
        return MissionResponseAction.from_dict(json.loads(row["raw_json"]))

    def get_action_for_trace(self, trace_id: str) -> MissionResponseAction | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT raw_json FROM mission_response_actions
                WHERE trace_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return MissionResponseAction.from_dict(json.loads(row["raw_json"]))

    # -------------------------------------------------------------------------
    # Outcomes
    # -------------------------------------------------------------------------

    def save_outcome(self, outcome: MissionResponseOutcome) -> MissionResponseOutcome:
        with self._lock:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO mission_response_outcomes
                        (outcome_id, action_id, executed_at, raw_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        outcome.outcome_id,
                        outcome.action_id,
                        outcome.executed_at,
                        json.dumps(outcome.to_dict()),
                    ),
                )
        return outcome

    def list_outcomes(self, limit: int = 50) -> list[MissionResponseOutcome]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT raw_json FROM mission_response_outcomes ORDER BY executed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [MissionResponseOutcome.from_dict(json.loads(row["raw_json"])) for row in rows]

    def get_outcome_for_action(self, action_id: str) -> MissionResponseOutcome | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT raw_json FROM mission_response_outcomes
                WHERE action_id = ?
                ORDER BY executed_at DESC LIMIT 1
                """,
                (action_id,),
            ).fetchone()
        if row is None:
            return None
        return MissionResponseOutcome.from_dict(json.loads(row["raw_json"]))

    # -------------------------------------------------------------------------
    # Snapshots
    # -------------------------------------------------------------------------

    def save_snapshot(self, snapshot: MissionUtilitySnapshot) -> MissionUtilitySnapshot:
        with self._lock:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO mission_response_snapshots
                        (mission_id, scenario_pack, raw_json)
                    VALUES (?, ?, ?)
                    """,
                    (snapshot.mission_id, snapshot.scenario_pack, json.dumps(snapshot.to_dict())),
                )
        return snapshot

    def get_snapshot(self, mission_id: str, scenario_pack: str) -> MissionUtilitySnapshot | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT raw_json FROM mission_response_snapshots
                WHERE mission_id = ? AND scenario_pack = ?
                """,
                (mission_id, scenario_pack),
            ).fetchone()
        if row is None:
            return None
        return MissionUtilitySnapshot.from_dict(json.loads(row["raw_json"]))
