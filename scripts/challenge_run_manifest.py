from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "src" / "sim" / "data" / "challenge_runs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class ChallengeRunManifest:
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    created_at: str = field(default_factory=_utc_now)
    runner: str = "challenge_offline_batch"
    scenario_policy: str = "competition"
    scenario_hours: dict[str, float] = field(default_factory=dict)
    lanes: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    status: str = "running"
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChallengeRunManifestStore:
    def __init__(self, base_dir: str | Path = DEFAULT_RUNS_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.base_dir / "challenge_runs.sqlite3"
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS challenge_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    runner TEXT NOT NULL,
                    scenario_policy TEXT NOT NULL,
                    status TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_challenge_runs_created_at
                    ON challenge_runs(created_at DESC);
                """
            )

    def save(self, manifest: ChallengeRunManifest) -> ChallengeRunManifest:
        raw_json = json.dumps(manifest.to_dict(), indent=2)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO challenge_runs(run_id, created_at, runner, scenario_policy, status, raw_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.run_id,
                    manifest.created_at,
                    manifest.runner,
                    manifest.scenario_policy,
                    manifest.status,
                    raw_json,
                ),
            )
        (self.base_dir / f"{manifest.run_id}.json").write_text(raw_json, encoding="utf-8")
        return manifest

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT raw_json
                FROM challenge_runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(str(row["raw_json"])) for row in rows]
