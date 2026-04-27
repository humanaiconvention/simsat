#!/usr/bin/env python3
"""Snapshot the live ObservationVLA SQLite store back to the JSON files.

The export pipeline (scripts/export_gemma4_v3_mix.py) reads three legacy
JSON files (traces.json, outcomes.json, submission_cases.json), but live
writes from the operator-review API land in observation_vla.sqlite3 only.
After every batch_review.py session the JSON files drift from reality and
the next dataset rebuild silently uses stale labels.

This script flushes the current DB state back to the three JSON files so
the next export sees the new outcomes. Idempotent — safe to re-run.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "sim" / "data" / "observation_vla"
DB = DATA_DIR / "observation_vla.sqlite3"


def _dump(conn: sqlite3.Connection, table: str, key: str, out_path: Path,
          where: str = "", schema_version: int = 1) -> int:
    cur = conn.execute(f"SELECT raw_json FROM {table}{(' WHERE ' + where) if where else ''}")
    rows = [json.loads(r[0]) for r in cur.fetchall()]
    payload = {"schema_version": schema_version, key: rows}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def main() -> int:
    if not DB.exists():
        print(f"DB not found: {DB}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(DB)
    try:
        n_traces = _dump(conn, "observation_traces", "traces",
                         DATA_DIR / "traces.json")
        # is_current = 1 filters to the live outcome per (trace, decision)
        n_outcomes = _dump(conn, "observation_outcomes", "outcomes",
                           DATA_DIR / "outcomes.json", where="is_current=1")
        n_subs = _dump(conn, "observation_submission_cases", "cases",
                       DATA_DIR / "submission_cases.json")
    finally:
        conn.close()
    print(f"Wrote: traces={n_traces}  outcomes={n_outcomes}  submission_cases={n_subs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
