"""
In-process session store for HAIC convention sessions.

Keeps sessions in memory (with optional JSON persistence to disk).
Thread-safe for use across FastAPI async handlers and the sim process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .schemas import ConventionSession, SessionStatus

logger = logging.getLogger(__name__)

_STORE_PATH = os.environ.get("HAIC_SESSION_STORE", "")


class SessionStore:
    """Thread-safe in-memory store for ConventionSession objects."""

    def __init__(self, persist_path: str = ""):
        self._lock = threading.Lock()
        self._sessions: Dict[str, ConventionSession] = {}
        self._persist_path = persist_path or _STORE_PATH
        if self._persist_path:
            self._load()

    # ---- CRUD ----

    def create(self, session: ConventionSession) -> ConventionSession:
        with self._lock:
            self._sessions[session.session_id] = session
        self._maybe_persist()
        return session

    def get(self, session_id: str) -> Optional[ConventionSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def update(self, session: ConventionSession) -> ConventionSession:
        with self._lock:
            if session.session_id not in self._sessions:
                raise KeyError(f"Session {session.session_id} not found")
            self._sessions[session.session_id] = session
        self._maybe_persist()
        return session

    def list_all(self) -> List[ConventionSession]:
        with self._lock:
            return list(self._sessions.values())

    def list_by_status(self, status: SessionStatus) -> List[ConventionSession]:
        with self._lock:
            return [s for s in self._sessions.values() if s.status == status]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
        if existed:
            self._maybe_persist()
        return existed

    # ---- Persistence ----

    def _maybe_persist(self) -> None:
        if not self._persist_path:
            return
        try:
            with self._lock:
                data = {sid: s.to_dict() for sid, s in self._sessions.items()}
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to persist session store: %s", e)

    def _load(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, raw in data.items():
                # Reconstruct minimal session from dict
                session = ConventionSession(session_id=sid)
                session.status = SessionStatus(raw.get("status", SessionStatus.PENDING.value))
                session.created_at = raw.get("created_at", session.created_at)
                session.interview_turns = raw.get("interview_turns", [])
                session.participant_id = raw.get("participant_id")
                session.pog_verified = raw.get("pog_verified", False)
                session.viability_gates = raw.get("viability_gates", {})
                session.receipt_merkle_root = raw.get("receipt_merkle_root")
                session.entropy_delta = raw.get("entropy_delta")
                session.geometric_health_before = raw.get("geometric_health_before")
                session.geometric_health_after = raw.get("geometric_health_after")
                session.settlement_result = raw.get("settlement_result")
                self._sessions[sid] = session
            logger.info("Loaded %d sessions from %s", len(self._sessions), self._persist_path)
        except Exception as e:
            logger.warning("Failed to load session store: %s", e)


# Module-level singleton
_store: Optional[SessionStore] = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
