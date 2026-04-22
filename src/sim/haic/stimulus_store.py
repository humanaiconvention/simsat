"""Thread-safe in-memory store for HAIC grounding stimuli."""

from __future__ import annotations

import json
import os
import threading
import logging
from pathlib import Path
from typing import Optional

from .schemas import GroundingStimulus

DEFAULT_STIMULUS_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "haic" / "stimuli.json"
_STORE_PATH = os.environ.get("HAIC_STIMULUS_STORE", str(DEFAULT_STIMULUS_STORE_PATH))
logger = logging.getLogger(__name__)


class StimulusStore:
    def __init__(self, persist_path: str = "") -> None:
        self._lock = threading.Lock()
        self._stimuli: dict[str, GroundingStimulus] = {}
        self._persist_path = Path(persist_path or _STORE_PATH)
        self._load()

    def put(self, stimulus: GroundingStimulus) -> GroundingStimulus:
        with self._lock:
            self._stimuli[stimulus.stimulus_id] = stimulus
        self._maybe_persist()
        return stimulus

    def get(self, stimulus_id: str) -> Optional[GroundingStimulus]:
        with self._lock:
            return self._stimuli.get(stimulus_id)

    def list_recent(self, limit: int = 50) -> list[GroundingStimulus]:
        with self._lock:
            items = sorted(self._stimuli.values(), key=lambda s: s.created_at, reverse=True)
        return items[:limit]

    def delete(self, stimulus_id: str) -> bool:
        with self._lock:
            existed = stimulus_id in self._stimuli
            self._stimuli.pop(stimulus_id, None)
        if existed:
            self._maybe_persist()
        return existed

    def _maybe_persist(self) -> None:
        if not self._persist_path:
            return
        try:
            with self._lock:
                payload = {
                    "schema_version": 1,
                    "stimuli": {sid: stimulus.to_dict() for sid, stimulus in self._stimuli.items()},
                }
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._persist_path.with_suffix(f"{self._persist_path.suffix}.tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            tmp_path.replace(self._persist_path)
        except Exception as exc:
            logger.warning("Failed to persist stimulus store: %s", exc)

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with self._persist_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            raw_stimuli = payload.get("stimuli", payload)
            self._stimuli = {
                stimulus_id: GroundingStimulus.from_dict({"stimulus_id": stimulus_id, **raw})
                for stimulus_id, raw in raw_stimuli.items()
            }
        except Exception as exc:
            logger.warning("Failed to load stimulus store: %s", exc)
            self._stimuli = {}


_store: StimulusStore | None = None


def get_stimulus_store() -> StimulusStore:
    global _store
    if _store is None:
        _store = StimulusStore()
    return _store
