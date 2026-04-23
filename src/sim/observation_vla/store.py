from __future__ import annotations

import json
import threading
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
        self._lock = threading.Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _load_payload(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "records": []}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {"schema_version": 1, "records": []}
        payload.setdefault("schema_version", 1)
        payload.setdefault("records", [])
        return payload

    def _write_payload(self, payload: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.path)

    def _load_traces_payload(self) -> dict:
        if not self.traces_path.exists():
            return {"schema_version": 1, "traces": []}
        try:
            with self.traces_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {"schema_version": 1, "traces": []}
        payload.setdefault("schema_version", 1)
        payload.setdefault("traces", [])
        return payload

    def _write_traces_payload(self, payload: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.traces_path.with_suffix(f"{self.traces_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.traces_path)

    def _load_outcomes_payload(self) -> dict:
        if not self.outcomes_path.exists():
            return {"schema_version": 1, "outcomes": []}
        try:
            with self.outcomes_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {"schema_version": 1, "outcomes": []}
        payload.setdefault("schema_version", 1)
        payload.setdefault("outcomes", [])
        return payload

    def _write_outcomes_payload(self, payload: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.outcomes_path.with_suffix(f"{self.outcomes_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.outcomes_path)

    def _load_memory_payload(self) -> dict:
        if not self.memory_path.exists():
            return {"schema_version": 1, "states": []}
        try:
            with self.memory_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {"schema_version": 1, "states": []}
        payload.setdefault("schema_version", 1)
        payload.setdefault("states", [])
        return payload

    def _load_submission_cases_payload(self) -> dict:
        if not self.submission_cases_path.exists():
            return {"schema_version": 1, "cases": []}
        try:
            with self.submission_cases_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {"schema_version": 1, "cases": []}
        payload.setdefault("schema_version", 1)
        payload.setdefault("cases", [])
        return payload

    def _write_memory_payload(self, payload: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.memory_path.with_suffix(f"{self.memory_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.memory_path)

    def _write_submission_cases_payload(self, payload: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.submission_cases_path.with_suffix(f"{self.submission_cases_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.submission_cases_path)

    def save_record(self, record: ObservationAssessmentRecord) -> ObservationAssessmentRecord:
        with self._lock:
            payload = self._load_payload()
            records = payload.get("records", [])
            updated = False
            for idx, raw in enumerate(records):
                if raw.get("assessment", {}).get("assessment_id") == record.assessment.assessment_id:
                    records[idx] = record.to_dict()
                    updated = True
                    break
            if not updated:
                records.append(record.to_dict())
            payload["records"] = records
            self._write_payload(payload)
        return record

    def get_record(self, assessment_id: str) -> ObservationAssessmentRecord | None:
        with self._lock:
            payload = self._load_payload()
        for raw in payload.get("records", []):
            if raw.get("assessment", {}).get("assessment_id") == assessment_id:
                return ObservationAssessmentRecord.from_dict(raw)
        return None

    def list_records(self, limit: int = 50) -> list[ObservationAssessmentRecord]:
        with self._lock:
            payload = self._load_payload()
        records = [ObservationAssessmentRecord.from_dict(raw) for raw in payload.get("records", [])]
        records.sort(key=lambda record: record.assessment.created_at, reverse=True)
        return records[:limit]

    def save_trace(self, trace: ObservationTraceRecord) -> ObservationTraceRecord:
        with self._lock:
            payload = self._load_traces_payload()
            traces = payload.get("traces", [])
            updated = False
            for idx, raw in enumerate(traces):
                if raw.get("trace_id") == trace.trace_id:
                    traces[idx] = trace.to_dict()
                    updated = True
                    break
            if not updated:
                traces.append(trace.to_dict())
            payload["traces"] = traces
            self._write_traces_payload(payload)
        return trace

    def list_traces(self, limit: int = 50) -> list[ObservationTraceRecord]:
        with self._lock:
            payload = self._load_traces_payload()
        traces = [ObservationTraceRecord.from_dict(raw) for raw in payload.get("traces", [])]
        traces.sort(key=lambda trace: trace.created_at, reverse=True)
        return traces[:limit]

    def get_trace(self, trace_id: str) -> ObservationTraceRecord | None:
        with self._lock:
            payload = self._load_traces_payload()
        for raw in payload.get("traces", []):
            if raw.get("trace_id") == trace_id:
                return ObservationTraceRecord.from_dict(raw)
        return None

    def save_outcome(self, outcome: ObservationOutcome) -> ObservationOutcome:
        with self._lock:
            payload = self._load_outcomes_payload()
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
            self._write_outcomes_payload(payload)
        return outcome

    def list_outcomes(self, limit: int = 50, current_only: bool = True) -> list[ObservationOutcome]:
        with self._lock:
            payload = self._load_outcomes_payload()
        outcomes = [ObservationOutcome.from_dict(raw) for raw in payload.get("outcomes", [])]
        if current_only:
            outcomes = [outcome for outcome in outcomes if outcome.is_current]
        outcomes.sort(key=lambda outcome: outcome.registered_at, reverse=True)
        return outcomes[:limit]

    def get_outcome(self, outcome_id: str) -> ObservationOutcome | None:
        with self._lock:
            payload = self._load_outcomes_payload()
        for raw in payload.get("outcomes", []):
            if raw.get("outcome_id") == outcome_id:
                return ObservationOutcome.from_dict(raw)
        return None

    def get_outcome_for_trace(self, trace_id: str) -> ObservationOutcome | None:
        with self._lock:
            payload = self._load_outcomes_payload()
        matches = [
            ObservationOutcome.from_dict(raw)
            for raw in payload.get("outcomes", [])
            if raw.get("trace_id") == trace_id and raw.get("is_current", True)
        ]
        if not matches:
            matches = [
                ObservationOutcome.from_dict(raw)
                for raw in payload.get("outcomes", [])
                if raw.get("trace_id") == trace_id
            ]
            if not matches:
                return None
        matches.sort(key=lambda outcome: outcome.registered_at, reverse=True)
        return matches[0]

    def replace_memory_states(self, states: list[ObservationMemoryState]) -> None:
        with self._lock:
            payload = {
                "schema_version": 1,
                "states": [state.to_dict() for state in states],
            }
            self._write_memory_payload(payload)

    def save_memory_state(self, state: ObservationMemoryState) -> ObservationMemoryState:
        with self._lock:
            payload = self._load_memory_payload()
            states = payload.get("states", [])
            updated = False
            for idx, raw in enumerate(states):
                if raw.get("state_id") == state.state_id:
                    states[idx] = state.to_dict()
                    updated = True
                    break
            if not updated:
                states.append(state.to_dict())
            payload["states"] = states
            self._write_memory_payload(payload)
        return state

    def get_memory_state(self, scope: str, scope_id: str) -> ObservationMemoryState | None:
        with self._lock:
            payload = self._load_memory_payload()
        for raw in payload.get("states", []):
            if raw.get("scope") == scope and raw.get("scope_id") == scope_id:
                return ObservationMemoryState.from_dict(raw)
        return None

    def list_memory_states(self, scope: str | None = None, limit: int = 50) -> list[ObservationMemoryState]:
        with self._lock:
            payload = self._load_memory_payload()
        states = [ObservationMemoryState.from_dict(raw) for raw in payload.get("states", [])]
        if scope:
            states = [state for state in states if state.scope == scope]
        states.sort(key=lambda state: state.last_updated, reverse=True)
        return states[:limit]

    def save_submission_case(self, pin: SubmissionCasePin) -> SubmissionCasePin:
        with self._lock:
            payload = self._load_submission_cases_payload()
            cases = payload.get("cases", [])
            updated = False
            for idx, raw in enumerate(cases):
                if raw.get("scenario_pack") == pin.scenario_pack:
                    cases[idx] = pin.to_dict()
                    updated = True
                    break
            if not updated:
                cases.append(pin.to_dict())
            payload["cases"] = cases
            self._write_submission_cases_payload(payload)
        return pin

    def get_submission_case(self, scenario_pack: str) -> SubmissionCasePin | None:
        with self._lock:
            payload = self._load_submission_cases_payload()
        for raw in payload.get("cases", []):
            if raw.get("scenario_pack") == scenario_pack:
                return SubmissionCasePin.from_dict(raw)
        return None

    def list_submission_cases(self) -> list[SubmissionCasePin]:
        with self._lock:
            payload = self._load_submission_cases_payload()
        cases = [SubmissionCasePin.from_dict(raw) for raw in payload.get("cases", [])]
        cases.sort(key=lambda pin: pin.pinned_at, reverse=True)
        return cases
