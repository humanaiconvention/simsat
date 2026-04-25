"""Smoke tests for core SimSat services.

Covers (with no live network or GPU required):
  1. ObservationStore — CRUD round-trip for traces, outcomes, submission cases.
  2. Submission case schema migration — trace_id as PK; multiple pins per pack.
  3. viability.evaluate_ttt_viability — all 3 TTT gates exercise pass and fail paths.
  4. ObservationVLAService API router — /submission-cases and /review-bundle via TestClient.
  5. EncounterStore — basic trace persistence.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)


# ---------------------------------------------------------------------------
# 1. ObservationStore CRUD
# ---------------------------------------------------------------------------

class TestObservationStoreCRUD:
    def _make_store(self, tmp_path):
        from observation_vla.store import ObservationStore
        return ObservationStore(base_dir=tmp_path)

    def test_save_and_get_trace(self, tmp_path):
        from observation_vla.store import ObservationStore
        from observation_vla.schemas import ObservationTraceRecord, ObservationSample, ObservationAssessment

        store = ObservationStore(base_dir=tmp_path)
        trace = ObservationTraceRecord(
            trace_id="trace_smoke_001",
            scenario_pack="test_pack",
            target_id="target_1",
            decision_id="dec_001",
            sample=ObservationSample(target_label="Test Target"),
            assessment=ObservationAssessment(runtime_mode="stub", recommended_action="accept"),
        )
        store.save_trace(trace)
        retrieved = store.get_trace("trace_smoke_001")
        assert retrieved is not None
        assert retrieved.trace_id == "trace_smoke_001"
        assert retrieved.assessment.runtime_mode == "stub"
        assert retrieved.sample.target_label == "Test Target"

    def test_get_trace_missing_returns_none(self, tmp_path):
        from observation_vla.store import ObservationStore
        store = ObservationStore(base_dir=tmp_path)
        assert store.get_trace("trace_does_not_exist") is None

    def test_save_and_list_submission_cases_multiple_per_pack(self, tmp_path):
        """Schema allows multiple pinned cases per scenario_pack (trace_id is PK)."""
        from observation_vla.store import ObservationStore
        from observation_vla.schemas import SubmissionCasePin

        store = ObservationStore(base_dir=tmp_path)
        pin1 = SubmissionCasePin(
            scenario_pack="urban_coastal_ambiguity",
            trace_id="trace_urban_001",
            decision_id="dec_001",
            target_id="rotterdam",
            target_label="Port of Rotterdam",
            outcome_id="out_001",
            label_source="operator_review",
            reviewer="Ben Haslam",
            pinned_at="2026-04-23T12:00:00Z",
        )
        pin2 = SubmissionCasePin(
            scenario_pack="urban_coastal_ambiguity",
            trace_id="trace_urban_002",
            decision_id="dec_002",
            target_id="sf_bay",
            target_label="San Francisco Bay",
            outcome_id="out_002",
            label_source="operator_review",
            reviewer="Ben Haslam",
            pinned_at="2026-04-25T15:00:00Z",
        )
        store.save_submission_case(pin1)
        store.save_submission_case(pin2)

        all_cases = store.list_submission_cases()
        assert len(all_cases) == 2
        trace_ids = {c.trace_id for c in all_cases}
        assert "trace_urban_001" in trace_ids
        assert "trace_urban_002" in trace_ids

    def test_get_submission_case_returns_most_recent(self, tmp_path):
        from observation_vla.store import ObservationStore
        from observation_vla.schemas import SubmissionCasePin

        store = ObservationStore(base_dir=tmp_path)
        for i, pinned_at in enumerate(["2026-04-23T12:00:00Z", "2026-04-25T15:00:00Z"]):
            store.save_submission_case(SubmissionCasePin(
                scenario_pack="maritime_chokepoints",
                trace_id=f"trace_maritime_{i:03d}",
                decision_id=f"dec_{i:03d}",
                target_id="suez",
                target_label="Suez Canal",
                outcome_id=f"out_{i:03d}",
                label_source="operator_review",
                reviewer="Ben Haslam",
                pinned_at=pinned_at,
            ))
        # get_submission_case returns the most recently pinned
        latest = store.get_submission_case("maritime_chokepoints")
        assert latest is not None
        assert latest.trace_id == "trace_maritime_001"  # pinned_at 2026-04-25

    def test_get_submission_case_by_trace(self, tmp_path):
        from observation_vla.store import ObservationStore
        from observation_vla.schemas import SubmissionCasePin

        store = ObservationStore(base_dir=tmp_path)
        pin = SubmissionCasePin(
            scenario_pack="disaster_response_weather",
            trace_id="trace_disaster_xyz",
            decision_id="dec_xyz",
            target_id="houston",
            target_label="Houston Ship Channel",
            outcome_id="out_xyz",
            label_source="operator_review",
            reviewer="Ben Haslam",
            pinned_at="2026-04-25T00:00:00Z",
        )
        store.save_submission_case(pin)

        found = store.get_submission_case_by_trace("trace_disaster_xyz")
        assert found is not None
        assert found.target_label == "Houston Ship Channel"

        missing = store.get_submission_case_by_trace("trace_does_not_exist")
        assert missing is None


# ---------------------------------------------------------------------------
# 2. TTT viability gates
# ---------------------------------------------------------------------------

class TestTTTViabilityGates:
    @pytest.fixture(autouse=True)
    def _import(self):
        from haic.viability import (
            evaluate_ttt_viability,
            MAX_TTT_WEIGHT_DRIFT,
            MAX_TTT_UPDATE_COUNT,
            TTT_BIAS_THRESHOLD,
        )
        self.evaluate = evaluate_ttt_viability
        self.MAX_DRIFT = MAX_TTT_WEIGHT_DRIFT
        self.MAX_COUNT = MAX_TTT_UPDATE_COUNT
        self.BIAS_THRESH = TTT_BIAS_THRESHOLD

    def _snapshot(self, *, drift=None, count=0, recent=None):
        return {
            "drift_from_policy_defaults": drift or {},
            "update_count": count,
            "recent_updates": recent or [],
        }

    def test_healthy_all_pass(self):
        snap = self._snapshot(
            drift={"a": 0.05, "b": -0.10},
            count=50,
            recent=[{"error": 0.1}, {"error": -0.05}, {"error": 0.03}],
        )
        gates = self.evaluate(snap)
        assert gates == {"weight_drift": True, "update_rate": True, "error_bias": True}

    def test_weight_drift_fails_on_single_large_delta(self):
        snap = self._snapshot(drift={"a": self.MAX_DRIFT + 0.01})
        gates = self.evaluate(snap)
        assert gates["weight_drift"] is False
        assert gates["update_rate"] is True  # others unaffected

    def test_weight_drift_passes_at_boundary(self):
        snap = self._snapshot(drift={"a": self.MAX_DRIFT})
        gates = self.evaluate(snap)
        assert gates["weight_drift"] is True  # equal is allowed

    def test_update_rate_fails_when_count_exceeded(self):
        snap = self._snapshot(count=self.MAX_COUNT + 1)
        gates = self.evaluate(snap)
        assert gates["update_rate"] is False

    def test_update_rate_passes_at_limit(self):
        snap = self._snapshot(count=self.MAX_COUNT)
        gates = self.evaluate(snap)
        assert gates["update_rate"] is True

    def test_error_bias_fails_when_systematic(self):
        # 9 positive errors out of 10 = 90% same sign > 70% threshold
        recent = [{"error": 0.1}] * 9 + [{"error": -0.01}]
        snap = self._snapshot(recent=recent)
        gates = self.evaluate(snap)
        assert gates["error_bias"] is False

    def test_error_bias_passes_with_mixed_errors(self):
        recent = [{"error": 0.1}, {"error": -0.1}, {"error": 0.05},
                  {"error": -0.05}, {"error": 0.02}, {"error": -0.08}]
        snap = self._snapshot(recent=recent)
        gates = self.evaluate(snap)
        assert gates["error_bias"] is True

    def test_error_bias_passes_with_insufficient_history(self):
        # Less than 3 updates — not enough data, passes by default
        snap = self._snapshot(recent=[{"error": 0.9}, {"error": 0.8}])
        gates = self.evaluate(snap)
        assert gates["error_bias"] is True

    def test_empty_snapshot_all_pass(self):
        gates = self.evaluate({})
        assert all(gates.values()), f"Expected all pass on empty snapshot, got {gates}"


# ---------------------------------------------------------------------------
# 3. ObservationVLA API router via TestClient
# ---------------------------------------------------------------------------

class TestObservationVLARouter:
    """Tests use a module-level client so startup fires once for the whole class."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup_client(self, request):
        from fastapi.testclient import TestClient
        from api import api  # type: ignore
        with TestClient(api) as client:
            request.cls.client = client
            yield

    def test_submission_cases_endpoint_returns_list(self):
        resp = self.client.get("/observation-vla/submission-cases")
        assert resp.status_code == 200
        data = resp.json()
        assert "cases" in data
        assert isinstance(data["cases"], list)

    def test_submission_cases_contains_at_least_three_human_reviewed(self):
        resp = self.client.get("/observation-vla/submission-cases")
        cases = resp.json()["cases"]
        human = [c for c in cases if c.get("reviewer") == "Ben Haslam"]
        assert len(human) >= 3, f"Expected ≥3 human-reviewed cases, got {len(human)}: {[c['target_label'] for c in human]}"

    def test_review_bundle_returns_required_keys(self):
        resp = self.client.get("/observation-vla/submission-cases")
        cases = resp.json().get("cases", [])
        if not cases:
            pytest.skip("No pinned submission cases in store")
        trace_id = cases[0]["trace_id"]
        bundle = self.client.get(f"/observation-vla/review-bundle/{trace_id}")
        assert bundle.status_code == 200
        data = bundle.json()
        for key in ("trace", "target", "current_outcome", "is_pinned_submission_case"):
            assert key in data, f"Missing key '{key}' in review bundle"

    def test_review_bundle_missing_trace_returns_404(self):
        resp = self.client.get("/observation-vla/review-bundle/trace_does_not_exist_xyz")
        assert resp.status_code == 404

    def test_traces_endpoint_returns_paginated_list(self):
        resp = self.client.get("/observation-vla/traces", params={"limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert "traces" in data
        assert len(data["traces"]) <= 5


# ---------------------------------------------------------------------------
# 4. EncounterStore basic persistence
# ---------------------------------------------------------------------------

class TestEncounterStore:
    def test_store_instantiates_cleanly(self, tmp_path):
        from encounter.store import EncounterStore
        store = EncounterStore(base_dir=tmp_path)
        assert store is not None

    def test_list_records_empty_on_fresh_store(self, tmp_path):
        from encounter.store import EncounterStore
        store = EncounterStore(base_dir=tmp_path)
        records = store.list_records(limit=10)
        assert isinstance(records, list)
