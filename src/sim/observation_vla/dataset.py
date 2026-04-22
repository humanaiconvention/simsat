from __future__ import annotations

from .schemas import ObservationOutcome, ObservationTraceRecord


class ObservationDatasetBuilder:
    def build_example(
        self,
        trace: ObservationTraceRecord,
        outcome: ObservationOutcome | None = None,
    ) -> dict:
        return {
            "trace_id": trace.trace_id,
            "scenario_pack": trace.scenario_pack,
            "decision_id": trace.decision_id,
            "target_id": trace.target_id,
            "window_id": trace.window_id,
            "stimulus_id": trace.stimulus_id,
            "sample": trace.sample.to_dict(),
            "assessment": trace.assessment.to_dict(),
            "residual": trace.residual.to_dict(),
            "decision_before": trace.decision_before,
            "decision_after": trace.decision_after,
            "probe": trace.probe,
            "features": trace.features,
            "outcome": outcome.to_dict() if outcome is not None else None,
            "labelled": outcome is not None,
        }

    def build_dataset(
        self,
        traces: list[ObservationTraceRecord],
        outcomes_by_trace_id: dict[str, ObservationOutcome],
        labelled_only: bool = True,
    ) -> list[dict]:
        rows: list[dict] = []
        for trace in traces:
            outcome = outcomes_by_trace_id.get(trace.trace_id)
            if labelled_only and outcome is None:
                continue
            rows.append(self.build_example(trace, outcome=outcome))
        return rows
