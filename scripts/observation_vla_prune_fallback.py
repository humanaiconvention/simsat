from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "src" / "sim" / "data" / "observation_vla"
MISSION_DIR = ROOT / "src" / "sim" / "data" / "mission_response"


def _parse_iso_utc(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_payload(path: Path, key: str) -> tuple[dict, list[dict]]:
    if not path.exists():
        return {"schema_version": 1, key: []}, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", 1)
    payload.setdefault(key, [])
    return payload, list(payload.get(key, []))


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _backup_file(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamped = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{path.stem}_{stamped}{path.suffix}"
    shutil.copy2(path, backup_path)
    return backup_path


def _trace_target_label(trace: dict) -> str:
    sample = trace.get("sample") or {}
    return str(sample.get("target_label") or trace.get("target_id") or "unknown_target")


def _protected_trace_ids(outcomes: list[dict], submission_cases: list[dict], actions: list[dict]) -> set[str]:
    protected: set[str] = set()
    protected.update(str(outcome.get("trace_id")) for outcome in outcomes if outcome.get("trace_id"))
    protected.update(str(case.get("trace_id")) for case in submission_cases if case.get("trace_id"))
    protected.update(str(action.get("trace_id")) for action in actions if action.get("trace_id"))
    return protected


def _candidate_reason(trace: dict, protected_ids: set[str], keep_latest_ids: set[str]) -> str | None:
    trace_id = str(trace.get("trace_id") or "")
    runtime_mode = str((trace.get("assessment") or {}).get("runtime_mode") or "")
    if runtime_mode != "stub_fallback":
        return None
    if trace_id in protected_ids:
        return "protected_reference"
    if trace_id in keep_latest_ids:
        return "kept_latest_per_target"
    return "prune_candidate"


def _build_keep_latest_ids(traces: list[dict], keep_latest_per_target: int) -> set[str]:
    by_target: dict[str, list[dict]] = defaultdict(list)
    for trace in traces:
        target_label = _trace_target_label(trace)
        by_target[target_label].append(trace)
    keep_ids: set[str] = set()
    for target_label, target_traces in by_target.items():
        target_fallbacks = [
            trace for trace in target_traces
            if str((trace.get("assessment") or {}).get("runtime_mode") or "") == "stub_fallback"
        ]
        target_fallbacks.sort(
            key=lambda trace: (
                _parse_iso_utc(str(trace.get("created_at") or "")),
                str(trace.get("trace_id") or ""),
            ),
            reverse=True,
        )
        for trace in target_fallbacks[:keep_latest_per_target]:
            keep_ids.add(str(trace.get("trace_id") or ""))
    return keep_ids


def build_report_and_candidates(
    *,
    keep_latest_per_target: int,
) -> tuple[str, list[dict], dict[str, list[dict]], dict[str, list[dict]]]:
    traces_payload, traces = _load_payload(OBS_DIR / "traces.json", "traces")
    assessments_payload, assessments = _load_payload(OBS_DIR / "assessments.json", "records")
    outcomes_payload, outcomes = _load_payload(OBS_DIR / "outcomes.json", "outcomes")
    submission_payload, submission_cases = _load_payload(OBS_DIR / "submission_cases.json", "cases")
    actions_payload, actions = _load_payload(MISSION_DIR / "actions.json", "actions")

    protected_ids = _protected_trace_ids(outcomes, submission_cases, actions)
    keep_latest_ids = _build_keep_latest_ids(traces, keep_latest_per_target)

    candidates: list[dict] = []
    protected_reasons: dict[str, list[dict]] = defaultdict(list)
    prune_reasons: dict[str, list[dict]] = defaultdict(list)

    assessment_ids_by_trace = {
        str(trace.get("trace_id")): str((trace.get("assessment") or {}).get("assessment_id") or "")
        for trace in traces
    }
    assessment_records_by_id = {
        str((record.get("assessment") or {}).get("assessment_id") or ""): record
        for record in assessments
    }

    for trace in traces:
        trace_id = str(trace.get("trace_id") or "")
        reason = _candidate_reason(trace, protected_ids, keep_latest_ids)
        if reason is None:
            continue
        assessment_id = assessment_ids_by_trace.get(trace_id, "")
        entry = {
            "trace_id": trace_id,
            "target_label": _trace_target_label(trace),
            "scenario_pack": str(trace.get("scenario_pack") or "all"),
            "created_at": str(trace.get("created_at") or ""),
            "runtime_mode": str((trace.get("assessment") or {}).get("runtime_mode") or ""),
            "assessment_mode": str((trace.get("assessment") or {}).get("assessment_mode") or ""),
            "recommended_action": str((trace.get("assessment") or {}).get("recommended_action") or ""),
            "assessment_id": assessment_id,
            "has_assessment_record": assessment_id in assessment_records_by_id,
        }
        if reason == "prune_candidate":
            candidates.append(entry)
            prune_reasons[entry["target_label"]].append(entry)
        else:
            protected_reasons[entry["target_label"]].append({**entry, "reason": reason})

    per_target_counts = Counter(entry["target_label"] for entry in candidates)
    runtime_counts = Counter(str((trace.get("assessment") or {}).get("runtime_mode") or "unknown") for trace in traces)

    lines: list[str] = []
    lines.append("# ObservationVLA Fallback Prune Report")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}`")
    lines.append(f"- Total traces: `{len(traces)}`")
    lines.append(f"- Runtime distribution: `{', '.join(f'{key}={runtime_counts[key]}' for key in sorted(runtime_counts))}`")
    lines.append(f"- Protected trace references: `{len(protected_ids)}`")
    lines.append(f"- Keep latest fallback per target: `{keep_latest_per_target}`")
    lines.append(f"- Safe prune candidates: `{len(candidates)}`")
    lines.append("")
    lines.append("## Candidate Counts By Target")
    lines.append("")
    lines.append("| target | prune_candidates |")
    lines.append("| --- | --- |")
    for target_label, count in sorted(per_target_counts.items()):
        lines.append(f"| {target_label} | {count} |")
    if not per_target_counts:
        lines.append("| none | 0 |")
    lines.append("")
    lines.append("## Protected Fallback Rows")
    lines.append("")
    lines.append("These are fallback rows that the tool refuses to prune because they are referenced or retained as latest debugging examples.")
    lines.append("")
    lines.append("| target | trace | reason |")
    lines.append("| --- | --- | --- |")
    protected_total = 0
    for target_label in sorted(protected_reasons):
        for entry in sorted(protected_reasons[target_label], key=lambda item: item["trace_id"])[:5]:
            protected_total += 1
            lines.append(f"| {target_label} | {entry['trace_id']} | {entry['reason']} |")
    if protected_total == 0:
        lines.append("| none | - | - |")
    lines.append("")
    lines.append("## Sample Prune Candidates")
    lines.append("")
    lines.append("| scenario | target | trace | created_at | action | assessment_record |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    sample_rows = sorted(
        candidates,
        key=lambda item: (_parse_iso_utc(item["created_at"]), item["trace_id"]),
    )
    for entry in sample_rows[:20]:
        lines.append(
            f"| {entry['scenario_pack']} | {entry['target_label']} | {entry['trace_id']} | "
            f"{entry['created_at']} | {entry['recommended_action']} | {entry['has_assessment_record']} |"
        )
    if not sample_rows:
        lines.append("| none | - | - | - | - | - |")
    lines.append("")
    lines.append("## Safe Apply Rule")
    lines.append("")
    lines.append("- Only `stub_fallback` traces are considered.")
    lines.append("- Any trace with an ObservationVLA outcome is protected.")
    lines.append("- Any trace pinned as a submission case is protected.")
    lines.append("- Any trace referenced by a mission-response action is protected.")
    lines.append(f"- The newest `{keep_latest_per_target}` fallback traces per target are retained for debugging.")
    lines.append("- Matching orphaned assessment records are pruned alongside deleted traces.")
    lines.append("")

    return "\n".join(lines), candidates, {
        "traces": traces_payload,
        "assessments": assessments_payload,
        "outcomes": outcomes_payload,
        "submission_cases": submission_payload,
        "actions": actions_payload,
    }, {
        "traces": traces,
        "assessments": assessments,
        "outcomes": outcomes,
        "submission_cases": submission_cases,
        "actions": actions,
    }


def apply_prune(
    candidates: list[dict],
    payloads: dict[str, dict],
    raw_lists: dict[str, list[dict]],
    *,
    backup_dir: Path,
) -> tuple[int, int, list[Path]]:
    trace_ids_to_remove = {entry["trace_id"] for entry in candidates}
    assessment_ids_to_remove = {
        entry["assessment_id"]
        for entry in candidates
        if entry.get("assessment_id")
    }

    traces_path = OBS_DIR / "traces.json"
    assessments_path = OBS_DIR / "assessments.json"

    backups = [
        _backup_file(traces_path, backup_dir),
        _backup_file(assessments_path, backup_dir),
    ]

    payloads["traces"]["traces"] = [
        trace for trace in raw_lists["traces"]
        if str(trace.get("trace_id") or "") not in trace_ids_to_remove
    ]
    payloads["assessments"]["records"] = [
        record for record in raw_lists["assessments"]
        if str((record.get("assessment") or {}).get("assessment_id") or "") not in assessment_ids_to_remove
    ]

    _write_payload(traces_path, payloads["traces"])
    _write_payload(assessments_path, payloads["assessments"])
    return len(trace_ids_to_remove), len(assessment_ids_to_remove), backups


def main() -> None:
    parser = argparse.ArgumentParser(description="Report and safely prune old ObservationVLA stub_fallback traces.")
    parser.add_argument("--keep-latest-per-target", type=int, default=2)
    parser.add_argument("--report-output", default=str(ROOT / "OBSERVATION_VLA_PRUNE.md"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", default=str(OBS_DIR / "prune_backups"))
    args = parser.parse_args()

    report, candidates, payloads, raw_lists = build_report_and_candidates(
        keep_latest_per_target=args.keep_latest_per_target,
    )
    report_path = Path(args.report_output)
    report_path.write_text(report, encoding="utf-8")
    print(f"Fallback prune report written to {report_path}")

    if not args.apply:
        print(f"Dry run only. Safe prune candidates: {len(candidates)}")
        return

    removed_traces, removed_assessments, backups = apply_prune(
        candidates,
        payloads,
        raw_lists,
        backup_dir=Path(args.backup_dir),
    )
    print(f"Pruned traces: {removed_traces}")
    print(f"Pruned assessment records: {removed_assessments}")
    print("Backups:")
    for backup in backups:
        print(f"- {backup}")


if __name__ == "__main__":
    main()
