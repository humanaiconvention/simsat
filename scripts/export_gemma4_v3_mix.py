from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys


ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = ROOT / "src" / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from observation_vla.prompts import build_triage_prompt, expected_response_schema  # type: ignore
from observation_vla.schemas import ObservationSample  # type: ignore


FORMAT_VERSION = "simsat_gemma_v3_v1"
TRACE_PATH = ROOT / "src" / "sim" / "data" / "observation_vla" / "traces.json"
OUTCOME_PATH = ROOT / "src" / "sim" / "data" / "observation_vla" / "outcomes.json"
SUBMISSION_CASE_PATH = ROOT / "src" / "sim" / "data" / "observation_vla" / "submission_cases.json"
TARGET_PATH = ROOT / "src" / "sim" / "data" / "encounter" / "targets.json"
REVIEW_ASSETS_DIR = ROOT / "review_queue_assets"
SUBMISSION_ASSETS_DIR = ROOT / "submission_assets"


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    chunks: list[str] = []
    current: list[str] = []
    for char in lowered:
        if char.isalnum():
            current.append(char)
            continue
        if current:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    return "_".join(chunks)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_traces(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    return list(payload.get("traces", []))


def load_outcomes(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    return list(payload.get("outcomes", []))


def load_submission_cases(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    return list(payload.get("cases", []))


def load_targets(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    return {
        str(target.get("target_id")): target
        for target in payload.get("targets", [])
    }


def build_outcome_index(outcomes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    current_only = [outcome for outcome in outcomes if bool(outcome.get("is_current", True))]
    return {
        str(outcome.get("trace_id")): outcome
        for outcome in current_only
        if outcome.get("trace_id")
    }


def build_submission_case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("trace_id")): case
        for case in cases
        if case.get("trace_id")
    }


def is_model_backed_trace(trace: dict[str, Any]) -> bool:
    runtime_mode = str((trace.get("assessment") or {}).get("runtime_mode", ""))
    return runtime_mode not in {"stub", "stub_fallback", ""}


def is_stub_fallback_trace(trace: dict[str, Any]) -> bool:
    runtime_mode = str((trace.get("assessment") or {}).get("runtime_mode", ""))
    return runtime_mode == "stub_fallback"


def is_image_backed_trace(trace: dict[str, Any]) -> bool:
    assessment = trace.get("assessment") or {}
    sample = trace.get("sample") or {}
    return (
        str(assessment.get("assessment_mode")) == "image_conditioned"
        and int(assessment.get("image_count", 0)) > 0
        and bool(sample.get("images"))
    )


def resolve_review_asset(trace: dict[str, Any], review_assets_dir: Path) -> Path | None:
    trace_id = str(trace.get("trace_id") or "")
    if not trace_id or not review_assets_dir.exists():
        return None
    matches = sorted(review_assets_dir.glob(f"*{trace_id}.png"))
    return matches[0] if matches else None


def resolve_submission_asset(trace: dict[str, Any], submission_assets_dir: Path) -> Path | None:
    if not submission_assets_dir.exists():
        return None
    scenario_pack = str(trace.get("scenario_pack") or "all")
    sample = trace.get("sample") or {}
    target_label = str(sample.get("target_label") or trace.get("target_id") or "unknown")
    candidate = submission_assets_dir / f"{scenario_pack}_{_slugify(target_label)}.png"
    return candidate if candidate.exists() else None


def resolve_image_path(
    trace: dict[str, Any],
    review_assets_dir: Path,
    submission_assets_dir: Path,
) -> Path | None:
    if not is_image_backed_trace(trace):
        return None
    review_asset = resolve_review_asset(trace, review_assets_dir)
    if review_asset is not None:
        return review_asset
    return resolve_submission_asset(trace, submission_assets_dir)


def copy_image_to_export(src: Path, dst_dir: Path, trace: dict[str, Any]) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    trace_id = str(trace.get("trace_id") or "unknown_trace")
    filename_root = trace_id if trace_id.startswith("trace_") else f"trace_{trace_id}"
    destination = dst_dir / f"{filename_root}{src.suffix.lower()}"
    if not destination.exists():
        shutil.copy2(src, destination)
    return destination.relative_to(dst_dir.parent).as_posix()


def build_prompt_text(sample: dict[str, Any], target: dict[str, Any] | None) -> str:
    merged_sample = dict(sample)
    if target is not None:
        merged_sample.setdefault("target_label", target.get("label"))
        merged_sample["target_tags"] = list(
            merged_sample.get("target_tags") or target.get("tags", [])
        )
        merged_sample["target_metadata"] = dict(
            merged_sample.get("target_metadata") or target.get("metadata", {})
        )
    observation_sample = ObservationSample.from_dict(merged_sample)
    return build_triage_prompt(observation_sample)


def build_response_schema() -> dict[str, Any]:
    return expected_response_schema()


def normalize_target_json(trace: dict[str, Any], outcome: dict[str, Any] | None) -> dict[str, Any]:
    assessment = trace.get("assessment") or {}
    evidence = assessment.get("evidence") or {}
    target = {
        "usable_observation": bool(evidence.get("usable_observation", False)),
        "scene_match_score": _clamp(float(evidence.get("scene_match_score", 0.0))),
        "salience_score": _clamp(float(evidence.get("salience_score", 0.0))),
        "change_or_event_score": _clamp(float(evidence.get("change_or_event_score", 0.0))),
        "occlusion_or_cloud_risk": _clamp(float(evidence.get("occlusion_or_cloud_risk", 0.0))),
        "confidence": _clamp(float(evidence.get("confidence", 0.0))),
        "recommended_action": str(assessment.get("recommended_action", "refine")),
        "rationale_tags": [str(tag) for tag in evidence.get("rationale_tags", [])],
    }
    if outcome is not None:
        target["usable_observation"] = bool(outcome.get("useful", target["usable_observation"]))
        operator_action = outcome.get("operator_action")
        if operator_action:
            target["recommended_action"] = str(operator_action)
        target["rationale_tags"] = _dedupe_strings(
            [*target["rationale_tags"], f"label_source:{outcome.get('label_source', 'unknown')}"]
        )
    return target


def attach_review_labels(outcome: dict[str, Any] | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    return {
        "operator_action": outcome.get("operator_action"),
        "useful": outcome.get("useful"),
        "usefulness_score": outcome.get("usefulness_score"),
        "label_source": outcome.get("label_source", "simulated"),
        "reviewer": outcome.get("reviewer"),
        "review_status": outcome.get("review_status", "provisional"),
        "notes": outcome.get("notes"),
        "outcome_tags": list(outcome.get("outcome_tags", [])),
    }


def label_quality(trace: dict[str, Any], outcome: dict[str, Any] | None) -> str:
    if outcome is not None:
        if outcome.get("label_source") == "operator_review":
            return "reviewed"
        return "simulated"
    if is_model_backed_trace(trace):
        return "weak_model"
    if str((trace.get("assessment") or {}).get("runtime_mode", "")) == "stub":
        return "stub"
    return "unlabelled"


def training_weight_for_row(
    trace: dict[str, Any],
    outcome: dict[str, Any] | None,
    submission_case: dict[str, Any] | None,
) -> float:
    quality = label_quality(trace, outcome)
    if quality == "reviewed":
        return 8.0 if submission_case is not None else 6.0
    if quality == "simulated":
        return 1.5
    if quality == "weak_model":
        return 2.0
    if quality == "stub":
        return 0.5
    return 0.25


def build_base_row(
    *,
    trace: dict[str, Any],
    target: dict[str, Any] | None,
    outcome: dict[str, Any] | None,
    submission_case: dict[str, Any] | None,
    variant: str,
    split: str,
    image_path: str | None,
    prompt_text: str,
) -> dict[str, Any]:
    sample = trace.get("sample") or {}
    assessment = trace.get("assessment") or {}
    return {
        "record_id": f"simsat_{trace.get('trace_id')}_{variant}",
        "format_version": FORMAT_VERSION,
        "split": split,
        "variant": variant,
        "label_quality": label_quality(trace, outcome),
        "training_weight": training_weight_for_row(trace, outcome, submission_case),
        "trace_id": trace.get("trace_id"),
        "decision_id": trace.get("decision_id"),
        "window_id": trace.get("window_id"),
        "stimulus_id": trace.get("stimulus_id"),
        "scenario_pack": trace.get("scenario_pack"),
        "target_id": trace.get("target_id"),
        "target_label": sample.get("target_label") or (target or {}).get("label") or trace.get("target_id"),
        "target_tags": list(sample.get("target_tags", (target or {}).get("tags", []))),
        "target_metadata": dict(sample.get("target_metadata", (target or {}).get("metadata", {}))),
        "image_path": image_path,
        "has_image": bool(image_path),
        "prompt_text": prompt_text,
        "response_schema": build_response_schema(),
        "input_context": {
            "geometry": dict(sample.get("geometry", {})),
            "probe": dict(sample.get("probe", {})),
        },
        "target_json": normalize_target_json(trace, outcome),
        "review_labels": attach_review_labels(outcome),
        "source_assessment": {
            "model_id": assessment.get("model_id"),
            "runtime_mode": assessment.get("runtime_mode"),
            "assessment_mode": assessment.get("assessment_mode"),
            "created_at": assessment.get("created_at"),
        },
    }


def reviewed_eval_split(
    reviewed_rows: list[dict[str, Any]],
    holdout_per_scenario: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviewed_rows:
        rows_by_scenario[str(row.get("scenario_pack", "all"))].append(row)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for scenario_pack, scenario_rows in rows_by_scenario.items():
        ordered = sorted(
            scenario_rows,
            key=lambda row: (
                0 if row.get("review_labels", {}).get("label_source") == "operator_review" else 1,
                0 if row.get("has_image") else 1,
                row.get("target_label", ""),
                row.get("trace_id", ""),
            ),
        )
        holdout = ordered[:holdout_per_scenario]
        remaining = ordered[holdout_per_scenario:]
        for row in holdout:
            copy = dict(row)
            copy["split"] = "eval"
            copy["variant"] = "eval_reviewed"
            copy["training_weight"] = 0.0
            eval_rows.append(copy)
        for row in remaining:
            copy = dict(row)
            copy["split"] = "train"
            copy["variant"] = "multimodal_reviewed"
            train_rows.append(copy)
    return train_rows, eval_rows


def shortlist_eval_rows(
    rows: list[dict[str, Any]],
    per_target_limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("target_id"))].append(row)

    shortlist: list[dict[str, Any]] = []
    for target_id, target_rows in grouped.items():
        ordered = sorted(
            target_rows,
            key=lambda row: (
                0 if row["target_json"].get("recommended_action") in {"refine", "defer", "skip"} else 1,
                -float(row["target_json"].get("confidence", 0.0)),
                row.get("trace_id", ""),
            ),
        )
        for row in ordered[:per_target_limit]:
            copy = dict(row)
            copy["split"] = "eval"
            copy["variant"] = "eval_shortlist"
            copy["training_weight"] = 0.0
            shortlist.append(copy)
    return sorted(shortlist, key=lambda row: (row.get("scenario_pack", ""), row.get("target_label", "")))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(
    *,
    traces: list[dict[str, Any]],
    format_rows: list[dict[str, Any]],
    weak_rows: list[dict[str, Any]],
    reviewed_rows: list[dict[str, Any]],
    reviewed_eval_rows: list[dict[str, Any]],
    shortlist_rows: list[dict[str, Any]],
    copied_images: list[str],
) -> dict[str, Any]:
    def _per_scenario(rows: list[dict[str, Any]]) -> dict[str, int]:
        counter = Counter(str(row.get("scenario_pack", "all")) for row in rows)
        return dict(sorted(counter.items()))

    def _per_target(rows: list[dict[str, Any]]) -> dict[str, int]:
        counter = Counter(str(row.get("target_label", row.get("target_id", "unknown"))) for row in rows)
        return dict(sorted(counter.items()))

    return {
        "format_version": FORMAT_VERSION,
        "source_paths": {
            "traces": str(TRACE_PATH),
            "outcomes": str(OUTCOME_PATH),
            "submission_cases": str(SUBMISSION_CASE_PATH),
            "targets": str(TARGET_PATH),
            "review_assets_dir": str(REVIEW_ASSETS_DIR),
            "submission_assets_dir": str(SUBMISSION_ASSETS_DIR),
        },
        "counts": {
            "total_traces_seen": len(traces),
            "format_train": len(format_rows),
            "multimodal_weak": len(weak_rows),
            "multimodal_reviewed": len(reviewed_rows),
            "eval_reviewed": len(reviewed_eval_rows),
            "eval_shortlist": len(shortlist_rows),
            "copied_images": len(copied_images),
        },
        "scenario_counts": {
            "format_train": _per_scenario(format_rows),
            "multimodal_weak": _per_scenario(weak_rows),
            "multimodal_reviewed": _per_scenario(reviewed_rows),
            "eval_reviewed": _per_scenario(reviewed_eval_rows),
            "eval_shortlist": _per_scenario(shortlist_rows),
        },
        "target_counts": {
            "multimodal_weak": _per_target(weak_rows),
            "multimodal_reviewed": _per_target(reviewed_rows),
            "eval_shortlist": _per_target(shortlist_rows),
        },
        "action_distribution": {
            "format_train": dict(sorted(Counter(row["target_json"]["recommended_action"] for row in format_rows).items())),
            "multimodal_weak": dict(sorted(Counter(row["target_json"]["recommended_action"] for row in weak_rows).items())),
            "multimodal_reviewed": dict(sorted(Counter(row["target_json"]["recommended_action"] for row in reviewed_rows).items())),
            "eval_reviewed": dict(sorted(Counter(row["target_json"]["recommended_action"] for row in reviewed_eval_rows).items())),
        },
        "copied_images": copied_images,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]
    lines = [
        "# SimSat Gemma 4 v3 Export",
        "",
        "This bundle is intended for the next SimSat-focused Gemma run on Kaggle T4 or A100 hardware.",
        "",
        "## Files",
        "",
        f"- `simsat_format_train.jsonl`: `{counts['format_train']}` rows",
        f"- `simsat_multimodal_weak.jsonl`: `{counts['multimodal_weak']}` rows",
        f"- `simsat_multimodal_reviewed.jsonl`: `{counts['multimodal_reviewed']}` rows",
        f"- `simsat_eval_reviewed.jsonl`: `{counts['eval_reviewed']}` rows",
        f"- `simsat_eval_shortlist.jsonl`: `{counts['eval_shortlist']}` rows",
        f"- `images/`: `{counts['copied_images']}` copied assets",
        "",
        "## Notes",
        "",
        "- SimSat native action space is `accept|defer|refine|skip`.",
        "- Reviewed labels are sparse and intentionally preserved as a separate higher-quality slice.",
        "- Weak multimodal rows come from current image-backed SimSat assessments and should be treated as weak labels.",
        "- This export is Sentinel-first and does not depend on Mapbox.",
        "",
        "## Runtime Guidance",
        "",
        "- T4: start with `simsat_format_train.jsonl` plus `simsat_multimodal_reviewed.jsonl`, then add a capped subset of weak rows.",
        "- A100: use the full bundle and higher decoding limits for structured JSON generation.",
        "",
        "## Source Paths",
        "",
        f"- `{manifest['source_paths']['traces']}`",
        f"- `{manifest['source_paths']['outcomes']}`",
        f"- `{manifest['source_paths']['submission_cases']}`",
        f"- `{manifest['source_paths']['targets']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SimSat ObservationVLA rows for the next Gemma v3 training/eval cycle.")
    parser.add_argument("--output-dir", default=str(ROOT / "exports" / "gemma4_v3"))
    parser.add_argument("--reviewed-holdout-per-scenario", type=int, default=1)
    parser.add_argument("--shortlist-per-target", type=int, default=1)
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear the output directory before export.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)

    traces = load_traces(TRACE_PATH)
    outcomes = load_outcomes(OUTCOME_PATH)
    submission_cases = load_submission_cases(SUBMISSION_CASE_PATH)
    targets = load_targets(TARGET_PATH)
    outcome_index = build_outcome_index(outcomes)
    submission_case_index = build_submission_case_index(submission_cases)

    format_rows: list[dict[str, Any]] = []
    weak_multimodal_rows: list[dict[str, Any]] = []
    reviewed_multimodal_source_rows: list[dict[str, Any]] = []
    copied_images: list[str] = []
    copied_image_set: set[str] = set()

    for trace in traces:
        trace_id = str(trace.get("trace_id") or "")
        target = targets.get(str(trace.get("target_id") or ""))
        outcome = outcome_index.get(trace_id)
        submission_case = submission_case_index.get(trace_id)
        prompt_text = build_prompt_text(trace.get("sample") or {}, target)

        resolved_image = resolve_image_path(trace, REVIEW_ASSETS_DIR, SUBMISSION_ASSETS_DIR)
        exported_image_path: str | None = None
        if resolved_image is not None and not args.no_copy_images:
            exported_image_path = copy_image_to_export(resolved_image, images_dir, trace)
            if exported_image_path not in copied_image_set:
                copied_image_set.add(exported_image_path)
                copied_images.append(exported_image_path)
        elif resolved_image is not None:
            exported_image_path = resolved_image.as_posix()

        if not is_stub_fallback_trace(trace):
            format_rows.append(
                build_base_row(
                    trace=trace,
                    target=target,
                    outcome=outcome,
                    submission_case=submission_case,
                    variant="format_train",
                    split="train",
                    image_path=None,
                    prompt_text=prompt_text,
                )
            )

        if exported_image_path and outcome and outcome.get("label_source") == "operator_review":
            row = build_base_row(
                trace=trace,
                target=target,
                outcome=outcome,
                submission_case=submission_case,
                variant="multimodal_reviewed",
                split="train",
                image_path=exported_image_path,
                prompt_text=prompt_text,
            )
            reviewed_multimodal_source_rows.append(row)
        elif exported_image_path and is_model_backed_trace(trace):
            row = build_base_row(
                trace=trace,
                target=target,
                outcome=outcome,
                submission_case=submission_case,
                variant="multimodal_weak",
                split="train",
                image_path=exported_image_path,
                prompt_text=prompt_text,
            )
            weak_multimodal_rows.append(row)

    reviewed_multimodal_rows, reviewed_eval_rows = reviewed_eval_split(
        reviewed_multimodal_source_rows,
        holdout_per_scenario=max(args.reviewed_holdout_per_scenario, 0),
    )

    shortlist_rows = shortlist_eval_rows(
        [row for row in weak_multimodal_rows if row.get("review_labels") is None],
        per_target_limit=max(args.shortlist_per_target, 0),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "simsat_format_train.jsonl", format_rows)
    write_jsonl(output_dir / "simsat_multimodal_weak.jsonl", weak_multimodal_rows)
    write_jsonl(output_dir / "simsat_multimodal_reviewed.jsonl", reviewed_multimodal_rows)
    write_jsonl(output_dir / "simsat_eval_reviewed.jsonl", reviewed_eval_rows)
    write_jsonl(output_dir / "simsat_eval_shortlist.jsonl", shortlist_rows)

    manifest = build_manifest(
        traces=traces,
        format_rows=format_rows,
        weak_rows=weak_multimodal_rows,
        reviewed_rows=reviewed_multimodal_rows,
        reviewed_eval_rows=reviewed_eval_rows,
        shortlist_rows=shortlist_rows,
        copied_images=copied_images,
    )
    write_manifest(output_dir / "manifest.json", manifest)
    write_readme(output_dir / "README.md", manifest)

    print(f"Gemma v3 export written to {output_dir}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
