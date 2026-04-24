"""
Bridge `benhaslam/simsat-gemma4-v3` JSONL trace records into `EncounterRecord`
instances for `simsat_env.SimSatEnv`.

This is the "adapter plug" that converts the training corpus already published
on Kaggle (from the Entry B / General AI Track Gemma-4 work) into the shape the
MuZero-SimSat port from Phase 5 T3 consumes.

INPUT FORMAT (one line per record)
----------------------------------
{
    "record_id":    "simsat_trace_cbe073faf8554afb986b632128204650_multimodal_weak",
    "scenario_pack": "disaster_response_weather",
    "target_label":  "Houston Ship Channel",
    "prompt_text":   "You are an Earth-observation assessment AI... [full context]",
    "target_json": {
        "usable_observation":       true,
        "scene_match_score":        0.87,
        "salience_score":           0.74,
        "change_or_event_score":    0.55,
        "occlusion_or_cloud_risk":  0.04,
        "confidence":               0.82,
        "recommended_action":       "refine",
        "rationale_tags":           ["cloud_risk", "high_priority"]
    },
    "has_image":   true,
    "image_path":  "images/trace_cbe073faf8554afb986b632128204650.png"
}

OUTPUT
------
A pickled list of `EncounterRecord` instances, ready to feed into
`SimSatEnv(records=records, backend="offline_replay")`.

USAGE
-----
    python build_encounter_records.py \
        --jsonl /path/to/simsat_multimodal_weak.jsonl \
        --images-dir /path/to/images/ \
        --output /path/to/encounter_records.pkl

Multiple JSONL inputs can be merged:

    python build_encounter_records.py \
        --jsonl /path/to/simsat_format_train.jsonl \
        --jsonl /path/to/simsat_multimodal_weak.jsonl \
        --jsonl /path/to/simsat_eval_reviewed.jsonl \
        --images-dir /path/to/images/ \
        --output /path/to/all_records.pkl

DEPENDENCIES
------------
Stdlib + numpy + Pillow. Does NOT require torch, transformers, or the
simsat_env module — it only imports `EncounterRecord` lazily, with a fallback
that redefines the dataclass locally for standalone testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Import EncounterRecord, with a local fallback for standalone testing.        #
# --------------------------------------------------------------------------- #

try:
    # Preferred: use the dataclass from Phase 5 T3's simsat_env module.
    from simsat_env import EncounterRecord
    _REMOTE_RECORD = True
except ImportError:
    _REMOTE_RECORD = False

    @dataclass
    class EncounterRecord:  # type: ignore[no-redef]
        """Local fallback matching simsat_env.EncounterRecord.

        If you have simsat_env.py on your PYTHONPATH, the remote definition is
        used instead and this local copy is shadowed. Keep the two in sync.
        """
        window_id: str
        target_id: str
        scenario_pack: str
        tile: np.ndarray
        target_priority: str = "normal"
        target_tags: list[str] = field(default_factory=list)
        sentinel_available: Optional[bool] = None
        sentinel_cloud_cover: Optional[float] = None
        elevation_degrees: Optional[float] = None
        line_of_sight: Optional[bool] = None
        actual_action: Optional[str] = None
        utility_realized: Optional[float] = None
        useful: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Prompt text parsing                                                          #
# --------------------------------------------------------------------------- #

# The v3 notebook embeds structured metadata in plain-text prompts. We recover
# those fields by regex rather than rely on a separate probe/geometry blob.
# If the prompt template evolves, extend these patterns.

_CLOUD_RE = re.compile(r"cloud[_\s]*cover[^0-9-]*([\d]+(?:\.\d+)?)", re.IGNORECASE)
_ELEVATION_RE = re.compile(r"elevation[^0-9-]*([\d]+(?:\.\d+)?)", re.IGNORECASE)
_SENTINEL_AVAILABLE_RE = re.compile(r"sentinel[_\s]*available[^a-z]*(true|false|yes|no)", re.IGNORECASE)
_LINE_OF_SIGHT_RE = re.compile(r"(line[_\s]of[_\s]sight|target[_\s]visible)[^a-z]*(true|false|yes|no)", re.IGNORECASE)
_PRIORITY_RE = re.compile(
    r"(?:target[_\s]?)?priority[^a-z]*(high[_\s]?priority|high[_\s]?visibility|normal|low)",
    re.IGNORECASE,
)


def _str_to_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "yes", "1", "t", "y")


def parse_prompt_metadata(prompt_text: str) -> dict[str, Any]:
    """Extract probe + geometry signals from the prompt text via regex.

    Returns a dict with whatever was found; missing fields stay out of the dict.
    Regexes are tolerant: trailing units ("%"), various label forms, and mixed
    separators all work.
    """
    md: dict[str, Any] = {}

    if m := _CLOUD_RE.search(prompt_text):
        try:
            md["sentinel_cloud_cover"] = float(m.group(1))
        except ValueError:
            pass

    if m := _ELEVATION_RE.search(prompt_text):
        try:
            md["elevation_degrees"] = float(m.group(1))
        except ValueError:
            pass

    if m := _SENTINEL_AVAILABLE_RE.search(prompt_text):
        md["sentinel_available"] = _str_to_bool(m.group(1))

    if m := _LINE_OF_SIGHT_RE.search(prompt_text):
        md["line_of_sight"] = _str_to_bool(m.group(2))

    if m := _PRIORITY_RE.search(prompt_text):
        raw = m.group(1).lower().replace(" ", "_")
        # Normalize to canonical tokens the trust model expects
        if raw in ("high_priority", "highpriority"):
            md["target_priority"] = "high_priority"
        elif raw in ("high_visibility", "highvisibility"):
            md["target_priority"] = "high_visibility"
        else:
            md["target_priority"] = raw

    return md


# --------------------------------------------------------------------------- #
# Image loading                                                                #
# --------------------------------------------------------------------------- #

OBS_H: int = 64
OBS_W: int = 64


def load_tile_image(image_path: Path, *, target_shape: tuple[int, int] = (OBS_H, OBS_W)) -> np.ndarray:
    """Load a PNG tile and return it as (1, H, W) float32 in raw Sentinel units.

    We do NOT normalize here — SimSatEnv does the 0..1 scaling downstream
    (`SENTINEL_NORM` division in `_tile_to_obs`). Keeping records in raw units
    lets a single corpus serve multiple normalization schemes.

    Missing file returns a zeros tile and logs a warning.
    """
    if not image_path.exists():
        logger.warning("image missing: %s — using zeros tile", image_path)
        return np.zeros((1, *target_shape), dtype=np.float32)

    from PIL import Image

    with Image.open(image_path) as im:
        if im.mode not in ("L", "I"):
            im = im.convert("L")  # grayscale luminance collapse
        if im.size != (target_shape[1], target_shape[0]):
            im = im.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        arr = np.array(im, dtype=np.float32)
    # Map 0..255 byte to pseudo-Sentinel scale so SENTINEL_NORM/10_000 division
    # lands the observation in a reasonable [0, 1] band. Ben: tune this if the
    # original v3 tiles are uint16 reflectance values rather than 8-bit PNG.
    arr = arr * (2500.0 / 255.0)
    return arr[np.newaxis, ...]


# --------------------------------------------------------------------------- #
# Record conversion                                                            #
# --------------------------------------------------------------------------- #

_TARGET_ID_SUFFIX_RE = re.compile(r"_(multimodal_weak|multimodal_reviewed|format_train|eval_reviewed|eval_shortlist)$")


def _trace_id_from_record_id(record_id: str) -> str:
    """Strip the phase suffix from simsat_trace_<uuid>_<suffix> → simsat_trace_<uuid>."""
    return _TARGET_ID_SUFFIX_RE.sub("", record_id)


def _target_slug(target_label: str) -> str:
    """Canonicalize a target label to a stable slug: 'Houston Ship Channel' → 'houston_ship_channel'."""
    return re.sub(r"[^\w]+", "_", target_label.strip().lower()).strip("_")


def v3_record_to_encounter(
    record: dict[str, Any],
    images_dir: Optional[Path] = None,
) -> EncounterRecord:
    """Convert one simsat-gemma4-v3 JSONL record into an `EncounterRecord`."""
    record_id = record.get("record_id", "unknown")
    trace_id = _trace_id_from_record_id(record_id)
    target_label = record.get("target_label", "unknown")
    target_id = _target_slug(target_label)
    scenario_pack = record.get("scenario_pack", "all")
    prompt_text = record.get("prompt_text", "") or ""
    target_json = record.get("target_json") or {}

    md = parse_prompt_metadata(prompt_text)

    # Load tile image if available
    tile: np.ndarray
    image_path_str = record.get("image_path")
    has_image = bool(record.get("has_image"))
    if has_image and image_path_str and images_dir is not None:
        image_path = (images_dir / image_path_str).resolve() if not Path(image_path_str).is_absolute() else Path(image_path_str)
        # The v3 corpus stores just "trace_<uuid>.png" under images_dir, not prefixed
        # with "images/" when image_path is already "images/trace_<uuid>.png". Try
        # both path shapes so the bridge is tolerant.
        if not image_path.exists():
            alt = images_dir / Path(image_path_str).name
            if alt.exists():
                image_path = alt
        tile = load_tile_image(image_path)
    else:
        tile = np.zeros((1, OBS_H, OBS_W), dtype=np.float32)

    # Labels from target_json — preserve ground truth for supervised stage-2
    action = target_json.get("recommended_action")
    usefulness = target_json.get("scene_match_score") or target_json.get("confidence")
    useful_flag = target_json.get("usable_observation") if target_json.get("usable_observation") is not None else None

    # Rationale tags → target_tags
    tags = list(target_json.get("rationale_tags", []) or [])

    return EncounterRecord(
        window_id=trace_id,
        target_id=target_id,
        scenario_pack=scenario_pack,
        tile=tile,
        target_priority=md.get("target_priority", "normal"),
        target_tags=tags,
        sentinel_available=md.get("sentinel_available"),
        sentinel_cloud_cover=md.get("sentinel_cloud_cover"),
        elevation_degrees=md.get("elevation_degrees"),
        line_of_sight=md.get("line_of_sight"),
        actual_action=action,
        utility_realized=float(usefulness) if isinstance(usefulness, (int, float)) else None,
        useful=bool(useful_flag) if useful_flag is not None else None,
    )


# --------------------------------------------------------------------------- #
# JSONL loading                                                                #
# --------------------------------------------------------------------------- #

def load_records_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one dict per non-empty line of a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d: invalid JSON line skipped (%s)", path, lineno, exc)


def build_corpus(
    jsonl_paths: list[Path],
    images_dir: Optional[Path] = None,
    *,
    scenario_pack_filter: Optional[str] = None,
) -> list[EncounterRecord]:
    """Load records from one or more JSONL files, return an ordered list."""
    all_records: list[EncounterRecord] = []
    per_file_stats: dict[str, int] = {}
    for path in jsonl_paths:
        before = len(all_records)
        for record_dict in load_records_from_jsonl(path):
            if scenario_pack_filter and record_dict.get("scenario_pack") != scenario_pack_filter:
                continue
            try:
                rec = v3_record_to_encounter(record_dict, images_dir=images_dir)
                all_records.append(rec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("record conversion failed for %s: %s",
                               record_dict.get("record_id", "?"), exc)
        per_file_stats[path.name] = len(all_records) - before
    logger.info("loaded records per file: %s", per_file_stats)
    return all_records


# --------------------------------------------------------------------------- #
# Summary helpers                                                              #
# --------------------------------------------------------------------------- #

def summarize(records: list[EncounterRecord]) -> dict[str, Any]:
    """Compute per-scenario-pack and per-action distribution stats."""
    by_pack: dict[str, int] = {}
    by_action: dict[str, int] = {}
    has_tile_nonzero = 0

    for r in records:
        by_pack[r.scenario_pack] = by_pack.get(r.scenario_pack, 0) + 1
        action = r.actual_action or "unlabeled"
        by_action[action] = by_action.get(action, 0) + 1
        if np.any(r.tile != 0):
            has_tile_nonzero += 1

    return {
        "total_records": len(records),
        "by_scenario_pack": dict(sorted(by_pack.items())),
        "by_action": dict(sorted(by_action.items())),
        "records_with_nonzero_tile": has_tile_nonzero,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bridge simsat-gemma4-v3 JSONL records into EncounterRecord pickle."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        required=True,
        help="Path to a simsat-gemma4-v3 JSONL file. Repeatable to merge multiple.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing trace_*.png tiles. Omit for text-only records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the pickled list[EncounterRecord].",
    )
    parser.add_argument(
        "--scenario-pack",
        default=None,
        help="Optional filter: only convert records for this scenario_pack.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a summary JSON (counts, distributions).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    records = build_corpus(
        args.jsonl,
        images_dir=args.images_dir,
        scenario_pack_filter=args.scenario_pack,
    )

    summary = summarize(records)
    logger.info("summary: %s", summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(records, f)
    logger.info("wrote %d records → %s", len(records), args.output)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("wrote summary → %s", args.summary_json)

    print(f"[build_encounter_records] records={len(records)} "
          f"scenarios={summary['by_scenario_pack']} actions={summary['by_action']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
