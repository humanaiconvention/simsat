# `simsat-gemma4-v3` record schema (reference)

Documents the JSONL shape T4's corpus bridge consumes. Derived from the attached
`notebook.ipynb` and `simsat_run.log` of the `benhaslam/kaggle-simsat-gemma4-v3`
Kaggle kernel. If you update the v3 dataset and the record shape changes, also
update `build_encounter_records.py::v3_record_to_encounter` and the regex
constants at the top of the file.

## JSONL files in `benhaslam/simsat-gemma4-v3`

| File | Purpose | `has_image` | Count (observed) |
|---|---|---|---|
| `simsat_format_train.jsonl` | Text-only JSON-format fine-tune | `False` | 66 |
| `simsat_multimodal_weak.jsonl` | Multimodal (CLIP-scored pseudo-labels) | `True` | 53 |
| `simsat_multimodal_reviewed.jsonl` | Multimodal + operator-reviewed | `True` | (count unknown) |
| `simsat_eval_reviewed.jsonl` | 3 expert-reviewed eval cases (pinned) | varies | 3 |
| `simsat_eval_shortlist.jsonl` | 10 shortlist eval cases (1/target) | varies | 10 |

Plus `images/trace_<uuid>.png` (~60 tiles) and `manifest.json` + `README.md`.

## One record example

```json
{
  "record_id": "simsat_trace_cbe073faf8554afb986b632128204650_multimodal_weak",
  "scenario_pack": "disaster_response_weather",
  "target_label": "Houston Ship Channel",
  "prompt_text": "You are an Earth-observation assessment AI ... [full context]",
  "target_json": {
    "usable_observation": true,
    "scene_match_score": 0.87,
    "salience_score": 0.74,
    "change_or_event_score": 0.55,
    "occlusion_or_cloud_risk": 0.04,
    "confidence": 0.82,
    "recommended_action": "refine",
    "rationale_tags": ["cloud_risk", "high_priority"]
  },
  "has_image": true,
  "image_path": "images/trace_cbe073faf8554afb986b632128204650.png"
}
```

## Field-by-field mapping → `EncounterRecord`

| JSONL field | `EncounterRecord` field | Extraction notes |
|---|---|---|
| `record_id` | `window_id` | Strip suffix `_{multimodal_weak \| multimodal_reviewed \| format_train \| eval_reviewed \| eval_shortlist}` |
| `target_label` | `target_id` | Slugify: `"Houston Ship Channel"` → `"houston_ship_channel"` |
| `target_label` | (also preserved in logs) | — |
| `scenario_pack` | `scenario_pack` | Pass-through |
| `prompt_text` | (parsed) | Regex-extracts cloud cover, elevation, sentinel_available, line_of_sight, target_priority |
| `target_json.recommended_action` | `actual_action` | `"accept" \| "defer" \| "refine" \| "skip"` |
| `target_json.scene_match_score` or `target_json.confidence` | `utility_realized` | First non-null float in (0, 1] |
| `target_json.usable_observation` | `useful` | Bool cast |
| `target_json.rationale_tags` | `target_tags` | List pass-through |
| `has_image` + `image_path` | `tile` | Load PNG → (1, 64, 64) float32; zeros tile if missing |
| (regex) | `sentinel_available` | `"sentinel_available: true"` → `True` |
| (regex) | `sentinel_cloud_cover` | `"cloud_cover: 4.5"` → `4.5` (percent) |
| (regex) | `elevation_degrees` | `"elevation: 72.3"` → `72.3` |
| (regex) | `line_of_sight` | `"line_of_sight: true"` or `"target_visible: true"` → `True` |
| (regex) | `target_priority` | Normalizes `"high_priority" \| "high_visibility" \| "normal"` |

## Prompt metadata regex patterns

Tolerant to trailing `%`, underscores vs spaces, various label forms:

```python
_CLOUD_RE             = r"cloud[_\s]*cover[^0-9-]*([\d]+(?:\.\d+)?)"
_ELEVATION_RE         = r"elevation[^0-9-]*([\d]+(?:\.\d+)?)"
_SENTINEL_AVAILABLE_RE = r"sentinel[_\s]*available[^a-z]*(true|false|yes|no)"
_LINE_OF_SIGHT_RE     = r"(line[_\s]of[_\s]sight|target[_\s]visible)[^a-z]*(true|false|yes|no)"
_PRIORITY_RE          = r"(?:target[_\s]?)?priority[^a-z]*(high[_\s]?priority|high[_\s]?visibility|normal|low)"
```

If your prompt template includes fields these patterns don't catch (e.g., `off_nadir_degrees`, `recency_hours`), extend the regex set and the parser function.

## Edge cases the bridge handles

- **Missing image file**: logs a warning, inserts zeros tile (shape `(1, 64, 64)` float32). Non-fatal so JSONL file with broken `image_path` entries still produces a complete record list.
- **Missing `target_json`**: record is still loaded, but `actual_action`, `utility_realized`, `useful` all end up `None`. Stage-1 pretrain can still use the tile + scenario_pack signal even without ground-truth action.
- **Prompt metadata missing a field**: regex just doesn't match; `EncounterRecord` default (`None` or `"normal"`) applies.
- **`record_id` without a recognized suffix**: entire `record_id` is used as `window_id`; no stripping.
- **Mixed JSONL sources**: `build_corpus` accepts multiple paths and merges in order. Useful for training on `format_train + multimodal_weak + multimodal_reviewed` then evaluating on `eval_shortlist`.

## Known issues in the upstream Kaggle run

(Tracked in the project doc; documented here so future bridge users know what to expect.)

- **All v3 predictions return `PARSE_ERROR`** with response `{"usable_` (9 chars). The v3 notebook attempts a fix (`max_new_tokens=512`) but the attached log shows it didn't take — generation stops mid-token at `usable_`. Likely cause: tokenizer emits `<end_of_turn>` or similar stop token right after the partial `usable_observation` key.
- **Training loss does decrease** (Phase 1: 1.54 → 0.98; Phase 2: 1.47 → 1.17), so adaptation works. It's the generation-termination logic that's broken.
- **The MuZero path does not share this failure mode** — MuZero emits a categorical policy distribution, no JSON parsing required. This is one of the stronger technical arguments for preferring MuZero over VLM-emits-JSON as the SimSat submission backbone.
