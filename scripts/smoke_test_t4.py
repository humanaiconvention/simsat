"""End-to-end smoke test for T4: corpus bridge + tile encoders.

Runs entirely in the sandbox — no torch, no transformers, no running Django,
no real LFM2.5 weights. Uses synthetic JSONL records matching the
simsat-gemma4-v3 schema plus synthetic tile PNGs.

Verifies:

  1. `parse_prompt_metadata` extracts cloud cover, elevation, sentinel
     availability, line-of-sight, and target priority from prompt text.
  2. `v3_record_to_encounter` maps a v3 JSONL record correctly onto an
     `EncounterRecord` (action, utility, useful flag, tile shape).
  3. `build_corpus` loads multiple JSONL files and merges into one list.
  4. `summarize` produces per-scenario-pack and per-action distributions.
  5. `IdentityEncoder` passes through pixel tiles unchanged.
  6. `DummyEncoder` returns stable `embed_dim`-vectors, deterministic across calls.
  7. `LFM2VLEncoderStub.encode()` raises NotImplementedError (intentional).
  8. Pickled output file round-trips back to a list of EncounterRecord.

Usage:
    python scripts/smoke_test_build_corpus.py

Exit code 0 on green, 1 on any assertion failure.
"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

# Make src/sim/muzero and scripts/ importable
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from build_encounter_records import (
    OBS_H, OBS_W,
    EncounterRecord,
    build_corpus,
    load_records_from_jsonl,
    load_tile_image,
    parse_prompt_metadata,
    summarize,
    v3_record_to_encounter,
)
from sim.muzero.tile_encoder import (
    DummyEncoder,
    IdentityEncoder,
    LFM2VLEncoderStub,
    build_encoder,
)


# --------------------------------------------------------------------------- #
# Synthetic corpus fixtures                                                    #
# --------------------------------------------------------------------------- #

SYNTHETIC_TARGETS = {
    "Houston Ship Channel":  ("disaster_response_weather", "houston_ship_channel", "accept", 0.92, True),
    "Suez Canal":            ("maritime_chokepoints",     "suez_canal",            "accept", 0.95, True),
    "San Francisco Bay":     ("urban_coastal_ambiguity",  "san_francisco_bay",     "accept", 0.90, True),
    "New Orleans Delta":     ("disaster_response_weather", "new_orleans_delta",    "refine", 0.67, False),
    "Panama Canal":          ("maritime_chokepoints",     "panama_canal",          "refine", 0.71, True),
    "Port of Rotterdam":     ("urban_coastal_ambiguity",  "port_of_rotterdam",     "refine", 0.68, False),
    "Shenzhen Bay":          ("urban_coastal_ambiguity",  "shenzhen_bay",          "defer",  0.55, False),
}


def _synthetic_prompt(target_label: str, cloud: float, elev: float, avail: bool, los: bool,
                      priority: str = "high_priority") -> str:
    """Emit a prompt text that parse_prompt_metadata can extract metadata from."""
    return (
        "You are an Earth-observation assessment AI for the SimSat mission planner. "
        f"Target: {target_label}. "
        f"priority: {priority}. "
        f"sentinel_available: {str(avail).lower()}. "
        f"cloud_cover: {cloud:.1f}. "
        f"elevation: {elev:.1f}. "
        f"line_of_sight: {str(los).lower()}. "
        "Respond only with valid JSON matching the schema."
    )


def _synthetic_record(idx: int, target_label: str, suffix: str) -> dict:
    pack, slug, action, util, useful = SYNTHETIC_TARGETS[target_label]
    rng = np.random.default_rng(seed=idx)
    cloud = float(rng.uniform(0.0, 30.0))
    elev = float(rng.uniform(30.0, 80.0))
    return {
        "record_id": f"simsat_trace_{idx:032x}_{suffix}",
        "scenario_pack": pack,
        "target_label": target_label,
        "prompt_text": _synthetic_prompt(target_label, cloud, elev, True, True),
        "target_json": {
            "usable_observation": useful,
            "scene_match_score": util,
            "salience_score": 0.80,
            "change_or_event_score": 0.50,
            "occlusion_or_cloud_risk": cloud / 100.0,
            "confidence": util,
            "recommended_action": action,
            "rationale_tags": ["high_priority", "sentinel_available"] if useful else ["cloud_risk"],
        },
        "has_image": True,
        "image_path": f"images/trace_{idx:032x}.png",
    }


def _write_synthetic_corpus(tmp_dir: Path) -> tuple[Path, Path, Path]:
    """Create a synthetic v3-shaped corpus. Returns (jsonl_path, images_dir, all_jsonl)."""
    from PIL import Image

    images_dir = tmp_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Write 6 synthetic records across all 3 scenario packs
    records = []
    target_names = list(SYNTHETIC_TARGETS.keys())
    for i in range(6):
        rec = _synthetic_record(i, target_names[i % len(target_names)], "multimodal_weak")
        records.append(rec)
        # Generate a synthetic tile PNG (grayscale random)
        tile = np.random.default_rng(seed=i).integers(0, 256, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(tile, mode="L")
        img.save(images_dir / f"trace_{i:032x}.png")

    jsonl_path = tmp_dir / "simsat_multimodal_weak.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Also write a smaller eval_reviewed file (no images, different suffix)
    reviewed_records = [
        _synthetic_record(100, "Houston Ship Channel",  "eval_reviewed"),
        _synthetic_record(101, "Suez Canal",            "eval_reviewed"),
        _synthetic_record(102, "San Francisco Bay",     "eval_reviewed"),
    ]
    for r in reviewed_records:
        r["has_image"] = False
        r["image_path"] = None

    reviewed_path = tmp_dir / "simsat_eval_reviewed.jsonl"
    with open(reviewed_path, "w", encoding="utf-8") as f:
        for r in reviewed_records:
            f.write(json.dumps(r) + "\n")

    return jsonl_path, images_dir, reviewed_path


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

def test_parse_prompt_metadata() -> None:
    # Use values that won't suffer from float-format rounding surprises
    prompt = _synthetic_prompt("Suez Canal", cloud=4.5, elev=72.3, avail=True, los=True)
    md = parse_prompt_metadata(prompt)
    # Tolerance 0.2 absorbs the ":.1f" format rounding in _synthetic_prompt
    assert abs(md.get("sentinel_cloud_cover", 0) - 4.5) < 0.2, f"cloud got {md}"
    assert abs(md.get("elevation_degrees", 0) - 72.3) < 0.2, f"elev got {md}"
    assert md.get("sentinel_available") is True, f"avail got {md}"
    assert md.get("line_of_sight") is True, f"los got {md}"
    assert md.get("target_priority") == "high_priority", f"priority got {md}"
    print(f"  \u2713 parse_prompt_metadata: {md}")


def test_v3_record_to_encounter() -> None:
    rec_dict = _synthetic_record(idx=5, target_label="Suez Canal", suffix="multimodal_weak")
    rec = v3_record_to_encounter(rec_dict, images_dir=None)  # no images dir -> zeros tile

    assert rec.target_id == "suez_canal", f"target_id got {rec.target_id}"
    assert rec.scenario_pack == "maritime_chokepoints", f"scenario_pack got {rec.scenario_pack}"
    assert rec.actual_action == "accept", f"action got {rec.actual_action}"
    assert rec.useful is True, f"useful got {rec.useful}"
    assert rec.target_priority == "high_priority", f"priority got {rec.target_priority}"
    assert rec.tile.shape == (1, OBS_H, OBS_W), f"tile shape got {rec.tile.shape}"
    assert set(rec.target_tags) >= {"high_priority"}, f"tags got {rec.target_tags}"
    print(f"  \u2713 v3_record_to_encounter: target={rec.target_id} "
          f"action={rec.actual_action} util={rec.utility_realized}")


def test_build_corpus_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        weak_path, images_dir, reviewed_path = _write_synthetic_corpus(tmp)

        # Build from multiple JSONL, with images
        records = build_corpus(
            [weak_path, reviewed_path],
            images_dir=images_dir,
        )
        assert len(records) == 9, f"expected 6 + 3 = 9 records, got {len(records)}"

        # Invariant (independent of filename prefix):
        #   - 6 records have has_image=True in JSONL → should have non-zero tiles
        #   - 3 records (eval_reviewed) have has_image=False → should have zero tiles
        nonzero_tile_count = sum(1 for r in records if np.any(r.tile != 0))
        zero_tile_count    = sum(1 for r in records if not np.any(r.tile != 0))
        assert nonzero_tile_count == 6, f"expected 6 non-zero tiles (multimodal_weak), got {nonzero_tile_count}"
        assert zero_tile_count    == 3, f"expected 3 zero tiles (eval_reviewed has_image=False), got {zero_tile_count}"

        summary = summarize(records)
        assert summary["total_records"] == 9
        assert summary["by_scenario_pack"]
        assert summary["by_action"]
        print(f"  \u2713 build_corpus: {summary}")

        # Round-trip pickle
        pkl_path = tmp / "records.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(records, f)
        with open(pkl_path, "rb") as f:
            reloaded = pickle.load(f)
        assert len(reloaded) == 9
        assert reloaded[0].target_id == records[0].target_id
        assert np.array_equal(reloaded[0].tile, records[0].tile)
        print(f"  \u2713 pickle round-trip: {len(reloaded)} records intact")


def test_scenario_pack_filter() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        weak_path, images_dir, reviewed_path = _write_synthetic_corpus(tmp)

        maritime_only = build_corpus(
            [weak_path, reviewed_path],
            images_dir=images_dir,
            scenario_pack_filter="maritime_chokepoints",
        )
        # Of 9 records, Suez (target "Suez Canal") + Panama Canal appear in maritime pack.
        # In our cycling pattern at i=1 (Suez), i=4 (Panama), and i=101 (Suez reviewed) map to maritime.
        assert all(r.scenario_pack == "maritime_chokepoints" for r in maritime_only)
        assert len(maritime_only) >= 1
        print(f"  \u2713 scenario_pack_filter: {len(maritime_only)} maritime records")


def test_tile_image_load() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Write a known-value tile
        from PIL import Image
        data = np.full((64, 64), 128, dtype=np.uint8)
        Image.fromarray(data, mode="L").save(tmp / "test.png")

        tile = load_tile_image(tmp / "test.png")
        assert tile.shape == (1, OBS_H, OBS_W), f"tile shape got {tile.shape}"
        # 128/255 * 2500 ≈ 1254 — should be in that range
        assert 1200 < tile.mean() < 1350, f"tile mean out of expected band: {tile.mean()}"
        print(f"  \u2713 load_tile_image: shape {tile.shape}, mean {tile.mean():.1f}")

        # Missing file → zeros
        zeros = load_tile_image(tmp / "nope.png")
        assert np.all(zeros == 0)
        print(f"  \u2713 missing tile → zeros tile")


def test_identity_encoder() -> None:
    enc = IdentityEncoder()
    assert enc.embed_dim is None, f"embed_dim got {enc.embed_dim}"
    tile = np.ones((1, 64, 64), dtype=np.float32) * 0.5
    out = enc.encode(tile)
    assert np.array_equal(out, tile), "IdentityEncoder should pass through"
    print(f"  \u2713 IdentityEncoder: embed_dim=None, pass-through")


def test_dummy_encoder() -> None:
    enc = DummyEncoder(embed_dim=64)
    tile_a = np.ones((1, 64, 64), dtype=np.float32) * 0.5
    tile_b = np.ones((1, 64, 64), dtype=np.float32) * 0.9
    tile_a2 = np.ones((1, 64, 64), dtype=np.float32) * 0.5  # same as a

    emb_a = enc.encode(tile_a)
    emb_b = enc.encode(tile_b)
    emb_a2 = enc.encode(tile_a2)

    assert emb_a.shape == (64,), f"embed shape got {emb_a.shape}"
    assert np.array_equal(emb_a, emb_a2), "DummyEncoder must be deterministic on same input"
    assert not np.array_equal(emb_a, emb_b), "DummyEncoder should differ on different inputs"
    print(f"  \u2713 DummyEncoder: shape={emb_a.shape}, deterministic={np.allclose(emb_a, emb_a2)}, "
          f"distinguishes={not np.allclose(emb_a, emb_b)}")


def test_lfm2vl_stub_raises() -> None:
    enc = LFM2VLEncoderStub(embed_dim=768)
    tile = np.ones((1, 64, 64), dtype=np.float32)
    try:
        enc.encode(tile)
        raise AssertionError("LFM2VLEncoderStub.encode should have raised NotImplementedError")
    except NotImplementedError as exc:
        msg = str(exc)
        assert "not yet wired" in msg.lower() or "integration" in msg.lower()
        print(f"  \u2713 LFM2VLEncoderStub.encode() raises NotImplementedError as expected")


def test_build_encoder_factory() -> None:
    identity = build_encoder("identity")
    assert isinstance(identity, IdentityEncoder)

    dummy = build_encoder("dummy", embed_dim=32)
    assert isinstance(dummy, DummyEncoder)
    assert dummy.embed_dim == 32

    lfm = build_encoder("lfm2vl")
    assert isinstance(lfm, LFM2VLEncoderStub)

    try:
        build_encoder("unknown_kind")
        raise AssertionError("should have raised for unknown kind")
    except ValueError:
        pass

    print(f"  \u2713 build_encoder factory: identity / dummy / lfm2vl / invalid-rejected")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 70)
    print("Phase 5 T4 smoke tests — corpus bridge + tile encoders")
    print("=" * 70)

    print("\n[1] parse_prompt_metadata")
    test_parse_prompt_metadata()

    print("\n[2] v3_record_to_encounter")
    test_v3_record_to_encounter()

    print("\n[3] build_corpus end-to-end")
    test_build_corpus_end_to_end()

    print("\n[4] scenario_pack filter")
    test_scenario_pack_filter()

    print("\n[5] load_tile_image")
    test_tile_image_load()

    print("\n[6] IdentityEncoder")
    test_identity_encoder()

    print("\n[7] DummyEncoder (deterministic + distinguishes)")
    test_dummy_encoder()

    print("\n[8] LFM2VLEncoderStub.encode() raises")
    test_lfm2vl_stub_raises()

    print("\n[9] build_encoder factory")
    test_build_encoder_factory()

    print("\n" + "=" * 70)
    print("All T4 smoke tests PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
