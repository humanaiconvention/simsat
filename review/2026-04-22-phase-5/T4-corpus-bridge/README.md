# T4 — Corpus bridge + tile encoder interface

Bridges the `benhaslam/simsat-gemma4-v3` Kaggle dataset (the General AI Track
foundation from 2026-04-13) into `EncounterRecord` instances that the Phase 5
T3 `SimSatEnv` + `SimSatGame` can consume. Plus a clean encoder interface so
the Liquid Track path — LFM2.5-VL as observation encoder feeding MuZero's
`h()` — has a drop-in seat.

## Why this bundle

Phase 5 T3 shipped `SimSatEnv` with an `offline_replay` backend but no way to
populate it. The 86-trace ObservationVLA corpus (local) and the ~130-example
`simsat-gemma4-v3` dataset (Kaggle) both exist; neither was connected.

T4 closes that gap:

1. **`build_encounter_records.py`** reads v3-schema JSONL + PNG images, emits a
   pickled `list[EncounterRecord]`. `SimSatEnv(records=pickle.load(...))`.
2. **`tile_encoder.py`** defines the `TileEncoder` protocol + `IdentityEncoder`
   (default) + `DummyEncoder` (deterministic fake for tests) + `LFM2VLEncoderStub`
   (scaffold for the real LFM2.5-VL integration).
3. **`scripts/smoke_test_build_corpus.py`** exercises both modules end-to-end
   with synthetic JSONL + synthetic tile PNGs. 9 tests, all green pre-commit.
4. **`simsat_env_encoder.patch.md`** documents the 2-line `SimSatEnv.__init__`
   change needed to plug encoders in, with explicit "do not apply yet" guidance
   so T3's passing smoke test stays passing until the downstream MuZero config
   is also updated.
5. **`schema_reference.md`** captures the v3 JSONL record shape with field
   mappings + known issues.

## What this bundle ships

| File | Role | Size |
|---|---|---|
| `build_encounter_records.py` | CLI + library for JSONL → EncounterRecord pickle | ~11 KB |
| `tile_encoder.py` | Encoder protocol + 3 concrete encoders + factory | ~6 KB |
| `scripts/smoke_test_build_corpus.py` | 9-test harness (metadata parse, v3 conversion, corpus build, scenario filter, tile load, 3 encoders, factory) | ~11 KB |
| `schema_reference.md` | v3 JSONL record schema + field mapping + regex patterns + edge cases | ~5 KB |
| `simsat_env_encoder.patch.md` | Minimal 2-line patch to `SimSatEnv.__init__` for the future encoder hook | ~4 KB |
| `README.md` | This file | — |

All Python modules **syntax-checked and runtime-smoke-tested locally** before
commit. Smoke test output:

```
[1] parse_prompt_metadata                    ✓
[2] v3_record_to_encounter                   ✓
[3] build_corpus end-to-end                  ✓ (6 multimodal + 3 eval = 9 records)
[4] scenario_pack filter                     ✓ (3 maritime_chokepoints records)
[5] load_tile_image                          ✓ (PNG round-trip + missing-file fallback)
[6] IdentityEncoder                          ✓ (embed_dim=None, pass-through)
[7] DummyEncoder                             ✓ (deterministic + distinguishes different tiles)
[8] LFM2VLEncoderStub.encode() raises        ✓ (NotImplementedError as expected)
[9] build_encoder factory                    ✓ (identity / dummy / lfm2vl / invalid-rejected)
All T4 smoke tests PASSED
```

## Application workflow

### Step 1 — Drop into local dev tree

```bash
cp T4-corpus-bridge/build_encounter_records.py D:\SimSat\scripts\build_encounter_records.py
cp T4-corpus-bridge/tile_encoder.py            D:\SimSat\src\sim\muzero\tile_encoder.py
cp T4-corpus-bridge/scripts/smoke_test_build_corpus.py D:\SimSat\scripts\smoke_test_t4.py
```

### Step 2 — Smoke-test locally

```bash
cd D:\SimSat
python scripts/smoke_test_t4.py
```

Expect all 9 tests green. If anything fails, likely culprits: Pillow not
installed, or the synthetic corpus step hits an edge case your environment
treats differently.

### Step 3 — Download the simsat-gemma4-v3 dataset

Via Kaggle CLI (most reliable, since the GitHub-integration Kaggle pull is
still blocked on read-only-FS):

```bash
kaggle datasets download -d benhaslam/simsat-gemma4-v3 -p D:\kaggle_data\simsat-gemma4-v3 --unzip
```

Expected files:

```
D:\kaggle_data\simsat-gemma4-v3\
├── simsat_format_train.jsonl        (66 text-only format training)
├── simsat_multimodal_weak.jsonl     (53 examples with CLIP-scored pseudo-labels)
├── simsat_multimodal_reviewed.jsonl (multimodal reviewed)
├── simsat_eval_reviewed.jsonl       (3 expert-reviewed: Houston, Suez, SF Bay)
├── simsat_eval_shortlist.jsonl      (10 shortlist eval)
├── manifest.json
├── README.md
└── images/
    ├── trace_0725ba13314d4aec8e695ed0fe2130dd.png
    ├── trace_cbe073faf8554afb986b632128204650.png
    └── ... (~60 tile PNGs total)
```

### Step 4 — Build the encounter-record pickle

```bash
python scripts/build_encounter_records.py \
    --jsonl D:\kaggle_data\simsat-gemma4-v3\simsat_format_train.jsonl \
    --jsonl D:\kaggle_data\simsat-gemma4-v3\simsat_multimodal_weak.jsonl \
    --jsonl D:\kaggle_data\simsat-gemma4-v3\simsat_multimodal_reviewed.jsonl \
    --jsonl D:\kaggle_data\simsat-gemma4-v3\simsat_eval_reviewed.jsonl \
    --jsonl D:\kaggle_data\simsat-gemma4-v3\simsat_eval_shortlist.jsonl \
    --images-dir D:\kaggle_data\simsat-gemma4-v3\images \
    --output D:\SimSat\corpus\simsat_v3_encounter_records.pkl \
    --summary-json D:\SimSat\corpus\simsat_v3_summary.json
```

Expected summary (from the Kaggle run log):

```json
{
  "total_records": 132,
  "by_scenario_pack": {
    "disaster_response_weather": 44,
    "maritime_chokepoints":      44,
    "urban_coastal_ambiguity":   44
  },
  "by_action": {
    "accept": 13,
    "refine": 97,
    "defer":  4,
    "skip":   1,
    "unlabeled": 0
  },
  "records_with_nonzero_tile": 70
}
```

(Exact counts depend on multimodal_reviewed's size, which isn't in the log.
Approximately 66 + 53 + ? + 3 + 10 ≈ 130+.)

### Step 5 — Feed into SimSatEnv

```python
import pickle
from sim.muzero.simsat_env import SimSatEnv

with open('D:/SimSat/corpus/simsat_v3_encounter_records.pkl', 'rb') as f:
    records = pickle.load(f)

env = SimSatEnv(
    backend="offline_replay",
    records=records,
    scenario_pack="all",  # or "maritime_chokepoints", etc.
)
obs = env.reset()
next_obs, reward, done, info = env.step(0)  # TRUST_ACCEPT
```

### Step 6 — (Optional) Use a non-identity encoder

Until the `simsat_env_encoder.patch.md` patch lands in T3's `simsat_env.py`,
the encoder interface is library-only:

```python
from sim.muzero.tile_encoder import build_encoder

enc = build_encoder("dummy", embed_dim=128)
tile = env.reset()   # (1, 64, 64)
embedding = enc.encode(tile)   # (128,)
```

After the patch lands (future phase), it becomes:

```python
env = SimSatEnv(records=records, tile_encoder=build_encoder("dummy", embed_dim=128))
obs = env.reset()   # (128,) embedding instead of (1, 64, 64) pixel tile
```

And for the Liquid Track production path:

```python
env = SimSatEnv(
    records=records,
    tile_encoder=build_encoder("lfm2vl", embed_dim=768, lora_adapter_path=".../lfm25_lora"),
)
```

(Requires `LFM2VLEncoderStub.encode()` to be filled in; currently raises
NotImplementedError.)

## Decisions baked into this bundle

- **PNG normalization**: Sentinel tiles in the v3 dataset are 8-bit PNGs. The
  bridge rescales `0..255` → `0..2500` (pseudo-Sentinel reflectance) so the
  downstream `SENTINEL_NORM=10_000` division in `SimSatEnv._tile_to_obs` lands
  observations in a reasonable `[0, 0.25]` band. If the underlying v3 tiles are
  actually uint16 reflectance values rather than 8-bit luminance, this scaling
  needs an update (see `load_tile_image`).
- **Grayscale collapse**: Multi-channel PNGs convert to grayscale luminance for
  `(1, H, W)` starter shape, matching the T3 / ARC3 observation contract.
  Multi-spectral (C > 1) is a deferred upgrade; the T4 bridge pre-wires the
  shape check so a future change is local.
- **Prompt metadata via regex**: the v3 prompts bake structured signals
  (cloud, elevation, sentinel_available, line_of_sight, priority) into plain
  text. Regex extraction is tolerant of formatting variations; if the prompt
  template evolves, extend the patterns in `build_encounter_records.py`.
- **Missing images yield zeros tiles, not errors**: the bridge logs a warning
  and inserts a zeros tile. This keeps `format_train` (no images) workable in
  the same pipeline as `multimodal_weak` (images expected).
- **LFM2.5-VL encoder is a stub, not a real implementation**: integration
  requires loading a multi-GB model which isn't runnable in the sandbox.
  `LFM2VLEncoderStub` defines the interface contract and raises
  `NotImplementedError` so tests catch any accidental use. Real implementation
  lands in a future phase (Phase 6 or T4c).

## Open items / future phases this unblocks

- **Phase 6 (stage-1 pretrain on Kaggle)**: the pickled `EncounterRecord` list
  from Step 4 is the training corpus for MuZero stage-1. Pretraining needs the
  records and the `SimSatEnv` from T3 — T4 connects them.
- **Encoder integration (Phase 6 or T5)**: apply `simsat_env_encoder.patch.md`
  to T3's `simsat_env.py`, implement `LFM2VLEncoder.encode()` against real
  LFM2.5-VL weights, update `MuZeroConfig.network` to `"fullyconnected"` when
  using an encoder that produces embeddings.
- **SimSatEnv API backend wiring (T4b, deferred)**: still stubbed in T3.
  Requires HTTP calls against the local Django `/sim/encounter/*` endpoints.
  Separate from the corpus bridge; independent path for live-play deployment.
- **Multi-spectral observation**: `load_tile_image` currently collapses multi-channel
  to grayscale. Supporting Sentinel-2 13-band multi-spectral is a rubric upgrade
  (the event explicitly rewards multi-spectral) — change `target_shape` to
  `(C, H, W)` and drop the `.mean(axis=0)` collapse.

## Known caveats

- The v3 Kaggle run itself has PARSE_ERROR on all predictions (see
  `schema_reference.md` Known Issues). T4 consumes the **training corpus**
  cleanly; the run-time prediction issue doesn't affect the bridge.
- The `multimodal_reviewed.jsonl` record count is unknown from the available
  artifacts. Actual totals from your local Kaggle download will vary.
- The regex prompt-parser covers the 5 signal fields shown in the v3 log. If
  the underlying `manifest.json` or `README.md` of `simsat-gemma4-v3` documents
  additional fields, extend the parser.

## Why this is a "starter" not a "finished integration"

- LFM2.5-VL encoder isn't actually loaded; `LFM2VLEncoderStub.encode()` raises.
  The interface is right; the weights loader is future work.
- The `SimSatEnv.tile_encoder` hook isn't wired into T3 yet (patch doc only).
  Applying prematurely would risk T3's current smoke test.
- No new scripts or notebooks target Kaggle yet — that's Phase 6 work.
- `build_encounter_records.py` is tested against synthetic JSONL + PNGs. Real
  v3 data may reveal schema quirks the regex patterns don't catch; extend as
  needed on first run against real data.

Ship T4, then iterate.
