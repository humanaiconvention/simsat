# Wiring `tile_encoder` into `SimSatEnv` (Phase 5 T3)

The T3 bundle (`simsat_env.py`) doesn't yet accept a `tile_encoder` parameter.
This patch shows the minimal two-line change needed to plug T4's encoder
interface into the env. **Do not apply yet** — wait until the LFM2.5-VL
integration phase (Phase 6 or equivalent) to avoid churning the T3 bundle that
the Phase 5 T3 smoke tests already validate.

## Minimal change

In `review/2026-04-22-phase-5/T3-muzero-simsat-port/simsat_env.py`:

### 1. Add `tile_encoder` kwarg to `SimSatEnv.__init__`

```python
# Before
def __init__(
    self,
    backend: str = "offline_replay",
    records: Optional[list[EncounterRecord]] = None,
    scenario_pack: str = "all",
    api_base_url: Optional[str] = None,
    step_penalty: float = -0.01,
    materialize_not_useful_penalty: float = -0.10,
    refine_cost: float = -0.05,
    skip_missed_penalty: float = -0.01,
    seed: Optional[int] = None,
) -> None:
```

```python
# After — one new parameter
def __init__(
    self,
    backend: str = "offline_replay",
    records: Optional[list[EncounterRecord]] = None,
    scenario_pack: str = "all",
    api_base_url: Optional[str] = None,
    step_penalty: float = -0.01,
    materialize_not_useful_penalty: float = -0.10,
    refine_cost: float = -0.05,
    skip_missed_penalty: float = -0.01,
    seed: Optional[int] = None,
    tile_encoder = None,   # NEW — optional TileEncoder protocol instance
) -> None:
    ...
    self.tile_encoder = tile_encoder
```

### 2. Apply the encoder in `_tile_to_obs`

```python
# Before (T3)
def _tile_to_obs(self, tile: np.ndarray) -> np.ndarray:
    arr = np.asarray(tile, dtype=np.float32)
    # ... shape handling + resize ...
    arr = arr / SENTINEL_NORM
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)
```

```python
# After (T4-compatible)
def _tile_to_obs(self, tile: np.ndarray) -> np.ndarray:
    arr = np.asarray(tile, dtype=np.float32)
    # ... shape handling + resize unchanged ...
    arr = arr / SENTINEL_NORM
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr.astype(np.float32)
    if self.tile_encoder is not None and getattr(self.tile_encoder, "embed_dim", None) is not None:
        return self.tile_encoder.encode(arr)   # returns (embed_dim,) embedding
    return arr  # (1, OBS_H, OBS_W) pixel tile unchanged
```

The `getattr(self.tile_encoder, "embed_dim", None) is not None` guard lets an
`IdentityEncoder` (embed_dim=None) be passed in as an explicit no-op without
forcing the encoder path.

## Observation shape shift

With a `DummyEncoder(embed_dim=128)` or `LFM2VLEncoder(embed_dim=768)` plugged
in, the env returns 1-D embeddings instead of 3-D pixel tiles. The MuZero side
adapts by:

1. Switching `MuZeroConfig.network` from `"impala"` to `"fullyconnected"`.
2. Setting `MuZeroConfig.observation_shape = (embed_dim,)` (the shape muzero-general
   expects for FC networks).
3. Tuning `fc_representation_layers` to accept the embedding.

These changes live in `simsat_game.py::MuZeroConfig`. Don't modify T3 until
this patch lands.

## When to apply

- **Not now**: Phase 5 T3 smoke tests pass with the current `simsat_env.py`. Landing this patch without the downstream MuZero config changes would break any code path that assumes `(1, 64, 64)` observations.
- **In Phase 6 or a T5 bundle**: alongside the MuZero config changes and an actual LFM2.5-VL loader. Apply as an amendment to T3 (re-push `simsat_env.py` with the new kwarg) and add a `SimSatMuZeroConfigLFM` class to `simsat_game.py` that wires the FC network + embed_dim.
- **Parity safeguard**: when this patch lands, the T3 smoke tests should still pass because `tile_encoder=None` (default) preserves the current behavior bit-for-bit.

## LFM2.5-VL integration checklist (future work, not T4)

When the real LFM2.5-VL encoder replaces `LFM2VLEncoderStub.encode`:

1. Load LFM2.5-VL base weights + LoRA adapter in `__init__` (transformers + peft).
2. Preprocess the incoming tile (0..1 float → PIL → processor).
3. Run through vision tower, extract pooled embedding.
4. Convert to numpy float32, shape `(embed_dim,)`.
5. Cache the model globally (module-level singleton) so instantiating `SimSatEnv` multiple times doesn't reload weights.
6. Add a `.to(device)` pathway and handle CUDA-OOM fallback to CPU.

Reference: `kaggle-simsat-gemma4-v3` build script shows a similar bitsandbytes + LoRA loading pattern for Gemma-4. The LFM2.5-VL integration should mirror that approach (4-bit NF4 on SM ≥ 7, float16 otherwise).
