# SimSat — DiLoCo Round 0 Learner Run Note

**Status as of 2026-04-27.** First real DiLoCo continuation run on Kaggle T4.
Switches the SimSat training path from "fresh QLoRA from scratch" to
"continue from DiLoCo global adapter, export adapter-fragment deltas".

This is a **plumbing-validation round.** Do not expect it to fix the v9/v10
accept-bias by itself — see "Concerns" at the bottom.

---

## Inputs verified

| Item | Value | Status |
|---|---|---|
| Global adapter (round 0 seed) | `D:\diloco_lab\state\global_round_000000` | seeded from `simsat-gemma4-v10-adapter` (sha256 in `diloco_seed_manifest.json`) |
| Adapter base model | `google/gemma-4-E2B-it` | matches runner default |
| Adapter target modules | 7× `*_proj.linear` | matches notebook `GEMMA4_KAGGLE_NOTES.md` Fix #6 |
| Source bundle | `D:\diloco_lab\dist\diloco_lab_source.zip` | regenerate with `python scripts/package_kaggle_source.py` if stale |
| Round-0 bundle | `D:\diloco_lab\dist\global_round_000000.zip` | regenerate with `python scripts/package_global_adapter.py --round-id 0` |
| Train JSONL (Kaggle dataset) | `benhaslam/simsat-gemma4-v1` → `simsat_train.jsonl` (294 rows) | ChatML messages format — verified row 1 matches `_row_messages()` shape |
| Logical dataset id (manifest only) | `simsat-gemma4-v3-reviewed` | per Codex handoff; used as a string label in `diloco_learner_summary.json`, not a Kaggle slug |

**Note on dataset version naming:** the Kaggle dataset slug is still
`benhaslam/simsat-gemma4-v1`, not v3. The `--dataset-id simsat-gemma4-v3-reviewed`
flag is the DiLoCo experiment-manifest label (free-form string used for
provenance in the outbox), not a Kaggle dataset reference. They can coexist;
the runner attaches whichever JSONL it discovers under `/kaggle/input` and
records the manifest label separately. Once we regenerate the dataset after
the review-queue labeling pass, push it as a new version under the same
`benhaslam/simsat-gemma4-v1` slug (or a new `-v2` slug) and update the
manifest label accordingly.

## Compatibility check vs. `GEMMA4_KAGGLE_NOTES.md`

The standing SimSat Kaggle notes describe 12 fixes for the SFTTrainer-based
fresh-LoRA run. The new runner (`continue_gemma4_adapter.py`) uses a custom
torch loop — no TRL, no SFTTrainer. Status of each fix:

| # | Fix | In runner? |
|---|---|---|
| 1 | Use Kaggle-attached Gemma-4 model path | ✓ `_auto_model_id` globs `/kaggle/input/**/gemma*e2b*/1` |
| 2 | Pin to single T4 (`CUDA_VISIBLE_DEVICES=0`) | ✓ set in `main()` |
| 3 | Uninstall Unsloth before TRL import | N/A — runner doesn't use TRL, but `_maybe_install` still uninstalls it defensively |
| 4 | No Unsloth on Gemma-4 at all | N/A — runner uses raw transformers |
| 5 | float16 not bfloat16 | ✓ `bnb_4bit_compute_dtype=torch.float16` |
| 6 | LoRA targets `*.linear` | ✓ inherited from seed adapter's `adapter_config.json` |
| 7 | Drop `max_seq_length` from SFTConfig | N/A — no SFTConfig |
| 8 | Use `processing_class=` not `tokenizer=` | N/A — no SFTTrainer |
| 9 | Use `dtype=` not `torch_dtype=` | ✓ `_load_base_model` tries `dtype=` first, falls back to `torch_dtype=` |
| 10 | Disable AMP `fp16=True` | N/A — manual loop, no GradScaler |
| 11 | `enable_input_require_grads()` AFTER PEFT wrap | ✓ `_attach_trainable_adapter` calls it after `PeftModel.from_pretrained` |
| 12 | `gradient_checkpointing_kwargs={"use_reentrant": False}` | ✓ same function applies it |

**No conflicts.** The runner is a clean rewrite that preserves every
load-time fix the SimSat v3 kernel learned the hard way.

---

## Exact Kaggle inputs to attach

In the Kaggle notebook editor → "Add data" → attach all four:

1. **Gemma-4 model** — `google/gemma-4` (transformers variant, e2b-it/1)
   resolves to `/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1`
2. **DiLoCo source bundle** — upload `D:\diloco_lab\dist\diloco_lab_source.zip`
   as a private dataset (e.g. `benhaslam/diloco-lab-src`)
3. **DiLoCo round-0 adapter** — upload `D:\diloco_lab\dist\global_round_000000.zip`
   as a private dataset (e.g. `benhaslam/diloco-global-round-000000`)
4. **SimSat training JSONL** — `benhaslam/simsat-gemma4-v1` (existing, has
   `simsat_train.jsonl`)

## Exact Kaggle cell

Single cell, run after attaching the 4 inputs above:

```python
import subprocess, sys, zipfile
from pathlib import Path

SRC = Path("/kaggle/working/diloco_lab_src")
if not SRC.exists():
    zips = list(Path("/kaggle/input").rglob("diloco_lab_source.zip"))
    if not zips:
        raise RuntimeError("Attach diloco_lab_source.zip as a Kaggle input")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(SRC)

GLOBAL = Path("/kaggle/working/global_adapter")
if not GLOBAL.exists():
    zips = list(Path("/kaggle/input").rglob("global_round_000000.zip"))
    if not zips:
        raise RuntimeError("Attach global_round_000000.zip as a Kaggle input")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(GLOBAL)

subprocess.check_call([
    sys.executable,
    str(SRC / "kaggle" / "continue_gemma4_adapter.py"),
    "--base-adapter", str(GLOBAL / "global_round_000000"),
    "--project", "simsat",
    "--dataset-id", "simsat-gemma4-v3-reviewed",
    "--learner-id", "kaggle-t4-simsat-round0-a",
    "--round-id", "0",
    "--max-steps", "120",
    "--lr", "5e-5",
])
```

Outputs land at:
- `/kaggle/working/diloco_continued_adapter/`  (the new continued LoRA + tokenizer + `diloco_learner_summary.json`)
- `/kaggle/working/diloco_outbox/<experiment>/round-000000/<learner>/`  (fragment deltas for inbox merge)

## Exact local sync / eval after the Kaggle run

**Sync** — `sync_kaggle_adapter.py` is hardcoded to the v1 training kernel
slug, so it does not apply to a new DiLoCo learner kernel. Pull manually:

```powershell
# 1) Continued adapter — eval-ready
$Round0 = "D:\SimSat\weights\diloco-round0-continued"
mkdir $Round0 -ErrorAction SilentlyContinue
kaggle kernels output <your-kernel-slug> -p $Round0
# Locate the actual adapter dir (kaggle output preserves /kaggle/working layout):
#   $Round0\diloco_continued_adapter\

# 2) Outbox fragments — copy into D:\diloco_lab\inbox so the orchestrator can merge
$Outbox = "$Round0\diloco_outbox"
robocopy $Outbox D:\diloco_lab\inbox /E
```

**Eval** — same command as v10/v11; only the LoRA path changes:

```powershell
$env:OBSERVATION_VLA_BACKEND   = "gemma4"
$env:OBSERVATION_VLM_BASE_MODEL = "google/gemma-4-e2b-it"
$env:OBSERVATION_VLM_MODE      = "lora"
$env:OBSERVATION_VLM_LORA_PATH = "D:\SimSat\weights\diloco-round0-continued\diloco_continued_adapter"
$env:OBSERVATION_VLM_MODEL_LABEL = "diloco-round0-continued"
cd D:\SimSat
python scripts\observation_vla_eval.py --inprocess
```

The eval flow is unchanged — `TransformersVLMAdapter` loads any LoRA at
the path you give it. Pinned cases, MAE, bucketed agreement, and action
agreement print as usual.

---

## Concerns

### 1. This round will not fix accept-bias on its own.

The training data is the same 294-row v9/v10 dataset (~70% refine in row
count, but only 8 operator-reviewed cases drive the high-weight signal,
and 4 of those have no PNG asset). 120 steps of continuation training
on the same data, starting from a v10 seed, will not move the decision
boundary in a meaningful way. **Treat this as a wiring-validation round**:
prove the seed-adapter-load → continue-train → outbox-export pipeline
works end-to-end. Eval numbers will be approximately v10's
(MAE≈0.16, bucketed≈0.57).

### 2. Eval coverage is unchanged and remains the headline weakness.

The pinned eval set (`simsat_eval_reviewed.jsonl`, n=3 +
`simsat_eval_shortlist.jsonl`, n=10) is small. After this round, do
**not** make accept/refine claims based on eval-only numbers. The
review-queue labeling work in flight (`D:\SimSat\review_queue.txt`,
30 candidates) is the prerequisite for a meaningful v11 retraining
round on broader, refine-balanced ground truth.

### 3. The `--dataset-id simsat-gemma4-v3-reviewed` label overstates current state.

The Kaggle dataset is still `benhaslam/simsat-gemma4-v1` (294 rows, v9
composition). The `-v3-reviewed` suffix in the manifest will appear in
the outbox provenance and could be misread as "this learner trained on
a refreshed reviewed-only dataset". After the review-queue labeling
session, regenerate the JSONL (`REFINE_BOOST=1.5 python
datasets/simsat-gemma4-v1/prepare_dataset.py`), bump the Kaggle dataset
version, and **then** the manifest label will accurately reflect the
underlying data. For round 0, accept the slight provenance noise —
do not rename the seed adapter or re-export.

### 4. Adapters

Per constraints: this run writes to `/kaggle/working/diloco_continued_adapter`
on Kaggle and `D:\SimSat\weights\diloco-round0-continued\` locally.
Existing `D:\SimSat\weights\simsat-gemma4-v10-adapter\` is untouched.

---

## Files changed

- **NEW:** `D:\SimSat\notebooks\DILOCO_ROUND0_RUN_NOTE.md` (this file)

No code, no adapters, no other artifacts modified. Nothing under
`D:\arc3` touched.
