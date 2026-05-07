# SimSat LFM2.5-VL-450M v1 — Kaggle T4 Handoff

**Purpose:** train a LoRA adapter on `LiquidAI/LFM2.5-VL-450M` over the
operator-reviewed Sentinel-2 SimSat encounter pool, so the Liquid Track
submission can claim "documented methodology + measurable improvement over
base + publicly shared weights and training code" — the three boxes the
rubric asks for.

## What's in this directory

| File | Purpose |
|---|---|
| `simsat_lfm_v1_training.py` | The full Python source (split into CELLs). Edit this, never edit `notebook.ipynb` directly. |
| `build_notebook.py` | Validates `simsat_lfm_v1_training.py` against 17 invariants (target_modules, sanity gates, etc.) and rebuilds `notebook.ipynb`. |
| `kernel-metadata.json` | Kaggle kernel config — points at `benhaslam/simsat-lfm2-5-vl-v1-training`, attaches `benhaslam/simsat-lfm-v1` dataset. |
| `push.py` | Driver: prepares dataset → pushes to Kaggle → rebuilds notebook → pushes kernel. |
| `notebook.ipynb` | Built artifact (do not hand-edit). |

## Authoritative LoRA config (do not change without reading source)

Pulled verbatim from
[`Liquid4All/leap-finetune/src/leap_finetune/training_configs/peft_configs.py`](https://github.com/Liquid4All/leap-finetune/blob/main/src/leap_finetune/training_configs/peft_configs.py)
(`DEFAULT_VLM_LORA`):

```python
LFM_MODULES                   = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]
VISION_TOWER_MODULES          = ["fc1", "fc2"]
MULTI_MODAL_PROJECTOR_MODULES = ["linear_1", "linear_2"]
target_modules = LFM_MODULES + VISION_TOWER_MODULES + MULTI_MODAL_PROJECTOR_MODULES

r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
```

Important: LFM uses `out_proj` (not `o_proj` like Gemma), and `in_proj` for
the conv blocks in its hybrid attention/conv architecture. Don't substitute
Gemma-style names.

## Run All — minimal happy path

```bash
# from repo root
cd D:/SimSat

# 1. (Re)build the local dataset directory + verify counts
python datasets/simsat-lfm-v1/prepare_dataset.py
# Expected:
#   Wrote 109 rows -> simsat_lfm_train.jsonl
#   Wrote 32 rows -> simsat_lfm_holdout.jsonl   (8 per class — TRUE held-out)
#   Wrote 4 rows -> simsat_lfm_eval.jsonl       (legacy, defer/refine only)
#   Images: 145 referenced, 0 missing

# 2. Push dataset + kernel to Kaggle
python notebooks/kaggle-simsat-lfm-v1/push.py
# (use --dataset-only or --kernel-only or --dry-run if you want)

# 3. Open the kernel page and click "Run All"
# https://www.kaggle.com/code/benhaslam/simsat-lfm2-5-vl-v1-training
```

The kernel runs the **complete pipeline in one Run All**:

1. Load base LFM2.5-VL-450M
2. Generate base-model predictions on the 32-row hold-out (saved to `/kaggle/working/base_predictions.json`)
3. Apply LoRA + train 3 epochs on 109 rows
4. Save adapter + post-save sanity gate
5. Generate tuned-model predictions on the same 32-row hold-out (saved to `/kaggle/working/tuned_predictions.json`)
6. Compute and print side-by-side metrics: parse rate, exact-action agreement, useful agreement, score MAE — overall + per-class
7. Save the eval report to `/kaggle/working/holdout_eval_report.json`

Expected wall-clock on Kaggle T4: **~10 min base eval + 30–60 min training + ~10 min tuned eval = ~50–80 min total**.

## Sanity gates that fire if something is wrong

The notebook has three concentric audits, designed after the Gemma-4 v1–v10
null-training disaster:

1. **Pre-train target match audit (Cell 2):** counts how many model modules
   each `target_modules` name actually matched. If total = 0, the kernel
   raises `RuntimeError` immediately — no wasted compute on a dead LoRA.

2. **Per-cell collator smoke test (Cell 4):** runs the data collator on a
   single sample and prints input/label shapes plus the loss-token ratio.
   If `loss-token ratio` is near zero, labels are over-masked and training
   would do nothing.

3. **Post-save adapter sanity gate (Cell 7):** loads the saved
   `adapter_model.safetensors`, asserts:
   - non-zero tensor count
   - at least one `lora_B` tensor with non-zero values (zero `lora_B` ⇒ no
     learning)
   - prints coverage across text-LM / vision-tower / projector

## Action distribution (from local prepare run)

```
Train:   accept=30  defer=28  refine=27  skip=24    (109 total — balanced enough)
Holdout: accept=8   defer=8   refine=8   skip=8     (32 total — perfectly balanced, TRUE held-out)
Legacy:  defer=2    refine=2                        (4 total — kept for back-compat)
```

The 32-row stratified hold-out exists because the existing 4-row legacy eval
covers only `defer` and `refine` — not enough surface to claim "measurable
improvement over base" across all 4 classes (the rubric requirement).

## After the training run completes

1. Check the post-save sanity gate output in the kernel logs — confirm
   `lora_B tensors non-zero > 0` and coverage shows all three components.
2. Run the smoke-test inference cell — verify the model emits valid JSON
   (the schema check is part of Cell 8).
3. To publish weights: edit Cell 9, change `if False:` → `if True:`, set
   the `HF_TOKEN` Kaggle secret, and re-run that cell only. It will create
   `HumanAIConvention/simsat-lfm25vl-450m-v1` and upload the adapter dir.
4. Build the base-vs-tuned eval kernel (`kaggle-simsat-lfm-v1-eval/`,
   parallel to `kaggle-simsat-v11-eval/`). The eval should compare the
   same `simsat_lfm_eval.jsonl` (or a larger subset) on the base model
   vs. the LoRA-tuned model and report exact-action agreement, useful
   agreement, and MAE.
5. Update `SUBMISSION_BRIEF.md` and `CHALLENGE_ENTRY.md` Liquid Track
   sections — replace the "encoder weights are at Liquid AI's published
   production checkpoint, frozen" disclaimer with the new fine-tune
   evidence.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Dataset versioning footgun** — Kaggle attaches the dataset *version* that was current when the kernel was last pushed; new dataset versions do NOT auto-refresh on the kernel | If you push an updated dataset (e.g. `push.py --dataset-only` to re-upload images), you MUST also re-push the kernel (`push.py --kernel-only`) so Kaggle re-snapshots the dataset attachment to the latest version. The error symptom is `AssertionError: Missing /kaggle/input/simsat-lfm-v1/simsat_lfm_train.jsonl` even though the dataset clearly exists on Kaggle. |
| LFM module names differ from leap-finetune in this transformers version | Pre-train target match audit (Cell 2) catches this before training spends compute |
| LoRA trains zero parameters | Post-save sanity gate (Cell 7) catches this and refuses to claim a successful run |
| OOM on T4 (16 GB) with 450M model + activations | bf16 + grad checkpointing + batch 1 + grad accum 8 — already configured |
| Image tokens accidentally included in loss | Collator masks `image_token_id` (396) in labels |
| Pad tokens in loss | Collator masks `pad_token_id` to -100 |
| Network flake during HF model download | `enable_internet=True` in kernel-metadata; HF downloads are retried by `transformers` |

## If the kernel fails

1. **Read the cell where it failed and the traceback.** Don't re-run blindly.
2. If the **pre-train target match audit** says 0 matched: LFM module
   naming changed in the transformers version Kaggle pinned. Run a quick
   diagnostic cell:
   ```python
   for n, _m in model.named_modules():
       if any(t in n for t in TARGET_MODULES):
           print(n)
   ```
   Inspect the output and update `LFM_MODULES` / `VISION_TOWER_MODULES` /
   `MULTI_MODAL_PROJECTOR_MODULES` accordingly.
3. If **OOM on T4:** drop `gradient_accumulation_steps` to 4, or reduce
   `max_length` to 1024.
4. If **loss is NaN early in training:** the bf16 path is occasionally
   unstable; switch `bf16=True` → `fp16=True, bf16=False` and add
   `dtype=torch.float16` to the model load (lower numerical headroom but
   stable on T4).

## Reference

- Liquid AI's official VLM SFT recipe: <https://docs.liquid.ai/customization/finetuning-frameworks/trl>
- `leap-finetune` source: <https://github.com/Liquid4All/leap-finetune>
- LFM2.5-VL-450M model card: <https://huggingface.co/LiquidAI/LFM2.5-VL-450M>
- LFM2.5-VL config (architectures, model_type): <https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/config.json>
- Cookbook satellite VLM example: <https://github.com/Liquid4All/cookbook/tree/main/examples/satellite-vlm>
