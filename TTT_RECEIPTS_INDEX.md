# TTT Receipts — Index

Seven published TTT receipts on a real LFM2.5-VL-450M checkpoint document
the full development arc of the VLA-layer test-time training lane. This
file is the navigational index; each receipt is a JSON artifact in
`.kaggle_output/` plus a section in [`LFM_TTT_POC.md`](./LFM_TTT_POC.md).

## The seven receipts

| # | Receipt | Hardware | Result | Receipt JSON |
|---|---|---|---|---|
| 1 | **Proof-of-life (5 steps)** | BEAST RTX 2080 8 GB | 5/5 attempted steps applied; 0 viability gates triggered. Step 6 OOM at 8 GB ceiling (caught). Mechanism real. | [`ttt_proof_of_life_receipt.json`](./.kaggle_output/ttt_proof_of_life_receipt.json) |
| 2 | **Long-horizon stability v1 (30 steps, 8-row probe)** | BEAST RTX 2080 8 GB | 28/30 applied, 0 OOM (allocator hygiene fix), parse_rate 1.000 throughout, lora_delta_l2 monotonic 0.0008 → 0.0116, steady-state by step 10. | [`extended_ttt_receipt.json`](./.kaggle_output/extended_ttt_receipt.json) |
| 3 | **Long-horizon stability v2 (50 steps, 16-row probe)** | BEAST RTX 2080 8 GB | 48/50 applied, 0 OOM, same steady-state pattern with tighter probe. Trust-layer TTT has 100-cycle evidence; VLA-layer now has 50-cycle. | [`extended_ttt_v2_receipt.json`](./.kaggle_output/extended_ttt_v2_receipt.json) |
| 4 | **Class-targeted v1 (full-CE loss) — negative result** | BEAST RTX 2080 8 GB | Skip class regressed 0.375 → 0.000. Diagnosed: full-assistant CE diluted action signal across ~99 non-action tokens. Loss-formulation issue identified. | [`class_targeted_ttt_receipt.json`](./.kaggle_output/class_targeted_ttt_receipt.json) |
| 5 | **Class-targeted v2 (action-weighted CE) — skip lift ⭐** | BEAST RTX 2080 8 GB | **Skip 0.375 → 0.750 (+37.5 pp)** in 16 steps. Loss drove 1.15 → 0.0002. Architectural claim validated on a single class. | [`class_targeted_ttt_v2_receipt.json`](./.kaggle_output/class_targeted_ttt_v2_receipt.json) |
| 6 | **Class-targeted v2 (action-weighted CE) — defer lift ⭐⭐** | BEAST RTX 2080 8 GB | **Defer 0.125 → 0.875 (+75 pp)** in 16 steps. Two-class generalization confirmed. | [`class_targeted_ttt_v2_defer_receipt.json`](./.kaggle_output/class_targeted_ttt_v2_defer_receipt.json) |
| 7 | **Stratified TTT v2 — robustness check** | BEAST RTX 2080 8 GB | Balanced 4-per-class stream over 16 steps; full 32-row holdout pre/post. **Net +3.1 pp overall, no catastrophic regression**: skip +37.5 pp, defer/refine each −12.5 pp, accept flat. | [`stratified_ttt_v2_receipt.json`](./.kaggle_output/stratified_ttt_v2_receipt.json) |

## Headline numbers

| Metric | Value | Receipt |
|---|---|---|
| Largest single-class lift | **+75 pp** (defer 0.125 → 0.875) | #6 |
| Two-class average lift | **+56.25 pp** (skip + defer) | #5, #6 |
| Stratified-stream net | +3.1 pp (no class catastrophically regressed) | #7 |
| Longest stable run | 50 steps, 0 divergence, parse 1.000 throughout | #3 |
| Total LoRA gradient steps applied across all 7 receipts | ~140 | all |

## Architectural conclusion

Free-running TTT on a class-mixed stream with full-CE loss reaches a
steady state without lifting (#2, #3). Action-token-weighted CE on a
class-targeted stream lifts the target class (+37.5 / +75 pp) (#5, #6).
Stratified TTT preserves overall accuracy with mild redistribution (#7).
The negative result (#4) identified the loss-formulation issue; the
positive results (#5, #6) confirmed the fix. The stability flank
(#1, #2, #3) confirms the loop runs cleanly under sustained operation.

The architectural claim — *runtime TTT under viability gates lifts
target-class accuracy per pass without ground-side retraining* — is
empirically validated across this development arc on a real LFM2.5-VL
checkpoint.

The remaining work is the prize-hardware lane (NVIDIA Orin 16 GB):
long-horizon (100+ cycle) runs under live encounter outcomes with the
confirmed-outcome signal source from orbit retrospective. The 7 receipts
above demonstrate the loop is ready for that lane.

## Reproduction

Each receipt was produced by a script in
[`notebooks/kaggle-simsat-lfm-v1/`](./notebooks/kaggle-simsat-lfm-v1/):

| Receipt | Script |
|---|---|
| #1 | `ttt_proof_of_life.py` |
| #2 | `extended_ttt_run.py` |
| #3 | `extended_ttt_run_v2.py` |
| #4 | `class_targeted_ttt.py` |
| #5 | `class_targeted_ttt_v2.py` |
| #6 | `class_targeted_ttt_v2_defer.py` |
| #7 | `stratified_ttt_v2.py` |

All scripts are committed and runnable on a CUDA-capable machine with the
v3 adapter checkpoint at `.kaggle_output_v3/simsat-lfm25vl-450m-v3-adapter/`.
The adapter is publicly downloadable from
[`HumanAIConvention/simsat-lfm25vl-450m-v3`](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3).
