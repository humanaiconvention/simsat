#!/usr/bin/env bash
# MuZero Stage 2 BC seed sweep — 10 seeds × (train + eval) on LFM2.5-VL-450M
# encoder. Confidence bands on val_acc + per-pack action distribution under
# the canonical Liquid Track encoder, not the offline SigLIP fallback.
#
# Wall clock: ~5-10 min/seed × 10 = 50-100 min on RTX 2080.
#
# Run from D:\SimSat root.
set -euo pipefail

SEEDS=(7 13 23 42 100 137 256 1024 2026 9999)
ENCODER="LiquidAI/LFM2.5-VL-450M"
LOG_DIR=/tmp/muzero_seed_sweep
mkdir -p "$LOG_DIR"

echo "=== MuZero seed sweep — ${#SEEDS[@]} seeds × LFM2.5-VL-450M ==="
echo "[$(date +%H:%M:%S)] start"

for s in "${SEEDS[@]}"; do
    OUT_DIR="weights/muzero/stage2_bc_seed${s}_lfm"
    EVAL_OUT="MUZERO_LFM_EVAL_seed${s}_lfm.md"
    echo "--- seed=${s} ---"

    echo "[$(date +%H:%M:%S)] training seed=${s}..."
    python scripts/muzero_stage2_pretrain.py \
        --seed "${s}" \
        --max-aug-ratio 4.0 \
        --encoder "${ENCODER}" \
        --out-dir "${OUT_DIR}" \
        > "${LOG_DIR}/train_seed${s}_lfm.log" 2>&1

    echo "[$(date +%H:%M:%S)] evaluating seed=${s}..."
    python scripts/muzero_lfm_eval.py \
        --policy bc \
        --policy-head "${OUT_DIR}/policy_head.pt" \
        --markdown \
        > "${EVAL_OUT}" 2> "${LOG_DIR}/eval_seed${s}_lfm.log"

    echo "[$(date +%H:%M:%S)] seed=${s} done"
done

echo "=== MuZero seed sweep complete ==="
ls -la weights/muzero/stage2_bc_seed*_lfm/policy_head.pt 2>&1
ls -la MUZERO_LFM_EVAL_seed*_lfm.md 2>&1
echo "[$(date +%H:%M:%S)] all done"
