#!/usr/bin/env bash
# MuZero Stage 2 v3 training run.
#
# What changed from v2:
#   - Defer corpus: 3 original → 13 cases (3 original + 10 auto-labeled via
#     cloud_cover>=80% AND target_visible heuristic, added 2026-05-05).
#   - With 13 defer originals and max_aug_ratio=4.0:
#       effective defer = min(48, ceil(13 * 5.0)) = 48
#     Defer now reaches the target-per-class for the first time.
#     v2 had 3 originals → cap(3*5)=15, min(48,15)=15 — still the minority.
#     v3 should break the 0/75 defer prediction.
#
# Output goes to weights/muzero/stage2_bc_v3/ so v2 is preserved at
# weights/muzero/stage2_bc/ (the canonical default path).
#
# Run from repo root:
#   bash scripts/muzero_run_stage2_v3.sh
#
# To evaluate v3 after training:
#   python scripts/muzero_lfm_eval.py --policy bc \
#       --policy-head weights/muzero/stage2_bc_v3/policy_head.pt
#
# Compare reward and defer recall vs v2 to confirm improvement.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$SCRIPT_DIR/muzero_stage2_pretrain.py" \
    --out-dir "$REPO_ROOT/weights/muzero/stage2_bc_v3" \
    --max-aug-ratio 4.0 \
    --target-per-class 48 \
    --epochs 60 \
    --lr 1e-3 \
    --seed 42 \
    "$@"

echo ""
echo "v3 training complete. Evaluate with:"
echo "  python scripts/muzero_lfm_eval.py --policy bc \\"
echo "      --policy-head weights/muzero/stage2_bc_v3/policy_head.pt"
