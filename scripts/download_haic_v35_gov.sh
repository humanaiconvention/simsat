#!/usr/bin/env bash
# Download HAIC v35-gov weights from Kaggle for Entry B
#
# Populates ./weights/haic-v35-gov/ with base / adapter / gguf / eval subfolders
# so src/sim/observation_vla/gemma4_haic_local.py can load them.
#
# Requires:
#   - kaggle CLI installed   (pip install kaggle)
#   - ~/.kaggle/kaggle.json  (API token, chmod 600)
#
# Usage:
#   cd D:\SimSat   (or repo root)
#   bash scripts/download_haic_v35_gov.sh

set -euo pipefail

DEST="${HAIC_GEMMA4_WEIGHTS_DIR:-./weights/haic-v35-gov}"
KERNEL_UNSLOTH="benhaslam/haic-gemma4-v35-gov-unsloth"
KERNEL_DATASET="benhaslam/haic-gemma4-v35-gov-dataset-generator"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "[ERROR] kaggle CLI not found. Install with: pip install kaggle"
    echo "        Then place your API token at ~/.kaggle/kaggle.json (chmod 600)"
    exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" && ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.config/kaggle}/kaggle.json" ]]; then
    echo "[WARN] Kaggle API token not found at ~/.kaggle/kaggle.json — download will fail if the kernel is private."
    echo "       Get one at https://www.kaggle.com/settings (Account -> Create New Token)."
fi

RAW_DIR="$DEST/_raw_kernel_output"
mkdir -p "$RAW_DIR"

echo "=== Downloading main training kernel output ==="
echo "  $KERNEL_UNSLOTH -> $RAW_DIR"
kaggle kernels output "$KERNEL_UNSLOTH" -p "$RAW_DIR"

echo ""
echo "=== Organizing weights into predictable layout ==="

BASE="$DEST/base"
ADAPTER="$DEST/adapter"
GGUF="$DEST/gguf"
EVAL="$DEST/eval"
mkdir -p "$BASE" "$ADAPTER" "$GGUF" "$EVAL"

# Merged full model (5 safetensors shards)
for f in model-*-of-*.safetensors model.safetensors.index.json config.json; do
    if ls "$RAW_DIR"/$f 1>/dev/null 2>&1; then
        cp -f "$RAW_DIR"/$f "$BASE/" 2>/dev/null || true
    fi
done

# Tokenizer + chat template go next to base (and to adapter for convenience)
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do
    if [[ -f "$RAW_DIR/$f" ]]; then
        cp -f "$RAW_DIR/$f" "$BASE/" 2>/dev/null || true
        cp -f "$RAW_DIR/$f" "$ADAPTER/" 2>/dev/null || true
    fi
done

# LoRA adapter
for f in adapter_config.json adapter_model.safetensors; do
    if [[ -f "$RAW_DIR/$f" ]]; then
        cp -f "$RAW_DIR/$f" "$ADAPTER/"
    fi
done

# GGUF quantized variants (the unsloth run also emits a haic-gemma4-v35-gov-gguf/ subfolder)
if [[ -d "$RAW_DIR/haic-gemma4-v35-gov-gguf" ]]; then
    cp -f "$RAW_DIR/haic-gemma4-v35-gov-gguf"/*.gguf "$GGUF/" 2>/dev/null || true
    cp -f "$RAW_DIR/haic-gemma4-v35-gov-gguf"/chat_template.jinja "$GGUF/" 2>/dev/null || true
fi
for f in *.gguf; do
    if [[ -f "$RAW_DIR/$f" ]]; then
        cp -f "$RAW_DIR/$f" "$GGUF/"
    fi
done

# Evaluation artifacts
for f in haic_v35_gov_full_results.json prism_gemma4_v35_gov.json trainer_state.json training_args.bin; do
    if [[ -f "$RAW_DIR/$f" ]]; then
        cp -f "$RAW_DIR/$f" "$EVAL/"
    fi
done

echo ""
echo "=== Summary ==="
echo "Base (merged safetensors):"
ls -la "$BASE" 2>/dev/null | tail -n +4 || echo "  (empty)"
echo ""
echo "Adapter (LoRA):"
ls -la "$ADAPTER" 2>/dev/null | tail -n +4 || echo "  (empty)"
echo ""
echo "GGUF (llama.cpp):"
ls -la "$GGUF" 2>/dev/null | tail -n +4 || echo "  (empty)"
echo ""
echo "Eval artifacts:"
ls -la "$EVAL" 2>/dev/null | tail -n +4 || echo "  (empty)"

echo ""
echo "=== Done ==="
echo "Point the sim service at these weights by setting:"
echo "  OBSERVATION_VLA_BACKEND=gemma4_haic_local"
echo "  HAIC_GEMMA4_WEIGHTS_DIR=$(cd "$DEST" && pwd)"
echo ""
echo "Optional: download training-data snapshots too (for the viability-grounding demo)"
echo "  kaggle kernels output $KERNEL_DATASET -p $DEST/dataset_snapshots"
