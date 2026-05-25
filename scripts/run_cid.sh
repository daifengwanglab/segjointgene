#!/usr/bin/env bash
# Run Simulation + UNet + dynamic_segmentation_CID self-training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_DIR="${1:-./data}"
SUB_PATH="${2:-basic}"
MAX_EPOCHS="${3:-100}"
shift 3 2>/dev/null || true
python3 main.py \
  --sub_path "$SUB_PATH" \
  --net_epoch "$MAX_EPOCHS" \
  paths.data_dir="$DATA_DIR" \
  "$@"
