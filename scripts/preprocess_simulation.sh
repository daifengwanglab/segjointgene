#!/usr/bin/env bash
# Preprocess Simulation raw data into NPZ patches (train/val/test).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_DIR="${1:-./data}"
FORCE="${2:-}"
if [[ "$FORCE" == "--force" ]]; then
  python3 scripts/preprocess_simulation.py --data_dir "$DATA_DIR" --force
else
  python3 scripts/preprocess_simulation.py --data_dir "$DATA_DIR"
fi
