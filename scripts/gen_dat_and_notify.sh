#!/usr/bin/env bash
set -euo pipefail

# Generate FPGA .dat parameters into a versioned directory and (optionally) atomically notify the queue-mode runner.
#
# Usage:
#   ./scripts/gen_dat_and_notify.sh [version] [checkpoint] [out_dir_base] [update_flag]
#
# Examples:
#   ./scripts/gen_dat_and_notify.sh v2
#   ./scripts/gen_dat_and_notify.sh v3 data/model/Model_1_MG_50.pt data/dat /tmp/rsvp_param_update.json

VERSION="${1:-v2}"
CHECKPOINT="${2:-data/model/Model_1_MG_50.pt}"
OUT_DIR_BASE="${3:-data/dat}"
UPDATE_FLAG="${4:-/tmp/rsvp_param_update.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

echo "[gen] checkpoint:   $CHECKPOINT"
echo "[gen] out-dir base: $OUT_DIR_BASE"
echo "[gen] version:      $VERSION"
echo "[gen] update-flag:  $UPDATE_FLAG"

python3 python/gen_dat_from_pt.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR_BASE" \
  --version "$VERSION" \
  --update-flag "$UPDATE_FLAG"

echo "[done] generated: ${OUT_DIR_BASE%/*}/$(basename "$OUT_DIR_BASE")_${VERSION}"
