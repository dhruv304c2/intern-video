#!/bin/bash
# Run RDRecallTest over the sample index/query video sets, or your own.
# Usage:
#   ./run-recall.sh                              # yt mode, sample CSVs
#   ./run-recall.sh yt <index.csv> <query.csv>
#   ./run-recall.sh local <index_dir> <query_dir>
set -euo pipefail
cd "$(dirname "$0")"

MAX_SCENES="${MAX_SCENES:-50}"
MODE="yt"
INDEX="datasets/index_videos.csv"
QUERY="datasets/query_videos.csv"

if [ "${1:-}" = "yt" ] || [ "${1:-}" = "local" ]; then
    MODE="$1"
    INDEX="$2"
    QUERY="$3"
    shift 3
fi

.venv/bin/python main.py recall "$MODE" "$INDEX" "$QUERY" \
    --max-scenes "$MAX_SCENES" "$@"
