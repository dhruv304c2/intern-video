#!/bin/bash
# Run RDRecallTest over the sample index/query video sets.
set -euo pipefail
cd "$(dirname "$0")"

MAX_SCENES="${MAX_SCENES:-50}"

.venv/bin/python main.py datasets/index_videos.csv datasets/query_videos.csv \
    --max-scenes "$MAX_SCENES" "$@"
