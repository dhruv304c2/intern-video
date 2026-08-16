#!/bin/bash
# Run RDRecallTest over the sample index/query video sets.
set -euo pipefail
cd "$(dirname "$0")"

.venv/bin/python main.py datasets/index_videos.csv datasets/query_videos.csv
