#!/bin/bash
# Ingest the first video found in vids/ (with embedding). Run
# chroma-server.sh first and serve.sh separately (each in its own
# terminal) to browse the results as they land.
set -euo pipefail
cd "$(dirname "$0")"

video=$(find vids -maxdepth 1 -name '*.mp4' | head -n1)
if [ -z "$video" ]; then
    echo "no .mp4 found in vids/ - add one first" >&2
    exit 1
fi

source .venv/bin/activate
python -m encoder.ingest "$video" scenes \
    --index http://127.0.0.1:8001 --embed
