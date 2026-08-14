#!/bin/bash
# Ingest the first video found in vids/ (with embedding). Start
# chroma-server.sh first and serve.sh separately to browse results as
# they land.
#
# Usage: ./ingest.sh [start|stop]
set -euo pipefail
cd "$(dirname "$0")"
source pidlib.sh

case "${1:-start}" in
start)
    video=$(find vids -maxdepth 1 -name '*.mp4' | head -n1)
    if [ -z "$video" ]; then
        echo "no .mp4 found in vids/ - add one first" >&2
        exit 1
    fi
    source .venv/bin/activate
    start_daemon ingest \
        python -m encoder.ingest "$video" scenes \
        --index http://127.0.0.1:8001
    ;;
stop)
    stop_daemon ingest
    ;;
*)
    echo "usage: $0 [start|stop]" >&2
    exit 1
    ;;
esac
