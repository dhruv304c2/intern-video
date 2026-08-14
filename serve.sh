#!/bin/bash
# Launch the API, reading from the chroma-server.sh instance. Start
# chroma-server.sh first and ingest.sh alongside.
#
# Usage: ./serve.sh [start|stop]
set -euo pipefail
cd "$(dirname "$0")"
source pidlib.sh

case "${1:-start}" in
start)
    source .venv/bin/activate
    start_daemon serve python api.py http://127.0.0.1:8001
    ;;
stop)
    stop_daemon serve
    ;;
*)
    echo "usage: $0 [start|stop]" >&2
    exit 1
    ;;
esac
