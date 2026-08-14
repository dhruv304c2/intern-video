#!/bin/bash
# Serve the frontend (talks to api.py over HTTP - run serve.sh separately).
#
# Usage: ./serve.sh [start|stop]
set -euo pipefail
cd "$(dirname "$0")"
source ../pidlib.sh

case "${1:-start}" in
start)
    start_daemon frontend python3 -m http.server 5500
    ;;
stop)
    stop_daemon frontend
    ;;
*)
    echo "usage: $0 [start|stop]" >&2
    exit 1
    ;;
esac
