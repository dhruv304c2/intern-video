#!/bin/bash
# Chroma's own server - required so ingest.sh and serve.sh can run as
# separate processes: a bare on-disk store corrupts itself under
# concurrent multi-process reads/writes, but chroma run serializes access
# through one process. Start this first.
#
# Usage: ./chroma-server.sh [start|stop]
set -euo pipefail
cd "$(dirname "$0")"
source pidlib.sh

case "${1:-start}" in
start)
    source .venv/bin/activate
    start_daemon chroma-server \
        chroma run --path index --host 127.0.0.1 --port 8001
    ;;
stop)
    stop_daemon chroma-server
    ;;
*)
    echo "usage: $0 [start|stop]" >&2
    exit 1
    ;;
esac
