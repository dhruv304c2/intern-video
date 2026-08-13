#!/bin/bash
# Wipe the vector index and encoded scene clips. Stop chroma-server.sh
# first - it holds index/ open, so deleting it while running leaves the
# server with a stale/inconsistent view.
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -f "chroma run --path index" > /dev/null; then
    echo "chroma-server.sh is still running - stop it first" >&2
    exit 1
fi

rm -rf index scenes
echo "removed index/ and scenes/"
