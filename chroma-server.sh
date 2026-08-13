#!/bin/bash
# Chroma's own server - required so ingest.sh and serve.sh can run as
# separate processes: a bare on-disk store corrupts itself under
# concurrent multi-process reads/writes, but chroma run serializes access
# through one process. Run this first, in its own terminal.
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
chroma run --path index --host 127.0.0.1 --port 8001
