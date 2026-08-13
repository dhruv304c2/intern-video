#!/bin/bash
# Launch the API, reading from the chroma-server.sh instance. Run
# chroma-server.sh first and ingest.sh alongside (each in its own terminal).
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
python api.py http://127.0.0.1:8001
