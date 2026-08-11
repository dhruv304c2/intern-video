#!/usr/bin/env bash
# Splits videos in vids/ into scenes, classifies each into a bitrate category,
# and writes report.html. See README.md for one-time setup.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python build_report.py "$@"
