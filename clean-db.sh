#!/bin/bash
# Wipe the vector index and encoded scene clips.
set -euo pipefail
cd "$(dirname "$0")"

rm -rf .cache/index .cache/scenes
echo "removed .cache/index/ and .cache/scenes/"
