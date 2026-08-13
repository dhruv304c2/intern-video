#!/bin/bash
# Serve the frontend (talks to api.py over HTTP - run serve.sh separately).
cd "$(dirname "$0")"
python3 -m http.server 5500
