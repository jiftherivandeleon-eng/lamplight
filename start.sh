#!/bin/sh
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python 3.10+ first."; exit 1; }
command -v ollama >/dev/null || echo "Note: Ollama not found. Install it from ollama.com."
if [ ! -f .venv/ready ]; then
  echo "First start: setting up (needs internet)."
  python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && touch .venv/ready || { echo "Setup failed."; exit 1; }
fi
.venv/bin/python server.py
