#!/usr/bin/env bash
set -e

# Create venv and install pip dependencies
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environment created and requirements installed. Activate with: source .venv/bin/activate"