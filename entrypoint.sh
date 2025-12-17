#!/bin/bash
set -e

# Activate the virtual environment
source /app/.venv/bin/activate

# Run the multi_session.py script with all passed arguments
exec "$@"
