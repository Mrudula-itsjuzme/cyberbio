#!/usr/bin/env bash
# Run the test suite.
#
# Why this wrapper exists: this machine has a system-wide ROS 2 install that
# exports PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages globally. Pytest
# autoloads plugins it finds there, and those plugins fail to import in this venv
# (missing 'lark'), crashing collection before any test runs.
#
# This project uses no third-party pytest plugins, so disabling autoload and
# dropping the ROS path is safe and changes nothing about what is tested.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "No interpreter at $PYTHON. Create the venv first:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -e \".[chem,dev]\"" >&2
    exit 1
fi

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="" exec "$PYTHON" -m pytest "$@"
