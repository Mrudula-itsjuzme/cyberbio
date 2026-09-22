#!/bin/bash
# Gate: Attack pair DFT jobs must not run unless calibration has passed.
if grep -q "CALIBRATION_FAILED" calibration_status.json; then
    echo "Gate check failed: Calibration did not pass. Stopping."
    exit 1
fi
echo "Running attack pairs..."
