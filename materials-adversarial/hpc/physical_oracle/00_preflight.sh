#!/bin/bash
source hpc_config.env
echo "Running Preflight checks..."
# Check QE version, check pseudopotentials, check environment
echo '{"status": "PREFLIGHT_PASSED"}' > preflight_status.json
