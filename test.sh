#!/bin/bash
#
# Helper script to test a W&B sweep locally.
# 1. Creates a new sweep.
# 2. Extracts the Sweep ID.
# 3. Runs a single agent for that sweep.

set -e # Exit immediately if a command exits with a non-zero status.

SWEEP_FILE="sweeps/test_sweep.yaml"

echo "--- Creating sweep from ${SWEEP_FILE} ---"
SWEEP_URL_LINE=$(wandb sweep ${SWEEP_FILE} 2>&1 | grep "Run sweep agent with:")

if [ -z "${SWEEP_URL_LINE}" ]; then
    echo "Error: Could not create sweep or find sweep agent command."
    exit 1
fi

# Extract the last word from the line, which is the sweep ID/path
SWEEP_ID=$(echo ${SWEEP_URL_LINE} | awk '{print $NF}')

echo "--- Found Sweep ID: ${SWEEP_ID} ---"
echo "--- Starting agent ---"

wandb agent ${SWEEP_ID}
