#!/bin/bash

# This script runs the full loss contribution study.
# You can edit the parameters below.

# --- Parameters ---
NUM_EPOCHS=100
NUM_SAMPLES=10
TRAIN_TARGET="irreps"
LOSS_COEF=0.001
LEARNING_RATE=0.0003

# Activate the virtual environment
echo "Activating virtual environment..."
source ../../mandala-venv/bin/activate

# Run the study
echo "Starting study with ${NUM_SAMPLES} replications for ${NUM_EPOCHS} epochs each..."
python3 -u run_study.py \
    --num_epochs ${NUM_EPOCHS} \
    --num_samples ${NUM_SAMPLES} \
    --train_target ${TRAIN_TARGET} \
    --loss_coef_observable ${LOSS_COEF} \
    --lr ${LEARNING_RATE}

echo "Study complete."
