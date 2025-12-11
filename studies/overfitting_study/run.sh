#!/bin/bash

# Activate the virtual environment
# source ../../mandala-venv/bin/activate

# Run the study with ReduceLROnPlateau scheduler and a new output folder
python3 run_study.py \
    --num_samples 10 \
    --num_epochs 2000 \
    --lr 0.001 \
    --use_lr_scheduler \
    --lr_scheduler_patience 5 \
    --lr_scheduler_factor 0.5 \
    --output_folder overfitting_with_scheduler
