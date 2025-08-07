#!/bin/bash

# Activate the virtual environment
source ../../mandala-venv/bin/activate

# Run the study
python3 run_study.py --num_samples 10 --num_epochs 200 --lr 0.001
