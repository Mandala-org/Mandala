#!/bin/bash

# Activate the virtual environment
source ../../mandala-venv/bin/activate

# Run the study with 100 samples
python3 run_study.py --num_samples 100
