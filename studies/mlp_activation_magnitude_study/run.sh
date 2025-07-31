#!/bin/bash
source mandala-venv/bin/activate
python3 studies/mlp_activation_magnitude_study/run_study.py --separate_folder --hidden_base_dim=32 --inputs_sigma=1.0 --layer_weights_mult=0.5 --plot_log_scale
python3 studies/mlp_activation_magnitude_study/run_study.py --separate_folder --hidden_base_dim=32 --inputs_sigma=1.0 --layer_weights_mult=1.0 --plot_log_scale
python3 studies/mlp_activation_magnitude_study/run_study.py --separate_folder --hidden_base_dim=32 --inputs_sigma=1.0 --layer_weights_mult=2.0 --plot_log_scale
