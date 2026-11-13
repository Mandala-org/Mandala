# Paper Result Data

The CSV files in this directory contain completed W&B run summaries used by
`paper/generate_result_plots.py`.

- `zncusnses_envelope_ablation.csv`: envelope prediction factorization.
- `zncusnses_pair_radial_mlp_ablation.csv`: shared and pair-conditioned radial MLPs.
- `silicon_node_aggregation_ablation.csv`: average and attention aggregation.
- `siox_mature_energy_guidance_ablation.csv`: SiO$_2$ energy-guidance strengths merged from two sweeps.
- `zncusnses_mature_spectral_ablation.csv`: ZnCu$_2$Sn(SeS)$_2$ spectral guidance.

Run `paper/download_ablation_csvs.py` to refresh the source data from W&B.

The JSON files under `validation_relative_metrics/` contain the absolute
validation errors, validation-set reference magnitudes, and population-consistent
relative errors used in the demonstration tables. They were generated on
`rosi4` with `scripts/report/evaluate_validation_relative_metrics.py` using
the exact selected checkpoints and their historical deterministic splits.
