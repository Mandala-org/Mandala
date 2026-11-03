# Pipeline benchmarks

Tools that measure the modular pipeline and calibrate the cost model used by
`net.inspection` and MANDALA Studio.

| Script | Measures |
| --- | --- |
| `forward_timing.py` | Forward and training-step time, parameters, edges and peak memory of one model on a synthetic periodic crystal. Uses only interfaces shared with the pre-decomposition source, so it times both trees. |
| `stage_profile.py` | Time of every pipeline stage (CUDA events) next to its cost-model operation count and parameter count. |
| `fit_cost_constants.py` | Per-transform `seconds = overhead + rate x operations` fits (non-negative least squares) in the format of `src/net/cost_constants.json`. |
| `compare_timings.py` | Markdown table of base-versus-modular medians and ratios. |
| `run_pipeline_benchmarks.sh` | The frozen batch: A/B timings of the MANDALA models and of DeepH, DeepH-E3 and MACE-H at 32 and 64 atoms, stage profiles of six reference recipes at 32/64/128 atoms, and the constant fit. |

## Running the batch

The launcher takes no arguments, runs from the repository root, needs one GPU
and stops at the first failure. Running out of CUDA memory is not a failure:
the run is recorded with its peak memory and shown as OOM in the summary
(profiles that do not fit are left out of the constant fit). With Slurm:

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=06:00:00 --wrap "bash scripts/benchmarks/run_pipeline_benchmarks.sh"
```

Results are written to `benchmark_data/pipeline_modularity_v1/<job id>/`:

- `ab/*.json` and `ab_summary.md`: base (commit `3ec49ae`, extracted with
  `git archive`) versus this checkout, same configuration, structure and seed;
- `profiles/*.json`: per-stage measurements;
- `cost_constants.json`: copy it to `src/net/cost_constants.json` to calibrate
  the cost estimates for that GPU.

Both trees cache the structure-only stages (radial basis, harmonics, edge
roles) as their datasets do, so the training-step comparison matches training.
