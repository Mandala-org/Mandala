# Paper ablation sweeps

These sweeps are controlled, fresh-initialization comparisons. Each lean YAML
contains two settings crossed with seeds 41-45, for exactly ten runs. Shared model,
training, logging, and reproducibility settings come from `net.common.Config` and
`scripts/wandb_run.py`. Dataset membership is held fixed with
`data-split-seed=42`, and each run performs one held-out test pass.

Regenerate the standalone sweep files after editing the shared matrix:

```bash
mandala-venv/bin/python sweeps/paper_ablations/generate_sweeps.py
```

All W&B projects, sweeps, groups, run names, and checkpoint directories begin with
`paper_`. The runner derives `ablation_setting` and the deterministic run name from
the one varied treatment parameter using `ablation_setting_from`.

The first round consists of:

- ZnCuSnSeS: envelope factorization.
- ZnCuSnSeS: pair-conditioned radial MLP.
- Perturbed silicon: spherical-harmonic tensor-square edge encoding.
- ZnCuSnSeS: sum versus attention node aggregation.
- ZnCuSnSeS: separate shifted-self handling.
- SiOx: envelope factorization.
- Perturbed silicon: Gamma-point spectral guidance (`0` versus `0.003`).
- ZnCuSnSeS: Gamma-point spectral guidance (`0` versus `0.003`).
- SiOx: energy guidance using ground-truth density (`0` versus `0.003`).

The baseline labels emitted by the runner are `off`, `disabled`, `sum`, or `0`,
as appropriate. Boolean treatments are recorded as `disabled`/`enabled`; numeric
coefficients retain their numeric text. Run names have the form
`paper_<ablation>_<setting>_seed<seed>`.

The shared defaults include the selected two-layer attention backbone, rich edge
encoding, SH tensor-square features, split pair heads, separate shifted-self
handling, MSE matrix loss, strict edge matching, `val/hamiltonian_mae` scheduling
and checkpointing, a 12-hour wall-clock budget, and fresh initialization.
Dataset-specific YAML overrides are limited to paths, split sizes, cutoffs,
representations, learning rates, and the mechanism under test.

Create a sweep with `wandb sweep sweeps/paper_ablations/<file>.yaml`. Run enough
agents to complete its ten jobs; `run_cap: 10` prevents additional assignments.
To register all nine sweeps in one pass, run:

```bash
bash sweeps/paper_ablations/create_all_sweeps.sh
```

Record the nine printed sweep IDs. Each sweep contains ten 12-hour runs.

After a sweep finishes, validate its paired design with:

```bash
python -u scripts/report/aggregate_paper_ablation.py \
  --project paper_<ablation> \
  --group paper_<ablation> \
  --baseline <baseline-label> \
  --expected-seeds 5 \
  --output wandb_sweep_analysis/paper_<ablation>.json
```
