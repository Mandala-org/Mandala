# SiOx reference-architecture runs

Four from-scratch runs through MANDALA's `E3GNN`: DeepH, DeepH-E3, MACE-H,
and MANDALA with the SiOx paper recipe. These launchers prepare commands;
they never submit jobs themselves. Use an isolated checkout so other ongoing
experiments retain their code and environment.
Run every command below from that checkout's repository root in the Mandala
environment. No environment activation
is embedded in the launchers. `PYTHONPATH` points explicitly to this checkout,
including when the environment's editable installation points elsewhere.

## 1. Prepare data once

Run the lightweight manifest preparation from the isolated checkout:

```bash
bash scripts/train_siox_models/prepare.sh
```

Preparation reads the existing
`/bigdata/casus/wdm/hamiltonian_learning/data/SiOx_new` file list, repeats sorted
discovery and Python `random.Random(42)` shuffling, and freezes 90 training and
10 validation snapshots. All four models use those exact lists. This paper
comparison has no test split or test evaluation.

Preparation only freezes the source file lists and hashes. All four runs read
the original OpenMX pairs through `DatasetFactory`; no HDF5 conversion or CPU
preparation allocation is required. Optional format export remains a separate
shared data utility. Each preset explicitly declares its graph support,
representation, basis, target symmetrization and scientific policies.

Small selection/provenance files, resolved run configurations, runtime contracts,
segment summaries and shared CSV logs go under
`researchops/runs/siox_models_v1` (ignored by git). Preserve that directory
between submissions. Large checkpoints go under
`/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/siox_models_v1`,
with separate directories for each model. MANDALA has a separate snapshot cache
and never loads the paper's fitted checkpoint or the publication retraining runs.

## 2. Inspect and launch the four runs

After preparation, inspect any configuration without a GPU or creating a run:

```bash
PYTHONPATH=src:. python -m scripts.train_siox_models.run train --model mandala_paper --plan
PYTHONPATH=src:. python -m scripts.train_siox_models.run train --model mace_h --plan
```

The table describes the frozen scientific settings. Dataset membership and IO
paths override the reference presets; their architecture, objective, optimizer
and scheduler defaults remain intact.

| Launcher | Configuration | Initial LR | Epoch limit | Batch size |
| --- | --- | --- | --- | --- |
| `deeph.sh` | `configs/reference/deeph.yaml` | 0.001 | 4000 | 3 |
| `deeph_e3.sh` | `configs/reference/deeph_e3.yaml` | 0.002 | 3000 | 1 |
| `mace_h.sh` | `configs/reference/mace_h.yaml` | 0.002 | 3000 | 1 |
| `mandala_paper.sh` | `configs/reference/mandala_paper.yaml` | 0.0027 | 4000 | 1 |

MANDALA retains the paper's two attention layers, hidden irreps, 10 Å cutoff,
128 radial functions, frozen radial envelope, MAE objective, LR scheduler and
spike-recovery settings. The initial LR comes from the selected sweep recipe;
weights and optimizer state start fresh. DeepH uses local-coordinate targets;
DeepH-E3 and MACE-H use global equivariant targets. MACE-H fits output statistics
on **training snapshots only**, retaining its configurable identity-gradient
scaling. The models' configured training losses, target policies and graph supports
differ, so their logged loss values are not directly comparable accuracy metrics.

Submit each launcher independently using a single GPU and 48-hour allocation
with the Mandala project profile. For the `runb1` helper used for the earlier
publication jobs, the commands are:

```bash
runb1 bash scripts/train_siox_models/deeph.sh
runb1 bash scripts/train_siox_models/deeph_e3.sh
runb1 bash scripts/train_siox_models/mace_h.sh
runb1 bash scripts/train_siox_models/mandala_paper.sh
```

`runb1` is a site shell helper, not shipped here; any wrapper that selects the
Mandala environment and a 48-hour GPU allocation works. The
repository launchers themselves simply run `python`, so they work inside an
explicit Slurm allocation too. Save scheduler stdout/stderr under the records
directory above. Full reference widths are retained. Completed segments, budgets and metrics must be read from the immutable
run records. This README is an invocation contract, not a live job-status page.

## 3. Continue after a segment

Resubmit the **same** launcher. It automatically finds its own
`latest_checkpoint.pt`, verifies the shared immutable data/configuration/code/runtime manifest,
and restores the full optimizer, scheduler and RNG state at an epoch boundary.
An exclusive lock prevents simultaneous writers for a model. It refuses an
existing run without a safe checkpoint, or a recovery without its original
manifest. For an explicit fresh/continue requirement use the Python CLI's
`--action start` or `--action continue`; the zero-argument launchers use `auto`.

Each segment requests a stop at the first completed epoch after 45 hours of
training, leaving three hours for preparation/checkpointing in a 48-hour job.
The budget begins at training, so unusually slow loading or epochs can still
overrun the allocation; the last safe boundary remains recoverable. No jobs
are automatically resubmitted. An interrupted epoch is replayed from its last
safe boundary. The epoch limit is cumulative across all submissions. Completed
runs produce `<model>.complete.json` and subsequent submissions exit harmlessly.

Python CLI options expose alternate input/checkpoint/records directories
and preparation counts for small checks. Changing scientific settings, inputs,
source code or runtime requires a new run namespace; do not delete manifests to
force a continuation. Keep the checkout fixed until these runs finish.
