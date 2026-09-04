# Pair-descriptor Hamiltonian workflow

These instructions apply to the repository-wide MANDALA pair-descriptor project.

## Cluster collaboration protocol

- Work locally; the user executes cluster workloads and returns small artifacts.
- Prepare cluster work in scientifically valid batches. Prefer one ordered batch
  launcher containing many independent/pre-registered jobs over asking the user
  to run and synchronize every small experiment.
- Batch together work whose later configurations do not depend on interpreting
  earlier results. Insert an explicit gate and return to the user only when an
  earlier result genuinely determines the next experiment or when a failure
  requires a code change.
- Each component operation still needs its own argparse Python CLI and frozen,
  zero-argument bash launcher. A batch launcher may call those launchers in a
  documented order and must stop on failure.
- CPU cluster jobs use 64 CPUs/workers by default. Use fewer only when a measured
  algorithmic or memory constraint is documented in the launcher/report.
- Cluster commands are given relative to the repository root. Do not include
  `cd /home/brzoza73/casus/mandala` or `git pull`; assume the working directory
  and repository checkout are already correct and current.
- Do not activate the ROSI virtual environment; it is activated by default.
- Do not submit Slurm jobs or run commands on the cluster. Provide commands for
  the user to execute.
- After a batch completes, copy/evaluate its compact artifacts together and use
  the combined evidence to choose the next result-dependent batch.

## Scientific batching rule

Precomputation, training-independent descriptor certification, deterministic
baselines, and pre-registered model grids may be batched when their definitions
are frozen in advance. Validation-selected promotions, architecture changes,
cutoff choices, and confirmatory test evaluations remain behind result-dependent
gates to prevent leakage and opportunistic tuning.
