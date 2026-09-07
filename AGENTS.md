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

## Directed Hamiltonian and Hermiticity protocol

- Preserve both members of every periodic offsite reverse pair as distinct
  supervised records: `(i, j, L)` and `(j, i, -L)`. Do not reduce the training
  cache to one canonical representative.
- Every mapper must make two independent raw forward evaluations for those two
  directed inputs. A mapper's normal `predict_offsite` path must not average a
  prediction with its reversed-input prediction and must not generate one
  direction analytically from the other. Shared model parameters are allowed;
  the directed examples, forward calls, and losses remain distinct.
- Fit/train against both raw directed targets. Do not impose Hermiticity inside
  the training forward pass or closed-form fit.
- Apply global Hermitian projection only after training, when evaluating or
  exporting physical predictions:
  `H_proj(i,j,L) = 0.5 * (H_raw(i,j,L) + H_raw(j,i,-L).T)`, with the reverse
  projected block set to its transpose. Treat onsite `(i,i,0)` analogously by
  projecting its raw block with its transpose at evaluation time.
- Report raw-direction metrics and raw Hermiticity residuals as diagnostics,
  plus the post-projection physical metrics used for model comparison. Never
  replace global pair projection with self-symmetrization of an offsite block.

## Independent onsite/offsite optimization protocol

- Treat onsite and offsite Hamiltonian blocks as separate supervised problems.
  They must have separate learned descriptor projections, model parameters,
  optimizers, batches, losses, checkpoints, and validation rankings; do not tune
  their relative weight in a joint objective.
- Continue the directed M3/M5 and other pair-local approaches for offsite blocks.
  Onsite models may use different model classes, including invariant species
  means, affine equivariant ridge maps, and onsite-only nonlinear equivariant
  residual models.
- Select onsite and offsite candidates independently using validation data.
  Compose them only after both choices are frozen. The composed predictor takes
  onsite blocks exclusively from the selected onsite model and offsite blocks
  exclusively from the selected directed offsite model, followed by the
  evaluation-only projections defined above.
