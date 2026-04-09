# E3MLP Investigation Plan

This is the fixed plan document for the project. It is not a progress log. Future evidence, decisions, plots, and overrides go only into `STUDY_LOG.md`.

## Objective

Design and validate a practical E3MLP family for Mandala that is:

- equivariant
- expressive
- stable to train
- reasonably fast
- replicable locally and on cluster

The final product is an informed WandB sweep for a `2`-layer E3GNN trained on density only, optimized for lowest possible energy error computed with ground-truth Hamiltonian, under a final search budget of `48h` on `16` single-GPU H100 jobs.

## Non-Negotiable Constraints

- Do not rely on conclusions from the old activation-magnitude study except as weak historical context.
- Do not use `S2Activation` in this project.
- Do not reuse the old study code; all new code and artifacts live under `studies/e3mlp_investigation/`.
- For real-data silicon experiments, use the default hidden irreps unless evidence says otherwise:
  - `32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e`
- For small/preliminary studies, use a smaller irreps that still goes up to `l=4`.
- Only increase preliminary-study width if the smaller setting is too noisy or fails to separate candidates.
- Only test the doubled hidden irreps if preliminary evidence supports it.
- The target practical depth is `4-6` layers, but every promising design must also be checked for stability at `10-12` layers.
- Every experiment must emit graphical artifacts and the most important ones must be referenced in the study log.
- New steps must always be evidence-driven; do not assume outcomes.
- Preliminary results never become permanent truth. Larger runs override them and that override must be recorded in the study log.

## Project Files

- Plan: [PLAN.md](/home/bartek/casus/mandala/studies/e3mlp_investigation/PLAN.md)
- Study log: [STUDY_LOG.md](/home/bartek/casus/mandala/studies/e3mlp_investigation/STUDY_LOG.md)
- Artifacts: [artifacts/](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts)
- Cache: [cache/](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache)
- Scripts: [scripts/](/home/bartek/casus/mandala/studies/e3mlp_investigation/scripts)
- Configs: [configs/](/home/bartek/casus/mandala/studies/e3mlp_investigation/configs)

## Core Hypothesis

Many E3MLP families likely fail or look weak without correct pre/post scaling, residual design, and initialization. Therefore the first stage is not "which nonlinear family wins" but "which families become viable once scaling, normalization, and initialization are treated seriously."

## Default Representation Policy

### Preliminary-study irreps

Use a smaller default irreps for synthetic, micro-stability, and smoke studies:

- `16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e`

This keeps:

- parity structure
- `l_max = 4`
- reasonable speed for many local iterations

### Real-data silicon irreps

Use:

- `32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e`

Only test a doubled silicon-width irreps if preliminary evidence indicates that width is a bottleneck rather than architecture or scaling.

## Candidate E3MLP Families

### Baseline families to include

- `NormAct`
- canonical `Gate`
- `GateScalarsMLP`
- `GateMagnitudes`
- residualized versions of the above
- bilinear self-tensor-product
- scalar+magnitude+self-TP gated
- invariant self-attention
- invariant MoE

### New simple ideas to add

1. `InvariantFiLME3MLP`
   - extract scalar invariants from scalars and non-scalar norms
   - predict per-copy gains and scalar-only biases
   - apply FiLM modulation before/after an equivariant linear map
2. `LayerScaleResidualE3MLP`
   - residual E3MLP with learnable per-irrep or per-copy residual scales initialized small
   - intended to stabilize `10-12` layer stacks

### New advanced ideas to add

1. `LowRankTPE3MLP`
   - linear branch plus low-rank tensor-product interaction branch
   - explicit interaction capacity with lower compute than dense TP everywhere
2. `InvariantHyperTPE3MLP`
   - invariants drive a small hypernetwork that modulates TP/path weights or branch coefficients
   - adaptive expressivity without full attention/MoE cost

## Experimental Principles

- Decompose the project into many small independent sub-experiments.
- Keep fast local experiments under roughly `10` minutes whenever possible.
- If an experiment deserves more time, prepare a cluster-ready bash script instead of guessing locally.
- Use dry-runs and smoke tests to validate scripts, then launch real runs in the background and work on non-overlapping tasks.
- Precompute and cache whatever is reused often and is expensive enough to matter.
- Keep local and cluster pipelines aligned so scaling up mostly means changing arguments.
- Always evaluate both quality and time cost.
- When idle time appears, invest it into more informative visualizations rather than waiting on a process.

## Stage Structure

### Stage A: Infrastructure and caching

Build reusable study infrastructure:

- a new experiment runner
- plotting utilities
- cached feature/pretext-dataset builders
- cluster job script templates
- a standard artifact layout

Outputs:

- reproducible configs
- cached tensors where useful
- plots produced automatically per run
- visual summary conventions that work both for synthetic and silicon experiments

### Stage B: Fast stability micro-studies

Purpose:

- identify viable E3MLP families after adding proper scaling, normalization, and initialization

Independent sub-experiments:

1. initialization scale sweep
2. pre/post scaling factor sweep
3. residual scale sweep
4. normalization placement sweep
5. gate activation sweep
6. `4-6` layer vs `10-12` layer stability check

Metrics:

- activation RMS per irrep and per layer
- gradient norms per layer
- update-to-weight ratio
- NaN/Inf occurrence
- forward/backward time

Required plots:

- layerwise activation magnitude by irrep
- layerwise gradient norm by irrep or block
- stability sweep summary heatmaps

Decision rule:

- keep families that are stable at `4-6` layers and at least not pathological at `10-12`

### Stage C: Fast synthetic expressivity studies

Purpose:

- separate shallow modulation capacity from true nonlinear interaction capacity

Independent sub-experiments:

1. fit equivariant linear teacher
2. fit equivariant quadratic / TP teacher
3. fit mixed linear+quadratic teacher
4. compare depth `2, 4, 6`

Metrics:

- train/val loss
- approximation error vs parameter count
- runtime

Required plots:

- fit curves
- prediction-vs-target scatter or density plots
- synthetic target/prediction visualizations

Decision rule:

- shortlist safe and expressive candidates

### Stage D: No-GNN silicon fitting studies

Purpose:

- evaluate E3MLPs in a realistic but simplified setting without message passing

Fixed rule:

- no GNN iterations
- one-shot or shallow structured processing only

Sub-experiments must vary how information is incorporated, including:

1. target-edge only
2. endpoint neighbor sum
3. endpoint neighbor average
4. endpoint neighbor attention
5. endpoint neighbor concatenation of fixed-size top-`K` summaries
6. E3MLP on node description, then aggregate, then E3MLP again
7. E3MLP on pair-message first, then aggregate

Targets:

1. Hamiltonian first
2. Density second

Losses to compare with all else fixed:

- MSE
- MAE
- Huber

Primary metrics:

- block MAE
- energy MAE
- runtime and memory

Required plots:

- per-irrep resolved errors
- per-distance resolved errors
- block magnitude vs error
- Hamiltonian/density target-prediction visualizations
- sweep summary plots across aggregation choices and losses

Decision rule:

- keep the most promising information-incorporation methods and E3MLP families for integrated runs

### Stage E: Integrated medium runs

Budget:

- `3-5` runs of roughly `1h` on an `8`-GPU node
- plus `2-3` WandB sweeps if the evidence suggests they are needed

Purpose:

- verify small-study claims jointly
- check ranking robustness
- reject approaches that only looked good in isolated tests

### Stage F: Final search preparation

Use accumulated evidence to define:

- a narrowed family set
- a narrowed scaling/normalization set
- a narrowed data-incorporation set
- a narrowed loss set

Then prepare the final WandB sweep for the real objective:

- `2`-layer E3GNN
- E3MLPs inside and on the head
- train on density only
- evaluate by energy error with ground-truth Hamiltonian
- cluster budget: `48h`, `16` single-GPU H100 jobs

## Decision Protocol

- Small preliminary runs guide the next step but do not settle claims.
- Whenever a next experiment depends on a preliminary result, record:
  - which claim is being used
  - that it is provisional
  - which larger run is planned to verify or overturn it
- If a larger run disagrees with a preliminary result, the larger run wins and the study log must explicitly note the override.

## Standard Output Per Experiment

Every experiment should produce at least:

- a config file or exact CLI
- a machine-readable metrics file
- at least one plot
- a short result summary appended to `STUDY_LOG.md`

Preferred plots:

- layerwise activation magnitude
- layerwise gradient norm
- train/val curves
- runtime vs error
- per-target error histograms
- error vs block magnitude / distance / edge class
- per-irrep resolved error plots
- target/prediction matrix visualizations for representative samples
- synthetic-data visual summaries where applicable

## Scalability Rules

- One codepath for local and cluster execution.
- Scale up with arguments, not rewrites.
- Cache precomputed tensors/features when reuse is high and correctness is unchanged.
- Keep experiment IDs and artifact paths deterministic.
- Favor embarrassingly parallel sub-experiments where possible.

## Immediate Next Actions After Plan Approval

1. set up the new study runner and logging/artifact conventions
2. initialize the study log with baseline assumptions and open questions
3. implement the first fast stability micro-study with proper scaling sweeps
4. generate plots automatically and append the first evidence entry to the study log
