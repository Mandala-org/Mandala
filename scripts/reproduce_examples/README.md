# Reproducible numerical-example training, version 1

Prepared for review on 2026-10-20. These files do not submit jobs. No ROSI
training has been launched. The three recipes initialize all learned weights
from scratch; historical checkpoints and W&B are not needed at execution time.

## Review and invocation

From the repository root in the project Python environment:

```bash
bash scripts/reproduce_examples/train_zncusnses.sh --plan
bash scripts/reproduce_examples/train_sio2.sh --plan
bash scripts/reproduce_examples/train_silicon.sh --plan
```

`--plan` prints the resolved recipe without creating run files or using a GPU.
After approval, the corresponding zero-argument commands train one segment:

```bash
bash scripts/reproduce_examples/train_zncusnses.sh
bash scripts/reproduce_examples/train_sio2.sh
bash scripts/reproduce_examples/train_silicon.sh
```

Run each in its own single-GPU allocation. Do not run them sequentially inside
one 48-hour allocation. Repeating the same command continues its own latest
recovery checkpoint. `--action start` insists on a new run; `--action continue`
insists on an existing recovery checkpoint. Neither mode forks the run or
resets its optimizer. There is no automatic submission or requeue loop.

Use the same immutable checkout and software environment for all segments.
The ROSI main checkout may be on the pair-Hamiltonian branch: use the approved
isolated GNN workspace rather than switching an active checkout. The project
environment is supplied by the execution profile; these scripts do not activate
or modify environments. B200 is the proposed initial target for all three.

## Frozen training choices

| Recipe | Targets / loss | Initial LR | Seed / split seed | Train / validation | Total epoch cap |
| --- | --- | ---: | --- | --- | ---: |
| `zncusnses.json` | H / MSE | 0.010803383053097332 | 44 / 44 | 80 / 20, scale 1 | 3000 |
| `sio2.json` | H / MAE | 0.0027 | 42 / 42 | 90 / 10 | 4000 |
| `silicon.json` | H, D, S / MAE + reference-partner energy guidance, coefficient 0.003 | 0.001 | 43 / 43 | 80 / 20, scale 1 | 5000 |

These are cumulative epoch caps, not epochs per allocation. All three retain
the actual selected checkpoint's architecture, tensor products, activations,
cutoffs, output scales, AdamW optimizer, scheduler parameters, and spike rollback
settings. `val/loss_total` is used consistently for scheduling, rollback, and
best-model selection. Silicon retains the selected checkpoint's
`train_observables_on_gt=True` and `rescale_density_to_num_electrons=False`;
some older shell launchers disagree with the actual saved configuration.

The LR changes are intentional, frozen proposals for scratch training:

- ZnCuSnSeS uses the verified original `dandy-sweep-34` sweep LR.
- Silicon uses the verified `leafy-sweep-15` sweep LR rather than its final
  0.0003 fine-tuning LR.
- SiO2 uses the successful MAE-stage LR of 0.0027. The original sweep LR was
  0.016313923722773627, selected with **MSE**, so it is not transferred blindly
  to scratch **MAE** training. The selected final reset used 0.0002.

No new search or validation-selected LR changes are performed by these scripts.
Starting with all final objectives active is a new, reproducible training
experiment; it is not claimed to reproduce the historical optimization path.

Each segment budgets 45 hours, including dataset/model preparation, and stops
after a complete training/validation epoch. Request 48 hours externally to leave
headroom for an epoch overrun and checkpoint I/O. This is an epoch-boundary
budget, not a hard timeout; an unexpectedly long epoch can exceed it. Scheduler
termination recovers the previous complete boundary. Continue the same command
until the cumulative epoch cap. The cap does not guarantee convergence; review
the saved learning curves before proposing a separately documented extension.

## What the historical models actually did

The `*_source.json` files were extracted from the three local
`checkpoints/paper_models/` files, not copied from the manuscript appendix.
`source_sweep_configs.json` contains bounded W&B configuration exports for the
named source sweeps. `selected_run_configs.json` records the selected W&B runs.

1. **ZnCuSnSeS:** the fresh grid run `dandy-sweep-34` was continued as
   `dandy-sweep-34-restart47h5`, then as
   `dandy-sweep-34-restart47h5-lr2em3`. The selected checkpoint has epoch 861,
   global step 68960, MSE loss, and the smaller l<=6 representation recorded in
   `zncusnses.json`, not the three-times-wider representation in the appendix.
   Its saved configuration says LR 0.002, but the saved optimizer LR is
   **0.0013504228816371665**. Full-state continuation restores optimizer LR;
   changing the CLI/config LR alone does not reset it.
2. **SiO2:** `dutiful-sweep-12` was a fresh MSE sweep run. It was continued through
   `restart47h9-pat60`, then compatibility-initialized into a MAE run at LR
   0.0027, then compatibility-initialized again at LR 0.0002. The selected
   checkpoint has epoch 757, global step 68220, and optimizer LR 0.00005.
3. **Silicon:** the source workflow transferred a silicon-300K model into the
   perturbed H-only `likely-sweep-22` search, added H/D/S heads in the
   `sleek-sweep-8` search, and performed further resets/continuations before the
   observable-guided `leafy-sweep-15` search (LR 0.001). The final selected run
   `leafy-sweep-15-half_gt-coef3em3-lr3em4-47h25` is another compatibility
   initialization at LR 0.0003. Its checkpoint has epoch 764, global step
   61200, and optimizer LR 0.0003. This was not a single scratch H/D/S run.

Intermediate ancestry above is documented by the historical launchers and
sweep definitions; the selected checkpoints and exported W&B records directly
verify the final configurations and the quoted source learning rates. The
complete old data membership and every intermediate training trajectory are
not recoverable from those checkpoints alone.

## Continuation audit and the new opt-in path

The original `compatibility=True` path initializes compatible weights into a
new optimizer/scheduler/run; it is fine-tuning, not continuation. The original
`compatibility=False` path passes the checkpoint to Lightning and restores
model, optimizer, scheduler, and training-loop state. Its configured `max_epochs`
is a total cap. The original path nevertheless lacked guarantees needed here:

- `ArtifactCheckpointCallback` did not serialize its best-score ranking.
- It saved latest state during validation, before the epoch's plateau-scheduler
  update in the inspected Lightning version.
- Global RNG state and epoch-independent shuffle order were not recovered by
  the application.
- Exception checkpointing could overwrite latest with a partial epoch.
- Split/run manifests could be overwritten without rejecting changed inputs.

The recipes opt into `epoch_boundary_resume=True`. This selects the new
`EpochBoundaryCheckpoint` callback in the existing training entrypoint instead
of the heavy artifact-report callback. It:

- saves full state atomically before each epoch and at normal segment end;
- preserves the all-time best ranking, optimizer, scheduler, rollback state,
  and Python/NumPy/PyTorch/CUDA RNG states;
- bundles the matching best checkpoint in recovery state, so rollback cannot
  accidentally use a best-model file written by an abandoned epoch;
- restores RNG after startup, and shuffles by absolute epoch with a private
  sampler generator;
- replays an interrupted epoch from its beginning, without retaining its
  partial optimizer updates;
- rejects best-model files as recovery checkpoints;
- disables startup sanity validation so it cannot perturb a resumed run;
- retains the original manifest and writes a separate manifest per continuation.

At epoch start Lightning has already marked the next epoch ready but has not
reset its previous batch counters. The callback normalizes that serialized
boundary to the completed-epoch representation and fails on an unfamiliar
schema. This is tested with local Lightning 2.6.1 / PyTorch 2.5.1 CPU. Before
long ROSI execution, check the same focused continuation tests in its actual
environment. Cross-version checkpoint migration is deliberately not automatic.

The wrapper locks each model against concurrent writers and rejects changes to
the recipe, code, installed package versions, GPU type, or recorded runtime
settings. Manifests reject changed split identities, input sizes, or mtimes.
Large input files are not content-hashed; preserve the immutable dataset itself.
The supplied envelopes are copied into this directory and SHA256-verified.
They are fixed historical fitted preprocessing artifacts, not re-fitted here.

Zero DataLoader workers and disabled cache-warmup shuffling avoid process/RNG
ambiguities. Cached/precomputed features remain enabled. Detailed plotting and
per-irrep reporting are deferred to evaluation to avoid consuming training time.
W&B runs offline per segment; checkpoints/manifests are authoritative and require
no network. Logs can be synchronized later if desired.

CPU tests verify exact uninterrupted-versus-resumed parameters, AdamW moments,
scheduler state, global steps, and best-model preservation, including a simulated
mid-epoch crash, plus an actual E3GNN continuation. CUDA scatter reductions can
remain nondeterministic: deterministic mode warns, and bitwise GPU equality is
not promised. Pinning hardware/software and preserving continuation state still
removes the application-level sources of restart drift.

## Data conventions and execution gates

All data/checkpoints stay under the documented bigdata root. Outputs are new:
`/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/publication_retrain_v1/`.
Existing models are never overwritten. Recovery uses `<model>/latest_checkpoint.pt`;
the selected model is `<model>/best_model.pt`.

The current loader retains native spin=0 density normalization. These fresh
models therefore use the corrected current convention, not the old doubled
density targets. The silicon energy/density results cannot be compared blindly
to the old numerical values. No normalization or physical-constraint method is
changed by this work.

The historical silicon metric JSON records 16 validation structures although
the old launcher requested 20 and tolerated incomplete data. The new recipe
explicitly requests the intended 80/20 and **fails** if fewer than 100 scale-1
structures are available. It never silently substitutes a smaller validation
population. Verify that inventory before long execution; if only 96 exist,
review and freeze an explicit 80/16 recipe before launching. Historical exact
split membership is not asserted. SiO2 selects 100 of its larger corpus.

These recipes preserve the numerical-example train/validation design and do not
introduce a test split or evaluate test data after every segment. Histograms and
other reviewer points are outside this change.

## Focused verification

```bash
python -m pytest -q tests/workflow/test_epoch_boundary_resume.py
python -m pytest -q tests/unit/scripts/test_reproduce_examples.py tests/unit/scripts/test_train.py tests/unit/net/test_time_budget.py
```

The two opt-in CUDA continuation cases can be run inside the target GPU
allocation with `MANDALA_RUN_GPU_TESTS=1` and the same CUBLAS setting as the
launchers. They compare clean and interrupted/resumed stochastic training.

Before approval, review the three initial LRs, total epoch caps, silicon data
inventory requirement, corrected density convention, and offline logging choice.
After approval, environment/inventory checks precede long GPU training.
