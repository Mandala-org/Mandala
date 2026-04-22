# HPO / Logging Refactor Design

This document proposes a clean split between:

- dataset preparation
- single-run training
- logging
- HPO dispatch
- HPC utilities

The main goals are:

- keep [scripts/train_silicon.py](/home/bartek/casus/mandala/scripts/train_silicon.py) thin
- support `wandb` sweeps and `optuna` studies through separate dispatchers
- support offline logging on HPC and later manual sync
- support parallel Optuna agents through a shared PostgreSQL database
- make it straightforward to add a SiO2 dataset entrypoint without duplicating training code

## Architecture

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "nodeSpacing": 40, "rankSpacing": 45}} }%%
flowchart TD
    A["scripts/train_silicon.py<br/>silicon CLI"]
    B["scripts/train_sio2.py<br/>SiO2 CLI"]
    C["src/training/dispatch.py<br/>mode dispatch"]

    D["src/training/run_single.py<br/>shared training run"]
    E["src/training/run_wandb_agent.py<br/>WandB agent"]
    F["src/training/run_optuna_agent.py<br/>Optuna agent"]

    G["src/training/logging.py<br/>logger + offline mode"]
    H["src/training/trainer_factory.py<br/>model, callbacks, trainer"]
    I["src/training/objective.py<br/>metric extraction"]

    J["src/training/data/common.py<br/>shared data helpers"]
    K["src/training/data/silicon.py<br/>silicon data prep"]
    L["src/training/data/sio2.py<br/>SiO2 data prep"]

    M["scripts/optuna_create_study.py<br/>study bootstrap"]
    N["scripts/optuna_postgres_job.sh<br/>PostgreSQL service"]
    O["scripts/wandb_agent_offline.sh<br/>offline WandB worker"]
    P["scripts/wandb_sync_offline.sh<br/>manual sync"]

    A --> C
    B --> C
    C --> D
    C --> E
    C --> F

    D --> G
    D --> H
    D --> I
    H --> J
    H --> K
    H --> L
    K --> J
    L --> J

    E --> G
    O --> E
    P --> G

    F --> G
    F --> M
    N --> M
    N --> F

    style A width:220px
    style B width:220px
    style C width:220px
    style D width:220px
    style E width:220px
    style F width:220px
    style G width:220px
    style H width:220px
    style I width:220px
    style J width:220px
    style K width:220px
    style L width:220px
    style M width:220px
    style N width:220px
    style O width:220px
    style P width:220px
```

## File Responsibilities

### [scripts/train_silicon.py](/home/bartek/casus/mandala/scripts/train_silicon.py)

Silicon-specific CLI entrypoint. It should only:

- parse silicon dataset CLI arguments
- parse training/logging/HPO mode flags
- build a `RunRequest`
- call the dispatch layer

It should not contain:

- WandB-specific control flow
- Optuna-specific control flow
- trainer assembly logic
- dataset construction details beyond argument parsing

### [scripts/train_sio2.py](/home/bartek/casus/mandala/scripts/train_sio2.py)

Same role as `train_silicon.py`, but for SiO2-specific snapshot discovery and split semantics.

### [src/training/dispatch.py](/home/bartek/casus/mandala/src/training/dispatch.py)

Top-level backend dispatcher. Chooses between:

- single run
- WandB agent mode
- Optuna agent mode

This is where we keep the policy of "what kind of run is this?"

### [src/training/run_single.py](/home/bartek/casus/mandala/src/training/run_single.py)

Owns one training run from prepared request to result:

- prepare dataset context
- create logger
- create callbacks/trainer/model
- run `trainer.fit`
- collect result metrics

This is the single source of truth for training behavior.

### [src/training/run_wandb_agent.py](/home/bartek/casus/mandala/src/training/run_wandb_agent.py)

Owns WandB-sweep-specific behavior:

- read sweep-provided config overrides
- merge them into the base request
- run one training job through `run_single`

This stays small because WandB itself manages sweep scheduling.

### [src/training/run_optuna_agent.py](/home/bartek/casus/mandala/src/training/run_optuna_agent.py)

Owns Optuna-worker behavior:

- connect to a shared Optuna study in PostgreSQL
- fetch and execute trials
- report metrics and pruning state
- run one training job per trial via `run_single`

This is the key piece for parallel HPC Optuna jobs.

### [src/training/logging.py](/home/bartek/casus/mandala/src/training/logging.py)

Builds logging backends and handles offline policy:

- WandB logger creation
- offline vs online mode
- run naming
- save directory conventions
- later sync metadata

This is where the no-internet HPC workflow should live.

### [src/training/trainer_factory.py](/home/bartek/casus/mandala/src/training/trainer_factory.py)

Creates:

- `E3GNN`
- callbacks
- Lightning trainer

This keeps trainer assembly separate from dispatch and dataset concerns.

### [src/training/objective.py](/home/bartek/casus/mandala/src/training/objective.py)

Defines how objective metrics are read from a completed run, for example:

- `val/energy_mae`
- `val/loss`

This avoids backend-specific code reaching into Lightning internals.

### [src/training/data/common.py](/home/bartek/casus/mandala/src/training/data/common.py)

Shared dataset helpers:

- snapshot pair discovery helpers
- split validation
- DataLoader construction
- common dataclasses for prepared training context

### [src/training/data/silicon.py](/home/bartek/casus/mandala/src/training/data/silicon.py)

Silicon dataset preparation:

- global split mode
- temperature split mode
- `DatasetFactory` usage
- mapper creation
- logging metadata for silicon studies

### [src/training/data/sio2.py](/home/bartek/casus/mandala/src/training/data/sio2.py)

SiO2 dataset preparation:

- SiO2 snapshot discovery
- SiO2-specific split logic
- mapper / dataset creation using the same shared interfaces

### [scripts/optuna_create_study.py](/home/bartek/casus/mandala/scripts/optuna_create_study.py)

Bootstrap utility for Optuna studies:

- create the study if it does not exist
- validate the Optuna YAML spec
- print the resolved storage URL and study name

This is useful before launching many agents.

### [scripts/optuna_agent.py](/home/bartek/casus/mandala/scripts/optuna_agent.py)

Thin CLI wrapper around `run_optuna_agent`.

Typical HPC usage: one SLURM job runs one agent process.

### [scripts/optuna_postgres_job.sh](/home/bartek/casus/mandala/scripts/optuna_postgres_job.sh)

HPC utility to start a PostgreSQL service job for Optuna shared storage.

This is likely the cleanest way to support many parallel jobs on a no-internet cluster.

### [scripts/wandb_agent_offline.sh](/home/bartek/casus/mandala/scripts/wandb_agent_offline.sh)

HPC wrapper for WandB agent runs in offline mode:

- set `WANDB_MODE=offline`
- set run/cache dirs
- start one agent process

### [scripts/wandb_sync_offline.sh](/home/bartek/casus/mandala/scripts/wandb_sync_offline.sh)

Manual sync utility for later internet-enabled environments:

- locate offline WandB run directories
- call `wandb sync`
- optionally filter by project/run directory

## Core Dataclasses

### `RunRequest`

Shared request object used by dispatchers.

Fields:

- `dataset_family`: `silicon | sio2`
- `cfg`: resolved `Config`
- `resume_checkpoint`
- `run_name`
- `run_dir`
- `logging_backend`: `wandb | none`
- `logging_mode`: `online | offline | disabled`
- `hpo_backend`: `none | wandb | optuna`
- `objective_metric`
- `extra_callbacks`

### `PreparedData`

Dataset-preparation output.

Fields:

- `train_loader`
- `val_loader`
- `mapper`
- `train_ds`
- `val_ds`
- `dataset_summary`

### `TrainResult`

Single-run output.

Fields:

- `run_name`
- `run_dir`
- `metrics`
- `best_checkpoint`
- `latest_checkpoint`
- `final_checkpoint`

## API By File

### `scripts/train_silicon.py`

#### `setup_argparse() -> argparse.Namespace`
Parse silicon CLI options and generic training/backend flags.

#### `main() -> None`
Build a `RunRequest` and call `dispatch_run`.

### `scripts/train_sio2.py`

#### `setup_argparse() -> argparse.Namespace`
Parse SiO2 CLI options and generic training/backend flags.

#### `main() -> None`
Build a `RunRequest` and call `dispatch_run`.

### `src/training/dispatch.py`

#### `dispatch_run(request: RunRequest) -> TrainResult | None`
Main dispatcher. Route to single run, WandB agent, or Optuna agent.

#### `validate_dispatch_request(request: RunRequest) -> None`
Validate mutually dependent dispatch options before execution.

### `src/training/run_single.py`

#### `run_single_training(request: RunRequest) -> TrainResult`
Execute one full training run.

#### `prepare_data(request: RunRequest) -> PreparedData`
Select dataset-family-specific preparation and return loaders plus mapper.

#### `prepare_run_dir(request: RunRequest) -> Path`
Resolve run/checkpoint/artifact directory layout.

### `src/training/run_wandb_agent.py`

#### `run_wandb_agent(request: RunRequest) -> TrainResult`
Run one WandB-agent-provided configuration through the shared training path.

#### `merge_wandb_config(request: RunRequest, wandb_config: dict) -> RunRequest`
Apply sweep-provided overrides onto the base request.

### `src/training/run_optuna_agent.py`

#### `run_optuna_agent(request: RunRequest) -> None`
Main Optuna worker loop. Connect to study and execute trials.

#### `load_optuna_spec(path: str) -> dict`
Load and validate the Optuna study YAML.

#### `create_optuna_study(spec: dict, storage_url: str, load_if_exists: bool) -> Any`
Create or open the Optuna study.

#### `objective_from_trial(base_request: RunRequest, trial: Any, spec: dict) -> float`
Map one Optuna trial to one training run and return the objective value.

#### `apply_trial_overrides(request: RunRequest, trial: Any, spec: dict) -> RunRequest`
Sample parameters from a trial and merge them into the request.

#### `build_optuna_sampler(spec: dict) -> Any`
Construct TPE, random, or grid sampler from YAML.

#### `build_optuna_pruner(spec: dict) -> Any | None`
Construct pruner configuration, for example Hyperband.

#### `build_pruning_callback(trial: Any, metric_name: str) -> pl.Callback`
Return the callback that reports intermediate metrics and supports pruning.

### `src/training/logging.py`

#### `build_logger(request: RunRequest) -> Any | None`
Create WandB logger or return `None`.

#### `resolve_logging_mode(request: RunRequest) -> str`
Resolve `online`, `offline`, or `disabled`.

#### `configure_wandb_environment(mode: str, run_dir: Path) -> None`
Set `WANDB_MODE`, cache dirs, and local output dirs consistently.

#### `build_run_name(request: RunRequest, trial_number: int | None = None) -> str`
Generate stable run names for single runs and HPO trials.

#### `collect_offline_sync_metadata(run_dir: Path, logger: Any | None) -> None`
Save enough metadata to sync later.

### `src/training/trainer_factory.py`

#### `build_model(cfg: Config, mapper: Any) -> E3GNN`
Create the model.

#### `build_callbacks(request: RunRequest) -> list`
Build benchmark, artifacts, checkpoints, pruning callbacks, and so on.

#### `build_trainer(request: RunRequest, logger: Any | None, callbacks: list) -> pl.Trainer`
Create the Lightning trainer.

### `src/training/objective.py`

#### `extract_objective_metric(result_metrics: dict, metric_name: str) -> float`
Read the requested objective from the final metrics dict.

#### `extract_result_metrics(trainer: pl.Trainer) -> dict[str, float]`
Normalize callback metrics from Lightning into plain Python floats.

### `src/training/data/common.py`

#### `build_dataloader(ds: Any, cfg: Config, accelerator: str, shuffle: bool) -> DataLoader`
Shared DataLoader construction.

#### `validate_split_counts(num_train: int | None, num_val: int | None) -> None`
Validate global-split arguments.

#### `discover_snapshot_pairs(root: Path, matrix_name: str) -> list[tuple[Path, Path]]`
Shared recursive snapshot discovery helper.

### `src/training/data/silicon.py`

#### `prepare_silicon_data(request: RunRequest) -> PreparedData`
Build silicon datasets, loaders, mapper, and summary metadata.

#### `build_silicon_train_val_pairs(request: RunRequest) -> tuple[list, list]`
Resolve train/val pairs according to silicon split mode.

#### `log_silicon_dataset_summary(prepared: PreparedData, request: RunRequest) -> None`
Run silicon-specific study logging for dataset metadata.

### `src/training/data/sio2.py`

#### `prepare_sio2_data(request: RunRequest) -> PreparedData`
Build SiO2 datasets, loaders, mapper, and summary metadata.

#### `build_sio2_train_val_pairs(request: RunRequest) -> tuple[list, list]`
Resolve SiO2 train/val pairs according to its dataset layout.

## HPC Operation Model

### WandB on No-Internet HPC

Recommended mode:

- run WandB in offline mode on the cluster
- store run files locally in the run directory
- sync later from a machine with internet access

Operationally:

1. launch training or agent jobs with `WANDB_MODE=offline`
2. keep all WandB output under the run/checkpoint directory
3. after the run, copy or mount those directories on an internet-enabled machine
4. run `wandb sync ...`

This should be a first-class mode in `src/training/logging.py`, not an ad hoc shell detail.

### Optuna on HPC

Recommended mode:

- run a dedicated PostgreSQL job on the cluster
- point all Optuna agents at the same DB URL
- run many independent Optuna agent jobs in parallel

Operationally:

1. start `optuna_postgres_job.sh` as a service job
2. run `optuna_create_study.py` once
3. submit many `optuna_agent.py` jobs
4. each agent connects to the same study and pulls available trials

This mirrors the "many WandB agents" workflow, but with the scheduler controlled by Optuna through the shared DB.

### Why PostgreSQL Rather Than SQLite

For HPC parallel jobs, PostgreSQL is the right default because:

- SQLite is fragile under many concurrent writers
- PostgreSQL supports many agents safely
- study state survives worker restarts more robustly

## Suggested CLI Shapes

### Single run

```bash
python scripts/train_silicon.py --wandb-project mandala-silicon-main-study-port
```

### WandB agent on offline HPC

```bash
WANDB_MODE=offline \
python scripts/train_silicon.py --dispatch wandb-agent
```

### Create Optuna study

```bash
python scripts/optuna_create_study.py \
  --spec sweeps/train_silicon_density_energy_bayes_high_lr_smaller_main_optuna.yaml \
  --storage postgresql://USER:PASS@HOST:5432/mandala_optuna
```

### Run one Optuna agent

```bash
python scripts/optuna_agent.py \
  --spec sweeps/train_silicon_density_energy_bayes_high_lr_smaller_main_optuna.yaml \
  --storage postgresql://USER:PASS@HOST:5432/mandala_optuna
```

## Recommended First Implementation Order

1. extract the shared single-run training path
2. extract silicon dataset preparation
3. add logging backend abstraction with offline WandB support
4. add dispatcher layer
5. rewire `train_silicon.py` through the dispatcher
6. add Optuna study bootstrap and Optuna agent worker
7. add SiO2 entrypoint using the same shared runner
