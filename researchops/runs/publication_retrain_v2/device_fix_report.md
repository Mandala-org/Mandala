# Publication retrain v2: runtime device serialization

Branch: `referee-retrain-v2`.

## Cause and fix

`scripts/train.py:_resolve_accelerator_and_device` (lines 378 and 384)
replaces `cfg.device` with `torch.device("cuda:0")` or `torch.device("cpu")`.
This happens in the shared `run_training` path before dataset construction;
the v1 and v2 wrappers both call that path. The recipe JSON supplies strings,
not device objects. `src/data/factory.py:170` resolves `cfg.to_mapping()`.
Previously `Config.to_mapping` normalized `torch.dtype` but left `torch.device`
objects in the exported dictionary, so OmegaConf rejected its `device` entry
at `src/net/configuration.py:242`.

The two-line fix in `src/net/common.py:88` converts devices to strings at the
existing recursive serialization boundary. No tensors, device selection,
precision, recipe, or training numerical settings are changed by this fix.
The regression test in `tests/unit/net/test_modular_configuration.py` covers
CPU and indexed CUDA devices, including `dataset_device`, resolution, and
round-trip device equivalence, without allocating GPU memory.

`--plan` only prints recipe JSON, so it cannot reproduce dataset construction.
The pre-fix CPU reproduction on ROSI used the same calls as
`verify_cpu_datasets.py`, restricted to silicon. It reached the exact reported
`UnsupportedValueType: Value 'device' ... full_key: device` exception.

## Verification commands

On ROSI, from `/data/home2/brzoza73/casus/mandala_worktrees/retrain_v2`:

```sh
source ~/scripts/profiles/mandala.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest -q tests/unit/scripts tests/unit/net/test_modular_configuration.py
python -u researchops/runs/publication_retrain_v2/verify_cpu_datasets.py
python -u researchops/runs/publication_retrain_v2/verify_cpu_datasets.py --resolve-only
```

Pytest: **96 passed, 52 subtests passed, 1 warning, 12.13 seconds**.
The warning was NVML initialization on the CPU login node.
Full silicon and SiO2 datasets completed (80/10/6 and 80/10/10). The full Zn
load was intentionally interrupted after configuration resolution and initial
graph loading: some uncached snapshots took about 90 seconds each. To reserve
time for submission and monitoring, `--resolve-only` calls the actual resolver
with actual recipe metadata and stops immediately after it succeeds, before
cold snapshot loading. It does not replace or simulate the resolver.
Resolution results for all three recipes and replacement job IDs are in the
companion `device_fix_verification.json` when verification completes.
Raw ROSI logs: `device_fix_pytest.log`, `device_fix_cpu_datasets.log`,
`device_fix_cpu_resolution.log`. The full-load log ends with the intentional
Zn interruption; this is not another configuration failure.

On Terok:

```sh
git diff --check
git fetch github develop && git rebase github/develop
git show 20b825e:src/net/common.py | sha256sum
git show github/develop:src/net/common.py | sha256sum
```

Both original common.py files have SHA256
`a561a93df64d5de0d70118fc853d544735fa330d4f9c0a7a2cfc4e44b16d8e02`.
Current develop is `41ce77f65c4b0f942532853e3a3e7b0d7868d493` and needs the same fix.

## Commits and choices

- `1ba5179`: original fix and unit test, pushed early for preservation.
- `1186303`: the same fix after the required rebase onto current develop.
- `c0aea51`: ancestry-only `ours` merge of the already-published `1ba5179` tip,
  retaining the rebased tree and allowing an ordinary fast-forward push.
  Develop's history had been rewritten; no force-push was used.
- `69543d0` (rebased copy `e59d8a6`): reusable CPU dataset reproduction for all
  three frozen v2 recipes.
- `0238f12`: preserve the published ancestry after the final required rebase
  check; its tree is identical to the tested/published `69543d0` tree.

- `a669df6`: optional resolver-boundary check to avoid expensive cold snapshot
  loading when verifying this configuration failure.

The required rebase incorporates existing develop changes; those are upstream
changes, not edits made for this fix. No files under `paper/` were edited and
no README was edited for this task. No branch was merged into develop.

CPU verification overrides only `device="cpu"` and `gpus=0`; it uses the full
declared manifests and does not create training runs or frozen contracts.
Silicon and SiO2 fully loaded; full Zn loading is not verified. All three
recipes execute the real factory configuration resolution in boundary mode.
No retrain recipe, dataset split, launcher resource setting, or run name was
changed. Full GPU training and final model quality remain unverified until the
resubmitted jobs run.

## Replacement chains

Submitted from source commit `a669df6dd18633ee55957f89a11090a167a82025` with
`python researchops/runs/publication_retrain_v2/submit.py --append` after a
`--plan` check confirmed `gpu-h100-casus` / `casus`. All 12 resulting sbatch
commands were checked for that partition/account, 45-hour segments, and nice
1000. The existing launcher retains one GPU, four CPU cores, and 128 GB per job.

| Material | Segment 1 | Segment 2 | Segment 3 | Segment 4 |
| --- | --- | --- | --- | --- |
| Silicon | 837005 | 837006 | 837007 | 837008 |
| SiO2 | 837009 | 837010 | 837011 | 837012 |
| ZnCuSnSeS | 837013 | 837014 | 837015 | 837016 |

Registered successfully as `mandala/publication-retrain-v2: attempt 4`:

```sh
~/second_brain/bin/sb exp attempt --project mandala --exp publication-retrain-v2 \
  --jobs 837005 837006 837007 837008 837009 837010 837011 837012 837013 837014 837015 837016 \
  --reason 'Shared training device selection stores torch.device in Config; Config.to_mapping now serializes devices to strings before OmegaConf resolution.' \
  --by fixer
```

Cancelled the nine obsolete pending downstream jobs (836880-836882,
836896-836898, 836900-836902). Contrary to the initial brief, they had remained
in Slurm with Dependency/DependencyNeverSatisfied rather than being dropped.
No unrelated jobs were changed.

GitHub validation of tested source tree `69543d0`: both CPU jobs (Python 3.10
and 3.11) passed in Tests run
https://github.com/Mandala-org/Mandala/actions/runs/42756825123;
Documentation passed in
https://github.com/Mandala-org/Mandala/actions/runs/42756825075.
The later helper change is outside `src/` and `scripts/`; production source
is unchanged from that tested tree. The final evidence/submission commit is
also outside the source paths used by the frozen training contracts.

## Startup monitoring result

Final monitoring snapshot: **2026-11-10 05:07:21 UTC**. Silicon and
SiO2 had each run for **10m53s** and reached **epoch 2** and **epoch 3**;
Zn had run for **10m23s** and was still loading its training snapshots
(18/80 at the last progress update). All three first segments were RUNNING
on gh003, all nine remaining segments were PENDING on their intended
dependencies, and none of the three logs contained a traceback.

Zn passed the actual configuration boundary on CPU and had entered
snapshot loading on GPU, but its complete GPU dataset load and first epoch
were not yet verified. No complete retrain or final scientific result is
claimed. Full startup monitoring snapshots, resource arguments, and
registration output are retained in `device_fix_verification.json`.
