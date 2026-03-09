# PySCF Baseline (PBC-Only)

This folder contains a cheap, reproducible periodic (PBC) PySCF baseline workflow for:

- `data/small/H2O/original/H2O.info.out`

## What the baseline does

- Reads geometry/lattice and reference energies from the OpenMX `.info.out` snapshot.
- Runs a periodic baseline on a k-point mesh (`heuristic` by default; PySCF for `rhf`/`rks`).
- Writes shift-resolved AO matrices for lattice translations.
- Enforces shell-count-matched basis size from OpenMX `orbital_set`.

For this H2O snapshot, the shell counts are:

- `H: 3s2p`
- `O: 3s3p2d`

## Files

- `calc_pyscf_baseline.py`: main PBC runner.
- `run_h2o_original_baseline.sh`: simple launcher.
- `results/`: target for generated baseline data.

## Usage

From repo root (ultra-cheap default, full heuristic prediction):

```bash
bash data/pyscf_baseline/run_h2o_original_baseline.sh
```

This defaults to `METHOD=heuristic` and writes Hamiltonian/overlap/density artifacts.
On this setup it is expected to run roughly under 1 second.

Direct Python run:

```bash
python data/pyscf_baseline/calc_pyscf_baseline.py \
  --info-path data/small/H2O/original/H2O.info.out \
  --output-dir data/pyscf_baseline/results \
  --run-name h2o_original_rhf_openmx_like \
  --method heuristic \
  --basis-mode openmx_like \
  --kmesh 1 1 1
```

Optional parse-only mode:

```bash
DRY_RUN=1 bash data/pyscf_baseline/run_h2o_original_baseline.sh
```

PySCF HF run (slower):

```bash
METHOD=rhf python data/pyscf_baseline/calc_pyscf_baseline.py \
  --info-path data/small/H2O/original/H2O.info.out \
  --output-dir data/pyscf_baseline/results \
  --run-name h2o_pbc_rhf_k111 \
  --method rhf \
  --basis-mode openmx_like \
  --kmesh 1 1 1
```

PySCF DFT run (slowest):

```bash
METHOD=rks python data/pyscf_baseline/calc_pyscf_baseline.py \
  --info-path data/small/H2O/original/H2O.info.out \
  --output-dir data/pyscf_baseline/results \
  --run-name h2o_pbc_pbe_k111 \
  --method rks \
  --xc pbe,pbe \
  --grid-level 1 \
  --basis-mode openmx_like \
  --kmesh 1 1 1
```

## Output artifacts

For `--run-name <name>`, the runner writes:

- `<name>.json`: settings, parsed snapshot metadata, SCF summary.
- `<name>.npz`: packed arrays including:
  - central-cell matrices: `hamiltonian_ao`, `overlap_ao`, `dm_ao`, `hcore_ao`
  - shift-resolved matrices: `shifts`, `hamiltonian_shifted`, `overlap_shifted`, `density_shifted`
  - k-space metadata: `kmesh`, `kpts_abs`
  - orbital data: `mo_coeff`, `mo_occ`, `mo_energy_hartree`
- Standalone arrays:
  - `<name>.hamiltonian_ao.npy`
  - `<name>.overlap_ao.npy`
  - `<name>.density_ao.npy`
  - `<name>.shifts.npy`
  - `<name>.hamiltonian_shifted.npy`
  - `<name>.overlap_shifted.npy`
  - `<name>.density_shifted.npy`
- `<name>.xyz`: geometry extracted from the snapshot.

## Loading in repo code

Use the parser in `src/data/pyscf_baseline_parser.py`:

```python
from data.pyscf_baseline_parser import load_pyscf_baseline, load_pyscf_snapshot

payload = load_pyscf_baseline("data/pyscf_baseline/results/<run>.npz")
snapshot = load_pyscf_snapshot("data/pyscf_baseline/results/<run>.npz")
```

The loader is PBC-only and requires shift-resolved tensors (`shifts`, `*_shifted`).
It builds `BlockMatrix` directly from `(shift, block)` data (no dense fallback path).

## Method options and rough cost

Current default is heuristic:

- `METHOD=heuristic` -> geometry-only, no SCF, no PySCF dependency

PySCF HF option:

- `METHOD=rhf` -> periodic `KRHF` (or `KUHF` if `SPIN != 0`)

PySCF DFT option:

- `METHOD=rks XC=...` -> periodic `KRKS` (or `KUKS` if `SPIN != 0`)

Rough wall-time multipliers for this H2O setup (80 AOs, same convergence settings):

- `heuristic`, `kmesh 1x1x1`: `~0.01x` to `~0.05x` vs HF (typically sub-second)
- `rhf`, `kmesh 1x1x1`: `1.0x` baseline
- `rks lda,vwn`, `kmesh 1x1x1`, `grid_level 0`: `1.5x` to `2.5x`
- `rks pbe,pbe`, `kmesh 1x1x1`, `grid_level 1`: `2.0x` to `4.0x`
- Any method with `kmesh 2x2x2` vs `1x1x1`: about `8x` for k-point count, often close to `6x` to `10x` wall time
- Any method with `kmesh 3x3x3` vs `1x1x1`: about `27x` k-points, often close to `20x` to `35x` wall time

Use these as planning estimates; exact runtime depends strongly on CPU, BLAS, and SCF convergence behavior.
