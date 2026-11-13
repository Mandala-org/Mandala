Demonstration models
====================

Model configurations
--------------------

The three models share two message-passing layers, separate-weight tensor
products, the rich edge encoder with harmonic tensor square, invariant
attention aggregation, split per-pair heads with separate shifted-self
handling and no spectral loss. They differ in:

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Setting
     - ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2`
     - SiO\ :sub:`2`
     - Perturbed Si
   * - Targets
     - H
     - H
     - H, D, S
   * - Cutoff / :math:`\ell_{\max}`
     - 11 Å / 6
     - 10 Å / 4
     - 8 Å / 4
   * - Neck / head depth
     - 2 / 1
     - 1 / 1
     - 2 / 2
   * - Head block
     - NormAct
     - NormAct
     - FiLM
   * - Distance envelope
     - multiply prediction
     - multiply prediction
     - off
   * - Band-energy guidance
     - off
     - off
     - :math:`3\times10^{-3}`, reference partner
   * - Split (train/val), seed
     - 80/20, 44
     - 90/10, 42
     - 80/16, 43

The complete configurations of the evaluated checkpoints are in
``paper/data/evaluated_model_configs/{zncu2sn_ses2,sio2,silicon_perturbed}.yaml``.
The paper appendix lists their hidden irreps; e.g. the SiO\ :sub:`2` model
uses
``128x0e+64x0o+32x1e+32x1o+24x2e+24x2o+16x3e+16x3o+8x4e``.

Training recipes
----------------

``scripts/reproduce_examples`` contains one frozen recipe per material
(``zncusnses.json``, ``sio2.json``, ``silicon.json``) with the architecture,
objective, optimizer, scheduler, spike-rollback settings, split sizes and
seeds of the selected models, the fitted distance envelopes
(``*_envelope.json``, verified by SHA-256 at start-up) and one launcher per
material. Each recipe initializes all weights from scratch.

These are fresh training experiments with the selected architectures and
objectives, not a replay of the historical fine-tuning and restart sequence.
The source learning rates and checkpoint ancestry are documented in
``scripts/reproduce_examples/README.md``. In particular, silicon uses the
current native spin=0 density convention, whereas the historical targets were
doubled; its density and energy errors cannot be compared directly with the
historical values below.

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 16 16

   * - Recipe
     - Targets / loss
     - Initial LR
     - Seeds (train / split)
     - Epoch cap
   * - ``zncusnses.json``
     - H / MSE
     - 0.0108
     - 44 / 44
     - 3000
   * - ``sio2.json``
     - H / MAE
     - 0.0027
     - 42 / 42
     - 4000
   * - ``silicon.json``
     - H, D, S / MAE + band energy (:math:`3\times10^{-3}`)
     - 0.001
     - 43 / 43
     - 5000

1. **Point the recipe to your data.** Edit the ``training`` block of the
   recipe — ``data_path``, ``checkpoint_dir`` — and ``save_dir`` and
   ``snapshot_cache_dir`` in ``config``:

   The hard-coded ``/bigdata/...`` locations are the authors' HPC paths;
   replace them with your local paths for the deposited datasets and training
   outputs (a RODARE data deposit is being prepared).

   .. code-block:: json

      "training": {
        "dataset_kind": "silicon_scales",
        "data_path": "/data/perturbed_Si",
        "checkpoint_dir": "/runs/publication",
        "scales": [1],
        "num_train_per_scale": 80,
        "num_val_per_scale": 16
      }

2. **Inspect the resolved plan** (no files are written, no GPU is used):

   .. code-block:: bash

      bash scripts/reproduce_examples/train_silicon.sh --plan

3. **Train.** Run each launcher in its own single-GPU allocation:

   .. code-block:: bash

      bash scripts/reproduce_examples/train_zncusnses.sh
      bash scripts/reproduce_examples/train_sio2.sh
      bash scripts/reproduce_examples/train_silicon.sh

   A segment budgets 45 hours and stops at an epoch boundary;
   request 48 hours to leave room for the final epoch and checkpoint I/O.
   The final complete epoch can overrun the budget.
   Re-running the same command continues the run from
   ``<checkpoint_dir>/<run>/latest_checkpoint.pt`` with the full optimizer,
   scheduler, RNG and model-selection state, until the cumulative epoch cap is
   reached. ``--action start`` refuses to touch an existing run;
   ``--action continue`` requires one.

   Every run writes a contract (``<run>.contract.json``) with the SHA-256 of
   the recipe and of every source file, the installed package versions, the
   GPU and the determinism settings. Continuation with a different recipe,
   code or runtime is refused. Input manifests also reject changed split
   identities, sizes or modification times. Preserve the immutable data:
   large inputs are not content-hashed, and CUDA scatter reductions can still
   be nondeterministic.

Validation metrics
------------------

Compute the absolute and relative validation errors of a newly trained
checkpoint with the deterministic validation split of its recipe. Run all
commands from the repository root. The ``/runs/...`` and ``/data/...`` paths
below are examples to replace with your checkpoint and dataset paths:

.. code-block:: bash

   PYTHONPATH=src:. python scripts/report/evaluate_validation_relative_metrics.py \
     --checkpoint /runs/publication/silicon/best_model.pt \
     --dataset-kind silicon_scales --data-path /data/perturbed_Si \
     --scales '[1]' --num-train-per-scale 80 --num-val-per-scale 16 \
     --data-split-seed 43 --output-json silicon_metrics.json

For SiO\ :sub:`2`, change ``--dataset-kind`` to ``siox``, select its checkpoint
and data root, and replace the silicon split options with
``--num-train 90 --num-val 10 --data-split-seed 42``. For ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2`,
use ``--dataset-kind ZnCuSnSeS``, its checkpoint and data root, and
``--scales '[1]' --num-train-per-scale 80 --num-val-per-scale 20 --data-split-seed 44``.
The relative error divides the mean per-structure
validation MAE by the mean absolute reference value over the same validation
structures. Exact historical split membership is not asserted by the fresh
recipes. The values of the published checkpoints are stored in
``paper/data/validation_relative_metrics``:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Material
     - Metric
     - Value (relative)
   * - ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2`
     - Hamiltonian MAE
     - 3.012 meV (2.95 %)
   * - SiO\ :sub:`2`
     - Hamiltonian MAE
     - 1.996 meV (1.67 %)
   * - SiO\ :sub:`2`
     - Band-energy MAE, :math:`\Tr(D^{\text{ref}}H^{\text{pred}})`
     - 0.146 eV/atom (0.193 %)
   * - Silicon
     - Hamiltonian MAE
     - 0.4352 meV (0.355 %)
   * - Silicon
     - Density-matrix MAE
     - :math:`5.758\times10^{-5}` (1.95 %)
   * - Silicon
     - Overlap MAE
     - :math:`1.390\times10^{-5}` (0.109 %)
   * - Silicon
     - Band-energy MAE
     - 0.0207 eV/atom (0.0631 %)

Hamiltonian errors are shift-resolved block errors after Hermitian
projection, in eV; density and overlap errors are dimensionless.

Figures
-------

``scripts/evaluate_checkpoint_materials.py`` produces the per-structure
figures — block heat maps of the first atoms, element correlation, relative
block error versus distance, density of states and band structure — for a
checkpoint and one validation structure:

.. code-block:: bash

   PYTHONPATH=src:. python scripts/evaluate_checkpoint_materials.py \
     --checkpoint /runs/publication/silicon/best_model.pt \
     --mode snapshot --snapshot-path /data/perturbed_Si/scale_1/Si_scale_1_090 \
     --output-dir figures/silicon \
     --path-string LGXUG --dos-kmesh 4x4x4 --use-gt-overlap-for-eigs --max-atoms 8 \
     --hamiltonian-clim 0.01 --density-clim 0.01

The paper uses validation structure no. 94 for ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2`,
``13_67164000`` for SiO\ :sub:`2` (DOS with :math:`\sigma=0.2` eV on a
:math:`2\times2\times2` mesh) and no. 90 for silicon (path
:math:`L\to\Gamma\to X\to U\to\Gamma`, tetrahedron DOS on :math:`4\times4\times4`).
Run the script with ``--help`` for all plotting options (colour limits,
energy windows, DOS method, k-meshes).

The silicon band-structure and DOS figure is generated by the command above;
the rendered figure is not bundled in this checkout.

.. figure:: ../../paper/figures/results/paper_zncusnses_hamiltonian_correlation.png
   :width: 70%

   Correlation of all 5 345 640 Hamiltonian elements of the representative
   ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2` validation structure (MAE 3.029 meV,
   :math:`R^2=0.999460`).

Paper figures and tables
------------------------

The manuscript sources are not part of this repository. The data behind the
paper's figures and tables are in ``paper/data`` and the figures are
regenerated with

.. code-block:: bash

   python paper/generate_result_plots.py

which writes to ``paper/figures``.
