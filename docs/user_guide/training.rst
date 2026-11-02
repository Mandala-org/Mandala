Training
========

The trainer
-----------

``scripts.train`` trains any configuration:

.. code-block:: bash

   PYTHONPATH=src:. python -m scripts.train --config CONFIG.yaml --output RUN_DIR

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Option
     - Meaning
   * - ``--config``
     - YAML configuration (flat, recipe or both).
   * - ``--set KEY=VALUE``
     - Dotted override; repeatable.
   * - ``--output``
     - Run directory; its name becomes the run name.
   * - ``--resume``
     - Continue from ``RUN_DIR/latest_checkpoint.pt``.
   * - ``--dry-run``
     - Build everything and take one forward/backward step without writing.
   * - ``--accelerator``
     - ``auto``, ``cpu`` or ``gpu``.
   * - ``--max-wall-clock-seconds``, ``--wall-clock-scope``
     - Stop at the next epoch boundary after this budget, measured over the
       whole ``session`` or the ``fit`` only.
   * - ``--log-dir``
     - Directory for logger output.

What a run contains
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Content
   * - ``run_manifest.json``
     - Resolved configuration, input file hashes, split membership,
       fitted-statistics provenance, source and dependency fingerprints,
       hardware and environment.
   * - ``split_manifest.json``
     - The exact snapshot paths of every split.
   * - ``latest_checkpoint.pt``
     - Complete training state at the last epoch boundary: model, optimizer,
       scheduler, RNG states, model-selection ranking.
   * - ``best_model.pt``
     - The weights selected by ``checkpoint_monitor``, for evaluation.

Continuation and reproducibility
--------------------------------

.. code-block:: bash

   PYTHONPATH=src:. python -m scripts.train --config CONFIG.yaml --output RUN_DIR \
     --resume RUN_DIR/latest_checkpoint.pt

Continuation restores the optimizer, scheduler, Python/NumPy/PyTorch/CUDA RNG
states and the best-model ranking and replays an interrupted epoch from its
beginning, so a run split into several allocations follows the same
trajectory as an uninterrupted one. The run specification is immutable:
resuming with a changed configuration, input set or fitted statistics is
refused. Fitted statistics are restored, never refitted.

Long jobs on a scheduler are typically run as repeated segments of the same
command with ``--max-wall-clock-seconds`` slightly below the allocation; each
segment ends cleanly at an epoch boundary and the next one resumes.

Observable and spectral guidance in practice
--------------------------------------------

Band-energy guidance for a Hamiltonian-only model, contracting the prediction
with the reference density:

.. code-block:: yaml

   matrix_targets: [hamiltonian]
   train_on_energy: true
   train_observables_on_gt: true
   loss_coef_observables: 1.0e-4
   observable_loss_kind: mae

Joint operators with fully predicted observables:

.. code-block:: yaml

   matrix_targets: [hamiltonian, overlap, density]
   train_on_energy: true
   train_on_num_electrons: true
   train_observables_on_gt: false
   loss_coef_observables: 3.0e-3

Spectral guidance on a :math:`2\times2\times2` mesh around the Fermi level:

.. code-block:: yaml

   spectral_loss_enabled: true
   spectral_loss_coef: 3.0e-3
   spectral_loss_kmesh: 2x2x2
   spectral_loss_window_ev: 10.0
   spectral_loss_taper_ev: 2.0
   spectral_loss_kind: huber
   spectral_loss_huber_delta_ev: 0.1
   spectral_fermi_cache_path: runs/fermi_cache.pt

Reference Fermi levels and overlap factorizations can be computed once for a
dataset:

.. code-block:: bash

   PYTHONPATH=src:. python scripts/precompute_spectral_fermi_cache.py --help

Fitted distance envelopes
-------------------------

An envelope JSON contains a ``family`` and per-pair parameters ``theta``
(log-transformed where positive) and, for ``slater_exp_quad_soft_wall``, the
largest fitted distance ``x_max``. Enable it with

.. code-block:: yaml

   hamiltonian_envelope_mode: multiply_prediction
   hamiltonian_envelope_path: scripts/reproduce_examples/sio2_envelope.json

``loss_weighting_mode: envelope_inverse_sqrt_clipped`` additionally uses the
envelope to re-weight edges in the loss.

Monitoring
----------

Metrics are logged per epoch with Weights & Biases (``wandb_project``;
``WANDB_MODE=offline`` keeps logs local). Besides the objective terms, MANDALA
logs element-wise MAE/RMSE per operator (``*_elementwise``), per species pair
and per irrep (``log_per_pair_loss_metrics``, ``log_per_irrep_metrics``),
Hamiltonian error contributions by irrep and pair, observables with predicted
and reference partners, gradient norms and per-stage activation magnitudes
(``log_grad_norm``, ``log_activation_mag``).

Hyperparameter search
---------------------

Every flat field is a sweep dimension. ``scripts/wandb_run.py`` is a W&B
sweep agent entry point and ``scripts/optuna_agent.py`` an Optuna one; the
paper's ablation sweeps in ``sweeps/paper_ablations`` are complete examples
(:doc:`../reproduction/ablations`).

Fine-tuning
-----------

``freeze_backbone_train_heads_only: true`` freezes every stage except those
with ``role: readout``. This adapts a trained network to a new operator or a
new chemistry quickly while keeping its learned representation.
