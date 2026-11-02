Inference and analysis
======================

Predicting operators
--------------------

From a reference snapshot
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from data.snapshot import Snapshot
   from net.common import Config
   from net.evaluation import restore_model_for_snapshot, predict_snapshot

   snap = Snapshot.from_openmx("calc/HS.out", "calc/Si.out", convention="e3nn",
                               cfg=Config(verbosity=0))
   model = restore_model_for_snapshot("runs/silicon/best_model.pt", snap, device="cuda")
   predicted = predict_snapshot(model, snap)      # {"hamiltonian": BlockMatrix, ...}

Only the structure of ``snap`` is used: atoms, positions and cell build the
graph; the orbital configuration builds the mapper. ``predict_snapshot``
returns physical operators — envelopes are applied — on the edges of the
model graph.

From a structure only
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from data.structure_inference import build_model_input_from_structure

   x = build_model_input_from_structure(
       atoms=("Si",) * 8, positions=positions, box=lattice,   # Å, lattice rows
       cfg=model.cfg, mapper=model.mapper,
   )
   predicted = model.predict_matrices(x)

No electronic-structure file is needed. A model trained on stored DFT support
(``graph_source: stored``) needs an explicit radius for new structures,
since geometry alone cannot recover a stored support.

Lower-level calls
~~~~~~~~~~~~~~~~~

``model(x)`` returns :class:`~data.block_matrix.IrrepsBlockData` — irrep
coefficients per atom pair — and accepts ``physical_derivatives=True`` to
keep the geometry differentiable. ``model.predicted_irreps_to_block_matrices``
converts them, and ``model.pipeline(x, return_trace=True)`` exposes every
intermediate feature.

Post-processing
---------------

.. code-block:: python

   from core.sparse_math import trace_matmul_sparse_block_matrix
   from net.evaluation import predictions_to_snapshot

   H = predicted["hamiltonian"]
   H_herm = 0.5 * (H + H.adjoint())                         # Hermitian projection
   band_energy = trace_matmul_sparse_block_matrix(snap.density, H)   # Tr(D_ref H)

   combined = predictions_to_snapshot(predicted, snap, reference_matrices=("overlap", "density"))
   bands = combined.get_band_structure(path="GXMGR", npoints=200)
   bands.eigenvalues, bands.linear_k, bands.tick_positions, bands.tick_labels

``predictions_to_snapshot`` combines predicted operators with explicitly
chosen reference operators — typically the overlap, which is cheap to compute
exactly for a localized basis and keeps the generalized eigenproblem well
conditioned.

The evaluation example puts these steps together and plots the bands:

.. literalinclude:: ../../examples/quickstart/evaluate.py
   :language: python
   :lines: 14-

Exporting predictions
---------------------

``scripts/export_prediction.py`` writes predicted operators for downstream
codes:

.. code-block:: bash

   # OpenMX-compatible text sections from an XYZ/extxyz structure
   PYTHONPATH=src:. python scripts/export_prediction.py \
     --checkpoint runs/silicon/best_model.pt --structure input.xyz \
     --output predicted.matrix --device cpu

   # periodic HDF5 interchange with coverage masks and provenance
   PYTHONPATH=src:. python scripts/export_prediction.py --format native \
     --checkpoint runs/silicon/best_model.pt --structure geometry_dir \
     --graph-radius 7.2 --output predictions/

The OpenMX writer derives the periodic neighbour graph from the model cutoff,
builds the full :math:`(2\,\mathrm{CpyCell}+1)^3` translation table in OpenMX's
``Generation_ATV`` order and writes the predicted H/S/D sections with their
translation mapping. It is also available in Python:

.. code-block:: python

   from data.openmx_writer import export_model_openmx
   export_model_openmx(model, atoms=atoms, positions=positions,
                       lattice=lattice, output_path="predicted.matrix")

The native exporter writes global-frame matrices, per-entry coverage masks,
geometry and checkpoint provenance.

Reports
-------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Script
     - Output
   * - ``scripts/evaluate_checkpoint_materials.py``
     - Block heat maps, element correlation, relative block error versus
       distance, DOS and band structure for one structure.
   * - ``scripts/evaluate_checkpoint_snapshot.py``
     - Matrix and observable metrics of one snapshot.
   * - ``scripts/evaluate_operator_checkpoint.py``
     - Metrics of a recipe checkpoint on a split.
   * - ``scripts/report/evaluate_validation_relative_metrics.py``
     - Absolute and population-consistent relative validation errors.
   * - ``scripts/evaluate_openmx_density_grid.py``
     - Real-space electron density on a grid from a density matrix.
   * - ``scripts/benchmark_silicon_cif_scaling.py``
     - Inference time and memory versus system size.

All of them accept ``--help``.

Derivatives
-----------

.. code-block:: python

   forces = model.predict_forces(x)       # -dE_b/dR for E_b = Tr(DH)
   stress = model.predict_stress(x)

Both differentiate the operator-derived band energy through the complete
network, including the graph geometry.
