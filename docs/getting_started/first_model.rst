Your first model
================

This tutorial trains, continues and evaluates a compact MANDALA model on the
two silicon calculations bundled with the repository: the ideal eight-atom
diamond cell for training and a thermally perturbed cell for validation. It
runs on a laptop CPU.

The configuration
-----------------

.. literalinclude:: ../../examples/first_model/silicon.yaml
   :language: yaml

The flat fields select the default network (:doc:`../theory/architecture`)
with two attention message-passing layers and features up to :math:`\ell=4`
— enough for the *d*–*d* couplings of the ``s2p2d1`` basis
(:doc:`../theory/operators`). All three operators are predicted jointly and
trained with an element-wise MAE (``loss_l1_fraction: 1.0``).

Check it
--------

.. code-block:: bash

   PYTHONPATH=src:. python -m scripts.train --config examples/first_model/silicon.yaml \
     --output runs/silicon --accelerator cpu --dry-run

The dry run parses both calculations, builds the 8 Å periodic graphs, builds
the model, takes one optimization step and prints the parameter count, the
loss and the split sizes — without writing anything.

Train
-----

.. code-block:: bash

   PYTHONPATH=src:. python -m scripts.train --config examples/first_model/silicon.yaml \
     --output runs/silicon --accelerator cpu

Training reads both calculations once (about two minutes for the 50 MB
text dumps) and then takes one to two seconds per epoch on an eight-core CPU.
The objective is the sum of the element-wise MAEs of H (Hartree), S and D:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Epoch
     - Training loss
     - Validation loss
   * - 1
     - 0.0327
     - 0.0250
   * - 100
     - 0.00937
     - 0.0126
   * - 200
     - 0.00481
     - 0.0114
   * - 400
     - 0.00296
     - 0.0111
   * - 600
     - 0.00290
     - 0.0111

With a single training structure the model fits it closely while the
validation error levels off: generalization to new geometries comes from
training on many structures, which is exactly what the paper models do.

The run directory contains ``run_manifest.json`` (resolved configuration,
input hashes, environment), ``split_manifest.json``,
``latest_checkpoint.pt`` and ``best_model.pt``.

Continue
--------

An interrupted run resumes exactly where it stopped — same optimizer state,
learning-rate schedule, random streams and best-model ranking. Repeat the
original command with ``--resume``:

.. code-block:: bash

   PYTHONPATH=src:. python -m scripts.train --config examples/first_model/silicon.yaml \
     --output runs/silicon --accelerator cpu --resume runs/silicon/latest_checkpoint.pt

Evaluate
--------

.. code-block:: bash

   PYTHONPATH=src:. python examples/quickstart/evaluate.py \
     --checkpoint runs/silicon/best_model.pt \
     --structure data/small/Si_perturbed_scale1_078 --plot bands.png

.. code-block:: text

   compared blocks: 696
   hamiltonian  MAE   84.1019 meV  max 3717.2480 meV
   overlap      MAE    0.0066   max    0.1580
   density      MAE    0.0011   max    0.0392
   band energy Tr(D_ref H) per atom: reference -16.4240 eV, predicted -15.3572 eV

The script compares every block that the model graph and the reference
calculation share, evaluates the band energy with the reference density
through the sparse trace :math:`\sum\operatorname{tr}(H_{ij}^{\bm L}D_{ji}^{-\bm L})`,
and with ``--plot`` draws the bands of the predicted Hamiltonian, combined with
the reference overlap, on top of the reference bands.

Going further
-------------

* A model trained on 80 perturbed structures for a few days on a GPU reaches
  a Hamiltonian MAE of 0.44 meV on held-out structures
  (:doc:`../reproduction/models`).
* :doc:`../user_guide/training` adds band-energy and spectral guidance.
* :doc:`../extending/assembling` trains a network built from new layers on
  the same data.
