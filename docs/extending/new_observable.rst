Adding an observable objective
==============================

Why add an observable?
----------------------

A small average error over matrix entries need not imply a small error in a
quantity derived from the whole operator. An additional differentiable
objective can emphasize the quantity a downstream calculation needs, while
retaining matrix supervision. The built-in band-energy, electron-count and
spectral terms are described in :doc:`../theory/objectives`; this example adds
a new term without changing a network layer or the training loop.

We supervise the mean onsite Hamiltonian trace per atom:

.. math::

   a(H)=\frac{1}{N_{\rm atoms}}\sum_i\Tr H^{\bm R=0}_{ii},\qquad
   \mathcal L_{\rm trace}=[a(H^{\rm pred})-a(H^{\rm ref})]^2,\qquad
   \mathcal L=\mathcal L_{\rm matrix}+\lambda_{\rm trace}\mathcal L_{\rm trace}.

This scalar measures the aggregate onsite level shift in the chosen orbital
basis. It is invariant under rotations and inversion of the orbital frames.
With a nonorthogonal basis it is **not** the band energy or the sum of the
generalized eigenvalues. Only zero-shift onsite blocks contribute; a periodic
image with the same atom indices is not an onsite block. Both labels and
predictions use the same basis and units.

1. Define the term
------------------

``examples/layer_cookbook/new_observable.py`` implements the scalar and its
squared error. The function receives one structure's graph ``inputs``, plus
``predictions`` and ``targets`` mapping operator names to physical
:class:`~data.block_matrix.BlockMatrix` objects. Hamiltonians here are in
Hartree, so the loss has units Hartree²/atom². The reference trace is computed
from the reference matrix; no extra label file is needed.

.. literalinclude:: ../../examples/layer_cookbook/new_observable.py
   :language: python
   :pyobject: onsite_trace_per_atom

.. literalinclude:: ../../examples/layer_cookbook/new_observable.py
   :language: python
   :pyobject: hamiltonian_trace_loss

Use tensor operations that retain autograd, and return one real scalar tensor.
This example requires complete, finite onsite reference blocks and raises an
error for missing or masked onsite data. A missing label must not silently
become a zero trace.

2. Register it
---------------

The decorator ``@register_objective_term("hamiltonian_trace")`` registers the
function with the hook in ``src/net/objectives.py``. Import the module before
constructing the model, just as for custom layers. The example's entrypoint
does this automatically. ``src/net/e3gnn.py`` evaluates the registered terms
for training, validation and testing and adds their weighted values to the
standard loss. Independent structures are averaged within a batch.

The hook accepts stateless functions. Optional keyword arguments can be set in
the term's ``options`` mapping. Unknown names, duplicate terms, invalid
weights, nonscalar returns and nonfinite losses raise errors. Existing runs
without ``objective.terms`` retain their original objectives.

Restoring this example's checkpoint also requires the registration import.
For instance, ``examples/quickstart/evaluate.py`` accepts
``--plugin examples/layer_cookbook/new_observable.py``; importing that module
registers both the objective and the cookbook layers without starting training.

3. Enable it in YAML and train
------------------------------

``examples/layer_cookbook/new_observable.yaml`` is an overlay on
``examples/layer_cookbook/cookbook.yaml``:

.. literalinclude:: ../../examples/layer_cookbook/new_observable.yaml
   :language: yaml

``terms`` adds the new objective to the existing masked matrix MSE. ``weight``
sets :math:`\lambda_{\rm trace}`. The trainer logs the unweighted loss as
``train/loss_term_hamiltonian_trace`` and its contribution as
``train/loss_term_hamiltonian_trace_weighted``, with analogous ``val/`` and
``test/`` names.

Run from the repository root, in the installed project environment. If the
OpenMX files are still Git LFS pointers, retrieve just these inputs first:

.. code-block:: bash

   git lfs install
   git lfs pull --include='data/small/Si/HS.out,data/small/Si/Si.out,data/small/Si_old/HS.out,data/small/Si_old/Si.out'
   OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src:. \
     python examples/layer_cookbook/new_observable.py \
       --output runs/cookbook_trace --accelerator cpu

The wrapper merges the overlay, invokes the standard ``scripts/train.py``
trainer, reloads the final ``latest_checkpoint.pt`` weights, and measures the
new loss on the same training structure before and after optimization. The
output directory must be new. Checkpoints and the resolved recipe and split
manifests are saved in ``runs/cookbook_trace/``.

The inputs are the bundled eight-atom silicon calculations in
``data/small/Si`` (training) and ``data/small/Si_old`` (validation), retrieved
from Git LFS. The perturbed validation directory used in :doc:`assembling`
is not bundled, so this example uses ``Si_old`` instead. No external dataset
or test fixture is required.

On a CPU with Python 3.10, PyTorch 2.5.1+cu121, e3nn 0.6.0, seed 7 and
float32, the two-epoch run printed:

.. code-block:: text

   Hamiltonian-trace loss before training: 8.3718232810e-02 Hartree^2/atom^2
   Hamiltonian-trace loss after 2 epochs: 5.0784446299e-02 Hartree^2/atom^2

These are training-structure losses, measured in evaluation mode at both
endpoints. They demonstrate optimization, not validation accuracy; floating
point results can vary with the software and hardware.

The cookbook model is tiny and inaccurate on purpose. It uses only one
training structure and two epochs here so that the extension mechanism is
quick to inspect and run. Its trace improvement is not evidence of useful
Hamiltonians or of the publication models' accuracy.

Verification
------------

The cookbook's own check exercises the objective through the model's graph
step, verifies its configured contribution to the total loss, and backpropagates
the observable alone to check finite, nonzero model gradients:

.. code-block:: bash

   OMP_NUM_THREADS=4 PYTHONPATH=src:. python examples/layer_cookbook/verify.py

``tests/unit/net/test_objective_terms.py`` independently checks the trace,
reference-label dependence, analytic gradients, rotations, improper rotations,
periodic self-images, missing onsite labels and registration errors.
``tests/workflow/test_modular_training.py`` checks the new objective with
packed targets, batching and explicitly masked reference entries.
