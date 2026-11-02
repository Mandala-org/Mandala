Operator readout: ``pair_mixture_readout``
==========================================

Role
----

The readout turns final edge (and node) features into the irrep coefficients
of the operator blocks. The irreps it must produce are fixed by the orbital
bases of the two atoms (:doc:`../theory/operators`); the readout decides how
parameters are shared between species pairs and edge classes.

Built-in members of this kind: ``partitioned_irrep_readout``
(:class:`~net.heads.DeepHead`: necks per edge class, one projection per
species pair in ``split`` mode or a single pair-conditioned projection in
``shared_conditioned`` mode), and ``irrep_linear`` onto
``${layout.coupled_irreps}`` followed by ``shell_tensor_decode`` (DeepH-E3).

The new layer
-------------

A separate readout per species pair has :math:`N^2` times the parameters of a
shared one and learns rare pairs from few examples; a fully shared readout
cannot distinguish chemistry. The pair-mixture readout interpolates with a
*rank* :math:`K`:

.. math::

   \bm x_{ij}=\sum_{k=1}^{K}c_k(a_i,a_j)\;W_k\,\bm e_{ij},

where :math:`W_1,\dots,W_K` are shared equivariant linear maps onto the
coupled irreps and :math:`c(a,b)\in\mathbb R^K` is a learned embedding of the
ordered species pair. :math:`K=1` is a single shared readout up to a per-pair
scale; :math:`K=N^2` can represent one independent readout per pair. In
between, pairs share a low-dimensional basis of readouts, and related
pairs — say Zn–S and Zn–Se — can learn similar coefficients.

The coefficients start at :math:`1/K`, so the initial readout is the average
of the :math:`K` maps for every pair.

**Symmetry.** Linear maps between equal irreps (checklist 3a) weighted by
invariant coefficients (3c). The output irreps are exactly the requested
``irreps_out``, which ``shell_tensor_decode`` checks against the operator
codec.

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: PairMixtureReadout

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _pair_mixture_readout

Usage
-----

.. code-block:: yaml

   - id: readout
     role: readout
     transform: pair_mixture_readout
     inputs: [edge_1, edge_type_idx]
     outputs: [coupled]
     options: {irreps_out: '${layout.coupled_irreps}', num_pairs: '${species.pair_count}', rank: 4}
   - id: decode
     role: readout
     transform: shell_tensor_decode
     inputs: [coupled]
     outputs: [hamiltonian_native]
     options: {shell_pairs: '${layout.shell_pairs}', spinful: '${layout.spinful}'}

``${layout.coupled_irreps}`` and ``${species.pair_count}`` are resolved from
the training data, so the same recipe works for any chemistry.

Variations
----------

* Condition the coefficients on the edge class (onsite, shifted self,
  offsite) as well, mirroring the class-specific necks of the default head.
* Make :math:`c` a function of invariants of :math:`\bm e_{ij}` in addition to
  the species pair: a mixture of experts whose gating depends on the
  environment.
