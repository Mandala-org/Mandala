Edge update: ``endpoint_bilinear_edge_update``
==============================================

Role
----

Operators live on edges: the off-diagonal block :math:`H_{ij}^{\bm L}`
depends on both atoms *and* on the bond between them. Edge updates bring the
latest node information into the pair state that the readout consumes.

Built-in members of this kind: ``endpoint_irrep_edge`` (MANDALA: convolution
of :math:`\bm h_i\oplus\bm h_j\oplus\bm e_{ij}` with the bond harmonics,
``edge_update_node_combine``), ``endpoint_tensor_product_edge`` (DeepH-E3)
and ``endpoint_scalar_edge`` (DeepH).

The new layer
-------------

The bilinear edge update couples the two endpoint states with a fully
connected tensor product and adds a linear map of the previous edge state:

.. math::

   \bm e'_{ij}=W_{\text{skip}}\,\bm e_{ij}+\mathrm{TP}_{\text{fc}}\big(\bm h_i,\bm h_j\big).

A concatenation :math:`\bm h_i\oplus\bm h_j` followed by linear maps can only
produce irreps already present in the node states. The product
:math:`\bm h_i\otimes\bm h_j` creates new ones — for instance
``1o ⊗ 1o → 0e + 1e + 2e`` and ``1o ⊗ 2e → 1o + 2o + 3o`` — and these are
exactly the parities needed for *p*–*p* and *p*–*d* blocks
(:doc:`../theory/operators`). The output layout may therefore be richer than
the input edge layout, which is what allows the readout below to reach every
coupled irrep of the operator.

The product is not symmetric in :math:`i` and :math:`j`: the edges
:math:`(i,j,\bm L)` and :math:`(j,i,-\bm L)` receive different states, as the
directed blocks they predict are different objects (related by
Hermiticity only after projection).

**Symmetry.** A Clebsch–Gordan product of two equivariant inputs
(checklist 3b) plus an equivariant linear map (3a).

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: EndpointBilinearEdgeUpdate

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _endpoint_bilinear_edge_update

Usage
-----

.. code-block:: yaml

   - id: edge_update
     role: message
     transform: endpoint_bilinear_edge_update
     inputs: [node_2, edge_0, edge_index]
     outputs: [edge_1]
     options: {irreps_out: 16x0e+8x1o+8x1e+8x2e+8x2o+4x3o+4x3e+4x4e}

Variations
----------

* Weight the product by a radial function of :math:`r_{ij}` so that distant
  pairs couple more weakly.
* Symmetrize the product, :math:`\tfrac12[\mathrm{TP}(\bm h_i,\bm h_j)+\mathrm{TP}(\bm h_j,\bm h_i)]`,
  and add an antisymmetric part separately, which makes the Hermitian and
  anti-Hermitian components of the predicted blocks explicit.
