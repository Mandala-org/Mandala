Edge encoder: ``radial_angular_encoder``
========================================

Role
----

The edge encoder produces the initial pair state :math:`\bm e_{ij}^{(0)}`.
Unlike nodes, edges have a direction, so their initial state can already
contain non-scalar irreps built from the spherical harmonics of
:math:`\hat{\bm r}_{ij}`.

Built-in members of this kind: MANDALA's ``edge_encoder`` block (an
``embedding`` of the pair type, a ``tensor_square`` of the harmonics, a
``concat`` with the radial basis and an ``fc_tensor_product`` of the two:
:math:`\mathrm{TP}(\mathrm{Emb}(t),\bm\varphi\oplus\mathrm{TS}(Y))`), and
``gaussian_basis`` used directly as a scalar edge state (DeepH-E3 preset).

The new layer
-------------

The radial–angular encoder gives every output copy :math:`(\ell,k)` its own
learned radial profile:

.. math::

   \bm e_{ij}^{(\ell,k)}=R_{\ell k}\big(\bm\varphi(r_{ij})\big)\;Y^{\ell}(\hat{\bm r}_{ij}),
   \qquad R=\mathrm{MLP}(\bm\varphi):\mathbb R^{n}\to\mathbb R^{\sum_\ell m_\ell}.

This is the most general equivariant function of a single displacement vector
that is linear in the angular basis: by Schur's lemma any equivariant map
from a vector to a copy of degree :math:`\ell` is
:math:`f(r)\,Y^\ell(\hat{\bm r})`. The encoder therefore spans exactly the
space of "two-body" edge features, with :math:`m_\ell` independent radial
functions per degree.

**Symmetry.** :math:`R_{\ell k}` is a function of invariants; multiplying the
irrep :math:`Y^\ell` by an invariant gives a copy of the same irrep
(checklist items 2 and 3c). Requested irreps must appear in the angular basis;
the factory rejects, e.g., ``1e`` from harmonics that only contain ``1o``.

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: RadialAngularEncoder

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _radial_angular_encoder

Usage
-----

.. code-block:: yaml

   - id: angular
     transform: spherical_harmonics
     inputs: [edge_vectors]
     outputs: [angular]
     options: {irreps: 1x0e+1x1o+1x2e+1x3o+1x4e}
   - id: edge_encoder
     role: encoder
     transform: radial_angular_encoder
     inputs: [radial, angular]
     outputs: [edge_0]
     options: {irreps_out: 16x0e+8x1o+8x2e+4x3o+4x4e}

Variations
----------

* Condition the radial MLP on the species pair (concatenate a pair-type
  embedding to :math:`\bm\varphi`), giving each bond type its own radial
  profiles.
* Append the tensor square of the harmonics to reach ``1e``, ``2o``… directly
  in the encoder.
