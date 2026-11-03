Equivariant normalization: ``degree_rms_norm``
==============================================

Role
----

Normalization keeps feature magnitudes in a range where nonlinearities and
optimizers work well, decoupling the scale of each layer's output from the
scale of its input. As with nonlinearities, the statistics used for
normalizing must be invariant, and means may only be subtracted from even
scalars.

Built-in members of this kind: ``layer_norm`` (equivariant layer
normalization, ``e3layernorm`` in the flat configuration), ``graph_irrep_norm``
(per-graph normalization as in DeepH-E3), and the per-copy
``EquivariantRMSNorm`` used by ``e3mlp_pre_norm``.

The new layer
-------------

``degree_rms_norm`` normalizes each *irrep term* — all :math:`m` copies of
one :math:`(\ell,p)` — of every entity by its own root-mean-square value:

.. math::

   \bm y_c=\gamma_c\,\frac{\bm x_c}{\sqrt{\frac{1}{m(2\ell+1)}\sum_{c'\in\text{term}}\|\bm x_{c'}\|^2+\epsilon}}\;(+\beta_c\ \text{for }\texttt{0e}).

It sits between the existing variants: per-copy RMS normalization fixes the
norm of every copy individually and erases the relative strength of copies,
while graph-level normalization couples atoms of the same structure. Per-term
normalization preserves the *relative* magnitudes of copies within a degree
and treats each atom independently, so predictions do not depend on which
other structures share a batch.

**Symmetry.** :math:`\sum_{c'}\|\bm x_{c'}\|^2` is invariant, so dividing by
it is multiplication by an invariant (checklist 6). The bias is restricted to
even scalars (checklist 4).

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: DegreeRMSNorm

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: expand_per_copy

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _degree_rms_norm

Usage
-----

.. code-block:: yaml

   - id: norm_1
     transform: degree_rms_norm
     inputs: [mixed_1]
     outputs: [normalized_1]
     options: {eps: 1.0e-6}

Variations
----------

* Normalize per degree :math:`\ell` across both parities, or across all
  non-scalar irreps jointly, to control how much relative scale is preserved.
* Track running statistics over the training set, as batch normalization does,
  and freeze them for inference.
