Equivariant nonlinearity: ``squash_activation``
===============================================

Role
----

Without nonlinearities a stack of equivariant layers collapses to a
polynomial of fixed degree in the input. Elementwise activations, however,
break equivariance for non-scalar irreps: :math:`\tanh` applied to each
component of a vector does not commute with rotations. Equivariant
nonlinearities therefore act on vectors only through their *norms*, through
invariant gates, or through functions on the sphere.

Built-in members of this kind: ``normact``, ``gate_magnitudes``,
``gate_scalars_mlp`` and ``s2act`` (the ``irrep_nonlinearity`` stage), the
e3nn gate inside ``equivariant_convolution``, and ``irrep_activation``.

The new layer
-------------

The *squash* function of capsule networks rescales each copy of an irrep
according to its own norm:

.. math::

   \bm x_c\ \longmapsto\ \frac{\|\bm x_c\|}{\kappa_c^2+\|\bm x_c\|^2}\,\bm x_c,
   \qquad \kappa_c=\operatorname{softplus}(\theta_c)+10^{-3}.

The output norm :math:`\|\bm x_c\|^2/(\kappa_c^2+\|\bm x_c\|^2)` grows
quadratically for small inputs and saturates at one for large ones: the
layer suppresses weak, noisy directions and caps large ones, with a learned
knee :math:`\kappa_c` per copy. Even scalars, which admit any pointwise
function, use SiLU.

**Symmetry.**

* *Rotations.* The scale factor depends on :math:`\|\bm x_c\|`, which is
  invariant, so the map multiplies each copy by an invariant (checklist 3c).
* *Inversion.* An odd-parity copy changes sign, :math:`\bm x_c\to-\bm x_c`.
  The squash is an odd function of :math:`\bm x_c`, so its output changes
  sign too. This makes the same formula valid for pseudoscalars ``0o``,
  where a function like SiLU would be wrong (checklist 5).

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: SquashActivation

``zero_safe_norm`` returns zero with a zero gradient at the origin, so the
layer has well-defined gradients even for vanishing copies.

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _squash_activation

Usage
-----

.. code-block:: yaml

   - id: activation_1
     transform: squash_activation
     inputs: [normalized_1]
     outputs: [node_1]

Variations
----------

* Use a gate that depends on *all* invariants of the entity (as
  ``gate_scalars_mlp`` does) instead of each copy's own norm, which lets
  scalars switch vectors on and off.
* Replace the rational squash by :math:`\tanh(\|\bm x\|/\kappa)\,\bm x/\|\bm x\|`
  for a bounded, norm-preserving-at-small-scale variant.
