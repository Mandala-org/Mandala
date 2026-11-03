Equivariant MLP block: ``equivariant_glu``
==========================================

Role
----

Point-wise blocks transform the features of each entity without looking at
its neighbours: they mix channels, change the irrep layout and add
expressiveness between message-passing steps. In MANDALA they appear as the
pre/post maps and refinements of every update and as the necks and
projections of the readout (:class:`~net.common.E3MLP` and its variants).

Built-in members of this kind: the ``e3mlp`` stage with the variants
``basic``, ``normact``, ``gate``, ``gatemagnitudes``, ``film``,
``resnormact``, ``resgatemagnitudes`` and ``bilinear`` (optionally routed:
one copy per edge role or species pair), and ``irrep_linear``.

The new layer
-------------

Gated linear units (GLUs) are among the most effective feed-forward blocks in
sequence models: a value path is multiplied elementwise by a sigmoid gate
computed from the same input. The equivariant GLU keeps that structure but
lets the gate see only invariants and act on whole copies:

.. math::

   \bm y=\bm x+\big(W_v\bm x\big)\odot_{\text{copy}}\sigma\big(\mathrm{MLP}(\iota(\bm x))\big),

where :math:`W_v` is an equivariant linear map (``o3.Linear``),
:math:`\iota(\bm x)` are the per-copy invariants of :math:`\bm x` and the MLP
outputs one gate per *output* copy. The residual is used when input and
output irreps coincide.

The gate sees the invariants of *all* input copies, so a scalar channel can
open or close a vector channel, and the norm of an ``2e`` copy can modulate
a ``1o`` output — couplings that a norm-based activation of each copy alone
cannot express, at the cost of a single linear layer and a small MLP.

**Symmetry.** :math:`W_v` mixes only copies of equal irreps (checklist 3a);
gates are functions of invariants (2) applied to whole copies (3c).

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: EquivariantGLU

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _equivariant_glu

Usage
-----

.. code-block:: yaml

   - id: mlp_1
     transform: equivariant_glu
     inputs: [aggregated_1]
     outputs: [mixed_1]
     options: {irreps_out: 32x0e+16x1o+16x1e+8x2e+8x2o+8x3o+8x3e+4x4e}

``o3.Linear`` maps input irreps that are absent from the output to nothing
and output irreps that are absent from the input to zero; the factory
reports the requested ``irreps_out`` as the output contract.

Variations
----------

* Stack two GLUs with a hidden layout wider than the input, as in
  transformer feed-forward blocks.
* Add a bilinear term :math:`\mathrm{TP}(W_1\bm x,W_2\bm x)` to the value
  path for self-interaction between degrees (the ``bilinear`` E3MLP variant
  does this with a residual).
