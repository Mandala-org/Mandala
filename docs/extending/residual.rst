Residual update: ``rezero_residual``
====================================

Role
----

Residual connections combine a layer's update with its input,
:math:`\bm h'=\bm h+\Delta\bm h`. They make deep networks trainable and let a
layer learn a correction rather than a replacement. In a pipeline the old and
new states are separate slots, so the residual is an explicit stage.

Built-in members of this kind: ``node_update_residual`` and
``edge_update_residual`` of the MANDALA blocks, the scaled residuals of the
``res*`` E3MLP variants, ``irrep_residual`` (with a projection when layouts
differ) and ``add``.

The new layer
-------------

ReZero multiplies the update by a learned scalar initialized at zero. The
equivariant version learns one coefficient per irrep copy:

.. math::

   \bm h'_c=\bm h_c+\alpha_c\,\Delta\bm h_c,\qquad \alpha_c\big|_{\text{init}}=0 .

At initialization every ReZero block is the identity, so a network of any
depth starts as a shallow one and the signal propagates undistorted; each
copy then learns how much of the update to admit. Per-copy coefficients let
the network open scalar channels early and higher-degree channels later.

**Symmetry.** :math:`\alpha_c` multiplies whole copies (checklist 3c) and both
inputs must have identical layouts, which the factory enforces.

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: ReZeroResidual

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _rezero_residual

Usage
-----

.. code-block:: yaml

   - id: residual_2
     transform: rezero_residual
     inputs: [node_1, update_2]
     outputs: [node_2]
     options: {initial: 0.0}

Because slots are write-once, ``node_1`` remains available after this stage;
long skip connections across several layers are written the same way.

Variations
----------

* Make :math:`\alpha` a function of invariants of :math:`\bm h` (a highway
  gate) so the mixing adapts to the local environment.
* Initialize at a small positive value (LayerScale) instead of zero when the
  update is already well scaled.
