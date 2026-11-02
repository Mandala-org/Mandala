Output envelope: ``learned_pair_envelope``
==========================================

Role
----

Matrix elements of localized orbitals decay roughly exponentially with
distance and span several orders of magnitude within one cutoff sphere. An
envelope multiplies the network output by a known radial profile so that the
network only has to learn a slowly varying, order-one correction.

Built-in members of this kind: the fitted Slater envelopes
(``hamiltonian_envelope_mode``, ``hamiltonian_envelope_path``) of the flat
network and ``gaussian_envelope`` with ``scale_features`` in recipes.

The new layer
-------------

Instead of fitting the envelope beforehand, ``learned_pair_envelope`` makes
the Slater-type profile part of the model:

.. math::

   \bm x_{ij}\longmapsto A_p\,(1+r_{ij})^{\nu_p}\,e^{-\beta_p r_{ij}}\;\bm x_{ij},
   \qquad p=(a_i,a_j),

with three parameters per ordered species pair, stored as
:math:`\log A_p`, :math:`\nu_p` and :math:`\beta_p=\operatorname{softplus}(\theta_p)\ge0`.
At initialization :math:`A=1`, :math:`\nu=0` and :math:`\beta\approx0`, so
the layer starts as the identity and learns the decay jointly with the rest
of the network. In log space the envelope is linear in its parameters,

.. math::

   \log E_p(r)=\log A_p+\nu_p\log(1+r)-\beta_p r,

which keeps its gradients well conditioned. The paper's ablation found a 3.8 %
lower median test Hamiltonian MAE from a fitted envelope; a learned envelope
removes the separate fitting step and adapts the profile to the objective.

**Symmetry.** One invariant factor per edge multiplies every component
(checklist 3c); equivariance is unaffected whatever the parameters.

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: LearnedPairEnvelope

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _learned_pair_envelope

Usage
-----

.. code-block:: yaml

   - id: envelope
     role: readout
     transform: learned_pair_envelope
     inputs: [coupled, edge_lengths, edge_type_idx]
     outputs: [coupled_scaled]
     options: {num_pairs: '${species.pair_count}'}

Variations
----------

* Initialize :math:`(A,\nu,\beta)` from a fit to the training targets, as the
  fitted envelopes do, and fine-tune them.
* Use separate envelopes per irrep degree :math:`L`: higher-:math:`L`
  couplings involve more anisotropic orbitals and often decay differently.
