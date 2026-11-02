Tensor-product convolution: ``depthwise_convolution``
=====================================================

Role
----

The convolution creates messages: it couples the state of a neighbour with
the geometry of the bond through a Clebsch–Gordan tensor product, weighted by
a learned function of distance. It is the only place where angular momentum
is *created* from geometry during message passing, and it dominates the cost
of the network.

Built-in members of this kind: :class:`~net.layers.EquiConv` inside the
MANDALA update blocks (separate-weight or fully connected product, gate,
radial channel weights), ``radial_tensor_product`` (factorized product, gate,
radial scaling) and the neighbour-density product of ``neighbor_density``.

The new layer
-------------

The depthwise ("uvu") convolution couples every channel :math:`u` of the
sender with the spherical harmonics, keeping channels separate, and
generates *one weight per channel and per coupling path* from the distance:

.. math::

   \big(\bm m_{ij}\big)^{L}_{u}=\sum_{\ell_1,\ell_2}
   w_{u}^{(\ell_1\ell_2\to L)}\!\big(\bm\varphi(r_{ij})\big)\,
   \Big(\bm h_{i,u}^{\ell_1}\otimes Y^{\ell_2}(\hat{\bm r}_{ij})\Big)^{L},
   \qquad L\le L_{\max}.

This is the continuous-filter convolution of NequIP and MACE. Compared with
the MANDALA default it trades the dense channel mixing of the separate-weight
product for a fully distance-dependent filter on every path, at a cost that
grows only linearly in the multiplicity. Channel mixing is then delegated to
a cheap equivariant linear layer (here: :doc:`mlp_block`).

The output irreps are assembled from all reachable paths and sorted; the
instruction list is re-indexed with the sorting permutation so that each path
writes into its sorted slot.

**Symmetry.** The weights are functions of :math:`r` (invariant), and the
coupling is a Clebsch–Gordan product (checklist 3b). Sender features are
gathered with ``edge_index[0]``; messages live on the edge support.

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: DepthwiseConvolution

``internal_weights=False, shared_weights=False`` makes e3nn accept one weight
vector per edge; ``self.tp.weight_numel`` is the number of weights the radial
network must emit.

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _depthwise_convolution

Usage
-----

.. code-block:: yaml

   - id: convolution_1
     role: message
     transform: depthwise_convolution
     inputs: [node_0, angular, radial, edge_index]
     outputs: [messages_1]
     options: {lmax: 4}

With ``node_0 = 32x0e`` and harmonics up to :math:`\ell=4` this produces
``32x0e+32x1o+32x2e+32x3o+32x4e``; in the second layer, with a node state
containing both parities, all parities up to :math:`L_{\max}` appear.

Variations
----------

* Feed the receiver state into the radial network as well, making the filter
  depend on the chemical environment of both atoms.
* Use the message support (``message_edge_index``, ``message_vectors``,
  ``message_lengths``) to pass messages on a different, e.g. shorter-range,
  graph than the one on which operators are predicted.
