Message aggregation: ``tempered_softmax_aggregation``
=====================================================

Role
----

Aggregation reduces the messages arriving at an atom into one equivariant
vector. It must be invariant to the order of neighbours and must handle a
variable number of them. The paper's silicon ablation shows that the choice
matters: replacing averaging by attention lowered the validation Hamiltonian
MAE by 10 %.

Built-in members of this kind: ``sum``, ``average`` and multi-head
``attention`` in the MANDALA node update (``node_update_message_agg``),
``indexed_aggregation`` and ``projected_attention`` as separate stages.

The new layer
-------------

Tempered softmax aggregation computes one invariant score per message and
normalizes it over the neighbourhood with a *learned temperature*:

.. math::

   s_{ij}=\mathrm{MLP}\big(\bm\varphi(r_{ij})\oplus\iota(\bm m_{ij})\big),\qquad
   \alpha_{ij}=\frac{\exp(s_{ij}/\tau)}{\sum_{k\in\mathcal N(j)}\exp(s_{kj}/\tau)},\qquad
   \bm h_j=\sum_{i\in\mathcal N(j)}\alpha_{ij}\,\bm m_{ij},

where :math:`\iota(\bm m)` collects one invariant per message copy (the value
of ``0e`` copies, the square of ``0o`` copies and the norm of all others).
The temperature interpolates between familiar reductions:

.. math::

   \tau\to\infty:\ \alpha_{ij}\to\frac1{|\mathcal N(j)|}\ \ (\text{mean}),\qquad
   \tau\to0:\ \alpha_{ij}\to\delta_{i,\arg\max_k s_{kj}}\ \ (\text{hard selection}).

Learning :math:`\log\tau` lets each layer choose how selective it is: an
early layer may average over a coordination shell, a later one may focus on
the single most relevant bond.

The maximum score per receiver is subtracted before exponentiation, which
keeps the softmax numerically stable for any number of neighbours.

**Symmetry.** Scores are functions of invariants only (checklist 2), so each
:math:`\alpha_{ij}` is invariant; a sum of equivariant messages weighted by
invariants is equivariant (3c, 3d). The softmax runs over the incoming edges
of each receiver, so it is also permutation invariant.

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: copy_invariants

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: TemperedSoftmaxAggregation

The fourth input, ``node_reference``, is any node feature; it fixes the
number of receivers (atoms without incoming edges receive zeros) and the
support of the output.

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _tempered_softmax_aggregation

Usage
-----

.. code-block:: yaml

   - id: aggregation_1
     role: message
     transform: tempered_softmax_aggregation
     inputs: [messages_1, radial, edge_index, node_0]
     outputs: [aggregated_1]
     options: {hidden: 32, temperature: 1.0}

Variations
----------

* Multiple heads with separate temperatures, concatenated or averaged.
* Add the receiver's own invariants to the score (a query, as in the built-in
  attention) to make the selection depend on the atom being updated.
