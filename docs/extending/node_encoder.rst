Node encoder: ``periodic_table_embedding``
==========================================

Role
----

The node encoder produces the initial state :math:`\bm h_i^{(0)}` of every
atom from its chemical species. Because the species is a label, the initial
state is invariant (``0e`` channels only); directional information enters
later through edges.

Built-in members of this kind: ``embedding`` (one learned vector per species,
the MANDALA default) and ``one_hot`` followed by ``irrep_linear`` (DeepH-E3,
MACE-H presets).

The new layer
-------------

A plain embedding table learns every element independently. Chemistry is
not independent: elements in the same group share valence configurations and
elements in the same period share core size. The periodic-table embedding
factorizes the initial state accordingly,

.. math::

   \bm h_i^{(0)}=\bm E_{\text{species}}[z_i]+\bm E_{\text{period}}\big[\mathrm{period}(z_i)\big]
   +\bm E_{\text{group}}\big[\mathrm{group}(z_i)\big]\ \in\ \mathbb R^{d}\ (d\times\texttt{0e}).

In a multicomponent material such as ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2`,
S and Se share the group-16 vector and Zn and Cu share the period-4 vector,
so information learned from one element transfers to its neighbours in the
table. The species-specific table keeps the full flexibility of a plain
embedding.

The species list is resolved from the training data and passed through the
recipe as ``elements: ${species.order}``, so the layer always agrees with the
index convention of ``node_type_idx``.

**Symmetry.** The output depends only on species labels: it is invariant
under every element of E(3) and equivariant under permutations of atoms.

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: period_and_group

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: PeriodicTableEmbedding

The lookup tables of period and group are buffers: they move with the model
between devices and are stored in checkpoints, but are not trained.

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _periodic_table_embedding

Usage
-----

.. code-block:: yaml

   - id: node_encoder
     role: encoder
     transform: periodic_table_embedding
     inputs: [node_type_idx]
     outputs: [node_0]
     options: {elements: '${species.order}', width: 32}

Variations
----------

* Add embeddings of tabulated invariant descriptors (electronegativity,
  covalent radius, valence-electron count) through a small MLP.
* Encode the *basis set* of each species (its shell list
  ``${species.shells}``), which distinguishes the same element computed with
  different orbital sets.
