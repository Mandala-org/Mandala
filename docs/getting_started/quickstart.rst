Quickstart
==========

This page takes the bundled eight-atom silicon calculation (OpenMX, PBE,
``Si7.0-s2p2d1`` basis: two *s*, two *p* and one *d* shell, 13 orbitals per
atom) from raw files to sparse operators, observables, irreducible
representations and a first model prediction.

Operators and observables
-------------------------

.. literalinclude:: ../../examples/quickstart/operators.py
   :language: python
   :lines: 8-

.. code-block:: bash

   PYTHONPATH=src:. python examples/quickstart/operators.py

.. code-block:: text

   atoms: 8 | species pairs: ['Si-Si']
   orbital irreps per species: {'Si': ['2x0e', '2x1o', '1x2e']}
   Si-Si blocks (edges, rows, cols): (1336, 13, 13)
   onsite blocks: 8 | total blocks: 1336
   blocks within 8 A: 792
   Tr(DS) = 15.999713 electrons | Tr(DH) = -135.7970 eV
   max |H - H^T| = 8.92e-11 Hartree
   Si-Si irreps: 9x0e+12x1o+5x1e+4x2o+9x2e+4x3o+1x3e+1x4e
   block -> irreps -> block error: 1.5e-08
   band structure: (60, 104) (k-points, bands)

A few things are worth noticing:

* A :class:`~data.block_matrix.BlockMatrix` stores one dense block per
  *directed periodic edge* :math:`(i, j, \bm L)` and groups them by ordered
  species pair (``"Si-Si"``). The 1336 blocks are all atom pairs whose
  orbitals overlap in the DFT calculation; 792 of them lie within 8 Å.
* The density matrix is stored per spin channel, so for this
  non-spin-polarized calculation :math:`\Tr(DS)=N_e/2=16` for 32 valence
  electrons.
* Each :math:`13\times13` block (169 numbers) is *exactly* the direct sum
  of the irreducible representations printed above. Their dimensions sum to
  169; the :doc:`operator theory <../theory/operators>` explains where they
  come from.
* The band structure is obtained from the generalized eigenvalue problem on
  the full overlap support; 104 bands are :math:`8\times13` orbitals.

A model on the same structure
-----------------------------

.. literalinclude:: ../../examples/quickstart/model.py
   :language: python
   :lines: 8-

.. code-block:: text

   parameters: 514156
   pipeline stages: 46
   composite blocks: ['edge_encoder', 'node_update_0', 'edge_update_0', 'node_update_1',
                      'edge_update_1', 'readout_hamiltonian', 'readout_overlap', 'readout_density']
   Si-Si irrep vectors: (376, 169)
   Si-Si blocks: (376, 13, 13)
   Tr(DS) of the untrained model: -6.163149356842041

The flat :class:`~net.common.Config` compiles into a
:class:`~net.pipeline.FeaturePipeline` of 46 stages grouped into the composite
blocks listed above: the edge encoder, two node/edge message-passing layers
and one readout per operator. Each block is a small graph of elementary
stages (convolution, aggregation, MLPs, activation, ...) that recipes can
replace one by one (:doc:`../architecture/pipeline`). ``model(x)``
returns :class:`~data.block_matrix.IrrepsBlockData` (irrep coefficients per
atom pair); :meth:`~net.e3gnn.E3GNN.predict_matrices` converts them into
physical :class:`~data.block_matrix.BlockMatrix` operators on the 376 edges of
the 6 Å graph. All three operators of an untrained model are, of course,
meaningless; :doc:`first_model` trains one.

Where to go next
----------------

* :doc:`first_model` trains, continues and evaluates a model from the command
  line.
* :doc:`../theory/symmetry` and :doc:`../theory/operators` develop the
  mathematics used above.
* :doc:`../extending/index` builds a complete network from new layers.
