Data
====

Supported sources
-----------------

One loader serves every workflow. A *snapshot* is a single geometry together
with its operators; it can come from any of the following sources.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Source
     - Files
     - Entry point
   * - OpenMX
     - ``HS.out`` (H, S, D text dump) and the OpenMX output file (``*.out``)
     - :meth:`Snapshot.from_openmx <data.snapshot.Snapshot.from_openmx>`
   * - Periodic HDF5
     - ``element.dat``, ``orbital_types.dat``, ``site_positions.dat``,
       ``lat.dat``, ``info.json``, ``hamiltonians.h5`` (and ``overlaps.h5``,
       ``density_matrixs.h5``)
     - :meth:`Snapshot.from_deeph_e3 <data.snapshot.Snapshot.from_deeph_e3>`
   * - FHI-aims
     - ``geometry.in``, ``basis-indices.out``, ``H/S/D_spin_01_kpt_000001.csc``
     - :meth:`Snapshot.from_fhiaims <data.snapshot.Snapshot.from_fhiaims>`
   * - PySCF
     - ``*.npz`` with real-space or k-space matrices and a JSON description
     - :meth:`Snapshot.from_pyscf <data.snapshot.Snapshot.from_pyscf>`,
       :meth:`~data.snapshot.Snapshot.from_pyscf_kspace`

For training, a snapshot is referenced by its **directory**; the loader
identifies the format from its contents (``HS.out`` → OpenMX,
``hamiltonians.h5`` + ``info.json`` → periodic HDF5, a single ``*.npz`` →
PySCF).

The periodic HDF5 interchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The HDF5 interchange stores positions as a :math:`3\times N` array, lattice
vectors as columns, lengths in Å and Hamiltonians in eV. Each matrix file
holds one dataset per block under the key ``[Rx, Ry, Rz, i, j]`` with
one-based atom indices. Spin–orbit coupled data are complex, spin-major full
matrices. ``info.json`` (or ``data.source_basis`` and
``data.source_cartesian_axes``) declares the orbital convention of the files
(``openmx`` or ``e3nn`` order, ``xyz`` or ``yzx`` axes) so that they are
normalized exactly once. OpenMX calculations convert with

.. code-block:: bash

   PYTHONPATH=src:. python scripts/convert_openmx_to_deeph_e3.py --help

The Snapshot
------------

:class:`~data.snapshot.Snapshot` bundles the three operators with positions,
lattice and scalar observables:

.. code-block:: python

   from data.snapshot import Snapshot
   from net.common import Config

   snap = Snapshot.from_openmx("calc/HS.out", "calc/Si.out", convention="e3nn",
                               cutoff_radius=8.0, cfg=Config(verbosity=0))

   snap.hamiltonian, snap.overlap, snap.density   # BlockMatrix
   snap.positions, snap.box                       # Å, lattice vectors as rows
   snap.get_energy()                              # Tr(DH)
   snap.get_number_of_electrons()                 # Tr(DS)

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Method
     - Purpose
   * - ``to_e3nn()``, ``to_openmx()``, ``to_fhiaims()``, ``to_pyscf()``
     - Change the orbital convention of all matrices.
   * - ``rotate(R)``
     - Rotate positions, cell and every operator block consistently.
   * - ``filter_by_distance(r)``
     - Keep blocks with :math:`\|\bm r_{ij}^{\bm L}\|\le r`.
   * - ``symmetrize_matrices()``
     - Hermitian projection of the selected operators.
   * - ``reduce_orbitals("1s1p")``
     - Restrict every species (or selected ones) to a subset of shells.
   * - ``to_k_space(...)``, ``get_band_structure(path=..., npoints=...)``
     - Bloch sums and generalized eigenvalues along a band path.
   * - ``dos(sigma=..., bin_width=..., E_min=..., E_max=...)``
     - Gaussian-broadened density of states.
   * - ``save(path)``, ``Snapshot.load(path)``
     - Binary round trip of the complete snapshot.
   * - ``export_to_deephe3(path)``
     - Write the periodic HDF5 interchange.

Block matrices and irrep vectors
--------------------------------

:class:`~data.block_matrix.BlockMatrix` is the sparse operator type:

.. code-block:: python

   H = snap.hamiltonian
   H.keys()                    # ['Si-Si'] — ordered species pairs
   H["Si-Si"]                  # (edges, 13, 13) dense blocks
   H.edges("Si-Si")            # (5, edges): Lx, Ly, Lz, i, j
   H.diag(), H.offdiag()       # onsite / all other blocks
   H.adjoint()                 # (H^L_ij)^† placed on (j, i, -L)
   (H - H.adjoint())           # arithmetic aligns edge orders automatically
   H.to_dense()                # supercell matrix over all stored shifts
   H.rotate(R), H.transpose()

:class:`~data.block_matrix.IrrepsBlockData` holds the same operator as irrep
vectors and is what the network outputs:

.. code-block:: python

   from core.block_irrep_mapper import BlockIrrepMapper
   mapper = BlockIrrepMapper(snap.density.orbital_cfg)
   vectors = H.to_vectors(mapper)      # IrrepsBlockData, vectors["Si-Si"]: (edges, 169)
   blocks = vectors.to_blocks(mapper)  # BlockMatrix again

Datasets
--------

:class:`~data.factory.DatasetFactory` registers snapshots per split and
creates :class:`~data.gnn_dataset.E3GNNDataset` objects that share one
orbital vocabulary and one mapper:

.. code-block:: python

   from data.factory import DatasetFactory

   factory = DatasetFactory(cfg, convention="e3nn")
   for path in train_dirs:
       factory.add_snapshot(path, purpose="train")
   for path in val_dirs:
       factory.add_snapshot(path, purpose="val")
   train, val, mapper = factory.create()
   x, y = train[0]          # graph inputs and targets of one structure

Each item is a :class:`~data.sample.GraphSample` that unpacks into inputs
``x`` (graph topology, geometry, species, edge partitions) and targets ``y``
(operators, energy, electron count). The collator batches several structures
into one disjoint graph union and separates predictions again before
observables are evaluated, so every structure keeps its own cell and
reference values.

Dataset construction enforces the scientific contract of a split:

* the species and shell vocabulary comes from the training split only, or is
  declared in advance with ``species.shells``; held-out structures with
  unknown species or different orbital layouts are rejected;
* splits must be disjoint;
* graph edges and target blocks must agree exactly (``require_exact_edge_match``);
* missing or non-finite target entries are masked from losses and statistics.

Graph supports
~~~~~~~~~~~~~~

The **output support** is the set of edges on which operators are predicted:
the stored DFT blocks (``graph_source: stored``) or all pairs within
``graph_radius`` (``graph_source: radius``). Further supports can be
declared for message passing and local neighbourhoods:

.. code-block:: yaml

   data:
     supports:
       message: {kind: subset, source: output, exclude_self: true, cutoff: 7.2}
       local:   {kind: radius, cutoff: 5.0}
     neighborhood_support: local

``kind: subset`` selects edges of another support; ``kind: radius`` builds an
independent periodic graph whose map onto output edges marks absent edges
with ``-1``. Supports are prepared per structure before batching, and the
batcher offsets every relation by the entity set it references.

Caching
~~~~~~~

Parsing large text dumps dominates data preparation. ``snapshot_cache_dir``
stores parsed snapshots; preprocessed graph samples are cached alongside and
invalidated automatically when the source files or the configuration that
produced them change. ``dataset_device: cuda`` keeps a prepared dataset on
the GPU for the whole run.
