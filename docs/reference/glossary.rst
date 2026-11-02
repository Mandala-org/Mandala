Glossary
========

.. glossary::

   Block
      The dense matrix :math:`X_{ij}^{\bm L}` of an operator between the
      orbitals of atom :math:`i` and atom :math:`j` translated by
      :math:`\bm L`.

   Copy
      One instance of an irrep inside a feature; ``16x1o`` has sixteen copies
      of the vector irrep. Invariant scalings act on whole copies.

   Coupled irreps
      The irreps of the packed edge vector covering every species pair,
      ``${layout.coupled_irreps}``; the readout target of recipes.

   Directed periodic edge
      The triple :math:`(i,j,\bm L)`. Both :math:`(i,j,\bm L)` and
      :math:`(j,i,-\bm L)` are stored and predicted.

   Edge class
      Onsite (:math:`i=j,\bm L=0`), shifted self (:math:`i=j,\bm L\ne0`) or
      offsite (:math:`i\ne j`).

   Envelope
      An invariant radial factor :math:`E_p(r)` multiplying predicted blocks of
      species pair :math:`p`.

   Hermitian projection
      :math:`X_{ij}^{\bm L}\leftarrow\tfrac12(X_{ij}^{\bm L}+(X_{ji}^{-\bm L})^\dagger)`.

   Irrep
      Irreducible representation :math:`(\ell,p)` of O(3), written ``"2e"``,
      ``"1o"``…

   Mapper
      :class:`~core.block_irrep_mapper.BlockIrrepMapper`, the Clebsch–Gordan
      change of basis between blocks and irrep vectors for every species pair.

   Observable guidance
      Training terms on quantities derived from predicted operators:
      :math:`\Tr(DH)`, :math:`\Tr(DS)` and band eigenvalues.

   Pipeline
      The ordered, typed list of stages of an E3GNN
      (:class:`~net.pipeline.FeaturePipeline`).

   Recipe
      A YAML configuration with an explicit ``pipeline`` section.

   Slot
      A named feature in a pipeline; written once by one stage.

   Snapshot
      One geometry with its H, S, D operators and metadata
      (:class:`~data.snapshot.Snapshot`).

   Stage
      One transformation in a pipeline: ``id``, ``transform``, ``inputs``,
      ``outputs``, ``options``.

   Support
      The index set of an edge-like feature (``output``, ``message``, …).
      Features on different supports cannot be combined.
