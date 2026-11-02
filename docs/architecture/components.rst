Component catalogue
===================

Built-in transformations grouped by the role they play in an E3GNN. Inputs
are listed in positional order. All transformations are ordinary
``torch.nn.Module`` instances and can be used outside a pipeline.

Geometry and embeddings
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Transformation
     - Inputs
     - Output and key options
   * - ``periodic_displacement``
     - positions, box, edge index, shifts, node batch
     - Differentiable :math:`\bm r_{ij}^{\bm L}` on a fixed support.
   * - ``vector_lengths``
     - vectors
     - :math:`\|\bm r\|`.
   * - ``gaussian_basis``
     - edge lengths
     - ``n_gaussians`` Gaussians on [``start``, ``stop``]; ``trainable``.
   * - ``cutoff_radial_expansion``
     - edge lengths
     - :math:`\sqrt{2/r_c}\,\sin(n\pi r/r_c)/r` times a polynomial envelope
       (``r_max``, ``num_basis``, ``cutoff_order``).
   * - ``spherical_harmonics``
     - edge vectors
     - :math:`Y^\ell(\hat{\bm r})` for ``irreps``; ``normalization``.
   * - ``bond_frames``, ``bond_harmonics``
     - edge index, vectors (+ frames, neighbourhood)
     - Local orthonormal bond frames and harmonics of neighbours expressed in them.
   * - ``embedding``, ``one_hot``
     - species index
     - Learned embedding (``num_embeddings``, ``width``) or one-hot features.
   * - ``species_irrep_embedding``
     - node species index
     - MANDALA node encoder, :math:`h^0=\mathrm{Emb}(z)`.
   * - ``pair_radial_angular_embedding``
     - pair index, radial, angular
     - MANDALA rich edge encoder,
       :math:`\mathrm{TP}(\mathrm{Emb}(t),\varphi\oplus\mathrm{TS}(Y))`.

Message passing
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Transformation
     - Inputs
     - Output and key options
   * - ``endpoint_irrep_node``, ``endpoint_irrep_edge``
     - node, edge, edge index, harmonics, radial, one-hot, pair index
     - MANDALA node/edge update blocks (:doc:`../theory/architecture`).
   * - ``endpoint_tensor_product_node``, ``endpoint_tensor_product_edge``
     - node, species, angular, edge, radial, edge index, node batch
     - DeepH-E3 style updates: gated tensor-product convolution,
       species self-connection, graph normalization.
   * - ``endpoint_gather``
     - node, edge, edge index
     - :math:`\bm h_i\oplus\bm h_j\oplus\bm e_{ij}`.
   * - ``radial_tensor_product``
     - features, angular, radial, edge batch
     - Factorized tensor product, gate, radial channel scaling
       (``irreps_out``, ``radial_hidden``, ``nonlin``).
   * - ``indexed_aggregation``
     - messages, edge index, node reference
     - ``sum`` or ``mean`` onto ``receiver`` 0 or 1.
   * - ``projected_attention``
     - messages, nodes, radial, edge index
     - MANDALA invariant multi-head attention reducer (``scalar_dim``, ``heads``).
   * - ``endpoint_message_aggregation``
     - as the node update
     - MANDALA attention node update as one stage.
   * - ``neighbor_density``, ``symmetric_product``
     - node attributes and features, edge attributes, radial, edge index
     - MACE-style atomic basis :math:`A_i=\sum_j R(r_{ij})Y(\hat{\bm r}_{ij})\otimes h_j`
       and its learned symmetric powers up to ``correlation``.
   * - ``angular_degree_expansion``
     - hidden stream, angular stream, species
     - Couples a node stream with a separately carried angular stream.
   * - ``scalar_gated_node``, ``endpoint_scalar_edge``, ``local_coordinate_message``
     - invariant node/edge features, bond neighbourhood relations
     - DeepH-style invariant and local-coordinate messages.
   * - ``gated_scalar_message``
     - invariant features
     - Gated MLP on scalars (``out_features``).

Feature operations
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Transformation
     - Inputs
     - Output and key options
   * - ``irrep_linear``
     - irreps
     - Equivariant linear map to ``irreps_out`` (``biases``).
   * - ``scalar_linear``, ``channel_linear``, ``bond_local_linear``
     - invariant / channel / bond-frame features
     - Dense maps where no rotation acts.
   * - ``irrep_activation``
     - irreps
     - ``kind: norm`` (NormActivation) or ``kind: gate``.
   * - ``graph_irrep_norm``
     - irreps, entity batch
     - Per-graph equivariant normalization.
   * - ``species_tensor_product``
     - irreps, species attributes
     - Species-conditioned self-connection.
   * - ``irrep_residual``, ``add``, ``concat``
     - two or more feature slots
     - Residual (with projection when layouts differ), sum, direct sum.
   * - ``gaussian_envelope``, ``scale_features``, ``pair_distance_affine``
     - lengths / values and invariant weights
     - Invariant edge weights and their application.

Frames and adapters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Transformation
     - Inputs
     - Output and key options
   * - ``irrep_cartesian_basis``
     - irreps
     - Re-express non-scalar irreps between Cartesian axis orders (``target``).
   * - ``irrep_to_bond``
     - irreps, edge index, bond frames
     - Rotate global irreps into the local bond frame (``endpoint``).

Readout and operator decoding
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Transformation
     - Inputs
     - Output and key options
   * - ``partitioned_irrep_readout``
     - node, edge, pair edges, edge partitions
     - MANDALA :class:`~net.heads.DeepHead` producing per-pair irrep vectors.
   * - ``partitioned_blocks_to_shells``
     - per-pair vectors
     - Repacks per-pair vectors into the shell codec layout.
   * - ``species_irrep_affine``
     - coupled irreps, pair index
     - Train-split fitted per-pair, per-irrep affine statistics.
   * - ``shell_tensor_decode``
     - coupled irreps
     - Clebsch–Gordan decoding of ``${layout.coupled_irreps}`` into native
       packed blocks (``shell_pairs``, ``spinful``).
   * - ``local_orbital_output``
     - bond-frame components
     - Bond-frame orbital entries, including spin–orbit components.

Reference presets
-----------------

``configs/reference`` composes these stages into published architectures and
``configs/hybrids`` mixes them:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Composition
   * - ``reference/mandala_paper.yaml``
     - Flat MANDALA configuration of the SiO\ :sub:`2` paper model.
   * - ``reference/deeph.yaml``
     - Invariant DeepH messages in local bond coordinates.
   * - ``reference/deeph_e3.yaml``
     - DeepH-E3 endpoint tensor-product updates and linear readout.
   * - ``reference/mace_h.yaml``
     - MACE-H neighbour densities, symmetric products and angular expansion.
   * - ``hybrids/tensor_messages_projected_attention.yaml``
     - Factorized radial tensor-product messages with MANDALA attention.
   * - ``hybrids/symmetric_products_attention_pair_readout.yaml``
     - MACE-style symmetric products with attention and a pair readout.
   * - ``hybrids/tensor_products_degree_expansion_shell_readout.yaml``
     - Tensor products, angular degree expansion and a shell readout.
   * - ``hybrids/equivariant_nodes_bond_frame_readout.yaml``
     - Equivariant node features read out in local bond frames.
