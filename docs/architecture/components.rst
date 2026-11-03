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
   * - ``soft_one_hot_basis``
     - edge lengths
     - e3nn radial basis (``number``, ``end``, ``basis``: gaussian, cosine,
       smooth_finite, fourier, bessel; ``cutoff``; ``scale``). MANDALA's radial basis.
   * - ``spherical_harmonics``
     - edge vectors
     - :math:`Y^\ell(\hat{\bm r})` for ``irreps``; ``normalization``;
       ``permutation`` evaluates in permuted axes and labels the output basis.
   * - ``edge_roles``
     - edge index, shifts
     - Route index: onsite, shifted self-image (``separate_shifted_self``), off-site.
   * - ``bond_frames``, ``bond_harmonics``
     - edge index, vectors (+ frames, neighbourhood)
     - Local orthonormal bond frames and harmonics of neighbours expressed in them.
   * - ``embedding``, ``one_hot``
     - species index
     - Learned embedding (``num_embeddings``, ``width``) or one-hot features.

Message passing
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Transformation
     - Inputs
     - Output and key options
   * - ``endpoint_concat``
     - node, edge, edge index
     - :math:`[\bm h_i,\bm h_j,\bm e_{ij}]`, :math:`[\bm h_i+\bm h_j,\bm e_{ij}]` or
       :math:`[\mathrm{TP}(\bm h_i,\bm h_j),\bm e_{ij}]` (``combine``; ``order``
       puts the receiver first).
   * - ``gather_rows``
     - values, index
     - Rows selected by an index, e.g. :math:`\bm h_i` per edge (``endpoint`` of a
       pair index).
   * - ``equivariant_convolution``
     - features, harmonics, radial [, pair index]
     - MANDALA gated convolution (``irreps_out``, ``tensor_product``, ``gate``,
       ``radial_layers``, ``radial_weight_granularity``, ``pair_embedding_dim``).
   * - ``factorized_tensor_product``
     - left [, right]
     - Tensor product with factorized path weights :math:`W_1[u,w]W_2[v,w]`, optionally
       gated (``gate``); one input gives the symmetric square (DeepH-E3).
   * - ``radial_channel_scale``
     - irreps, radial
     - Each irrep copy times a weight from an MLP of the radial features
       (``hidden``, ``granularity``).
   * - ``channelwise_tensor_product``
     - features, harmonics, radial
     - MACE convolution: channel-wise product with per-edge path weights from a
       radial MLP (``irreps_out``, ``radial_layers``).
   * - ``indexed_aggregation``
     - messages, edge index, node reference
     - ``sum`` or ``mean`` onto ``receiver`` 0 or 1.
   * - ``projected_attention``
     - messages, nodes, radial, edge index
     - MANDALA invariant multi-head attention reducer (``scalar_dim``, ``heads``).
   * - ``channel_reshape``, ``symmetric_contraction``
     - density / channel density, species attributes
     - MACE product basis: channel-first layout and learned symmetric powers up to
       ``correlation``.
   * - ``bond_endpoint_sum``, ``scalar_normalization``
     - bond-neighbourhood messages / invariant channels
     - DeepH local readout: sums into both bond endpoints; layer, batch, graph,
       instance or pair normalization (``segments``).
   * - ``gated_scalar_message``, ``scalar_mlp``
     - invariant features (scalars or bond-frame components)
     - :math:`\sigma(W_f x)\odot\mathrm{softplus}(W_s x)`; dense MLP.

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
   * - ``e3mlp``
     - irreps [, route index]
     - Equivariant MLP (``layers``, ``variant``, ``irreps_out``,
       ``activate_last``); one parameter copy per route category
       (``routes``, ``inactive_routes``).
   * - ``tensor_square``, ``fc_tensor_product``
     - irreps [, route index] / two irreps on the same entities
     - Symmetric square and fully connected product (``irreps_out``).
   * - ``irrep_nonlinearity``, ``layer_norm``, ``irrep_dropout``
     - irreps
     - MANDALA activations (``nonlin_kind``), per-entity normalization, copy dropout.
   * - ``self_connection``, ``residual``
     - previous, species one-hot, update / previous, update
     - Species-dependent skip path; residual sum (``mode``: add_if_matching, add, project,
       linear).
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
   * - ``add``, ``concat``, ``constant_scale``, ``irrep_sort``
     - feature slots
     - Sum, direct sum, ``values * factor / divisor``, irrep reordering.
   * - ``gaussian_envelope``, ``scale_features``, ``pair_distance_affine``
     - lengths / values and invariant weights
     - Invariant edge weights and their application.
   * - ``pair_envelope``, ``envelope_scale``
     - lengths and pair index / operator blocks and envelope
     - Fitted per species-pair distance envelope of the operator blocks and the
       scaling of predictions by it (``hamiltonian_envelope_mode`` lowers to these).

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
   * - ``onsite_from_nodes``
     - edge, node, edge index, roles
     - Onsite rows take their atom's features.
   * - ``pair_projection``
     - features, edge partitions [, roles]
     - Per species-pair (and per role) projection to orbital-pair irreps
       (``mode``: split or shared_conditioned; ``route_scales``, ``log_scale``).
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

The flat MANDALA configuration lowers to these stages, grouped into the
composite blocks ``edge_encoder``, ``node_update``, ``edge_update`` and
``readout`` (:doc:`pipeline`).

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
     - MACE-style symmetric products, a MANDALA attention update and a pair
       readout, written as two composite blocks.
   * - ``hybrids/tensor_products_degree_expansion_shell_readout.yaml``
     - Tensor products, angular degree expansion and a shell readout.
   * - ``hybrids/equivariant_nodes_bond_frame_readout.yaml``
     - Equivariant node features read out in local bond frames.
