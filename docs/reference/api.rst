API reference
=============

Model
-----

.. autoclass:: net.e3gnn.E3GNN
   :members: forward, predict_matrices, predicted_irreps_to_block_matrices,
             fit_training_data, load_from_checkpoint_with_mapper,
             predict_forces, predict_stress

.. autoclass:: net.common.Config
   :members: from_mapping, to_mapping, resolve
   :undoc-members: False

.. autofunction:: net.configuration.resolve_configuration

Pipeline
--------

.. autoclass:: net.pipeline.FeatureSpec
   :members: dim

.. autoclass:: net.pipeline.FeaturePipeline
   :members: forward

.. autofunction:: net.pipeline.register_transform

.. autofunction:: net.pipeline.build_transform

.. autofunction:: net.component_defaults.default_graph

Layers
------

.. autoclass:: net.encoders.NodeEncoder
.. autoclass:: net.encoders.EdgeEncoder
.. autoclass:: net.layers.EquiConv
.. autoclass:: net.layers.NodeUpdateBlock
.. autoclass:: net.layers.EdgeUpdateBlock
.. autoclass:: net.heads.DeepHead
.. autoclass:: net.common.E3MLP
.. autoclass:: net.common.RadialMLP
.. autoclass:: net.common.SeparateWeightTensorProduct
.. autofunction:: net.activations.make_nonlinearity
.. autoclass:: net.activations.GateMagnitudes
.. autoclass:: net.layer_norm.E3LayerNorm
.. autofunction:: net.e3mlp_variants.make_variant_block
.. autoclass:: net.transforms.equivariant.GaussianBasis
.. autoclass:: net.transforms.equivariant.GatedRadialTensorProduct
.. autoclass:: net.transforms.equivariant.GraphIrrepNorm
.. autoclass:: net.transforms.cluster.NeighborDensityInteraction
.. autoclass:: net.transforms.cluster.SymmetricProductContraction
.. autoclass:: net.transforms.cluster.AngularDegreeExpansion
.. autoclass:: net.transforms.composition.ProjectedAttention
.. autoclass:: net.training_policies.SpeciesIrrepAffine

Operators and data
------------------

.. autoclass:: data.snapshot.Snapshot
   :members: from_openmx, from_deeph_e3, from_fhiaims, from_pyscf, from_pyscf_kspace,
             to_e3nn, to_openmx, rotate, filter_by_distance, symmetrize_matrices,
             reduce_orbitals, get_energy, get_number_of_electrons, to_k_space,
             get_band_structure, dos, save, load, export_to_deephe3

.. autoclass:: data.block_matrix.BlockMatrix
   :members: edges, keys, diag, offdiag, adjoint, transpose, to_vectors, to_dense,
             from_dense, rotate, sparsify, save, load

.. autoclass:: data.block_matrix.IrrepsBlockData
   :members: to_blocks, rotate, from_packed, pack_matrix

.. autoclass:: data.factory.DatasetFactory
   :members: add_snapshot, create

.. autoclass:: data.gnn_dataset.E3GNNDataset

.. autofunction:: data.structure_inference.build_model_input_from_structure

Orbitals and irreps
-------------------

.. autoclass:: core.orbital_irrep_config.OrbitalIrrepConfig
   :members: from_yaml, to_dict, elements, block_dims

.. autoclass:: core.block_irrep_mapper.BlockIrrepMapper
   :members: blocks_to_vectors, vectors_to_blocks, get_pair_irreps, vector_dim, block_dims

.. autoclass:: core.operator_layout.ShellTensorCodec
   :members: from_species_shells, encode, decode, pack_blocks, unpack_blocks, layout_signature

.. autoclass:: core.operator_layout.OrbitalFrameRotation
   :members: rotate_blocks, to_coupling_basis, from_coupling_basis

Sparse algebra and reciprocal space
-----------------------------------

.. autofunction:: core.sparse_math.trace_matmul_sparse_block_matrix
.. autofunction:: core.sparse_math.trace_matmul_sparse_block_matrix_aligned
.. autofunction:: core.sparse_math.build_trace_alignment
.. autofunction:: core.periodic_fourier.shiftspace_to_kspace_dense
.. autofunction:: core.periodic_fourier.kspace_to_shiftspace_dense

Objectives
----------

.. autofunction:: net.spectral_loss.compute_spectral_eigenvalue_loss
.. autofunction:: net.observable_metrics.observable_loss
.. autofunction:: data.envelope.load_pair_envelope_table
.. autofunction:: data.envelope.evaluate_slater_soft_cutoff
.. autofunction:: data.envelope.evaluate_slater_exp_quad_soft_wall

Evaluation and export
---------------------

.. automodule:: net.evaluation
   :members: load_checkpoint, restore_config, restore_model_for_snapshot,
             predict_snapshot, predictions_to_snapshot, evaluate_sample,
             export_operator_prediction

.. autofunction:: data.openmx_writer.export_model_openmx
