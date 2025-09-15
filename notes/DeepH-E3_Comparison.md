  Comparative Report: Mandala vs. DeepH-E3

  This report provides a detailed comparison between the Mandala and DeepH-E3 frameworks,
  focusing on four key areas: data preparation, network architecture, network details, and the
  training routine.

  ---

  1. Data Preparation

  The process of converting raw DFT data into a graph format suitable for the GNN differs
  significantly between the two frameworks.

   * Input Data Source
       * Mandala: Directly consumes raw output files from DFT codes like OpenMX (.scfout,
         .info.out).
       * DeepH-E3: Requires a preliminary preprocessing step that converts DFT outputs into an
         intermediate HDF5-based format (hamiltonians.h5, overlaps.h5, etc.) and several .dat
         files containing structural information. This step is handled by a Julia script
         (openmx_get_data.jl).
       * Comparison: Different. Mandala's approach is more direct, while DeepH-E3 introduces an
         explicit preprocessing stage, separating data parsing from training.

   * Graph Construction
       * Mandala: Builds the graph based on geometry. All atoms within a specified cutoff radius
         are connected, including periodic images. This can result in multiple edges between the
         same pair of atoms if different periodic images fall within the cutoff.
       * DeepH-E3: Builds the graph based on the sparsity of the DFT matrices. Edges are created
         only for atom pairs (i, j) that have a non-zero matrix block A_ij(R) in the source
         hamiltonians.h5 file. This results in a one-to-one correspondence between a graph edge
         and a matrix block to be predicted.
       * Comparison: Different. This is a fundamental conceptual difference. Mandala's graph is
         defined by geometric proximity, whereas DeepH-E3's graph is defined by the electronic
         interactions present in the reference DFT calculation.

   * Edge Features
       * Mandala: Pre-computes higher-level edge features. This includes a radial basis expansion
          of the distance and the spherical harmonics of the displacement vector. These are
         passed as input to the network.
       * DeepH-E3: Provides more basic features to the network: the scalar distance and the 3D
         displacement vector. The radial basis expansion and spherical harmonics are then
         computed on-the-fly inside the network modules.
       * Comparison: Similar. Both frameworks use the same core geometric information. The main
         difference is the stage at which this information is processed into higher-level
         equivariant features (data prep vs. inside the model).

   * Target Handling
       * Mandala: Keeps the graph structure (x) and the target matrices (y) as separate objects
         in a tuple (x, y). The target matrices (H, S, D) are stored as BlockMatrix objects.
       * DeepH-E3: Integrates the targets directly into the graph structure. The target matrix
         block for each edge is decomposed into irreps, flattened into a vector, and stored as
         the label attribute on that specific edge in the torch_geometric.data.Data object.
       * Comparison: Different. This reflects the different graph construction philosophies. In
         DeepH-E3, since each edge is a matrix block, the target belongs to the edge. In
         Mandala, the graph is a representation of the atomic environment, and the model learns
         to predict the matrices for that environment.

  ---

  2. Network Architecture

  While both frameworks employ E(3)-equivariant message-passing networks, their architectural
  organization shows some differences.

   * Overall Structure
       * Mandala: The E3GNN model consists of an encoder followed by a series of MessageBlocks.
       * DeepH-E3: The Net model consists of an embedding layer followed by a series of
         interaction blocks, each comprising a NodeUpdateBlock and an EdgeUpdateBlock.
       * Comparison: Similar. Both architectures are based on the iterative refinement of node
         and edge features through stacked message-passing layers.

   * Interaction Blocks
       * Mandala: Encapsulates the edge and node update logic within a single MessageBlock
         module.
       * DeepH-E3: Defines NodeUpdateBlock and EdgeUpdateBlock as separate, distinct modules
         that are called sequentially within the main model's forward pass.
       * Comparison: Functionally the same. This is primarily a code organization difference.
         The underlying principle of updating edge features based on node states and then
         updating node features based on aggregated edge messages is identical.

   * Output Head
       * Mandala: Uses a DeepHead module, which is a multi-layer equivariant sub-network. It has
         a shared "trunk" followed by separate linear projections for each atom-pair type (e.g.,
         "Si-H", "H-H"). This allows the final prediction layer to be specialized for different
         chemical interactions.
       * DeepH-E3: Uses a single final Linear layer to project the final edge features into the
         irreps of the target matrix blocks.
       * Comparison: Dissimilar. Mandala's DeepHead is significantly more complex and
         expressive, providing specialized output layers for each interaction type. DeepH-E3
         uses a simpler, single projection for all edge types.

  ---

  3. Network Details

  The implementation details of the equivariant layers and operations are quite similar, as both
  frameworks build upon e3nn.

   * Equivariant Operations
       * Mandala: Primarily uses e3nn.o3.FullyConnectedTensorProduct.
       * DeepH-E3: Uses a custom SeparateWeightTensorProduct, where the tensor product weights
         are generated by a small MLP that takes the radial embedding as input.
       * Comparison: Similar. Both use tensor products as the core of their message-passing
         operations. The use of a separate MLP to generate TP weights in DeepH-E3 is a specific
         variant that makes the interaction explicitly distance-dependent.

   * Non-linearities
       * Mandala: Provides a flexible choice of equivariant activations, including
         NormActivation, S2Activation, and custom Gate implementations.
       * DeepH-E3: Primarily uses e3nn.nn.Gate.
       * Comparison: Similar. Both use standard equivariant gating mechanisms. Mandala offers
         more explicit flexibility in its configuration.

   * Basis Convention Conversion
       * Mandala: Handled by the OpenMXE3NNConverter class.
       * DeepH-E3: Handled by the Rotate class inside e3modules.py.
       * Comparison: Exactly the same. Both frameworks correctly identify the need to convert
         the spherical harmonic basis from the OpenMX convention to the e3nn convention and
         implement the conversion using the same set of permutation matrices.

  ---

  4. Training Routine

   * Framework
       * Mandala: Uses PyTorch Lightning, which abstracts the training loop and provides a
         structured framework for training, validation, and testing.
       * DeepH-E3: Implements a custom training loop from scratch within the DeepHE3Kernel
         class.
       * Comparison: Different. Mandala's use of a high-level framework like PyTorch Lightning
         separates the model definition from the engineering details of training, making the
         code more modular.

   * Loss Function
       * Mandala: Computes the Mean Squared Error by iterating through the dictionary of
         predicted blocks or irrep vectors.
       * DeepH-E3: Uses a MaskMSELoss on a single, large flattened vector containing all the
         irrep coefficients for an entire structure. A pre-computed mask is used to ignore
         zero-padded elements.
       * Comparison: Functionally the same. Both compute the MSE on the predicted irrep
         coefficients. The difference is in the implementation strategy (loop over blocks vs.
         masked loss on a flat vector).

   * Optimizer & Scheduler
       * Mandala: Adam optimizer with a ReduceLROnPlateau scheduler.
       * DeepH-E3: Adam optimizer with a custom RevertDecayLR scheduler that includes logic for
         reverting to the best model state upon a learning rate decay.
       * Comparison: Similar. Both use standard optimizers and learning rate scheduling based on
         validation performance.

  ---

  Proposed Tests for Comparative Study

  To further evaluate the mathematical and functional differences between the two frameworks,
  the following tests could be performed:

   1. Basis Conversion Equivalence: Create a set of random matrix blocks in the OpenMX basis and
      convert them to the e3nn basis using both Mandala's OpenMXE3NNConverter and DeepH-E3's
      Rotate class. The resulting matrices should be identical.
   2. Graph Construction Comparison: For a simple periodic structure, generate the graph using
      both Mandala's geometry-based approach and DeepH-E3's data-driven approach. A comparison of
      the resulting edge lists would formally demonstrate the fundamental difference in how the
      two frameworks represent atomic environments.
   3. Interaction Block Equivalence: Configure a single message-passing block from each framework
      with mathematically equivalent settings (e.g., same irreps, simple non-linearity, no skip
      connections). Given identical inputs, this test would verify if the core message-passing
      operations produce the same outputs.
   4. End-to-End Sanity Check: Train a minimal version of each model on a single, identical data
      point. Compare the magnitude and direction of the loss and gradient updates after one
      training step to ensure that both frameworks behave in a physically and mathematically
      plausible manner under similar conditions.
