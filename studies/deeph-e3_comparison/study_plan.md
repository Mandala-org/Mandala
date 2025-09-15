# Comparative Study Plan

  Here is the revised and more detailed plan, incorporating the comparison for both irreps and
  matrix element training targets.

  Folder: studies/deeph-e3_comparison/
  Script: run_study.py

  Experiment 1: End-to-End Basis Transformation Equivalence

  This experiment will test the hypothesis that the final objects used for loss calculation in
  both frameworks are mathematically equivalent, despite the different approaches to basis
  conversion.

   * Hypothesis: Given an identical input matrix block from OpenMX, the final prediction object
     (irrep vector or matrix block) and target object used for the loss calculation are in the
     same basis and are numerically identical.
   * Evaluation Criterion: The absolute and relative difference between the final objects
     produced by the two frameworks should be negligible (< 1e-6).

   * Methodology:
       1. Data Loading: Load the Hamiltonian matrix from the specified silicon snapshot. One
          version will be loaded "as-is" (DeepH-E3 style), and the other will be pre-converted to
          the e3nn basis (Mandala style).
       2. Part 1a: Training on Irreps (`train_target: 'irreps'`)
           * Mandala Path: Take a block from the pre-converted Hamiltonian and use
             BlockIrrepMapper.blocks_to_vectors to get the target irrep vector,
             target_vector_mandala.
           * DeepH-E3 Path: Take the corresponding "as-is" block and use the
             e3TensorDecomp.get_net_out logic to get the target irrep vector, target_vector_deeph.
           * Comparison: Compare target_vector_mandala and target_vector_deeph.
       3. Part 1b: Training on Matrix Elements (`train_target: 'matrix'`)
           * Mandala Path: The target is the pre-converted Hamiltonian block,
             target_matrix_mandala. The network predicts an irrep vector, which is converted back
             to a matrix pred_matrix_mandala using vectors_to_blocks. We will compare the targets
             directly.
           * DeepH-E3 Path: The target is the "as-is" OpenMX block, target_matrix_deeph. The
             network's prediction is converted to an OpenMX-basis matrix pred_matrix_deeph using
             get_H. We will compare the targets. To do this, I will convert target_matrix_deeph to
             the e3nn basis using its own Rotate class.
           * Comparison: Compare target_matrix_mandala and the converted target_matrix_deeph.
       4. Reporting:
           * Generate a YAML report (basis_equivalence_report.yaml) with separate sections for
             "Irreps Target" and "Matrix Target", detailing the max/mean absolute and relative
             differences.
           * Generate plots (basis_irreps.png, basis_matrix.png) for visual comparison.

  Experiment 2: End-to-End Rotational Equivariance

  This experiment will test the hypothesis that both frameworks are correctly E(3)-equivariant,
  regardless of their internal coordinate system conventions.


   * Hypothesis: A rotation of the atomic coordinates results in a predictable, equivariant
     transformation of the final predicted object (irrep vector or matrix block).
   * Evaluation Criterion: The prediction for a rotated structure must be identical to the
     analytically rotated prediction from the original structure. The difference should be
     negligible (< 1e-6).

   * Methodology:
       1. Setup: Load the silicon snapshot and instantiate minimal, untrained models for both
          frameworks. Define a random 3D rotation matrix R.
       2. Part 2a: Training on Irreps
           * For each framework, I will:
               1. Get the predicted irrep vector v1 for a block from the original structure.
               2. Get the predicted irrep vector v2 from the rotated structure.
               3. Analytically rotate v1 using the appropriate Wigner-D matrix to get v1_rot.
               4. Compare v2 and v1_rot.
       3. Part 2b: Training on Matrix Elements
           * For each framework, I will:
               1. Get the predicted matrix block P1 from the original structure.
               2. Get the predicted matrix block P2 from the rotated structure.
               3. Analytically rotate P1 using the Wigner-D matrices: P1_rot = D_row(R) @ P1 @
                  D_col(R)†.
               4. Compare P2 and P1_rot.
       4. Reporting:
           * Generate a YAML report (equivariance_report.yaml) with four sections
             (Mandala-Irreps, Mandala-Matrix, DeepH-E3-Irreps, DeepH-E3-Matrix), each containing
             the quantitative equivariance error.
           * Generate corresponding plots for visual comparison.
