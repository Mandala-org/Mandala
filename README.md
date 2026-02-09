# Mandala: SO(2) Tensor Product Acceleration Branch

This branch focuses on replacing the standard SO(3) tensor product path with an SO(2)-reduced formulation based on edge-frame rotation, following:

- "Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs"

Target objective:

- reduce tensor-product-like compute from `O(L^6)` to `O(L^3)` by rotating features into an edge-aligned frame, applying SO(2) operations, then rotating back.

## Where The New Pieces Are

- Edge-alignment rotation utility:
  - `src/net/common.py:508` -> `_rotation_matrix_align_y(...)`

- Full EquiformerV2 code imported from upstream implementation:
  - `src/net/equiformer_v2/`

- Our Equiformer-based SO(2) wrappers for Mandala irreps:
  - Main version:
    - `src/net/so2_ops_equiformer_direct_min.py`
  - Masked/index-precomputed variant:
    - `src/net/so2_ops_equiformer_direct_masked.py`

- Benchmark notebook:
  - `notebooks/demos/SO2 Tensor Product Benchmark.ipynb`

## Benchmark Scope

Current comparisons in the notebook are centered on:

- `FullyConnectedTensorProduct` (e3nn baseline)
- `SeparateWeightTensorProduct` (e3nn baseline)
- `SO2OpsEquiformerDirect` (ours)
- `SO2OpsEquiformerDirectMasked` (ours, masked pack/unpack path)
- Pure Equiformer SO(2) block timing (isolated rotate -> SO(2) -> rotate_inv)

## Notes

- `RotatedTensorProduct` remains useful as a sanity/reference wrapper.
- Fast experimental rotated TP classes were removed from this branch to keep the benchmark focused on Equiformer-based SO(2) paths.
