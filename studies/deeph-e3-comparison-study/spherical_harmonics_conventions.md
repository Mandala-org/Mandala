# Spherical Harmonics and Basis Conventions

This is the highest-risk mismatch area for your observed performance gap.

## 1. What must match mathematically

For parity with DeepH-E3, all of the following must be consistent:

1. Orbital basis ordering/sign convention for each `l` block (OpenMX real SH vs wiki/e3nn real SH).
2. Cartesian axis order expected by e3nn SH implementation.
3. Rotation-matrix conversion used in Wigner-D / equivariant transforms.
4. Whether SH API normalizes internally or vectors are normalized externally.
5. Whether target and prediction spaces are compared in identical basis at loss time.

If any of these differ, error can appear as structured artifacts (for example along "fake diagonals" in specific shifted blocks).

## 2. OpenMX -> wiki/e3nn orbital block basis matrices

DeepH-E3 uses explicit permutation matrices `U_l`:

- `external/DeepH-E3/deephe3/e3modules.py:44`
- `external/DeepH-E3/deephe3/e3modules.py:50`

Main src uses the same pattern and extends to `l=4`:

- `src/core/basis_converter.py:32`
- `src/core/basis_converter.py:37`

So for `l <= 3`, basis matrices are aligned between DeepH-E3 and main src.

## 3. Cartesian axis convention and y-z-x reorder

DeepH-E3 explicit behavior:

- Edge vector for SH is reordered to `(y, z, x)` from `[dist, dx, dy, dz]`:
  - `external/DeepH-E3/deephe3/model.py:401`
- Rotation matrix conversion does `R_e3nn = P R P^T` with `P = I[[1,2,0]]`:
  - `external/DeepH-E3/deephe3/e3modules.py:128`

Main src behavior:

- During OpenMX -> e3nn snapshot conversion, `positions`, `forces`, and `box` are right-multiplied by `I[[2,0,1]]`.
  - `src/data/snapshot.py:366`
- For row vectors, this right-multiply effectively maps `(x,y,z) -> (y,z,x)`.
- Then graph displacements are computed in this transformed frame and passed to SH with standard normalize/component mode.
  - `src/data/graph_features.py:86`
  - `src/data/graph_features.py:193`

Conclusion:

- DeepH-E3 and main src appear convention-compatible on axis ordering when no extra user transforms are introduced.

## 4. Minimal study divergence points

### 4.1 SH normalization path differs

Minimal does:

1. manual normalization `edge_vec_norm = edge_vec / ||edge_vec||`,
2. then `spherical_harmonics(..., normalize=False)`.

- `studies/minimal_overfit_study/overfit_water_minimal.py:664`
- `studies/minimal_overfit_study/overfit_water_minimal.py:670`

DeepH-E3/main src do:

- `spherical_harmonics(..., normalize=True, normalization='component')`.
  - `external/DeepH-E3/deephe3/model.py:355`
  - `src/data/graph_features.py:193`

Why this matters:

- Even if vectors are unit-normalized first, API-level `normalize=False` path is not guaranteed equivalent to `normalize=True` for all edge cases (especially around near-zero norms and self-edges).
- Your self-edge handling and manual zero-protection can create subtly different SH values than the reference path.

### 4.2 Additional user-facing convention knobs in minimal script

Minimal has `--xyz-permutation`, `--change-box`, `--box-convention` toggles:

- `studies/minimal_overfit_study/overfit_water_minimal.py:285`
- `studies/minimal_overfit_study/overfit_water_minimal.py:291`
- `studies/minimal_overfit_study/overfit_water_minimal.py:298`

These are powerful debugging options but they also increase surface area for accidental mismatch if not mirrored exactly for both graph and targets.

### 4.3 Target canonicalization after coordinate changes

Minimal correctly canonicalizes target matrices after possible position/box transforms:

- `studies/minimal_overfit_study/overfit_water_minimal.py:553`

This is good, but correctness still depends on the exact same displacement convention as graph features and SH input.

## 5. Output-space convention handling

DeepH-E3:

- Output decomposition/recomposition is tied to Wigner-3j and OpenMX/wiki conversion (`get_H`, `get_net_out`).
  - `external/DeepH-E3/deephe3/e3modules.py:315`
  - `external/DeepH-E3/deephe3/e3modules.py:348`

Main/minimal:

- Use `BlockIrrepMapper` and `ReducedTensorProducts` based change-of-basis matrices.
  - `src/core/block_irrep_mapper.py:61`

Implication:

- Even when mathematically equivalent, implementation path is different and must be verified numerically with dedicated parity tests.

## 6. Concrete convention parity checks to run first

1. SH parity check:
   - For identical `edge_disp`, compare minimal SH tensor vs main src SH tensor.
   - Expect near machine precision if conventions match exactly.
2. Basis conversion parity:
   - Compare `src/core/basis_converter.py` block conversion with DeepH-E3 `Rotate.openmx2wiki_H` on the same random blocks.
3. `get_net_out` parity:
   - For selected `(l_left, l_right)` channel blocks, compare DeepH-E3 `get_net_out` vs `BlockIrrepMapper.blocks_to_vectors` representation up to known ordering/permutation.
4. Round-trip parity:
   - `block -> vector -> block` in both frameworks, then compare in the same basis.

## 7. Main takeaway

Most likely convention-related risk in your current minimal script is not the OpenMX/wiki matrices themselves, but the SH preprocessing and normalization path plus optional coordinate/box toggles.

The highest-priority alignment target is:

- use the same SH API mode as DeepH-E3/main src,
- ensure graph and target use exactly one consistent axis/box convention,
- remove or lock non-essential convention toggles during parity experiments.
