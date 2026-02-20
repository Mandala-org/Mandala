# Data Pipeline Deep Dive

This note documents the end-to-end data path used by this repository, with emphasis on:

- `scripts/train_silicon.py`
- everything under `src/`
- convention transforms between OpenMX and e3nn
- all major mathematical operations used during parsing, feature construction, and training

## 1. End-to-end flow at a glance

1. `scripts/train_silicon.py` collects `(Si_DM, info.dat)` pairs from temperature folders.
2. `DatasetFactory` parses all `info.dat` files to build one global `OrbitalIrrepConfig`.
3. `BlockIrrepMapper` builds per-element-pair change-of-basis tensors (`q`) for block<->irrep conversion.
4. `E3GNNDataset` converts each pair into a `Snapshot`:
   - parse OpenMX blocks (H, S, D)
   - parse structure/forces/box from `info.dat`
   - optionally filter by cutoff
   - convert basis (`openmx` -> `e3nn` by default)
   - canonicalize edge ordering
5. `compute_graph_features` builds graph edges and geometric features:
   - edge index + periodic shifts
   - edge type ids
   - radial basis embedding
   - spherical harmonics
6. `E3GNN` runs node/edge encoders -> message passing -> per-matrix heads.
7. Head outputs (irrep vectors) are converted to block matrices, then losses are computed:
   - matrix reconstruction losses
   - optional observable losses (`Tr(DH)`, `Tr(DS)`)
   - optional force/stress gradients from `E = Tr(DH)`.

## 2. Data selection in `scripts/train_silicon.py`

For temperatures

$$
\mathcal{T} = \{T_{\min}, T_{\min}+\Delta T, \dots, T_{\max}\},
$$

training temperatures are

$$
\mathcal{T}_{\text{train}} = \{T \in \mathcal{T} \mid T \neq T_{\text{val}}\}
$$

except the degenerate case where all three are equal (then that temperature is kept).

At each temperature, paths matching `*/Si_DM` are sampled uniformly without replacement:

$$
n_{\text{sel}}(T) = \min\left(|\mathcal{S}_T|, n_{\text{snap}}\right),
$$

$$
\mathcal{S}_T^{\text{sel}} \sim \text{SampleWithoutReplacement}\left(\mathcal{S}_T, n_{\text{sel}}(T)\right).
$$

Each selected matrix path is paired with sibling `info.dat`.

## 3. Orbital configuration and global mapper

### 3.1 Orbital irreps from OpenMX info

`parse_info_out` builds per-element orbital specs like `2s2p1d` (internally converted to irreps such as `2x0e+2x1o+1x2e`).

`DatasetFactory` merges all snapshots' orbital sets into one project-wide config:

$$
\text{OrbCfg} = \bigcup_{s \in \text{train}\cup\text{val}} \text{OrbCfg}(s),
$$

with consistency checks (conflicts raise).

### 3.2 BlockIrrepMapper math

For each ordered element pair `(A,B)`, e3nn `ReducedTensorProducts` yields a change-of-basis tensor `q`.
Implementation flattens block dimensions:

- Block to vector:
$$
v = \operatorname{vec}(M)\, q^\top
$$
- Vector to block:
$$
\operatorname{vec}(M) = v\, q
$$

where $M \in \mathbb{R}^{d_A \times d_B}$, $v \in \mathbb{R}^{n_{\text{vec}}}$.

## 4. OpenMX parsing and matrix assembly

### 4.1 `info.dat` parser transform

`parse_info_out` extracts:

- element list and Cartesian positions from `<coordinates.forces ... >`
- forces
- fractional coordinates (if present)
- orbital occupancy metadata used to infer orbital irreps per element

When both Cartesian and fractional coordinates are available, the cell matrix is recovered by least squares:

$$
h^\star = \arg\min_h \|Fh - A\|_F^2,
$$

where $F \in \mathbb{R}^{N\times 3}$ are fractional coordinates and $A \in \mathbb{R}^{N\times 3}$ are Cartesian coordinates.
The implementation uses `torch.linalg.lstsq(F, A, driver="gelsd")`.

### 4.2 `Si_DM` parser transform

`parse_openmx_scfout` parses:

- Kohn-Sham Hamiltonian (`hamiltonian`)
- Overlap (`overlap`)
- Density (`density`, first spin block only)
- periodic translation map from "Overlap matrix with position operator x"

For each block header `(i,j,Rn)` it reads a dense block $B_{ij}^{Rn}$ of shape $d_i \times d_j$.

Edge key convention in sparse storage:

$$
e = (s_x, s_y, s_z, i, j).
$$

Blocks are grouped by element pair key `A-B`.

Optional density symmetrization:

$$
D \leftarrow D + D^\top.
$$

## 5. Convention transform: OpenMX <-> e3nn (critical section)

There are two coupled transforms:

1. orbital-block basis transform (inside matrix blocks)
2. Cartesian axis permutation (for positions/forces/box)

### 5.1 Orbital-block transform

For each angular momentum $l$, a fixed orthogonal matrix $U_l$ is used.
For OpenMX -> e3nn:

$$
M'_{AB} = U_A\, M_{AB}\, U_B^\top,
$$

where $U_A$ is block-diagonal, built from per-orbital $U_l$ matrices for element $A$.

Implemented permutations:

- $l=0:\ I_1$
- $l=1:\ I_3[[1,2,0]]$
- $l=2:\ I_5[[2,4,0,3,1]]$
- $l=3:\ I_7[[6,4,2,0,1,3,5]]$
- $l=4:\ I_9[[8,6,4,2,0,1,3,5,7]]$

Inverse (e3nn -> OpenMX) uses transposes.

### 5.2 Cartesian axis permutation

In `Snapshot._change_basis`:

- OpenMX -> e3nn:
  $$
  P_{\text{O}\to\text{E}} = I_3[[2,0,1]]
  $$
  applied as row-vector transform:
  $$
  r' = r\,P_{\text{O}\to\text{E}},\quad f' = f\,P_{\text{O}\to\text{E}},\quad h' = h\,P_{\text{O}\to\text{E}}.
  $$
- e3nn -> OpenMX:
  $$
  P_{\text{E}\to\text{O}} = I_3[[1,2,0]] = P_{\text{O}\to\text{E}}^{-1}.
  $$

So basis conversion is not only an orbital reorder; geometry axes are also permuted.

## 6. Snapshot-level processing

### 6.1 Physical observables from sparse blocks

Defined in `Snapshot` using sparse block traces:

$$
E = \operatorname{Tr}(D H),\qquad N_e = \operatorname{Tr}(D S).
$$

Sparse trace implementation uses reverse edges:

$$
\operatorname{Tr}(AB)
= \sum_{(s,i,j)} \operatorname{tr}\left(A_{(s,i,j)}\, B_{(-s,j,i)}\right).
$$

### 6.2 Distances and cutoff filtering

For edge $e=(s_x,s_y,s_z,i,j)$:

$$
\Delta r_e = r_j - r_i + s\,h,\quad s=[s_x,s_y,s_z].
$$

$$
d_e = \|\Delta r_e\|_2.
$$

`filter_by_distance(cutoff)` keeps edges where $d_e \le r_c$, and applies the same mask to H/S/D.

### 6.3 Canonical edge ordering

Edges are reordered per key as:

1. diagonal edges first (self edges with zero shift), sorted by atom index
2. off-diagonal edges sorted lexicographically by:
   $$
   (d_e, s_x, s_y, s_z, i, j).
   $$

This deterministic ordering is important because graph features are sorted using the same rule.

## 7. Geometry and graph feature construction

`compute_graph_features` uses ASE neighbor list with periodic shifts.

### 7.1 Edge set

- Off-diagonal edges from `neighbor_list("ijS", cutoff=self.cfg.cutoff_radius)`.
- Self edges are added explicitly for each node.

Combined edge tensor:

$$
\text{edge\_index} =
\begin{bmatrix}
\text{src}\\
\text{dst}
\end{bmatrix},
\quad
\text{edge\_shift} =
\begin{bmatrix}
s_x\\
s_y\\
s_z
\end{bmatrix}.
$$

### 7.2 Geometric features

Displacement for off-diagonal edges:

$$
\Delta r_{ij} = r_j - r_i + S_{ij} h.
$$

Lengths:

$$
r_{ij} = \|\Delta r_{ij}\|_2.
$$

Spherical harmonics feature:

$$
\phi^{\text{SH}}_{ij} = Y_{\ell m}(\widehat{\Delta r}_{ij})
$$

for irreps up to $\ell_{\max}$, with e3nn component normalization.

Radial feature via Gaussian soft one-hot basis on $[0,r_c]$:

$$
\phi^{\text{rad}}_{ij,k}
\propto
\exp\left(-\frac{(r_{ij}-\mu_k)^2}{2\sigma^2}\right),
\quad k=1,\dots,n_{\text{radial}}.
$$

Edge type id:

$$
t_{ij} = \text{index}(\text{element}_i-\text{element}_j).
$$

## 8. Dataset sample structure

Each dataset item is `(x, y)`:

- `x` contains graph inputs:
  - node type indices
  - positions, box, atoms
  - optional precomputed edges/features (`edge_index`, `edge_shift`, `edge_type_idx`, `edge_length_emb`, `edge_sh`, `num_self_edges`)
- `y` contains targets:
  - matrices (`hamiltonian`, `overlap`, `density`) either as blocks or irreps
  - observables (`energy`, `num_electrons`)
  - optional forces/stress tensors.

If `train_target == "irreps"`, targets are converted with mapper:

$$
y_M^{\text{irreps}} = \text{blocks\_to\_vectors}(y_M^{\text{block}}).
$$

## 9. Model pipeline equations (`E3GNN`)

### 9.1 Encoders

- Node encoder: learned embedding by element id (scalars).
- Edge encoder: tensor product of edge-type embedding and concatenated radial+SH features.

### 9.2 Message passing

Each layer runs edge update then node update.
Conceptually for edge $i\to j$:

$$
e_{ij}^{(l+1)} = \Psi_e\left(x_i^{(l)}, x_j^{(l)}, e_{ij}^{(l)}, Y(\widehat{r}_{ij}), \phi^{\text{rad}}_{ij}\right)
$$

Node update uses scatter-sum of incoming messages:

$$
m_j^{(l)} = \sum_{i \in \mathcal{N}(j)} \Psi_n\left(x_i^{(l)}, x_j^{(l)}, e_{ij}^{(l+1)}, Y(\widehat{r}_{ij}), \phi^{\text{rad}}_{ij}\right)
$$

$$
x_j^{(l+1)} = \Phi_n(m_j^{(l)}, x_j^{(l)}).
$$

### 9.3 Head/readout

For each matrix target name (`hamiltonian`, `overlap`, `density`):

1. trunk MLP on edge embeddings
2. per-pair projection to pair-specific irrep output
3. group by pair key and carry corresponding `(sx,sy,sz,i,j)` edges
4. wrap as `IrrepsBlockData`
5. convert to block matrices when needed:
   $$
   \hat{M}^{\text{block}} = \text{vectors\_to\_blocks}(\hat{M}^{\text{irreps}}).
   $$

## 10. Training loss math

For matrix target $m \in \{H,S,D\}$, pair key $k$:

$$
\mathcal{L}_{2,m,k} = \operatorname{mean}\left((\hat{T}_{m,k}-T_{m,k})^2\right),
\quad
\mathcal{L}_{1,m,k} = \operatorname{mean}\left(|\hat{T}_{m,k}-T_{m,k}|\right).
$$

Per-matrix aggregation:

$$
\mathcal{L}_{m}
= (1-\alpha)\sum_k \mathcal{L}_{2,m,k}
+ \alpha\sum_k \mathcal{L}_{1,m,k},
\quad \alpha=\text{loss\_l1\_fraction}.
$$

Matrix loss:

$$
\mathcal{L}_{\text{matrix}} = \sum_m \mathcal{L}_m.
$$

Observable predictions:

$$
\hat{E} = \operatorname{Tr}(\hat{D}\hat{H}),\quad
\hat{N} = \operatorname{Tr}(\hat{D}\hat{S}).
$$

Optional observable loss:

$$
\mathcal{L}_{\text{obs}}
=
\lambda_{\text{obs}}
\left(
\mathbf{1}_{E}\,\|\hat{E}-E\|_2^2
+
\mathbf{1}_{N}\,\|\hat{N}-N\|_2^2
\right).
$$

Total (before optional regularization):

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{matrix}} + \mathcal{L}_{\text{obs}}.
$$

If enabled:

$$
\mathcal{L}_{\text{total}}
\leftarrow
\mathcal{L}_{\text{total}}
+ \lambda_1\|\theta\|_1
+ \lambda_2\|\theta\|_2^2.
$$

## 11. Force and stress formulas

From predicted snapshot:

$$
F_i = -\frac{\partial E}{\partial r_i}.
$$

With cell matrix $h$, volume $\Omega = \det(h)$, and $G=\partial E/\partial h$:

$$
\sigma = \frac{1}{\Omega} h^\top G,
$$

optionally symmetrized:

$$
\sigma \leftarrow \frac{1}{2}(\sigma+\sigma^\top).
$$

## 12. Potential issues found

1. Import path setup in `train_silicon.py` likely points one directory too high.
   - `Path(__file__).resolve().parents[2]` resolves to `.../casus`, not repository root `.../casus/mandala`.
2. Sampling in `train_silicon.py` is stochastic with no explicit seeding despite `cfg.seed` existing.
3. `pin_memory` condition appears inverted (`cfg.gpus == 0`), which is usually suboptimal for GPU training.
4. Return-value mismatch when `precompute_edge_features=False`:
   - `compute_graph_features` returns 6 values, `E3GNN.forward` expects 7.
5. Partial-GT observable branch can reference `E_true`/`N_true` before assignment in some config combinations.
6. `E3GNNDataset.to()` uses `self.device` before initialization.
7. Snapshot serialization writes stress only when `box is not None` instead of `stress is not None`.
8. `Snapshot.from_openmx` gates stress assignment on `info.box.numel()` instead of stress tensor presence.
9. `BenchmarkCallback` summarizes `loader_times` (list of floats) with a function expecting list-of-dicts.
10. Density symmetrization uses $D \leftarrow D + D^\top$ (not average), which may double magnitude depending upstream conventions.
11. Cache key in `E3GNNDataset` does not include all transformation-affecting knobs (for example convention/target format), so stale-cache cross-contamination is possible.
12. `Snapshot._edge_displacements` docstring says minimal-image, but wrapping is commented out (current behavior is raw shifted displacement).

## 13. Key convention takeaway (OpenMX vs e3nn)

OpenMX->e3nn conversion in this code is a composite transform:

1. orbital-space permutation/sign transform per $l$ on every block
2. coordinate-axis permutation on `positions`, `forces`, and `box`

Ignoring either side leads to inconsistent geometry-vs-matrix alignment.
