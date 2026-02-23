# Ideas to Generalize an MLP from Scalars to Irreps

This note proposes a family of modules that map one irreps vector to one irreps vector:

- input: `x in Irreps_in`
- output: `y in Irreps_out` (often same as input irreps)
- if `num_layers == 1`, behavior should reduce exactly to `e3nn.o3.Linear(Irreps_in, Irreps_out)`.

It starts with approaches already close to what is explored in:

- `studies/mlp_activation_magnitude_study/run_study.py`

and then extends to more involved designs.

---

## Equivariance facts used repeatedly

Let `rho_in(g)` and `rho_out(g)` be representation matrices of a 3D rotation/inversion `g` for input/output irreps.
A map `f` is equivariant if:

$$
f(rho_in(g) x) = rho_out(g) f(x)
$$

Key closure rules:

1. Composition rule:
   If `f` and `h` are equivariant, then `h o f` is equivariant.
2. Sum rule:
   If `f` and `h` are equivariant to same output irreps, then `f + h` is equivariant.
3. Invariant-scaling rule:
   If `a(x)` is invariant scalar and `v(x)` is equivariant vector/tensor, then `a(x) * v(x)` is equivariant.
4. Linear intertwiners:
   `e3nn.o3.Linear` is equivariant by construction.
5. Tensor products:
   `e3nn.o3.TensorProduct` and `FullyConnectedTensorProduct` are equivariant by Clebsch-Gordan construction.

So the design strategy is simple: build blocks from these primitives and only use scalar nonlinear functions on invariant scalars.

---

## 1. Linear-only stack (baseline control)

### Definition

Layer:

$$
y = Lx, \quad L = \texttt{e3nn.o3.Linear}
$$

Deep variant:

$$
y = L_K L_{K-1} ... L_1 x
$$

### Activation idea

No activation. This is the strict linear baseline.

### Why equivariant

Each `L_i` is equivariant, and composition of equivariant maps is equivariant.

### Notes

- This is not nonlinear, but it gives the exact required `num_layers=1` behavior.
- Use as a sanity check and initialization target for richer designs.
- Can produce irreps not present in input? **No** (except optional constant scalar bias terms, which are not input-dependent features).

---

## 2. Linear + NormActivation (already explored in your study)

### Definition

Per block with irrep `(l,p)` and multiplicity channel `c`, write vector part as `v_{l,c}`.
Norm-based activation applies:

$$
v_{l,c} \to \phi_{l,c}\!\left(\|v_{l,c}\|\right) \cdot \frac{v_{l,c}}{\|v_{l,c}\| + \epsilon}
$$

and scalar irreps go through standard scalar nonlinearity.

Equivalent block: `Linear -> NormActivation`.

### Activation idea

Nonlinearity acts on invariant magnitudes, then rescales directions.

### Why equivariant

- `||v_{l,c}||` is rotation-invariant.
- Direction transforms covariantly as the same irrep.
- Invariant scalar times equivariant vector remains equivariant.

### Notes

- Very stable and simple.
- Often the first nonlinear equivariant block to try.
- Can produce irreps not present in input? **No** (the block rescales/warps existing irreps channels; it does not create new `l,p` types).

---

## 3. Linear + canonical e3nn Gate

### Definition

Split intermediate irreps into:

- scalar channels `s`
- gate-scalar channels `g`
- gated non-scalar channels `t`

Then:

$$
s' = phi(s), \quad
t' = \sigma(g) \odot t
$$

where `sigma(g)` are scalars broadcast across `m` components of each irrep copy.

### Activation idea

Learn scalar "switches" controlling non-scalar amplitudes.

### Why equivariant

- `phi(s)` and `sigma(g)` are scalar maps, hence invariant under rotation action.
- `t` transforms equivariantly.
- Multiplying equivariant `t` by invariant scalars preserves equivariance.

### Notes

- This is a standard strong baseline used in many e3nn models.
- Can produce irreps not present in input? **No** (gating is scalar modulation of existing non-scalars).

---

## 4. Linear + GateScalarsMLP (already explored)

### Definition

Use scalar subspace as context:

$$
u = \text{MLP}_{scalar}(s)
$$

and use `u` to gate all non-scalars:

$$
t'_i = u_i * t_i
$$

with optional scalar nonlinearity on `s`.

### Activation idea

Context-dependent gating from an internal scalar controller MLP.

### Why equivariant

- Scalar MLP consumes only scalars -> outputs scalars.
- These gates are invariant.
- Invariant gates modulate equivariant channels without breaking equivariance.

### Notes

- More expressive than fixed per-channel gate functions.
- Already present in your activation-magnitude study settings.
- Can produce irreps not present in input? **No** (scalar MLP only gates existing channels).

---

## 5. Linear + GateMagnitudes (already explored)

### Definition

For each irrep copy:

$$
r = ||v||, \quad
\alpha = \psi(r), \quad
v' = \alpha \cdot \frac{v}{r + \epsilon}
$$

Scalars can pass through separate scalar nonlinearity.

### Activation idea

"Radial" nonlinearity in irreps space; direction kept, radius transformed.

### Why equivariant

- Norm `r` is invariant.
- Direction transforms equivariantly.
- Scaling by invariant `alpha` preserves equivariance.

### Notes

- Closely related to NormActivation but with custom gating behavior.
- Can produce irreps not present in input? **No** (norm-based modulation preserves existing irrep types).

---

## 6. Linear + S2Activation

### Definition

For harmonics up to `l_max`, convert irreps coefficients to sampled function on sphere, apply pointwise scalar nonlinearity, project back:

$$
a_{lm} \to f(\theta,\phi) \to \phi\!\left(f(\theta,\phi)\right) \to a'_{lm}
$$

Equivalent block: `Linear -> S2Activation`.

### Activation idea

Nonlinearity in geometric (spherical signal) domain, not coefficient domain.

### Why equivariant

Rotations act as domain warps on sphere:

$$
f(\theta,\phi) \to f\!\left(g^{-1}\cdot(\theta,\phi)\right)
$$

Pointwise scalar nonlinearity commutes with this pullback action, and spherical projection is equivariant.

### Notes

- Expressive but more expensive.
- Needs compatible irreps structure/resolution choices.
- Can produce irreps not present in input? **Conditional / often yes** (pointwise nonlinearity on the sphere mixes harmonic orders and can populate additional `l` channels if your projection/truncation retains them).

---

## 7. Residual equivariant MLP blocks (pre-norm style)

### Definition

Block:

$$
h = \text{Act}(\text{Norm}(L_1 x)), \quad
y = x + L_2 h
$$

Stack many such blocks.

### Activation idea

Any equivariant activation family above can be plugged as `Act`.

### Why equivariant

- `L_1`, `Norm`, `Act`, `L_2` are equivariant.
- Residual sum uses equivariance sum rule.

### Notes

- Much better optimization for deep stacks.
- If output irreps differ, use projected skip: `y = P x + L_2 h`.
- Can produce irreps not present in input? **Conditional** (inherits this ability from the internal `Act` block; with norm/gate-style acts alone: no).

---

## 8. FiLM-style invariant modulation of irreps channels

### Definition

Extract scalar invariants `z` (typically from scalar irreps and/or norms of non-scalars), pass through scalar MLP to produce per-channel gains/biases:

$$
(gamma, beta) = \text{MLP}(z)
$$

Apply:

$$
v_{l,c}' = gamma_{l,c} * v_{l,c}
$$

and optionally for scalar channels:

$$
s_c' = gamma_{0,c} * s_c + beta_c
$$

### Activation idea

Data-dependent affine modulation (FiLM) generalized to irreps.

### Why equivariant

- `gamma`, `beta` are invariant scalars.
- Non-scalars are only scaled, not mixed across `m` in non-invariant ways.
- Scalar bias is allowed only on scalar irreps.

### Notes

- Powerful when you need adaptive behavior with controlled symmetry.
- Can produce irreps not present in input? **No** (FiLM-style gains/biases modulate existing channels only).

---

## 9. Bilinear irreps-MLP via self tensor products

### Definition

Create quadratic interactions:

$$
h_1 = L_a x,\quad h_2 = L_b x,\quad q = TP(h_1, h_2),\quad y = L_o Act(q)
$$

Optionally concatenate with linear path:

$$
y = L_{lin} x + L_{quad} Act(TP(L_a x, L_b x))
$$

### Activation idea

Nonlinearity plus explicit second-order interactions (MLP analogue of quadratic features).

### Why equivariant

- `L_a`, `L_b`, `L_o` are equivariant.
- `TP` is equivariant.
- `Act` built from invariant-scalar gating keeps equivariance.
- Sum of equivariant paths is equivariant.

### Notes

- Usually more expressive per layer than pure linear+activation.
- Can produce irreps not present in input? **Yes** (tensor products generate Clebsch-Gordan outputs beyond the original irrep set).

---

## 10. Higher-order polynomial equivariant MLP (tensor power ladder)

### Definition

Build features from multiple orders:

$$
h^{(1)} = L_1 x,\quad
h^{(2)} = TP(h^{(1)}, h^{(1)}),\quad
h^{(3)} = TP(h^{(2)}, h^{(1)}), ...
$$

Then mix selected orders with linear projections and gated activations.

### Activation idea

Generalized polynomial network in irreps space.

### Why equivariant

Each tensor-power term is equivariant by repeated equivariant TP application.
Mixing and gating use equivariant/invariant operations only.

### Notes

- Very expressive.
- Compute and memory can grow fast; prune irreps aggressively.
- Can produce irreps not present in input? **Yes** (higher-order tensor products expand reachable irrep set).

---

## 11. Invariant self-attention over multiplicity channels

### Definition

Treat multiplicity copies as "tokens" within each `l`.
Compute invariant attention scores from scalar contractions (e.g., dot products of invariant summaries):

$$
a_{ij} = \operatorname{softmax}_j\!\left(\operatorname{score}(z_i, z_j)\right)
$$

where `z_i` are invariant scalar summaries from each token.
Then:

$$
v_i' = \sum_j a_{ij} V_j
$$

with `V_j` equivariant value features.

### Activation idea

Nonlinearity is in attention weights (softmax over invariant logits), with equivariant value aggregation.

### Why equivariant

- Attention logits/weights are invariant scalars.
- Weighted sums of equivariant vectors with invariant weights remain equivariant.
- Must not use orientation-dependent (non-invariant) logits.

### Notes

- Enables dynamic cross-channel routing while preserving symmetry.
- More complex bookkeeping than gate-based designs.
- Can produce irreps not present in input? **No** (attention reweights/mixes multiplicity copies within existing irrep blocks).

---

## 12. Equivariant Neural ODE / continuous-depth irreps MLP

### Definition

Define an equivariant vector field:

$$
\frac{dx}{dt} = f_\theta(x,t)
$$

with `f_\theta` built from equivariant blocks above.
Integrate from `t0` to `t1`:

$$
y = \Phi_{t_1,t_0}(x)
$$

### Activation idea

Depth comes from integration trajectory; strong nonlinear behavior with parameter sharing or time conditioning.

### Why equivariant

If `f_\theta` is equivariant, then transformed state `rho(g) x(t)` satisfies the same ODE as the ODE started from `rho(g) x_0`.
By uniqueness of ODE solutions:

$$
\Phi_{t_1,t_0}\!\left(\rho(g)x_0\right) = \rho(g)\,\Phi_{t_1,t_0}(x_0)
$$

### Notes

- Very flexible and potentially parameter-efficient.
- Training can be expensive/sensitive depending on solver.
- Can produce irreps not present in input? **Conditional** (depends on whether `f_\theta` contains TP/polynomial generators; ODE wrapping alone does not add new irreps).

---

## 13. Invariant-routed mixture-of-experts (MoE) over equivariant experts

### Definition

Let experts `E_k(x)` be equivariant maps.
Compute invariant routing weights:

$$
r = softmax(MLP(invariants(x)))
$$

Output:

$$
y = \sum_k r_k E_k(x)
$$

### Activation idea

Nonlinearity from invariant router + diverse expert families.

### Why equivariant

- Each `E_k` is equivariant.
- `r_k` are invariant scalars.
- Invariant-weighted sum of equivariant outputs is equivariant.

### Notes

- Good for multimodal regimes where one block family is not enough.
- Router regularization matters (entropy/load balancing).
- Can produce irreps not present in input? **Conditional** (yes if at least one expert can; no if all experts are modulation-only families).

---

## 14. Scalar+magnitude MLP + non-scalar self-TP gating (custom variant)

### Definition

Given input split into scalars `s` and non-scalars `t`:

$$
u = \operatorname{MLP}\!\left([s,\; \|t\|_{\text{per-copy}}]\right)
$$

Use one part of `u` as transformed scalars and the remaining part as gates.
Build quadratic non-scalar features via self tensor product:

$$
q = \operatorname{TP}(t,t), \quad q_{ns} = \text{non-scalar part of } q
$$

Concatenate and gate:

$$
z_{ns} = [t,\; q_{ns}], \quad \tilde{z}_{ns,i} = \sigma(u_i)\, z_{ns,i}
$$

Then finalize with equivariant linear map:

$$
y = L\!\left([u_{\text{scalar}},\; \tilde{z}_{ns}]\right)
$$

### Activation idea

A scalar controller MLP learns context from invariants, while self-TP injects explicit quadratic equivariant interactions.

### Why equivariant

- `\|t\|` are invariants.
- Scalar MLP outputs invariants.
- `TP(t,t)` is equivariant.
- Invariant gates times equivariant channels remain equivariant.
- Final `Linear` is equivariant.

### Notes

- Matches the exact 6-step variant requested (scalar+magnitude MLP branch + non-scalar self-TP branch + gating + final linear).
- Can produce irreps not present in input? **Yes** (through the self-TP branch, limited to irreps reachable by Clebsch-Gordan products of non-scalar inputs).

---

## Practical module contract suggestion

To satisfy your exact requirement (`num_layers=1` equals `e3nn.o3.Linear`):

1. Constructor takes `num_layers`, `irreps_in`, `irreps_hidden`, `irreps_out`, `activation_kind`.
2. If `num_layers == 1`: instantiate exactly one `Linear(irreps_in, irreps_out)`.
3. If `num_layers > 1`: use:
   - first layer: `Linear(irreps_in, irreps_hidden)` + equivariant activation block
   - middle layers: repeated hidden blocks
   - last layer: `Linear(irreps_hidden, irreps_out)` (optional final activation depending on task).

This pattern cleanly unifies "linear when shallow" and "highly nonlinear when deep".

---

## Which approaches were already explored in your activation study

From `studies/mlp_activation_magnitude_study/run_study.py`, the explored activation families are:

- NormActivation (approach 2)
- GateScalarsMLP with multiple gate scalar activations (approach 4)
- GateMagnitudes with multiple gate scalar activations (approach 5)

All other approaches above are additional expansions beyond that study.
