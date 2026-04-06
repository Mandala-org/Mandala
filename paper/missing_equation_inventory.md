# Missing Equation Inventory for `main.tex`

This document collects mathematical transformations that are present in the Mandala framework but are not yet written explicitly in [main.tex](/home/bartek/casus/mandala/paper/main.tex).

Scope:

- include only equations that are currently missing from the manuscript
- write them in a LaTeX-ready form
- cover the merged feature set intended for `src/`
- exclude items already written in `main.tex`, such as:
  - the generalized eigenproblem
  - the density matrix definition
  - `N_e = \Tr(DS)`
  - `E = \Tr(DH)`
  - the overlap-gauge proof and `\mu_H`
  - the basic E(3) equivariance definition
  - the irrep tensor-product decomposition
  - the generic message-passing equations
  - the generic head equation
  - sparse trace, symmetrization, density normalization, and the global loss template

## 1. Periodic Graph Construction and Edge Partitioning

### 1.1 Edge set under a cutoff

This is implicit in the text, but not currently written as an equation.

```tex
\mathcal{E}
=
\left\{
(i,j,\bm{L}) \;\middle|\;
\left\|
\bm{r}_{ij}^{\bm{L}}
\right\|_2
\le r_{\mathrm{c}}
\right\},
\qquad
\bm{r}_{ij}^{\bm{L}}
=
\bm{R}_j-\bm{R}_i+\sum_{\alpha=1}^{3}L_{\alpha}\bm{a}_{\alpha}.
```

Suggested placement:

- `Numerical Implementation`
- `Software design goals, sparse data model, and periodic graph construction`

Code references:

- [graph_features.py:27](/home/bartek/casus/mandala/src/data/graph_features.py#L27)
- [graph_features.py:60](/home/bartek/casus/mandala/src/data/graph_features.py#L60)

### 1.2 Edge lengths and edge classes

The code distinguishes diagonal, shifted-self, and off-diagonal edges explicitly, but the paper does not yet formalize those sets.

```tex
r_{ij}^{\bm{L}}
=
\left\|
\bm{r}_{ij}^{\bm{L}}
\right\|_2.
```

```tex
\mathcal{E}_{\mathrm{diag}}
=
\left\{
(i,j,\bm{L}) \in \mathcal{E}
\;\middle|\;
i=j,\;
\bm{L}=\bm{0}
\right\},
```

```tex
\mathcal{E}_{\mathrm{shifted\_self}}
=
\left\{
(i,j,\bm{L}) \in \mathcal{E}
\;\middle|\;
i=j,\;
\bm{L}\neq \bm{0}
\right\},
```

```tex
\mathcal{E}_{\mathrm{offdiag}}
=
\left\{
(i,j,\bm{L}) \in \mathcal{E}
\;\middle|\;
i\neq j
\right\}.
```

Equivalent mask notation for one enumerated edge `e`:

```tex
\delta_e^{\mathrm{diag}}
=
\mathbf{1}[i_e=j_e]\mathbf{1}[\bm{L}_e=\bm{0}],
\qquad
\delta_e^{\mathrm{shifted\_self}}
=
\mathbf{1}[i_e=j_e]\mathbf{1}[\bm{L}_e\neq\bm{0}],
\qquad
\delta_e^{\mathrm{offdiag}}
=
\mathbf{1}[i_e\neq j_e].
```

Suggested placement:

- `Software design goals, sparse data model, and periodic graph construction`
- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [common.py:767](/home/bartek/casus/mandala/studies/minimal_overfit_observables/common.py#L767)
- [common.py:774](/home/bartek/casus/mandala/studies/minimal_overfit_observables/common.py#L774)
- [common.py:774](/home/bartek/casus/mandala/studies/minimal_overfit_study/common.py#L774)

### 1.3 Trace-alignment permutation

`main.tex` gives the sparse trace itself, but not the precomputed reverse-edge alignment used for efficient evaluation.

For each pair key `k = A-B`, define the reverse key `k^{\mathrm{rev}} = B-A` and the alignment map `P_k` by

```tex
P_k(n)=m
\quad\Longleftrightarrow\quad
\bigl(\bm{L}_{k,n},i_{k,n},j_{k,n}\bigr)
=
\bigl(-\bm{L}_{k^{\mathrm{rev}},m},j_{k^{\mathrm{rev}},m},i_{k^{\mathrm{rev}},m}\bigr).
```

Then the aligned trace contraction can be written as

```tex
\Tr(AB)
=
\sum_{k}
\sum_{n}
\mathrm{tr}
\!\left(
A_{k,n}\,
B_{k^{\mathrm{rev}},P_k(n)}
\right).
```

In vectorized batch notation:

```tex
\Tr(AB)
=
\sum_k
\mathrm{einsum}
\!\left(
\texttt{"bij,bji->"},
A_k,
B_{k^{\mathrm{rev}}}[P_k]
\right).
```

Suggested placement:

- `Sparse operator algebra, symmetrization, gauge correction, and density normalization`

Code references:

- [sparse_math.py:92](/home/bartek/casus/mandala/src/core/sparse_math.py#L92)
- [sparse_math.py:148](/home/bartek/casus/mandala/src/core/sparse_math.py#L148)

## 2. Representation Construction and Encoders

### 2.1 Automatically constructed hidden irreps

The paper mentions hidden irreps as a search dimension, but not the concrete default construction used by the framework.

```tex
\mathcal{H}_{\mathrm{hid}}
=
\bigoplus_{\ell=0}^{\ell_{\max}}
m_{\ell}\,D^{(\ell,+)}
\;\oplus\;
\bigoplus_{\ell=0}^{\ell_{\max}}
\mathbf{1}_{\mathrm{odd}}\,
m_{\ell}\,D^{(\ell,-)},
\qquad
m_{\ell}
=
\max\!\left(\left\lfloor \frac{d_0}{2^{\ell}} \right\rfloor,1\right),
```

where `d_0` is the base hidden multiplicity and `\mathbf{1}_{\mathrm{odd}}` toggles odd-parity channels.

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`
- or `Configuration-driven architecture search`

Code references:

- [common.py:195](/home/bartek/casus/mandala/src/net/common.py#L195)

### 2.2 Node encoder

The current manuscript states that node encoders embed chemical species into scalar channels, but the exact mapping is not written.

For atom `i` with atomic species index `z_i`:

```tex
\bm{h}_i^{(0)}
=
\mathrm{Emb}_{\mathrm{atom}}(z_i)
\in
\mathbb{R}^{d_0},
\qquad
\bm{h}_i^{(0)}
\sim
d_0 \times 0e.
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [encoders.py:49](/home/bartek/casus/mandala/src/net/encoders.py#L49)

### 2.3 Edge encoder, Mandala variant

The generic text says that edge encoders combine edge type, radial embeddings, and spherical harmonics, but the exact concatenation-plus-tensor-product map is not written.

Let `t_{ij}` denote the edge-type index and let `\bm{\phi}(r_{ij})` be the radial embedding. Then the Mandala-style edge encoder is

```tex
\bm{u}_{ij}
=
\mathrm{Emb}_{\mathrm{edge}}(t_{ij}),
\qquad
\bm{g}_{ij}
=
\bm{\phi}(r_{ij}) \oplus Y(\hat{\bm{r}}_{ij}),
```

```tex
\bm{e}_{ij}^{(0)}
=
\mathrm{TP}_{\mathrm{enc}}
\!\left(
\bm{u}_{ij},
\bm{g}_{ij}
\right).
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [encoders.py:125](/home/bartek/casus/mandala/src/net/encoders.py#L125)
- [encoders.py:166](/home/bartek/casus/mandala/src/net/encoders.py#L166)

### 2.4 Edge encoder, radial-only DeepH-E3 style

The alternative scalar-only edge encoder path is also not written explicitly.

```tex
\bm{e}_{ij}^{(0)}
=
W_{\mathrm{dist}}\,
\bm{\phi}(r_{ij}).
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [encoders.py:140](/home/bartek/casus/mandala/src/net/encoders.py#L140)
- [encoders.py:170](/home/bartek/casus/mandala/src/net/encoders.py#L170)

## 3. Equivariant Convolution, Radial Weighting, and Message Updates

### 3.1 Equivariant convolution with per-irrep radial weights

`main.tex` gives a generic message function, but not the concrete EquiConv factorization used in the implementation.

Given feature input `\bm{f}_{ij}` and spherical harmonics `Y(\hat{\bm{r}}_{ij})`, the equivariant convolution first forms

```tex
\bm{z}_{ij}
=
\mathrm{TP}
\!\left(
\bm{f}_{ij},
Y(\hat{\bm{r}}_{ij})
\right).
```

If a gate nonlinearity is enabled:

```tex
\widetilde{\bm{z}}_{ij}
=
\mathrm{Gate}(\bm{z}_{ij}),
```

otherwise `\widetilde{\bm{z}}_{ij} = \bm{z}_{ij}`.

The radial MLP then produces one scalar weight per output irrep block:

```tex
\bm{w}_{ij}
=
\mathrm{MLP}_{\mathrm{rad}}
\!\left(
\bm{\phi}(r_{ij})
\right).
```

If `\widetilde{\bm{z}}_{ij}` is decomposed into irrep slices `\widetilde{\bm{z}}_{ij}^{(a)}`, the weighted output is

```tex
\bm{y}_{ij}^{(a)}
=
w_{ij}^{(a)}\,
\widetilde{\bm{z}}_{ij}^{(a)},
\qquad
\bm{y}_{ij}
=
\bigoplus_a \bm{y}_{ij}^{(a)}.
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [layers.py:87](/home/bartek/casus/mandala/src/net/layers.py#L87)
- [layers.py:231](/home/bartek/casus/mandala/src/net/layers.py#L231)
- [layers.py:242](/home/bartek/casus/mandala/src/net/layers.py#L242)

### 3.2 Edge update block

The paper currently gives only the generic message-passing pattern. The concrete edge update used by the framework can be written as:

```tex
\bm{e}_{ij}^{\mathrm{pre}}
=
\mathrm{MLP}_{\mathrm{pre}}^{(e)}
\!\left(
\bm{e}_{ij}^{(t)}
\right),
```

```tex
\bm{f}_{ij}^{(t)}
=
\bm{h}_i^{(t)}
\oplus
\bm{h}_j^{(t)}
\oplus
\bm{e}_{ij}^{\mathrm{pre}},
```

```tex
\bm{q}_{ij}^{(t)}
=
\mathrm{EquiConv}^{(e)}
\!\left(
\bm{f}_{ij}^{(t)},
Y(\hat{\bm{r}}_{ij}),
\bm{\phi}(r_{ij})
\right),
```

```tex
\bm{e}_{ij}^{\mathrm{tmp}}
=
\mathrm{MLP}_{\mathrm{post}}^{(e)}
\!\left(
\bm{q}_{ij}^{(t)}
\right).
```

If the edge self-connection is enabled with one-hot edge type `\bm{c}_{ij}`:

```tex
\bm{s}_{ij}^{(e)}
=
\mathrm{TP}_{\mathrm{sc}}^{(e)}
\!\left(
\bm{e}_{ij}^{(t)},
\bm{c}_{ij}
\right),
```

and the full update becomes

```tex
\bm{e}_{ij}^{(t+1)}
=
\mathrm{Act}
\!\left(
\bm{e}_{ij}^{\mathrm{tmp}}
 +
\bm{s}_{ij}^{(e)}
\right)
+
\mathbf{1}_{\mathrm{res}}\,\bm{e}_{ij}^{(t)}.
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [layers.py:265](/home/bartek/casus/mandala/src/net/layers.py#L265)
- [layers.py:377](/home/bartek/casus/mandala/src/net/layers.py#L377)

### 3.3 Node update block

The concrete node update used by the framework is:

```tex
\bm{h}_{i}^{\mathrm{pre}}
=
\mathrm{MLP}_{\mathrm{pre}}^{(n)}
\!\left(
\bm{h}_{i}^{(t)}
\right),
```

```tex
\bm{f}_{ij}^{(t)}
=
\bm{h}_{i}^{\mathrm{pre}}
\oplus
\bm{h}_{j}^{\mathrm{pre}}
\oplus
\bm{e}_{ij}^{(t)},
```

```tex
\bm{m}_{ij}^{(t)}
=
\mathrm{EquiConv}^{(n)}
\!\left(
\bm{f}_{ij}^{(t)},
Y(\hat{\bm{r}}_{ij}),
\bm{\phi}(r_{ij})
\right),
```

```tex
\overline{\bm{m}}_{i}^{(t)}
=
\sum_{j \in \mathcal{N}(i)}
\bm{m}_{ji}^{(t)},
```

```tex
\bm{h}_{i}^{\mathrm{tmp}}
=
\mathrm{MLP}_{\mathrm{post}}^{(n)}
\!\left(
\overline{\bm{m}}_{i}^{(t)}
\right).
```

If the node self-connection is enabled with node one-hot `\bm{c}_i`:

```tex
\bm{s}_{i}^{(n)}
=
\mathrm{TP}_{\mathrm{sc}}^{(n)}
\!\left(
\bm{h}_{i}^{(t)},
\bm{c}_{i}
\right),
```

and the full node update becomes

```tex
\bm{h}_{i}^{(t+1)}
=
\mathrm{Act}
\!\left(
\bm{h}_{i}^{\mathrm{tmp}}
 +
\bm{s}_{i}^{(n)}
\right)
+
\mathbf{1}_{\mathrm{res}}\,\bm{h}_{i}^{(t)}.
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [layers.py:417](/home/bartek/casus/mandala/src/net/layers.py#L417)
- [layers.py:526](/home/bartek/casus/mandala/src/net/layers.py#L526)

### 3.4 Sequential message block

The actual message block order in the implementation is:

```tex
\bm{h}^{(t+1/2)}
=
\mathrm{NodeUpdate}^{(t)}
\!\left(
\bm{h}^{(t)},\bm{e}^{(t)}
\right),
```

```tex
\bm{e}^{(t+1)}
=
\mathrm{EdgeUpdate}^{(t)}
\!\left(
\bm{h}^{(t+1/2)},\bm{e}^{(t)}
\right),
```

```tex
\bm{h}^{(t+1)}
=
\bm{h}^{(t+1/2)}.
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [layers.py:571](/home/bartek/casus/mandala/src/net/layers.py#L571)
- [layers.py:652](/home/bartek/casus/mandala/src/net/layers.py#L652)

### 3.5 Separate-weight tensor-product variant

The manuscript mentions tensor-product variants, but not the explicit separate-weight form used in the framework.

For one admissible coupling path, the implementation builds a factorized weight tensor

```tex
W^{(\mathrm{path})}_{ab u}
=
W^{(1,\mathrm{path})}_{a u}\,
W^{(2,\mathrm{path})}_{b u},
```

so that the tensor product is evaluated with a pathwise factorized weight rather than one fully dense joint tensor.

Equivalently, the output along one admissible path can be written schematically as

```tex
\bm{z}^{(\mathrm{path})}
=
\mathrm{TP}
\!\left(
\bm{x},
\bm{y};
W^{(1,\mathrm{path})}\otimes W^{(2,\mathrm{path})}
\right).
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`
- `Configuration-driven architecture search`

Code references:

- [common.py:436](/home/bartek/casus/mandala/src/net/common.py#L436)
- [common.py:501](/home/bartek/casus/mandala/src/net/common.py#L501)

## 4. Readout-Head Variants Missing from the Paper

### 4.1 Head tensor-square preprocessing

The main text mentions tensor-square head variants but does not give the transformation explicitly.

For an edge feature `\bm{e}_{ij}^{(L)}`, define

```tex
\bm{t}_{ij}
=
\mathrm{TS}
\!\left(
\bm{e}_{ij}^{(L)}
\right),
```

and let the actual head input be

```tex
\bm{u}_{ij}
=
\begin{cases}
\bm{t}_{ij}, & \text{if tensor-square head preprocessing is enabled},\\
\bm{e}_{ij}^{(L)}, & \text{otherwise}.
\end{cases}
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [common.py:663](/home/bartek/casus/mandala/studies/minimal_overfit_observables/common.py#L663)
- [common.py:758](/home/bartek/casus/mandala/studies/minimal_overfit_observables/common.py#L758)
- [common.py:602](/home/bartek/casus/mandala/studies/minimal_overfit_study/common.py#L602)

### 4.2 Branching by diagonal / shifted-self / off-diagonal edge class

The generic head equation in `main.tex` does not yet capture the class-specific branches used by the richer head variants.

For a target operator `X`, define

```tex
\widehat{\bm{x}}_{e}^{X}
=
\begin{cases}
\Psi^{X}_{\mathrm{diag}}(\bm{u}_{e}), & e \in \mathcal{E}_{\mathrm{diag}},\\
\Psi^{X}_{\mathrm{shifted\_self}}(\bm{u}_{e}), & e \in \mathcal{E}_{\mathrm{shifted\_self}},\\
\Psi^{X}_{\mathrm{offdiag}}(\bm{u}_{e}), & e \in \mathcal{E}_{\mathrm{offdiag}}.
\end{cases}
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [common.py:677](/home/bartek/casus/mandala/studies/minimal_overfit_observables/common.py#L677)
- [common.py:767](/home/bartek/casus/mandala/studies/minimal_overfit_observables/common.py#L767)
- [common.py:621](/home/bartek/casus/mandala/studies/minimal_overfit_study/common.py#L621)

### 4.3 On-site override with node embeddings for zero-shift self-edges

This variant is present in the merged study feature set but absent from the manuscript.

For zero-shift self-edges:

```tex
\bm{u}_{e}
=
\begin{cases}
\bm{h}_{i}^{(L)}, & \text{if } e=(i,i,\bm{0}) \text{ and onsite node-head override is enabled},\\
\bm{e}_{e}^{(L)}, & \text{otherwise}.
\end{cases}
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [common.py:999](/home/bartek/casus/mandala/studies/minimal_overfit_observables/common.py#L999)

### 4.4 Scalar auxiliary branch for scalar output irreps

The scalar-MLP head variant is not written in the paper.

First construct an auxiliary scalar feature from the tensor-square representation:

```tex
\bm{s}_{e}^{\mathrm{aux}}
=
L_{\mathrm{aux}}^{X}
\!\left(
\bm{t}_{e}
\right),
```

then concatenate it with the radial embedding:

```tex
\bm{a}_{e}^{X}
=
\bm{s}_{e}^{\mathrm{aux}}
\oplus
\bm{\phi}(r_e),
```

and predict scalar output components by

```tex
\widehat{\bm{x}}_{e,\mathrm{scalar}}^{X}
=
\mathrm{MLP}_{\mathrm{scalar}}^{X}
\!\left(
\bm{a}_{e}^{X}
\right).
```

These scalar components are then merged with the non-scalar equivariant projection according to the fixed irrep term ordering of the target operator.

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`
- possibly `Configuration-driven architecture search`

Code references:

- [common.py:652](/home/bartek/casus/mandala/studies/minimal_overfit_study/common.py#L652)
- [common.py:718](/home/bartek/casus/mandala/studies/minimal_overfit_study/common.py#L718)
- [common.py:846](/home/bartek/casus/mandala/studies/minimal_overfit_study/common.py#L846)

### 4.5 Magnitude-factorized output branch

This is one of the most concrete missing mathematical transformations in the current paper.

The head predicts a normalized operator output together with a positive per-edge magnitude:

```tex
\widehat{\bm{x}}_{e,\mathrm{norm}}^{X}
=
\Psi_{\mathrm{dir}}^{X}(\bm{u}_{e}),
```

```tex
\bm{a}_{e}^{X}
=
L_{\mathrm{mag}}^{X}
\!\left(
\bm{t}_{e}
\right)
\oplus
\bm{\phi}(r_e),
```

```tex
\widehat{m}_{e}^{X}
=
\exp
\!\left(
\mathrm{MLP}_{\mathrm{mag}}^{X}
\!\left(
\bm{a}_{e}^{X}
\right)
\right),
```

and the actual reconstructed block prediction is

```tex
\widehat{X}_{e}^{\mathrm{actual}}
=
\widehat{m}_{e}^{X}\,
\widehat{X}_{e}^{\mathrm{norm}}.
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`
- `Configuration-driven architecture search`

Code references:

- [common.py:661](/home/bartek/casus/mandala/studies/minimal_overfit_study/common.py#L661)
- [common.py:873](/home/bartek/casus/mandala/studies/minimal_overfit_study/common.py#L873)
- [overfit_water_minimal.py:1029](/home/bartek/casus/mandala/studies/minimal_overfit_study/overfit_water_minimal.py#L1029)

## 5. Variant-Specific Equivariant MLP Equations Missing from the Paper

### 5.1 GateScalarsMLP nonlinearity

This nonlinearity is present in `src/` but not written mathematically in the paper.

Split a hidden vector into scalar and non-scalar parts:

```tex
\bm{x}
=
\bm{x}_{\mathrm{sc}}
\oplus
\bm{x}_{\mathrm{ns}}.
```

A scalar MLP predicts one gate per non-scalar irrep copy:

```tex
\bm{g}
=
\sigma_{\mathrm{gate}}
\!\left(
\mathrm{MLP}_{\mathrm{gate}}(\bm{x}_{\mathrm{sc}})
\right),
```

the scalar part is activated directly,

```tex
\widetilde{\bm{x}}_{\mathrm{sc}}
=
\sigma_{\mathrm{sc}}(\bm{x}_{\mathrm{sc}}),
```

and each non-scalar irrep copy `\bm{x}_{\mathrm{ns}}^{(a)}` is scaled by its gate:

```tex
\widetilde{\bm{x}}_{\mathrm{ns}}^{(a)}
=
g^{(a)}\,
\bm{x}_{\mathrm{ns}}^{(a)}.
```

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [activations.py:61](/home/bartek/casus/mandala/src/net/activations.py#L61)

### 5.2 GateMagnitudes nonlinearity

For each non-scalar irrep copy `\bm{x}_{\mathrm{ns}}^{(a)}`, define its norm

```tex
\rho^{(a)}
=
\left\|
\bm{x}_{\mathrm{ns}}^{(a)}
\right\|_2,
```

then scale the copy by an activated magnitude:

```tex
\widetilde{\bm{x}}_{\mathrm{ns}}^{(a)}
=
\frac{
\sigma_{\mathrm{gate}}(\rho^{(a)})
}{
\rho^{(a)}
}
\bm{x}_{\mathrm{ns}}^{(a)}.
```

Scalars are still transformed by `\sigma_{\mathrm{sc}}`.

Suggested placement:

- `E(3)-equivariant message passing, readout heads, and architecture variants`

Code references:

- [activations.py:163](/home/bartek/casus/mandala/src/net/activations.py#L163)

### 5.3 Residual E3-MLP

The residual variant can be written as

```tex
\bm{z}^{(0)}
=
L_{\mathrm{in}} \bm{x},
```

```tex
\bm{z}^{(n+1)}
=
\bm{z}^{(n)}
+
\lambda_{\mathrm{res}}\,
f_{\mathrm{eq}}
\!\left(
\bm{z}^{(n)}
\right),
```

```tex
\bm{y}
=
L_{\mathrm{out}} \bm{z}^{(N)}.
```

Suggested placement:

- `Configuration-driven architecture search`

Code references:

- [e3mlp_variants.py:581](/home/bartek/casus/mandala/studies/minimal_overfit_study/e3mlp_variants.py#L581)
- [e3mlp_variants.py:613](/home/bartek/casus/mandala/studies/minimal_overfit_study/e3mlp_variants.py#L613)

### 5.4 Bilinear self-tensor-product E3-MLP

The bilinear self-tensor-product variant is:

```tex
\bm{a}
=
L_a \bm{x},
\qquad
\bm{b}
=
L_b \bm{x},
```

```tex
\bm{q}
=
\mathrm{TP}_{\mathrm{self}}(\bm{a},\bm{b}),
```

```tex
\widetilde{\bm{q}}
=
f_{\mathrm{eq}}(\bm{q}),
```

```tex
\bm{y}
=
L_{\mathrm{skip}}\bm{x}
+
\lambda_{\mathrm{res}}\,L_{\mathrm{out}}\widetilde{\bm{q}}.
```

Suggested placement:

- `Configuration-driven architecture search`

Code references:

- [e3mlp_variants.py:685](/home/bartek/casus/mandala/studies/minimal_overfit_study/e3mlp_variants.py#L685)
- [e3mlp_variants.py:743](/home/bartek/casus/mandala/studies/minimal_overfit_study/e3mlp_variants.py#L743)

### 5.5 Invariant self-attention over multiplicity channels

This variant is named in the workflow inventory but not yet formalized in the paper.

For one non-scalar irrep block with multiplicity dimension `a=1,\dots,m`, let

```tex
\rho_a
=
\left\|
\bm{x}_a
\right\|_2,
```

and build invariant tokens

```tex
\bm{u}_a
=
\rho_a
\oplus
\bm{c},
\qquad
\bm{c}
=
\mathrm{MLP}_{\mathrm{ctx}}(\bm{x}_{\mathrm{sc}}).
```

Then

```tex
\bm{q}_a
=
W_Q \bm{u}_a,
\qquad
\bm{k}_a
=
W_K \bm{u}_a,
```

```tex
A_{ab}
=
\mathrm{softmax}_b
\!\left(
\frac{\bm{q}_a^{\top}\bm{k}_b}{\sqrt{d_{\mathrm{att}}}\,\tau}
\right),
```

and the multiplicity channels are mixed by

```tex
\widetilde{\bm{x}}_a
=
\sum_b A_{ab}\,\bm{x}_b
+
\mathbf{1}_{\mathrm{res}}\bm{x}_a.
```

Suggested placement:

- `Configuration-driven architecture search`

Code references:

- [e3mlp_variants.py:821](/home/bartek/casus/mandala/studies/minimal_overfit_study/e3mlp_variants.py#L821)
- [e3mlp_variants.py:941](/home/bartek/casus/mandala/studies/minimal_overfit_study/e3mlp_variants.py#L941)

### 5.6 Scalar-magnitude self-TP gated variant

This custom variant is not described mathematically in the manuscript at all.

Let the hidden feature be split into scalar and non-scalar parts:

```tex
\bm{x}
=
\bm{x}_{\mathrm{sc}}
\oplus
\bm{x}_{\mathrm{ns}}.
```

Define non-scalar copy magnitudes

```tex
\bm{\rho}
=
\mathrm{MagCopies}
\!\left(
\bm{x}_{\mathrm{ns}}
\right),
```

and invariant MLP input

```tex
\bm{u}
=
\bm{x}_{\mathrm{sc}}
\oplus
\bm{\rho}.
```

An invariant scalar MLP produces new scalar channels and gates:

```tex
\bm{v}
=
\mathrm{MLP}(\bm{u}),
\qquad
\bm{x}_{\mathrm{sc}}^{\mathrm{new}}
=
\bm{v}_{\mathrm{sc}},
\qquad
\bm{g}
=
\sigma_{\mathrm{gate}}(\bm{v}_{\mathrm{gate}}).
```

A self tensor product is formed on the non-scalar channels,

```tex
\bm{t}_{\mathrm{ns}}
=
\mathrm{TP}_{\mathrm{self}}
\!\left(
\bm{x}_{\mathrm{ns}},\bm{x}_{\mathrm{ns}}
\right),
```

its non-scalar part is concatenated with the original non-scalar channels,

```tex
\bm{z}_{\mathrm{ns}}
=
\bm{x}_{\mathrm{ns}}
\oplus
\bm{t}_{\mathrm{ns}},
```

and those copies are scaled by the learned gates:

```tex
\widetilde{\bm{z}}_{\mathrm{ns}}
=
\mathrm{ScaleByCopy}
\!\left(
\bm{z}_{\mathrm{ns}},\bm{g}
\right).
```

Finally,

```tex
\bm{y}
=
L_{\mathrm{out}}
\!\left(
\bm{x}_{\mathrm{sc}}^{\mathrm{new}}
\oplus
\widetilde{\bm{z}}_{\mathrm{ns}}
\right)
+
\mathbf{1}_{\mathrm{res}}\bm{x}.
```

Suggested placement:

- `Configuration-driven architecture search`

Code references:

- [e3mlp_variants.py:1181](/home/bartek/casus/mandala/studies/minimal_overfit_study/e3mlp_variants.py#L1181)
- [e3mlp_variants.py:1285](/home/bartek/casus/mandala/studies/minimal_overfit_study/e3mlp_variants.py#L1285)

## 6. Observable Evaluation Paths Missing from the Paper

### 6.1 Aligned sparse evaluation of energy and electron count

The manuscript gives `\widehat{E}=\Tr(\widehat{D}\widehat{H})` and `\widehat{N}_e=\Tr(\widehat{D}\widehat{S})`, but not the aligned sparse form actually used in the implementation.

```tex
\widehat{E}
=
\sum_k \sum_n
\mathrm{tr}
\!\left(
\widehat{H}_{k,n}\,
\widehat{D}_{k^{\mathrm{rev}},P_k(n)}
\right),
```

```tex
\widehat{N}_e
=
\sum_k \sum_n
\mathrm{tr}
\!\left(
\widehat{D}_{k,n}\,
\widehat{S}_{k^{\mathrm{rev}},P_k(n)}
\right).
```

Suggested placement:

- `Observable guidance with energies, electron counts, forces, and stress`

Code references:

- [train_silicon_minimal.py:2025](/home/bartek/casus/mandala/studies/minimal_silicon_study/train_silicon_minimal.py#L2025)
- [train_silicon_minimal.py:2041](/home/bartek/casus/mandala/studies/minimal_silicon_study/train_silicon_minimal.py#L2041)

### 6.2 Mixed observable-guidance traces with one reference operator

`main.tex` mentions these paths in prose, but not as explicit equations.

The diagnostic or partially guided observable traces are:

```tex
\widehat{E}_{H|D^{\mathrm{ref}}}
=
\Tr\!\left(
\widehat{H}\,D^{\mathrm{ref}}
\right),
\qquad
\widehat{E}_{D|H^{\mathrm{ref}}}
=
\Tr\!\left(
H^{\mathrm{ref}}\widehat{D}
\right),
```

```tex
\widehat{N}_{D|S^{\mathrm{ref}}}
=
\Tr\!\left(
\widehat{D}\,S^{\mathrm{ref}}
\right),
\qquad
\widehat{N}_{S|D^{\mathrm{ref}}}
=
\Tr\!\left(
D^{\mathrm{ref}}\widehat{S}
\right).
```

Suggested placement:

- `Observable guidance with energies, electron counts, forces, and stress`

Code references:

- [e3gnn.py:420](/home/bartek/casus/mandala/src/net/e3gnn.py#L420)
- [e3gnn.py:423](/home/bartek/casus/mandala/src/net/e3gnn.py#L423)
- [e3gnn.py:424](/home/bartek/casus/mandala/src/net/e3gnn.py#L424)

## 7. Suggested Insertion Order

If these equations are moved into the manuscript, the cleanest order is:

1. edge set, edge classes, and trace alignment
2. hidden-irrep construction and encoder equations
3. EquiConv and concrete node/edge update equations
4. separate-weight tensor-product variant
5. head tensor-square, class-split heads, scalar branch, onsite override, and magnitude factorization
6. variant-specific E3-MLP equations
7. aligned observable evaluation and mixed trace diagnostics

## 8. Priority Order for the Paper

If you only want the most important additions first, I would prioritize:

1. edge classes and trace alignment
2. EquiConv with radial weighting
3. concrete node/edge update equations
4. class-split head equations
5. magnitude-factorized head equations
6. aligned sparse observable equations

These six items would already make the methods section significantly more complete and much closer to the actual framework implementation.
