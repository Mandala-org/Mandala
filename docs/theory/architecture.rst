The MANDALA E3GNN
=================

This chapter derives the default MANDALA network layer by layer. The flat
:class:`~net.common.Config` compiles into exactly the stages described here
(:func:`net.component_defaults.default_graph`); every symbol below has a
configuration field next to it.

.. graphviz::

   digraph e3gnn {
     rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.32;
     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fillcolor="#f3f7fb", color="#315b7d"];
     edge [color="#5a8fbd", fontname="Helvetica", fontsize=9];
     geo [label="positions, lattice, edges (i, j, L)", shape=plaintext, fillcolor=none];
     rad [label="radial basis φ(r)\nn_radial Gaussians"];
     sh [label="spherical harmonics Y(r̂)\nl ≤ l_max"];
     z [label="species z_i, pair type t_ij", shape=plaintext, fillcolor=none];
     nenc [label="node encoder\nh⁰ = Emb(z)", fillcolor="#f2faf6", color="#2f8f6b"];
     eenc [label="edge encoder\ne⁰ = TP(Emb(t), φ ⊕ Y⊗Y)", fillcolor="#f2faf6", color="#2f8f6b"];
     subgraph cluster_layer { label="× num_layers_gnn"; fontname="Helvetica"; fontsize=10; color="#c9d6e3"; style=rounded;
       nupd [label="node update\nconvolution → aggregation → MLP → activation"];
       eupd [label="edge update\nconvolution with new node states"];
     }
     head [label="readout per operator\nneck → per pair / edge class projection", fillcolor="#fdf3ea", color="#d07b3f"];
     env [label="optional envelope  E_p(r) · x"];
     out [label="Qᵀ: irreps → H, S, D blocks", shape=plaintext, fillcolor=none];
     geo -> rad; geo -> sh; z -> nenc; z -> eenc; rad -> eenc; sh -> eenc;
     nenc -> nupd; eenc -> nupd; nupd -> eupd; eupd -> nupd [style=dashed, label="next layer"];
     eupd -> head; nupd -> head; head -> env -> out;
   }

Geometric embeddings
--------------------

**Radial basis.** Each edge length is expanded in ``n_radial`` Gaussians on
:math:`[0, r_c]`,

.. math::

   \varphi_n(r)=\frac{1}{1.12}\exp\!\Big[-\Big(\frac{r-\mu_n}{\Delta}\Big)^2\Big],
   \qquad \mu_n=\frac{n\,r_c}{n_{\text{radial}}-1},\quad\Delta=\mu_1-\mu_0 .

With ``pair_distance_normalization: pair_r0`` the argument is
:math:`r/r_0^{(a_ia_j)}`, a per-species-pair length taken from the envelope
(below), which aligns the radial resolution with each bond type.

**Angular basis.** :math:`Y(\hat{\bm r}_{ij})=\bigoplus_{\ell\le\ell_{\max}}Y^\ell`
with :math:`\ell_{\max}=` ``l_max``; non-scalar harmonics of onsite edges are
zero.

Encoders
--------

**Nodes.** :math:`\bm h_i^{(0)}=\mathrm{Emb}(z_i)\in\mathbb R^{d_0}` is a
learned species embedding of width ``hidden_base_dim`` (``d_0 x 0e``).

**Edges** (``edge_encoder_style: rich``). With a learned pair-type embedding
:math:`\bm u_{ij}=\mathrm{Emb}(t_{ij})` of width ``edge_type_emb_dim``,

.. math::

   \bm e_{ij}^{(0)}=\mathrm{TP}_{\text{fc}}\!\Big(\bm u_{ij},\ \bm\varphi(r_{ij})\oplus\mathrm{TS}\big(Y(\hat{\bm r}_{ij})\big)\Big),

where :math:`\mathrm{TS}(Y)=\mathrm{TP}(Y,Y)` is the e3nn tensor square
(``edge_encoder_use_sh_tensor_square``). The tensor square creates the
odd-:math:`\ell`/even-parity and even-:math:`\ell`/odd-parity combinations
(``1e``, ``2o``…) that a single harmonic lacks. ``edge_encoder_style:
distance`` uses a plain linear map of :math:`\bm\varphi` instead.

The equivariant convolution
---------------------------

Both update blocks are built around one radially weighted, gated tensor
product (the ``equivariant_convolution`` stage). Given a geometric feature
:math:`\bm f_{ij}` it computes

.. math::

   \bm z_{ij}=\mathrm{TP}\big(\bm f_{ij},Y(\hat{\bm r}_{ij})\big),\qquad
   \widetilde{\bm z}_{ij}=\mathrm{Gate}(\bm z_{ij}),\qquad
   \bm q_{ij}^{(c)}=w_c\big(\bm\varphi(r_{ij})\big)\,\widetilde{\bm z}_{ij}^{(c)} .

* :math:`\mathrm{TP}` is the separate-weight or fully connected product
  (``tp_type``); only irreps reachable by a selection rule are produced.
* :math:`\mathrm{Gate}` applies SiLU (``0e``) or tanh (``0o``) to scalars and
  multiplies every non-scalar copy by a sigmoid/tanh gate scalar computed in
  the same tensor product.
* :math:`w=\mathrm{MLP}_{\text{rad}}(\bm\varphi)` (hidden widths
  ``radial_layers``) emits one invariant weight per irrep term
  (``radial_weight_granularity: term``) or per copy (``copy``). With
  ``pair_conditioned_radial_mlp`` the MLP also receives the pair-type
  embedding, giving every element pair its own radial filter.

Node update
-----------

For layer :math:`t`, with :math:`\widetilde{\bm h}=\mathrm{E3MLP}_{\text{pre}}(\bm h^{(t)})`,

.. math::

   \bm q_{ij}=\mathrm{EquiConv}\big(\widetilde{\bm h}_i\oplus\widetilde{\bm h}_j\oplus\bm e_{ij}^{(t)}\big),
   \qquad
   \bar{\bm m}_j=\operatorname*{Agg}_{i\to j}\ \bm q_{ij},

messages flowing from the source :math:`i` to the receiver :math:`j` of each
directed edge. Three aggregations are available (``node_update_message_agg``):

.. math::

   \text{sum: }\sum_i\bm q_{ij},\qquad
   \text{average: }\frac1{|\mathcal N(j)|}\sum_i\bm q_{ij},\qquad
   \text{attention: }\frac1H\sum_{h=1}^{H}\sum_i\alpha^h_{ij}\,V^h\bm q_{ij},

with invariant multi-head attention weights

.. math::

   \alpha^h_{ij}=\operatorname*{softmax}_{i\in\mathcal N(j)}
   \frac{\langle Q^h\widetilde{\bm h}_j,\ K^h(\bm\varphi_{ij}\oplus\bm q_{ij})\rangle}{\sqrt{d_h}} .

:math:`Q^h, K^h` are equivariant linear maps onto ``0e`` channels (they read
only the invariant part of their input), so the scores are invariant, and
each head's values :math:`V^h` are equivariant linear maps of the messages.
``node_update_attention_scalar_dim`` sets :math:`Hd_h`,
``node_update_attention_heads`` sets :math:`H`.

The aggregated message is then post-processed:

.. math::

   \bm h^{(t+1)}_j=\mathcal R\Big(\mathrm{Drop}\big(\mathrm{LN}(\sigma(\mathrm{E3MLP}_{\text{post}}(\bar{\bm m}_j)+\bm s_j))\big)\Big)\ \big[+\,\bm h^{(t)}_j\big],

where :math:`\bm s_j=\mathrm{TP}_{\text{fc}}(\bm h^{(t)}_j,\mathrm{onehot}(z_j))` is
the optional species-dependent self-connection (``use_self_connection``),
:math:`\sigma` the equivariant nonlinearity (``nonlin_kind``), LN the optional
equivariant layer normalization (``e3layernorm``), Drop irrep dropout
(``dropout``), :math:`\mathcal R` a refinement E3MLP with
``internal_e3mlp_layers`` layers of variant ``internal_e3mlp_variant``, and
the bracket the optional residual (``node_update_residual``).

Edge update
-----------

Edges are updated *after* nodes, from the new node states:

.. math::

   \bm c_{ij}=\begin{cases}
     \bm h_i^{(t+1)}\oplus\bm h_j^{(t+1)} & \texttt{concat}\\
     \bm h_i^{(t+1)}+\bm h_j^{(t+1)} & \texttt{sum}\\
     \mathrm{TP}_{\text{fc}}(\bm h_i^{(t+1)},\bm h_j^{(t+1)}) & \texttt{tensor\_product}
   \end{cases}
   \qquad(\texttt{edge\_update\_node\_combine}),

.. math::

   \bm e_{ij}^{(t+1)}=\mathcal R\Big(\mathrm{Drop}\big(\mathrm{LN}(\sigma(\mathrm{E3MLP}_{\text{post}}(\mathrm{EquiConv}(\bm c_{ij}\oplus\mathrm{E3MLP}_{\text{pre}}(\bm e_{ij}^{(t)})))+\bm s_{ij}))\big)\Big)\ \big[+\,\bm e^{(t)}_{ij}\big],

with the pair-type self-connection
:math:`\bm s_{ij}=\mathrm{TP}_{\text{fc}}(\bm e^{(t)}_{ij},\mathrm{onehot}(t_{ij}))`.
Edge features are the natural carriers of pair information: after
``num_layers_gnn`` layers, :math:`\bm e_{ij}` summarizes the environment of
both atoms *and* the geometry of the bond.

Readout heads
-------------

One ``readout`` block per predicted operator maps final features to the
irreps of each species pair:

.. math::

   \widehat{\bm x}_{ij}^{\bm L,X}=s_c\,\Psi^{X}_{c,\,a_ia_j}\Big(\mathcal T\big(N^{X}_c(\bm u_{ij})\big)\Big),
   \qquad c\in\{\text{onsite},\ \text{shifted self},\ \text{offsite}\}.

* :math:`\bm u_{ij}` is the final edge feature, or for onsite blocks the final
  node feature :math:`\bm h_i` (``head_use_node_embeddings_for_self_edges``).
* :math:`N^X_c` is a *neck* E3MLP of depth ``neck_depth`` per edge class: an
  ``e3mlp`` stage routed by ``edge_roles``, with one parameter copy per class;
  shifted-self edges get their own copy when ``separate_shifted_self`` is set.
* :math:`\mathcal T` is an optional tensor square (``head_use_tensor_square``).
* :math:`\Psi^X_{c,a_ia_j}` projects onto the pair irreps (table in
  :doc:`operators`). ``head_pair_mode: split`` uses one projection per species
  pair; ``shared_conditioned`` uses a single projection of
  :math:`N\oplus\mathrm{onehot}(a_i,a_j)` and gathers each pair's channels.
* :math:`s_c` is a fixed output scale (``head_diag_output_scale``,
  ``head_offdiag_output_scale``). With ``head_use_mlp_log_scale`` the
  prediction is further multiplied by :math:`\exp(\mathrm{MLP}(N))`, an
  invariant learned magnitude.

Finally :math:`X_{ij}^{\bm L}=Q^\top_{\mathcal B_i\mathcal B_j}\widehat{\bm x}_{ij}^{\bm L,X}`.

Distance envelopes
------------------

Hamiltonian and overlap blocks decay roughly exponentially with distance, over
orders of magnitude. A fitted, pair-dependent envelope :math:`E_p(r)` can
factor that decay out of the learning problem
(``hamiltonian_envelope_mode``):

* ``multiply_prediction``: :math:`H^{\text{pred}}_{ij}=E_{a_ia_j}(r_{ij})\,\widehat H_{ij}`;
* ``normalize_target``: train :math:`\widehat H` against :math:`H^{\text{ref}}/E`
  and rescale for evaluation.

Two envelope families are provided, with per-pair parameters fitted to the
training targets and stored as JSON (``hamiltonian_envelope_path``):

.. math::

   E^{\text{soft cutoff}}(r)=A\,(1+r)^{\nu}e^{-br}\,\frac{1}{1+e^{(r-r_0)/\tau}},

.. math::

   E^{\text{exp-quad wall}}(r)=A\,(1+r)^{\nu}\exp\!\Big(-br-cr^2-\frac{\gamma}{r_c-r}\Big),
   \qquad r_c = x_{\max}+\delta .

Positive parameters are stored as logarithms. Because :math:`E` is an
invariant scalar per edge, the factorization does not affect equivariance.

The equivariant MLP and its variants
------------------------------------

:class:`~net.common.E3MLP` alternates equivariant linear layers with
nonlinear blocks; the block type is selected per position (``e3mlp_variant``
for pre/post maps, ``internal_e3mlp_variant`` for refinements,
``head_e3mlp_variant`` for readout projections). Writing
:math:`L` for a scaled ``o3.Linear``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Variant
     - Block
   * - ``basic``
     - :math:`\sigma_{\text{nonlin\_kind}}(L\bm x)`
   * - ``normact``
     - :math:`\bm y=L\bm x,\ \bm y_c\mapsto\operatorname{sigmoid}(\|\bm y_c\|)\,\bm y_c/\|\bm y_c\|`
   * - ``gate``
     - e3nn ``Gate`` on :math:`L\bm x` (extra gate scalars produced by :math:`L`)
   * - ``gatemagnitudes``
     - scalars :math:`\to f(\bm y_c)`, non-scalars :math:`\bm y_c\mapsto g(\|\bm y_c\|)\,\bm y_c/\|\bm y_c\|`
   * - ``film``
     - :math:`\bm y=L\bm x`; an MLP of the invariants of :math:`\bm y` emits a gain
       :math:`\gamma_c\in(0,2)` per copy and a bias per scalar:
       :math:`\bm y_c\mapsto\gamma_c\bm y_c\,(+\beta_c)`, followed by scalar activations
   * - ``resnormact``, ``resgatemagnitudes``
     - :math:`\bm x+\lambda_c\odot\mathrm{block}(\bm x)` with a learned per-copy
       scale initialized at ``e3mlp_residual_scale``
   * - ``bilinear``
     - :math:`\bm x+\lambda_c\odot L_3\,\mathrm{NormAct}\big(\mathrm{TP}(L_1\bm x, L_2\bm x)\big)`

The odd-scalar activations (``activation_odd_scalar``, ``activation_odd_gate``)
are made odd functions automatically so that pseudoscalars remain
pseudoscalars.

Equivariant nonlinearities
--------------------------

``nonlin_kind`` selects :math:`\sigma` in the update blocks:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Kind
     - Definition (per copy :math:`\bm x_c`)
   * - ``normact``
     - :math:`\bm x_c\mapsto f(\|\bm x_c\|)\,\bm x_c/\|\bm x_c\|` (e3nn ``NormActivation``)
   * - ``gate_magnitudes``
     - scalars :math:`\mapsto f(x)`; others :math:`\mapsto g(\|\bm x_c\|)\,\bm x_c/\|\bm x_c\|`
   * - ``gate_scalars_mlp``
     - scalars :math:`\mapsto f(x)`; others multiplied by gates from an MLP of all ``0e`` channels
   * - ``s2act``
     - expand to a function on the sphere (resolution ``s2act_res``), apply
       :math:`f` pointwise, project back (e3nn ``S2Activation``)

:math:`f` is ``activation_scalar`` and :math:`g` ``activation_gate``.

Cost
----

For a fixed cutoff the number of edges grows linearly with the number of
atoms, so graph construction and message passing scale as
:math:`\mathcal O(|\mathcal E|\,C_{\text{TP}})` with :math:`C_{\text{TP}}` the
cost of the configured tensor-product paths; readout and block reconstruction
scale with the number and size of the retained blocks. Dense diagonalization
for band structures is a separate post-processing step that scales cubically
with the number of orbitals.
