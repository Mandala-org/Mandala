Symmetry and representations
============================

This chapter fixes notation and collects the representation theory that every
MANDALA layer relies on. Readers familiar with e3nn may skim it; the
:ref:`operator transformation law <operator-law>` at the end is specific to
electronic-structure operators and worth reading in full.

The Euclidean group
-------------------

Atomic structures are acted upon by the Euclidean group
:math:`\mathrm{E}(3)=\mathbb R^3\rtimes\mathrm O(3)`. An element
:math:`g=(Q,\bm t)` with :math:`Q\in\mathrm O(3)` and :math:`\bm t\in\mathbb R^3`
moves atoms, and the lattice vectors of a periodic cell, as

.. math::

   \bm R_i \mapsto Q\bm R_i + \bm t, \qquad \bm a_\alpha \mapsto Q\bm a_\alpha .

Every orthogonal matrix is a rotation (:math:`\det Q=+1`) or a rotation
composed with the inversion :math:`-\mathbb 1` (:math:`\det Q=-1`). MANDALA
expresses all geometry through *relative* displacements

.. math::

   \bm r_{ij}^{\bm L} = \bm R_j - \bm R_i + \textstyle\sum_\alpha L_\alpha \bm a_\alpha,
   \qquad \bm L\in\mathbb Z^3,

which are unchanged by translations and transform as
:math:`\bm r\mapsto Q\bm r`. Translation invariance is therefore exact by
construction; the network only has to respect :math:`\mathrm O(3)`.

A function :math:`f` is **equivariant** when
:math:`f(g\cdot X)=\rho_{\text{out}}(g)\,f(X)` for representations
:math:`\rho_{\text{in}}, \rho_{\text{out}}` of the group, and **invariant**
when :math:`\rho_{\text{out}}` is trivial.

Irreducible representations
---------------------------

The irreducible representations (irreps) of :math:`\mathrm O(3)` are labelled
by a degree :math:`\ell\in\{0,1,2,\dots\}` and a parity :math:`p=\pm1`.
They act on :math:`2\ell+1` dimensional spaces through

.. math::

   D^{(\ell,p)}(Q) = p^{\,[\det Q = -1]}\; D^{\ell}\!\big(Q\det Q\big),

where :math:`D^\ell` is the real Wigner matrix of the proper rotation
:math:`Q\det Q`. e3nn writes these irreps as ``"0e"``, ``"1o"``, ``"2e"``…
(``e``: :math:`p=+1`, ``o``: :math:`p=-1`). Scalars are ``0e``,
pseudoscalars ``0o``, ordinary vectors ``1o`` and axial vectors ``1e``.

A feature vector carried by a node or an edge is a direct sum of irreps with
multiplicities,

.. math::

   \mathcal H=\bigoplus_{\ell,p} m_{\ell p}\,D^{(\ell,p)},
   \qquad\text{e.g. } \texttt{128x0e+64x1o+32x2e},

and :math:`\mathrm O(3)` acts on it block-diagonally. Each of the
:math:`m_{\ell p}` blocks is called a **copy** (or channel) of that irrep.

Spherical harmonics
-------------------

Directions enter the network through real spherical harmonics
:math:`Y^\ell(\hat{\bm r})\in\mathbb R^{2\ell+1}`, which satisfy

.. math::

   Y^\ell(Q\hat{\bm r}) = D^{(\ell,(-1)^\ell)}(Q)\,Y^\ell(\hat{\bm r}).

They transform as the irrep :math:`(\ell,(-1)^\ell)`, so the harmonic of a
displacement vector is ``0e+1o+2e+3o+…``. MANDALA uses component
normalization, :math:`\|Y^\ell\|^2 = 2\ell+1`, and evaluates every global
feature in the canonical Cartesian order :math:`(x,y,z)`. Onsite edges
(:math:`i=j`, :math:`\bm L=0`) have no direction; their non-scalar harmonics
are set to zero, which is the only choice compatible with equivariance.

Tensor products and Clebsch–Gordan coefficients
-----------------------------------------------

The product of two irreps decomposes into irreps,

.. math::

   D^{(\ell_1,p_1)}\otimes D^{(\ell_2,p_2)}
   =\bigoplus_{L=|\ell_1-\ell_2|}^{\ell_1+\ell_2} D^{(L,\,p_1p_2)},

and the change of basis is given by the Clebsch–Gordan coefficients
:math:`C^{L M}_{\ell_1m_1\,\ell_2m_2}`. For features
:math:`\bm x\in\mathcal H_1`, :math:`\bm y\in\mathcal H_2`, one coupling
*path* :math:`(\ell_1,\ell_2\to L)` produces

.. math::

   \big(\bm x\otimes\bm y\big)^{L}_M
   = \sum_{m_1 m_2} C^{LM}_{\ell_1m_1\,\ell_2m_2}\,x^{\ell_1}_{m_1}\,y^{\ell_2}_{m_2}.

A *weighted* tensor product attaches a learnable weight to every path and
pair of copies; the weight is an invariant, so the product is equivariant
for any weight values, including weights computed from invariant inputs such
as interatomic distances. MANDALA uses three weight layouts:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Layout
     - Weights per path
     - Used by
   * - fully connected (``uvw``)
     - :math:`m_1\,m_2\,m_{\text{out}}`
     - ``tp_type: fully_connected``, encoders, self-connections
   * - separate weight
     - :math:`W'\in\mathbb R^{m_1\times m_{\text{out}}}`,
       :math:`W''\in\mathbb R^{m_2\times m_{\text{out}}}`, outer product
     - ``tp_type: separate_weight`` (default)
   * - depthwise (``uvu``)
     - :math:`m_1\,m_2`, output multiplicity :math:`m_1`
     - NequIP/MACE-style convolutions, :doc:`../extending/convolution`

The separate-weight product uses the rank-one weight
:math:`w_{uvw}=W'_{uw}W''_{vw}`; it has far fewer parameters than the fully
connected product and is the MANDALA default for message passing.

.. admonition:: Selection rules
   :class: theorem

   A path :math:`(\ell_1,p_1)\otimes(\ell_2,p_2)\to(L,p)` exists if and only if
   :math:`|\ell_1-\ell_2|\le L\le\ell_1+\ell_2` and :math:`p=p_1p_2`. A layer
   asked to output an irrep that no path reaches either drops it or produces
   zeros; MANDALA's message layers keep only reachable outputs, while
   :class:`e3nn.o3.Linear` maps unmatched outputs to zero.

Equivariant building blocks
---------------------------

Every learnable map in a MANDALA network belongs to one of a few families,
each equivariant for a simple reason:

* **Linear maps between irreps** (``o3.Linear``). By Schur's lemma an
  equivariant linear map only mixes copies of the *same* irrep, with one
  weight matrix per irrep type and a bias only on ``0e``.
* **Tensor products** with invariant weights, as above.
* **Invariant scalings.** Multiplying a whole copy by an invariant number
  (a gate, an attention weight, an envelope, a normalization factor)
  commutes with :math:`D^{(\ell,p)}(Q)` because it is a multiple of the
  identity on that copy.
* **Pointwise functions of invariants.** Any function may be applied to
  ``0e`` channels; odd functions may be applied to ``0o`` channels.
* **Gather and scatter over graph indices.** Permuting entities commutes with
  acting on their features, and summing equivariant vectors is equivariant.

Compositions of equivariant maps are equivariant, which is why a pipeline of
validated stages is guaranteed to be equivariant as a whole.

.. _operator-law:

How electronic-structure operators transform
--------------------------------------------

Let atom :math:`i` carry atomic orbitals
:math:`\phi_{i n \ell m}(\bm r)=R_{n\ell}(|\bm r-\bm R_i|)\,Y^\ell_m(\widehat{\bm r-\bm R_i})`.
Under :math:`g=(Q,\bm t)` the orbitals of a shell of degree :math:`\ell`
mix through :math:`D^{(\ell,(-1)^\ell)}(Q)`. Writing
:math:`\mathcal D_i(Q)=\bigoplus_{\text{shells of }i}D^{(\ell,(-1)^\ell)}(Q)`
for the block-diagonal action on all orbitals of atom :math:`i`, every
one-electron operator block obeys

.. math::

   X_{ij}^{\bm L}\;\longmapsto\;\mathcal D_i(Q)\,X_{ij}^{\bm L}\,\mathcal D_j(Q)^{\top},
   \qquad X\in\{H,S,D\}.

The block of a pair is therefore an element of the tensor product of the two
orbital representations. Combined with the Clebsch–Gordan decomposition, this
is the statement that a block is a *direct sum of irreps* — the central fact
that turns operator learning into irrep regression. It also yields two
consequences used throughout MANDALA:

* **Hermiticity.** :math:`H_{ji}^{-\bm L}=\big(H_{ij}^{\bm L}\big)^{\dagger}`.
  Both directed blocks are predicted and can be projected onto the Hermitian
  subspace after prediction (:doc:`operators`).
* **Inversion.** Under :math:`Q=-\mathbb 1`, a block between shells
  :math:`\ell_1,\ell_2` picks up :math:`(-1)^{\ell_1+\ell_2}`; the parity of
  every coupled irrep is fixed accordingly, which is why hidden features must
  contain both parities when *p*–*d* blocks are predicted.

Spin–orbit coupled operators act on spinors and additionally carry the
spin-½ representation; their blocks are complex and transform with
:math:`\mathcal D_i\otimes U(Q)`. The shell codec (``codec.spinful: true``)
includes this action.

.. admonition:: Verifying the law numerically
   :class: contract

   ``examples/layer_cookbook/verify.py`` rotates and inverts a periodic
   structure (cell included), predicts :math:`H` with a complete model and
   compares every block with
   :math:`\mathcal D_i(Q)H_{ij}\mathcal D_j(Q)^\top`, before and after a
   training step. The same check is applied by the physics test-suite to the
   reference presets.
