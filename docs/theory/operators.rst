Sparse operators and their irreps
=================================

Periodic block structure
------------------------

In a localized basis every one-electron operator of a crystal is a family of
dense blocks between the orbitals of atom :math:`i` in the home cell and atom
:math:`j` in the cell translated by :math:`\bm L\in\mathbb Z^3`:

.. math::

   X_{ij}^{\bm L} = \big[\langle \phi_{i\mu}\,|\,\hat X\,|\,\phi_{j\nu}(\cdot-\textstyle\sum_\alpha L_\alpha\bm a_\alpha)\rangle\big]_{\mu\nu},
   \qquad X\in\{H, S, D\}.

The *directed periodic edge* :math:`(i,j,\bm L)` labels each block. MANDALA
stores blocks in :class:`~data.block_matrix.BlockMatrix`, grouped by ordered
species pair (``"Si-O"``) and indexed by an integer array with rows
:math:`(L_x, L_y, L_z, i, j)`. Edges fall into three classes that the readout
treats separately:

.. math::

   \mathcal E_{\text{onsite}}=\{i=j,\ \bm L=0\},\qquad
   \mathcal E_{\text{shifted self}}=\{i=j,\ \bm L\neq0\},\qquad
   \mathcal E_{\text{offsite}}=\{i\neq j\}.

Both directions :math:`(i,j,\bm L)` and :math:`(j,i,-\bm L)` are stored and
predicted. A cutoff radius :math:`r_c` selects the edges with
:math:`\|\bm r_{ij}^{\bm L}\|\le r_c`; the same edge set defines the model
graph, so graph edges and operator blocks are in one-to-one correspondence.

Units and conventions
~~~~~~~~~~~~~~~~~~~~~

* Lengths in Å; Hamiltonians in Hartree internally, reported in eV.
  :math:`S` and :math:`D` are dimensionless.
* Orbitals are converted from the native order and sign convention of each
  electronic-structure code into the e3nn real-spherical-harmonic convention
  (``convention="e3nn"``) or kept in native OpenMX order (``basis: openmx``,
  used by the packed-shell representation). Converters exist for OpenMX,
  FHI-aims and PySCF.
* The density matrix is stored per spin channel for non-spin-polarized
  calculations: :math:`\Tr(DS)=N_e/2`.

From blocks to irreps
---------------------

By the :ref:`operator transformation law <operator-law>`, a block between
atoms with orbital representations
:math:`\mathcal B_i=\bigoplus_a D^{(\ell_a,(-1)^{\ell_a})}` and
:math:`\mathcal B_j=\bigoplus_b D^{(\ell_b,(-1)^{\ell_b})}` lives in
:math:`\mathcal B_i\otimes\mathcal B_j`. Clebsch–Gordan coupling of every pair
of shells gives an orthogonal change of basis :math:`Q_{\mathcal B_i\mathcal B_j}`:

.. math::

   \bm x_{ij}^{\bm L} = Q_{\mathcal B_i\mathcal B_j}\,\operatorname{vec}\!\big(X_{ij}^{\bm L}\big),
   \qquad
   \operatorname{vec}\!\big(X_{ij}^{\bm L}\big) = Q_{\mathcal B_i\mathcal B_j}^{\top}\,\bm x_{ij}^{\bm L}.

For a shell pair the rows of :math:`Q` are

.. math::

   \big(\bm x^{(\ell_a\ell_b)}\big)^{L}_{M}=\sum_{m_am_b}C^{LM}_{\ell_am_a\,\ell_bm_b}\,X_{(a m_a),(b m_b)},

and :math:`\bm x` transforms *irreducibly*:
:math:`\bm x\mapsto\bigoplus_L D^{(L,p)}(Q)\,\bm x`. This is the
Wigner–Eckart theorem read in reverse: the Clebsch–Gordan coefficients carry
all geometric dependence of the block, and the irrep coefficients are what is
left for the network to learn. :class:`~core.block_irrep_mapper.BlockIrrepMapper`
materializes :math:`Q` for every species pair; it is shared by datasets,
readouts and evaluation, so data and predictions are always expressed in the
same basis.

Irrep atlas
~~~~~~~~~~~

Shell pairs decompose as follows (generated from e3nn during the build):

.. irrep-atlas::
   :shells: s p d f

Complete atom-pair blocks of the bases used in the paper demonstrations,
computed with the MANDALA mapper (Si and O: ``s2p2d1``; Zn and Cu:
``s2p2d2``):

.. irrep-atlas::
   :species: O=2x0e+2x1o+1x2e Si=2x0e+2x1o+1x2e

.. irrep-atlas::
   :species: Cu=2x0e+2x1o+2x2e Zn=2x0e+2x1o+2x2e

The highest degree in a block is :math:`\ell_a+\ell_b`; a network that must
predict *d*–*d* blocks needs ``4e`` features, and *p*–*d* blocks need
odd-parity ``1o``, ``2o`` and ``3o`` features. Hidden irreps should be chosen
with these tables at hand.

The packed shell codec
~~~~~~~~~~~~~~~~~~~~~~

The packed-shell representation (``data.representation: packed_shells``) uses
:class:`~core.operator_layout.ShellTensorCodec`, a second implementation of
the same coupling that additionally supports spin–orbit coupled (complex,
spin-major) blocks and native OpenMX orbital order. It packs every species
pair into one global edge vector whose irreps,
``${layout.coupled_irreps}``, are the union of all shell-pair couplings, with
an explicit mask for entries absent in a given pair. Recipes refer to its
shape through the resolved ``layout`` section.

Sparse operator algebra
-----------------------

Traces of products never require dense matrices. With the reverse edge
:math:`(j,i,-\bm L)` of :math:`(i,j,\bm L)`,

.. math::

   \Tr(AB) = \sum_{(i,j,\bm L)} \operatorname{tr}\!\big(A_{ij}^{\bm L}\,B_{ji}^{-\bm L}\big)
   = \sum_{k}\sum_{n}\operatorname{tr}\!\big(A_{k,n}\,B_{\bar k,P_k(n)}\big),

where :math:`k` runs over species pairs, :math:`\bar k` is the reversed pair
and :math:`P_k` is a precomputed permutation aligning every edge with its
reverse. This gives, per spin channel,

.. math::

   N_e = \Tr(DS), \qquad E_b = \Tr(DH),

implemented by :func:`~core.sparse_math.trace_matmul_sparse_block_matrix`
and its aligned, vectorized variants.

Hermitian projection
~~~~~~~~~~~~~~~~~~~~

A network predicts the two directed blocks of a pair independently. The
Hermitian part is recovered by the projection

.. math::

   X_{ij}^{\bm L}\leftarrow\tfrac12\Big(X_{ij}^{\bm L}+\big(X_{ji}^{-\bm L}\big)^{\dagger}\Big),

applied to onsite blocks as :math:`\tfrac12(X+X^\dagger)`. It is an
orthogonal projection, so it never increases the Frobenius error with respect
to a Hermitian reference. Configuration controls where it is used:
``symmetrize_*_targets`` (targets), ``symmetrize_output`` (training
predictions) and ``symmetrize_validation_output`` (validation predictions).

Energy-gauge alignment
~~~~~~~~~~~~~~~~~~~~~~

A rigid shift of the Kohn–Sham potential changes :math:`H` by
:math:`\mu S` without changing any physics. For error *reporting* against a
reference, MANDALA can remove the component of the error along :math:`S`:

.. math::

   \widetilde H = H^{\text{pred}} - \mu_H S, \qquad
   \mu_H = \frac{\langle H^{\text{pred}}-H^{\text{ref}},\,S\rangle_F}{\langle S, S\rangle_F}.

Electron-count normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A predicted density can be rescaled to a prescribed electron count,

.. math::

   D \leftarrow \beta D, \qquad \beta = \frac{N_e^{\text{target}}}{\Tr(DS)},

enabled by ``rescale_density_to_num_electrons``.

Reciprocal space and band structures
------------------------------------

For a crystal momentum :math:`\bm k`,

.. math::

   X(\bm k)=\sum_{\bm L}e^{\,i\bm k\cdot\bm R_{\bm L}}X^{\bm L},
   \qquad \bm R_{\bm L}=\textstyle\sum_\alpha L_\alpha\bm a_\alpha,

and band energies solve :math:`H(\bm k)\bm c=\varepsilon S(\bm k)\bm c`.
MANDALA factorizes :math:`S(\bm k)=LL^\dagger` (Cholesky) and diagonalizes
:math:`L^{-1}H(\bm k)L^{-\dagger}`. The overlap must be positive definite;
it is when :math:`S` is kept on its full support. For truncated or predicted
overlaps the optional conditioning

.. math::

   S = U\operatorname{diag}(s_i)U^\dagger\ \longmapsto\
   U\operatorname{diag}\!\big(\max(s_i,10^{-6})\big)U^\dagger

restores a factorizable matrix (``psd_cleanup=True`` in
:meth:`~data.snapshot.Snapshot.get_band_structure`). Both the Fourier sum and
the eigensolver are differentiable, which makes the spectrum a training
signal (:doc:`objectives`).
