Training objectives
===================

MANDALA trains on any combination of matrix elements and quantities derived
from the predicted operators. For predicted operators
:math:`\mathcal M\subseteq\{H,S,D\}` (``matrix_targets``) the objective is

.. math::

   \mathcal L=\sum_{X\in\mathcal M}\lambda_X\,\mathcal L_X^{\text{matrix}}
   +\lambda_{\text{obs}}\big(\mathcal L^{E}+\mathcal L^{N}\big)
   +\lambda_{\text{spec}}\,\mathcal L^{\text{spec}}
   +\lambda_1\|\theta\|_1+\lambda_2\|\theta\|_2^2 .

Every term is a differentiable function of the network parameters
:math:`\theta`, so gradients from observables flow back through the sparse
contractions and eigensolvers into every layer.

Matrix terms
------------

For the orbital-block representation,

.. math::

   \mathcal L_X^{\text{matrix}}=\eta\,\mathrm{MAE}(X^{\text{pred}},X^{\text{ref}})
   +(1-\eta)\,\mathrm{MSE}(X^{\text{pred}},X^{\text{ref}}),
   \qquad \eta=\texttt{loss\_l1\_fraction}.

The reduction over blocks is set by ``matrix_loss_reduction``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Reduction
     - Definition
   * - ``pair_sum``
     - Sum over ordered species pairs of the mean error within each pair.
   * - ``pair_mean``
     - Mean over species pairs of the same per-pair means.
   * - ``elements``
     - Every observed matrix entry of the structure weighted equally.

``pair_sum`` gives each chemical channel equal weight regardless of how many
blocks it has, which keeps rare element pairs from being drowned out in
multicomponent systems; ``elements`` matches the conventional element-wise
MAE that is reported as ``*_elementwise`` metrics.

Further controls:

* ``train_target: irreps`` supervises irrep coefficients instead of blocks;
  ``train_on_irrep_parts`` adds per-irrep matrix contributions.
* ``partial_train`` restricts supervision to ``diag`` (onsite),
  or ``offdiag`` (all non-onsite pairs) blocks. The former ``shifted_self``
  option is retained only when evaluating historical checkpoints.
* ``loss_weighting_mode`` re-weights edges: ``envelope_inverse_clipped``
  (:math:`w=1/E`), ``envelope_inverse_sqrt_clipped`` (:math:`w=E^{-1/2}`) or
  ``distance_short_range_bias`` (:math:`w=(1+r)^{-1/2}`), clipped to
  [``loss_weight_min``, ``loss_weight_max``]. Inverse-envelope weights
  equalize the influence of weak long-range blocks.

The packed-shell representation uses the masked objective declared in the
recipe's ``objective`` section, e.g. ``{kind: mse, epoch_weight: elements}``
(``mse`` or ``mae``; epochs weighted by matrix elements or edges). Entries that
are absent or non-finite are masked out of both the loss and fitted
statistics.

Observable guidance
-------------------

With :math:`E_b=\Tr(DH)` and :math:`N_e=\Tr(DS)` evaluated sparsely,

.. math::

   \mathcal L^{E}=\rho\big(E_b^{\text{pred}},E_b^{\text{ref}}\big),\qquad
   \mathcal L^{N}=\rho\big(N_e^{\text{pred}},N_e^{\text{ref}}\big),\qquad
   \rho\in\{\text{MAE},\text{MSE}\}\ (\texttt{observable\_loss\_kind}),

enabled by ``train_on_energy`` and ``train_on_num_electrons`` with weight
``loss_coef_observables``. Two routes exist:

**Fully predicted.** :math:`E_b^{\text{pred}}=\Tr(D^{\text{pred}}H^{\text{pred}})`;
requires predicting both operators.

**Reference partner** (``train_observables_on_gt: true``). Each predicted
operator is contracted with the *reference* partner and the two terms are
averaged,

.. math::

   \mathcal L^{E}=\tfrac12\Big[\rho\big(\Tr(D^{\text{ref}}H^{\text{pred}}),E_b^{\text{ref}}\big)
   +\rho\big(\Tr(D^{\text{pred}}H^{\text{ref}}),E_b^{\text{ref}}\big)\Big],

using whichever operators are predicted. This isolates the error of each
operator and lets a Hamiltonian-only model be guided by the band energy.

.. admonition:: Why guidance helps
   :class: theorem

   A matrix loss treats all entries as independent. The band energy is a
   *weighted* sum, :math:`\Tr(D^{\text{ref}}H)=\sum D^{\text{ref}}_{ji}H_{ij}`,
   which emphasizes exactly the entries that the occupied subspace is
   sensitive to. In the SiO\ :sub:`2` ablation a coefficient of
   :math:`10^{-4}` reduced the median per-atom band-energy error by 68 % at
   a 4.7 % cost in Hamiltonian MAE (:doc:`../reproduction/ablations`).

Spectral guidance
-----------------

For periodic systems the spectrum near the Fermi level is supervised
directly. On a uniform mesh :math:`\bm q_{n_1n_2n_3}=(n_1/N_1,n_2/N_2,n_3/N_3)`
(``spectral_loss_kmesh``) MANDALA assembles :math:`H(\bm q)` from the
predicted blocks and :math:`S(\bm q)` from the *reference* overlap, then solves

.. math::

   \varepsilon(\bm q)=\operatorname{eigvalsh}\big(L^{-1}H(\bm q)L^{-\dagger}\big),
   \qquad S(\bm q)=LL^\dagger .

Bands are matched by index. With the reference Fermi level :math:`E_F`,
half-width :math:`W` (``spectral_loss_window_ev``) and taper :math:`\tau`
(``spectral_loss_taper_ev``), each reference band gets the weight

.. math::

   w_{\bm qb}=\begin{cases}
   1 & |\varepsilon^{\text{ref}}_{\bm qb}-E_F|\le W-\tau\\
   \tfrac12\Big[1+\cos\Big(\pi\,\frac{|\varepsilon^{\text{ref}}_{\bm qb}-E_F|-(W-\tau)}{\tau}\Big)\Big] & W-\tau<|\cdot|\le W\\
   0 & \text{otherwise}
   \end{cases}

and

.. math::

   \mathcal L^{\text{spec}}=\frac{\sum_{\bm q,b}w_{\bm q}w_{\bm qb}\,
   \rho_\delta\big(\varepsilon^{\text{pred}}_{\bm qb}-\varepsilon^{\text{ref}}_{\bm qb}\big)}
   {\sum_{\bm q,b}w_{\bm q}w_{\bm qb}},

with :math:`\rho_\delta` a Huber (threshold ``spectral_loss_huber_delta_ev``),
absolute or squared error (``spectral_loss_kind``) and uniform
:math:`w_{\bm q}`. Using the reference overlap for both spectra attributes
the spectral error to the Hamiltonian alone. Reference factorizations,
spectra and Fermi levels can be cached (``spectral_fermi_cache_path``,
``scripts/precompute_spectral_fermi_cache.py``). In the
ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2` ablation this term reduced the median test
eigenvalue error from 0.579 eV to 0.039 eV with a neutral effect on the
Hamiltonian MAE.

Derivative objectives
---------------------

``model(x, physical_derivatives=True)`` keeps the graph geometry
differentiable. The band-energy derivatives

.. math::

   \bm F^{\text{band}}_I=-\frac{\partial E_b}{\partial\bm R_I},\qquad
   \sigma^{\text{band}}=\frac1\Omega\,h^\top\frac{\partial E_b}{\partial h},\qquad\Omega=\det h,

are available through :meth:`~net.e3gnn.E3GNN.get_forces` and
:meth:`~net.e3gnn.E3GNN.get_stress` (``enable_forces``, ``enable_stress``)
and as training terms (``train_on_forces``, ``train_on_stress`` with
``loss_coef_forces``, ``loss_coef_stress``). :math:`E_b` is the band energy
:math:`\Tr(DH)`, so these are derivatives of the band energy, which is the
quantity the operators define directly.

Fitted output statistics
------------------------

Pipelines may contain ``species_irrep_affine`` stages: per species pair and
per irrep field, a mean shift (scalars only) and a scale fitted exclusively on
the training split before optimization starts,

.. math::

   \bm x\mapsto\sigma\,\bm x+\mu .

``gradient_mode: identity`` computes the same forward value but passes the
coefficient gradient through unscaled,
:math:`\bm x+\operatorname{sg}\big[\bm x(\sigma-1)\big]+\mu`, which avoids
multiplying gradients by very large or small scales; ``physical`` uses the
exact affine derivative. The fitted buffers and the hash of the training
manifest are stored in every checkpoint.

Optimization and model selection
--------------------------------

The flat interface uses AdamW with gradient clipping (``grad_clip_val``),
a plateau scheduler on ``lr_scheduler_target`` and model selection on
``checkpoint_monitor``. ``revert_on_spike`` restores the best weights and
decays the learning rate when the monitored loss jumps by more than
``revert_spike_factor``. Recipes declare their optimizer and scheduler in
``optimization`` (``adam``/``adamw``; ``plateau`` with optional EMA smoothing,
or ``milestone_decay``). ``training.min_lr`` ends a run once the learning rate
falls below it.
