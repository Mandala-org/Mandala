Ablations and scaling
=====================

Protocol
--------

Every ablation compares two or three settings of *one* configuration field
across matched seeds, with all other settings, the dataset membership
(``data-split-seed 42``) and the 12-hour wall-clock budget held fixed. The
sweeps are W&B grid sweeps over ``(setting, seed)`` executed by
``scripts/wandb_run.py``; ``ablation-setting-from`` names the varied field so
that every run is labelled ``<ablation>_<setting>_seed<seed>``. Each run ends
with one held-out test evaluation.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Ablation
     - Varied field and settings
     - Sweep files (``sweeps/paper_ablations/``)
   * - SiO\ :sub:`2` energy guidance
     - ``loss_coef_observables``: 0, :math:`10^{-4}`, :math:`10^{-3}`
     - ``paper_siox_mature_energy_guidance_12h.yaml``,
       ``paper_siox_mature_energy_guidance_3em5_12h.yaml``,
       ``paper_additional_seeds_siox_mature_energy_guidance_12h.yaml``
   * - ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2` spectral guidance
     - ``spectral_loss_coef``: 0, :math:`10^{-3}` (:math:`2\times2\times2`, :math:`W=10` eV, :math:`\tau=2` eV, Huber 0.1 eV)
     - ``paper_zncusnses_mature_full_network_spectral_12h.yaml``,
       ``paper_additional_seeds_zncusnses_mature_full_network_spectral_12h.yaml``
   * - ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2` distance envelope
     - ``hamiltonian_envelope_mode``: off, multiply_prediction
     - ``paper_zncusnses_envelope_12h.yaml``,
       ``paper_additional_seeds_zncusnses_envelope_12h.yaml``
   * - Silicon node aggregation
     - ``node_update_message_agg``: average, attention
     - ``paper_silicon_node_aggregation_12h.yaml``,
       ``paper_additional_seeds_silicon_node_aggregation_12h.yaml``

The energy- and spectral-guidance studies ("mature") fine-tune the selected
demonstration models with each setting (``resume-from-checkpoint``,
``compatibility: true``), measuring the effect of the objective on an
already accurate network. The envelope and aggregation studies train from
scratch.

Running a sweep
---------------

1. Edit the dataset, cache and checkpoint paths (``data-path``,
   ``snapshot-cache-dir``, ``checkpoint-dir`` and, for mature studies,
   ``resume-from-checkpoint``) in the sweep file, or regenerate all sweep files
   from the shared matrix after editing it:

   .. code-block:: bash

      python sweeps/paper_ablations/generate_sweeps.py

2. Register the sweep and start agents until its ``run_cap`` is reached:

   .. code-block:: bash

      wandb sweep sweeps/paper_ablations/paper_silicon_node_aggregation_12h.yaml
      wandb agent <entity>/<project>/<sweep-id>

3. Check the paired design (every seed present for every setting):

   .. code-block:: bash

      PYTHONPATH=src:. python scripts/report/aggregate_paper_ablation.py \
        --entity <entity> --project paper_silicon_node_aggregation_12h \
        --group paper_silicon_node_aggregation_12h --baseline average \
        --expected-seeds 5 --output silicon_node_aggregation.json

4. Export the run summaries and draw the figures:

   .. code-block:: bash

      python paper/download_ablation_csvs.py
      python paper/generate_result_plots.py

``paper/data/*.csv`` contains the summaries of the published runs (seed,
setting, Hamiltonian MAE, observable MAE, split), so step 4 reproduces the
published figures directly from them.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Ablation
     - Effect (medians over matched seeds)
   * - Energy guidance, SiO\ :sub:`2` (test)
     - :math:`10^{-4}`: band-energy MAE 0.245 → 0.078 eV/atom (−68 %),
       Hamiltonian MAE +4.7 %; :math:`10^{-3}` is worse than :math:`10^{-4}`
       on both.
   * - Spectral guidance, ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2` (test, 10 seeds)
     - Eigenvalue MAE 0.579 → 0.0391 eV, paired mean improvement 92.1 %, all
       seeds improve; Hamiltonian MAE unchanged within its confidence interval.
   * - Distance envelope, ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2` (test)
     - Hamiltonian MAE −3.8 %, four of five seeds improve.
   * - Attention vs. average, silicon (validation)
     - Hamiltonian MAE −10.1 %.

.. figure:: ../../paper/figures/result_zncusnses_spectral_guidance_ablation.png
   :width: 100%

   Held-out Hamiltonian and eigenvalue MAE for ten matched seeds with and
   without spectral guidance.

.. figure:: ../../paper/figures/result_siox_energy_guidance_ablation.png
   :width: 100%

   Held-out Hamiltonian and per-atom band-energy MAE for the SiO\ :sub:`2`
   energy-guidance study.

Inference scaling
-----------------

Target-free inference on diamond-silicon supercells of 8 to 4096 atoms with
the silicon model:

.. code-block:: bash

   PYTHONPATH=src:. python scripts/generate_silicon_scaling_cifs.py --output-dir cifs/
   PYTHONPATH=src:. python scripts/benchmark_silicon_cif_scaling.py \
     --checkpoint /runs/publication/silicon/best_model.pt \
     --structures-dir cifs/ --output scaling.txt --device cuda \
     --warmup-runs 2 --atom-counts 8,64,512,4096
   PYTHONPATH=src:. python scripts/plot_silicon_cif_scaling.py \
     --input scaling.txt --output-dir figures/

Replace the checkpoint path with your trained silicon model. The CIFs and
text timing log are generated by the preceding commands; the benchmark writes
a text log, not JSON. ``--atom-counts`` selects the published range explicitly
(the benchmark's default also includes 32768 atoms).

The benchmark separately times graph construction, the forward pass and
block reconstruction (two warm-up calls, ten timed repetitions) and records
the peak memory of each phase. On one NVIDIA B200 GPU the total time grows
from 0.364 s (8 atoms) to 97.0 s (4096 atoms; 1 216 512 blocks), with the
forward pass taking 3.17 s of that at the largest size.

.. figure:: ../../paper/figures/results/silicon_cif_scaling_time_total.png
   :width: 60%

   Total inference wall time versus number of atoms.
