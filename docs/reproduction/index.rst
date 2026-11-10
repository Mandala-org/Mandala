Reproducing the paper
=====================

The MANDALA paper demonstrates the framework on three materials and reports
four controlled ablations and an inference-scaling benchmark. This section
maps every result to the files and commands that produce it.

.. list-table::
   :header-rows: 1
   :widths: 34 34 32

   * - Result
     - Produced by
     - Section
   * - ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2` Hamiltonian, correlation
     - ``scripts/reproduce_examples/zncusnses.json``
     - :doc:`models`
   * - SiO\ :sub:`2` Hamiltonian, band energy, DOS
     - ``scripts/reproduce_examples/sio2.json``
     - :doc:`models`
   * - Silicon H, D, S, band energy, bands, DOS, density
     - ``scripts/reproduce_examples/silicon.json``
     - :doc:`models`
   * - Energy, spectral, envelope and aggregation ablations
     - ``sweeps/paper_ablations``
     - :doc:`ablations`
   * - Inference scaling to 4096 atoms
     - ``scripts/*silicon_cif_scaling*.py``
     - :doc:`ablations`

Reference data
--------------

All operators were computed with OpenMX 3.9 (PBE, norm-conserving PBE19
pseudopotentials, non-spin-polarized, SCF threshold :math:`10^{-8}` Hartree):

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Dataset
     - Structures
     - Basis, grid, k-mesh, model cutoff
   * - ZnCu\ :sub:`2`\ Sn(SeS)\ :sub:`2`
     - 100 perturbed 32-atom cells
     - Zn10.0S-s2p2d2, Cu10.0S-s2p2d2, Sn7.0-s2p2d3f1, Se9.0-s4p3d3f2,
       S9.0-s4p3d3f2; 400 Ry; :math:`11^3`; 11 Å
   * - Amorphous SiO\ :sub:`2`
     - 165 MD configurations of 150 atoms (100 used)
     - Si9.0-s2p2d1, O7.0-s2p2d1; 600 Ry; :math:`\Gamma`; 10 Å
   * - Perturbed silicon
     - 96 eight-atom diamond cells (80 train / 16 validation), :math:`\sigma=0.02` Å
     - Si7.0-s2p2d1; 400 Ry; :math:`8^3`; 8 Å

Each structure is one directory containing ``HS.out`` and the OpenMX output
file. The training launchers discover structures in these layouts:

.. code-block:: text

   ZnCuSnSeS/scale_1/<structure>/{HS.out, ZnCuSeS.out}     dataset_kind: ZnCuSnSeS
   SiOx/<structure>/{HS.out, SiO2.out}                     dataset_kind: siox
   perturbed_Si/scale_1/<structure>/{HS.out, Si.out}       dataset_kind: silicon_scales

Environment
-----------

Install MANDALA as in :doc:`../getting_started/installation` with the CUDA
stack. The paper models were trained in single precision on one NVIDIA GPU
per model (B200 for the scaling benchmark). Training uses the deterministic
settings of the launchers (``PYTHONHASHSEED``,
``CUBLAS_WORKSPACE_CONFIG=:4096:8``, zero data-loader workers, fixed split and
training seeds).

Workflow overview
-----------------

.. graphviz::

   digraph repro {
     rankdir=LR; bgcolor="transparent";
     node [shape=box, style="rounded,filled", fillcolor="#f3f7fb", color="#315b7d", fontname="Helvetica", fontsize=10];
     edge [color="#5a8fbd"];
     data [label="OpenMX datasets"];
     recipe [label="frozen recipe\nscripts/reproduce_examples/*.json"];
     train [label="train_*.sh\n45 h segments,\nepoch-boundary resume"];
     ckpt [label="best_model.pt"];
     metrics [label="validation metrics\n(tables)"];
     figs [label="figures\n(blocks, correlation,\nDOS, bands, density)"];
     sweeps [label="ablation sweeps\n5 seeds × 2 settings"];
     plots [label="paper/generate_result_plots.py"];
     data -> train; recipe -> train; train -> ckpt; ckpt -> metrics; ckpt -> figs;
     data -> sweeps -> plots;
   }

.. toctree::
   :maxdepth: 1

   models
   ablations
