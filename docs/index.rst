MANDALA
=======

**MANDALA** learns the electronic structure of molecules and materials as
*operators*. Given atomic positions, species and (for crystals) a lattice, an
E(3)-equivariant graph neural network predicts the sparse, atom-pair-resolved
Hamiltonian :math:`H`, overlap :math:`S` and density matrix :math:`D` in a
localized atomic-orbital basis. Observables are computed from those operators
rather than by separate surrogate models:

.. math::

   N_e = \Tr(DS), \qquad E_b = \Tr(DH), \qquad
   H(\bm k)\,\bm c = \varepsilon\, S(\bm k)\,\bm c .

Because these quantities remain differentiable functions of the network
output, they can also *guide training*: band energies, electron counts and
eigenvalue spectra near the Fermi level enter the loss next to the matrix
elements themselves.

.. graphviz::

   digraph overview {
     rankdir=TB; bgcolor="transparent"; nodesep=0.5; ranksep=0.45;
     node [shape=box, style="rounded,filled", fillcolor="#f3f7fb", color="#315b7d", fontname="Helvetica", fontsize=11];
     edge [color="#5a8fbd", fontname="Helvetica", fontsize=9];
     { rank=same; dft; snap; pgraph; }
     { rank=same; net; irr; ops; obs; }
     dft [label="DFT output\nOpenMX · FHI-aims · PySCF\nperiodic HDF5"];
     snap [label="Snapshot\nBlockMatrix H, S, D\npositions, lattice"];
     pgraph [label="periodic graph\nedges (i, j, L)\nradial + angular features"];
     net [label="E3GNN\ntyped pipeline of\nequivariant layers", fillcolor="#f2faf6", color="#2f8f6b"];
     irr [label="irrep vectors\nper atom pair"];
     ops [label="sparse operators\nĤ, Ŝ, D̂"];
     obs [label="observables\nTr(DH), Tr(DS),\nbands, DOS"];
     dft -> snap [label="parse +\nbasis convention"];
     snap -> pgraph;
     pgraph -> net;
     net -> irr;
     irr -> ops [label="Q^T (Clebsch-Gordan)"];
     ops -> obs;
     obs -> net [style=dashed, label="observable guidance", constraint=false];
   }

What makes MANDALA distinctive
------------------------------

* **Operators, not numbers.** Hamiltonian, overlap and density blocks are
  predicted jointly from one equivariant latent representation and keep their
  exact transformation law under rotations, reflections and translations.
* **Physics in the loss.** Sparse trace contractions and differentiable
  generalized eigensolvers turn :math:`\Tr(DH)`, :math:`\Tr(DS)` and band
  eigenvalues into training objectives.
* **One typed pipeline for every architecture.** The default MANDALA network,
  DeepH, DeepH-E3, MACE-H and arbitrary hybrids are configurations of the same
  stage compiler. Each stage declares the entities, irreducible
  representations, coordinate frame and support it reads and writes; invalid
  compositions are rejected before allocation.
* **Extensible by registration.** A new layer is an ``nn.Module`` plus a
  factory that states its contract. The :doc:`layer cookbook <extending/index>`
  implements a new member of every layer family and certifies each one.
* **Reproducible by construction.** Runs freeze the resolved configuration,
  input hashes, split membership and fitted statistics; continuation resumes
  exactly at epoch boundaries.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart
   getting_started/first_model

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/data
   user_guide/configuration
   user_guide/training
   user_guide/inference

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/symmetry
   theory/operators
   theory/architecture
   theory/objectives

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/pipeline
   architecture/components

.. toctree::
   :maxdepth: 2
   :caption: Extending MANDALA

   extending/index
   extending/radial_basis
   extending/node_encoder
   extending/edge_encoder
   extending/convolution
   extending/aggregation
   extending/nonlinearity
   extending/normalization
   extending/mlp_block
   extending/residual
   extending/edge_update
   extending/readout
   extending/envelope
   extending/assembling

.. toctree::
   :maxdepth: 2
   :caption: Reproducing the paper

   reproduction/index
   reproduction/models
   reproduction/ablations

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/api
   reference/configuration
   reference/glossary
