Installation
============

MANDALA supports Python 3.10–3.13 and uses PyTorch, e3nn, torch-scatter and
PyTorch Lightning. The commands below create an isolated environment with the
tested PyTorch 2.5.1 stack.

Requirements
------------

* Python >=3.10,<3.14
* A C/C++ toolchain (only when torch-scatter has to be built from source)
* `Git LFS <https://git-lfs.com>`_ for the example electronic-structure data
* Optional: an NVIDIA GPU with a CUDA 12.1 compatible driver
* Optional, for building this documentation: Graphviz (``dot``)

Clone the repository
--------------------

.. code-block:: bash

   git clone git@github.com:Mandala-org/Mandala.git mandala
   cd mandala
   git lfs install
   git lfs pull

``git lfs pull`` fetches the bundled OpenMX, FHI-aims and PySCF example
calculations under ``data/``; they are used by the quickstart and the tests.

Create the environment
----------------------

CPU
~~~

.. code-block:: bash

   python3.10 -m venv mandala-venv
   source mandala-venv/bin/activate
   python -m pip install --upgrade pip wheel
   python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html --no-build-isolation
   python -m pip install -e '.[dev]'

CUDA 12.1
~~~~~~~~~

.. code-block:: bash

   python3.10 -m venv mandala-venv
   source mandala-venv/bin/activate
   python -m pip install --upgrade pip wheel
   python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   python -m pip install -e '.[dev]'

Only the graph stack (``torch-scatter`` and ``torch_geometric``) must be
selected for the exact PyTorch build and CUDA wheel tag; the remaining
dependencies follow pip resolution. ``torch-scatter`` contains compiled
kernels, while PyG itself can use its platform-independent wheel. PyTorch
CUDA wheels supply their runtime; a local toolkit is needed only to build
CUDA extensions from source. Record the driver and installed versions with
each run; MANDALA writes package versions into every run manifest.

Tested environments
-------------------

Setups on which MANDALA has run its test suite and a GPU training step. The
list is generated from ``docs/data/tested_environments.json`` with
``python scripts/report/render_tested_environments.py``; the JSON keeps the
full records.

.. include:: tested_environments_table.rst.inc

Notes:

* Only ``torch-scatter`` (and ``torch_geometric``) must match the exact
  PyTorch build and CUDA tag; everything else follows pip resolution.
* CUDA 13 wheels need an NVIDIA driver of the 580 family or NVIDIA's
  ``cuda-compat`` libraries on older drivers; on CUDA 12 drivers use the
  ``cu128`` wheels.
* Python 3.15 passes the CPU test suite, but no CUDA-enabled PyTorch wheel
  exists for it yet, so the package metadata stays at ``<3.15``.

Import layout
-------------

The library lives in ``src/`` as the top-level packages ``core``, ``data``,
``net``, ``analysis`` and ``utils``; command-line workflows live in
``scripts/``. All commands in this documentation are run from the repository
root with both directories on the import path:

.. code-block:: bash

   export PYTHONPATH=src:.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Package
     - Contents
   * - ``core``
     - Orbital/irrep bookkeeping (``OrbitalIrrepConfig``, ``BlockIrrepMapper``,
       ``ShellTensorCodec``), basis converters, sparse trace algebra and
       periodic Fourier transforms.
   * - ``data``
     - Parsers, ``Snapshot``/``BlockMatrix``/``IrrepsBlockData``, graph
       construction, datasets, envelopes and exporters.
   * - ``net``
     - ``E3GNN``, the typed ``FeaturePipeline``, all layers and
       transformations, objectives, spectral loss and checkpoint utilities.
   * - ``analysis``
     - Evaluation reports, density-of-states and band-structure analysis,
       real-space density grids and sweep analysis.
   * - ``scripts``
     - Training, export, evaluation and benchmarking command-line tools.

Verify the installation
-----------------------

The layer-cookbook certificate exercises e3nn, torch-scatter, the pipeline
compiler, the orbital codec and a complete model in about twenty seconds on a
laptop CPU:

.. code-block:: bash

   PYTHONPATH=src:. python examples/layer_cookbook/verify.py

It ends with::

   All cookbook layers and the cookbook model are E(3)-equivariant.

A focused selection of the scientific contract tests runs with pytest:

.. code-block:: bash

   PYTHONPATH=src:. python -m pytest -q \
     tests/physics/test_o3_model_contracts.py \
     tests/unit/net/test_hybrid_pipelines.py

Test markers select test families (``unit``, ``physics``, ``equivariance``,
``workflow``, ``integration``, ``slow``, ``gpu``, ``large_data``). GPU tests
run when ``MANDALA_RUN_GPU_TESTS=1`` is set, large-data tests when
``MANDALA_LARGE_DATA_ROOT`` points to the corresponding datasets.

Build this documentation
------------------------

The documentation is built inside the MANDALA environment, because the API
pages and the generated references import the package. With the environment
active (and Graphviz's ``dot`` on the path for the diagrams):

.. code-block:: bash

   python -m pip install sphinx sphinx-rtd-theme
   PYTHONPATH=src:. python -m sphinx -b html docs docs/_build/html

The configuration reference, the irrep atlas and the transformation catalogue
are generated from the source code during the build, so they always describe
the installed version.
