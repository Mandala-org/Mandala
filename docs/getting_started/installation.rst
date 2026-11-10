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

The table is generated from ``docs/data/tested_environments.json``. Records
marked ``reported`` describe earlier use supplied by the author, without a
fresh test run or an inferred pass count. Regenerate it with:

.. code-block:: bash

   python scripts/report/render_tested_environments.py

.. include:: tested_environments_table.rst.inc

Known version notes
~~~~~~~~~~~~~~~~~~~

* Python 3.14.10 passes the CPU contracts (701 passed, 68 skipped) and both
  GPU smoke tests with PyTorch 2.11.0+cu128 and the matching PyG scatter wheel
  on the A40, driver 550.163.01. PyTorch warns that TorchScript is deprecated
  on Python 3.14; the tested e3nn model still passes forward and backward.
* Python 3.15 has no cu128 torch wheel on the queried index (2026-11-10).
  This is a wheel availability limit; no cu128 tests ran for Python 3.15.

* Python 3.15.2 with PyTorch 2.15.0+cu134 passes the same CPU selection
  (701 passed, 68 skipped). h5py 3.16.0 has no cp315 wheel: it was built
  from source against locally extracted HDF5 1.14.5 libraries. Scatter
  was also built from source with CUDA 13.4. GPU execution on Python 3.15
  is not verified, so package metadata retains the ``<3.15`` upper bound.
* The verified CPU selection excludes GPU, slow, large-data, network and
  integration tests. It also excludes the large silicon spatial fixture and
  unshipped generated PySCF reference files. The 68 skips require optional
  upstream source checkouts; exact commands and results are in the JSON.
* PyTorch 2.7.1+cu128 runs the model and compiled scatter CUDA kernels on
  the A40 with driver 550.163.01 (advertised CUDA 12.4), without a launch error.
  This is a smoke check, not a guarantee for all kernels or architectures.
* CUDA 13.x wheels require driver family 580 or newer under NVIDIA's
  `minor-version compatibility rules
  <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>`_.
  With driver 550.163.01 (advertised CUDA 12.4), use the verified cu128
  wheels; upgrading Python alone does not resolve the driver limit.
* Native checkpoints contain Python Config objects. The loader explicitly
  retains trusted-payload loading across the PyTorch 2.6 ``weights_only``
  default change; load only checkpoints whose source you trust.
* The newest stable PyTorch selected on 2026-11-10 is 2.15.0, with cu134
  wheels for Python 3.13, 3.14 and 3.15. There is no matching PyG scatter
  wheel index. The follow-up probe builds scatter with nvcc 13.4.92,
  matching CUDA CRT/NVVM 13.4.92 and architectures 8.6, 9.0 and 10.0.
  Unpinned CRT/NVVM resolve to 13.5 and produce PTX 9.5 that the 13.4
  assembler rejects; both compiler components must be pinned.
* The earlier Terok cu134 failures remain in the table and JSON. Those
  records used CPU-only scatter after a CUDA 12.4 compiler mismatch.
  Their reproduction-setup CPU failure came from an incomplete CUDA mock;
  the follow-up isolates cuDNN runtime metadata as well as GPU metadata.
* ROSI's H100 has kernel driver 570.158.01, which also cannot initialize
  cu134 directly. The follow-up GPU probe uses NVIDIA's isolated
  ``cuda-compat-13-4`` libraries (user-space driver 615.71.10) through
  ``LD_LIBRARY_PATH``. NVIDIA lists driver 570 as compatible with this
  `forward-compatibility package
  <https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html>`_.
  The system driver is unchanged; this qualification is part of the stack.
  Python 3.13.18 passes both GPU smoke tests and all four GPU-marked tests
  (including epoch-boundary resume) with CUDA scatter kernels on this H100.

* The earlier Terok / PyTorch 2.11 run required source patches ("with patches,
  see notes" in the table). Their diff was not supplied; that report does not
  establish support for an unmodified 2.11 installation. The new
  Python 3.14/cu128 row is fresh evidence on the current source revision.

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
