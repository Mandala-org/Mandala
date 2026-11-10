#!/bin/bash
# Local extraction only: no apt install, sudo, or system changes.
set -eu
cd "$(dirname "$0")/../../.."
mkdir -p .venvs/hdf5/packages
(
    cd .venvs/hdf5/packages
    apt-get download libhdf5-dev libhdf5-310 libhdf5-hl-310 libaec0 libsz2
    for package in *.deb; do dpkg-deb -x "$package" ..; done
)
export PATH="$HOME/.local/bin:$PATH"
export HDF5_INCLUDEDIR="$PWD/.venvs/hdf5/usr/include/hdf5/serial"
export HDF5_LIBDIR="$PWD/.venvs/hdf5/usr/lib/x86_64-linux-gnu/hdf5/serial"
export LD_LIBRARY_PATH="$PWD/.venvs/hdf5/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv pip install --python .venvs/py315-latest-cu13x/bin/python h5py==3.16.0
