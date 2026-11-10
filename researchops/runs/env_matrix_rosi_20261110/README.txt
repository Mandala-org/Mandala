Follow-up environment verification, 2026-11-10, referee-r8-r9-v2.
verify_environment.py records every command, exit code, wall time and output tail.
Full local logs and JUnit XML are under ignored .venvs/logs/<environment>/.
CUDA source builds request 8.6 (A40), 9.0 (H100), 10.0 (B200).
NVIDIA CUDA 13 PyPI packages use un-suffixed names; nvidia-cuda-nvcc-cu13
is a placeholder, so the actual compiler is nvidia-cuda-nvcc==13.4.92.
No cross-worktree environments are modified.

The first H100 allocation revealed kernel driver 570.158.01 (CUDA 12.8),
not a CUDA 13 driver. The retry extracts NVIDIA cuda-compat-13-4 userspace
libraries (615.71.10) into .venvs and sets LD_LIBRARY_PATH for this job only.
This combination is listed as compatible in NVIDIA's forward compatibility
matrix. install_cuda_compat.py pins and checks the Debian archive SHA256.
No system driver or administrator-owned files are changed.

Python 3.15 has no h5py 3.16.0 wheel. prepare_hdf5_terok.sh extracts
Debian HDF5 1.14.5 development/runtime packages below .venvs/hdf5 and
builds h5py locally. HDF5_INCLUDEDIR, HDF5_LIBDIR and LD_LIBRARY_PATH
are captured by the subsequent Python 3.15 verification record.
