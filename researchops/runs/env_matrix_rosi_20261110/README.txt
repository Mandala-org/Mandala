Follow-up environment verification, 2026-11-10, referee-r8-r9-v2.
verify_environment.py records every command, exit code, wall time and output tail.
Full local logs and JUnit XML are under ignored .venvs/logs/<environment>/.
CUDA source builds request 8.6 (A40), 9.0 (H100), 10.0 (B200).
NVIDIA CUDA 13 PyPI packages use un-suffixed names; nvidia-cuda-nvcc-cu13
is a placeholder, so the actual compiler is nvidia-cuda-nvcc==13.4.92.
No cross-worktree environments are modified.
