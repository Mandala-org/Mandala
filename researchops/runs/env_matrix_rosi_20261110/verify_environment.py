"""Reproducible environment probe; run from the repository root (no login-node builds)."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import time
import tomllib
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True)
parser.add_argument("--python", required=True)
parser.add_argument("--torch", default="2.15.0")
parser.add_argument("--tag", default="cu134")
parser.add_argument("--gpu-only", action="store_true")
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()
root = Path.cwd()
run = root / "researchops/runs/env_matrix_rosi_20261110"
out = run / "results" / (args.env + ("-rosi" if args.gpu_only else "") + ".json")
logs = root / ".venvs/logs" / (args.env + ("-rosi" if args.gpu_only else ""))
logs.mkdir(parents=True, exist_ok=True)
out.parent.mkdir(parents=True, exist_ok=True)
venv = root / ".venvs" / args.env
py = str(venv / "bin/python")
env = os.environ.copy()
env.update(
    PYTHONPATH="src:.",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    WANDB_MODE="offline",
    MPLBACKEND="Agg",
    MAX_JOBS=str(args.workers),
)
previous = json.loads(out.read_text()) if out.exists() else {}
record = dict(
    name=args.env + ("-rosi" if args.gpu_only else ""),
    status="verified_20261110",
    date="2026-11-10",
    host=platform.node(),
    platform=platform.platform(),
    python=args.python,
    torch=args.torch + "+" + args.tag,
    cuda_wheel_tag=args.tag,
    source_revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    installation_commands=[],
    result="incomplete",
)

if previous:
    record["initial_attempts"] = previous.get("initial_attempts", [])
    if previous.get("result") in ["passed", "failed", "blocked"]:
        record["initial_attempts"].append(
            {
                k: v
                for k, v in previous.items()
                if k
                in [
                    "result",
                    "error",
                    "build_scatter_cuda",
                    "torch_scatter_build",
                    "cpu_tests",
                    "gpu_smoke",
                    "gpu_tests",
                ]
            }
        )

record["common_environment"] = {
    key: env[key]
    for key in [
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "WANDB_MODE",
        "MPLBACKEND",
        "MAX_JOBS",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "HDF5_INCLUDEDIR",
        "HDF5_LIBDIR",
        "MANDALA_CUDA_COMPAT",
    ]
    if key in env
}


def save():
    out.write_text(json.dumps(record, indent=2) + "\n")


def execute(label, command, extra=None):
    actual = env | (extra or {})
    start = time.monotonic()
    command_text = shlex.join(command)
    print(label + ": " + command_text, flush=True)
    with (logs / (label + ".log")).open("w") as stream:
        code = subprocess.call(
            command, stdout=stream, stderr=subprocess.STDOUT, env=actual
        )
    output = (logs / (label + ".log")).read_text(errors="replace")
    result = dict(
        command=command_text,
        environment=extra or {},
        exit_code=code,
        runtime_seconds=round(time.monotonic() - start, 3),
        output=output[-16000:],
    )
    print(label + ": exit " + str(code) + "\n" + output[-1800:], flush=True)
    record[label] = result
    if label.startswith(("install", "create", "build")):
        record["installation_commands"].append(command_text)
    save()
    return result


def require(result):
    if result["exit_code"]:
        record.update(result="blocked", error=result["output"])
        save()
        raise SystemExit(1)


if not (venv / "bin/python").exists():
    require(
        execute(
            "create",
            [
                "uv",
                "venv",
                "--seed",
                "--managed-python",
                "--python",
                args.python,
                str(venv),
            ],
        )
    )
require(
    execute(
        "install_torch",
        [
            "uv",
            "pip",
            "install",
            "--python",
            py,
            "torch==" + args.torch,
            "--index-url",
            "https://download.pytorch.org/whl/" + args.tag,
        ],
    )
)
scatter = execute(
    "install_scatter_wheel",
    [
        "uv",
        "pip",
        "install",
        "--python",
        py,
        "--reinstall-package",
        "torch-scatter",
        "torch-scatter",
        "torch_geometric",
        "--find-links",
        f"https://data.pyg.org/whl/torch-{args.torch}+{args.tag}.html",
        "--only-binary",
        "torch-scatter",
    ],
)
if scatter["exit_code"]:
    require(
        execute(
            "install_build_tools",
            ["uv", "pip", "install", "--python", py, "setuptools", "wheel", "ninja"],
        )
    )
    # CUDA 13 moved to un-suffixed NVIDIA PyPI distributions; cu13 is a placeholder.
    toolkit = execute(
        "install_cuda_toolkit",
        [
            "uv",
            "pip",
            "install",
            "--python",
            py,
            "nvidia-cuda-nvcc==13.4.92",
            "nvidia-cuda-runtime==13.4.92",
            "nvidia-nvvm==13.4.92",
            "nvidia-cuda-crt==13.4.92",
            "nvidia-cuda-cccl==13.3.5.0.6",
        ],
    )
    cuda = root / ".venvs" / ("cuda-" + args.env)
    cuda.mkdir(exist_ok=True)
    packages = next((venv / "lib").glob("python*/site-packages"))
    # Merge the split PyPI toolkit into CUDA_HOME without changing installed packages.
    for part in ["bin", "include", "lib", "lib64", "nvvm"]:
        dest = cuda / part
        dest.mkdir(exist_ok=True)
        for source in sorted(packages.glob("nvidia/**/" + part)):
            for child in source.iterdir():
                target = dest / child.name
                if not target.exists():
                    target.symlink_to(child)
    # Runtime wheels ship a versioned soname, but the linker needs libcudart.so.
    for child in (cuda / "lib").iterdir():
        target = cuda / "lib64" / child.name
        if not target.exists():
            target.symlink_to(child)
    cudart = cuda / "lib64/libcudart.so"
    if not cudart.exists():
        cudart.symlink_to(cuda / "lib64/libcudart.so.13")
    record["cuda_toolkit_files"] = [
        str(p.relative_to(packages)) for p in packages.glob("nvidia/**/nvcc")
    ]
    build_env = dict(
        CUDA_HOME=str(cuda),
        FORCE_CUDA="1",
        TORCH_CUDA_ARCH_LIST="8.6;9.0;10.0",
        PATH=str(cuda / "bin") + ":" + env["PATH"],
    )
    result = execute(
        "build_scatter_cuda",
        [
            py,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-deps",
            "--no-binary=torch-scatter",
            "torch-scatter==2.1.2",
        ],
        build_env,
    )
    record["torch_scatter_build"] = dict(
        kind="source",
        backend="cuda" if not result["exit_code"] else "cpu",
        cuda_build_error=result["output"] if result["exit_code"] else None,
    )
    if result["exit_code"]:
        require(
            execute(
                "build_scatter_cpu",
                [
                    py,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-build-isolation",
                    "--no-cache-dir",
                    "--no-deps",
                    "--no-binary=torch-scatter",
                    "torch-scatter==2.1.2",
                ],
                dict(FORCE_ONLY_CPU="1"),
            )
        )
else:
    record["torch_scatter_build"] = dict(kind="PyG wheel", backend="cuda")
deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
deps = [d for d in deps if not d.startswith(("torch-scatter", "torch>"))] + [
    "pytest",
    "pytest-xdist",
    "plotly",
]
require(
    execute("install_dependencies", ["uv", "pip", "install", "--python", py, *deps])
)
require(
    execute(
        "install_project",
        [
            py,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "--ignore-requires-python",
            "--no-deps",
        ],
    )
)
execute(
    "versions",
    [
        py,
        "-c",
        "import json,sys,importlib.metadata as m; print(json.dumps(dict(python=sys.version.split()[0], **{p:m.version(p) for p in ['torch','torch_scatter','torch_geometric','e3nn','numpy','scipy']})))",
    ],
)
if not record["versions"]["exit_code"]:
    record.update(json.loads(record["versions"]["output"].strip().splitlines()[-1]))
execute("freeze", ["uv", "pip", "freeze", "--python", py])
record["freeze"] = record["freeze"]["output"].splitlines()
smi = execute(
    "nvidia_smi",
    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
)
if not smi["exit_code"]:
    record["gpu"], record["driver"] = map(
        str.strip, smi["output"].splitlines()[0].split(",")
    )
execute(
    "cuda_driver_library",
    [
        py,
        "-c",
        "import ctypes; from pathlib import Path; lib=ctypes.CDLL('libcuda.so.1'); version=ctypes.c_int(); lib.cuDriverGetVersion(ctypes.byref(version)); print('CUDA driver API:',version.value); print('\\n'.join(line for line in Path('/proc/self/maps').read_text().splitlines() if 'libcuda.so' in line))",
    ],
)
execute(
    "import_check",
    [
        py,
        "-c",
        "import torch,torch_scatter,torch_geometric,e3nn; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CUDA unavailable')",
    ],
)
probe = execute(
    "scatter_cuda",
    [
        py,
        "-c",
        "import torch; from torch_scatter import scatter_max; print(scatter_max(torch.tensor([1.,3.,2.],device='cuda'),torch.tensor([0,0,1],device='cuda'))); torch.cuda.synchronize()",
    ],
)
record["torch_scatter_build"]["cuda_kernels_work"] = probe["exit_code"] == 0


def pytest(label, command, extra):
    xml = logs / (label + ".xml")
    result = execute(
        label, [py, "-m", "pytest", *command, "--junitxml=" + str(xml)], extra
    )
    if xml.exists():
        suites = ET.parse(xml).getroot().findall("testsuite")
        for key in ["tests", "failures", "errors", "skipped"]:
            result[key] = sum(int(s.attrib.get(key, 0)) for s in suites)
        result["junit_counts"] = {
            k: result[k] for k in ["tests", "failures", "errors", "skipped"]
        }
        # pytest 9 includes successful subtests in JUnit's tests count.
        summary = result["output"].splitlines()[-1]
        for word, key in [
            ("passed", "passed"),
            ("failed", "failures"),
            ("error", "errors"),
            ("skipped", "skipped"),
        ]:
            match = re.search(r"(\d+) " + word + r"s?\b", summary)
            result[key] = int(match.group(1)) if match else 0
        result["tests"] = sum(
            result[k] for k in ["passed", "failures", "errors", "skipped"]
        )
        subtests = re.search(r"(\d+) subtests passed", summary)
        result["subtests_passed"] = int(subtests.group(1)) if subtests else 0
    save()


if not args.gpu_only:
    pytest(
        "cpu_tests",
        [
            "-n",
            str(args.workers),
            "-q",
            "tests/unit",
            "tests/physics",
            "tests/equivariance",
            "-m",
            "(unit or physics or equivariance) and not (gpu or slow or large_data or network or integration)",
            "--ignore=tests/unit/data/test_snapshot_spatial.py",
            "--ignore=tests/physics/test_pyscf_rotation_reference_files.py",
        ],
        dict(CUDA_VISIBLE_DEVICES=""),
    )
if args.tag == "cu128" or args.gpu_only:
    pytest(
        "gpu_smoke",
        ["-q", "tests/gpu/test_cuda_smoke.py"],
        dict(MANDALA_RUN_GPU_TESTS="1"),
    )
else:
    record["gpu_smoke"] = {
        "exit_code": None,
        "reason": "Not run: Terok driver 550.163.01 cannot initialize CUDA 13.x.",
    }
if args.gpu_only:
    pytest(
        "gpu_tests",
        ["-n", str(args.workers), "-q", "tests", "-m", "gpu"],
        dict(MANDALA_RUN_GPU_TESTS="1"),
    )
checks = [
    record[k]["exit_code"]
    for k in ["import_check", "cpu_tests", "gpu_smoke", "gpu_tests"]
    if k in record and record[k]["exit_code"] is not None
]
record["result"] = "passed" if not any(checks) else "failed"
save()
