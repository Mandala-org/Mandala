"""Extract NVIDIA's user-space forward-compatibility libraries in this worktree.

No system installation or kernel driver change. Reference:
https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import tarfile
from urllib.request import urlretrieve

URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/"
    "x86_64/cuda-compat-13-4_615.71.10-1ubuntu1_amd64.deb"
)
SHA256 = "8f49a2bbca06447b2ce335320b7b6169fac9de5866958454849a75f7363d17c3"
destination = Path(".venvs/cuda-compat-13-4").resolve()
destination.mkdir(parents=True, exist_ok=True)
archive = destination / "cuda-compat.deb"
if not archive.exists():
    print("Downloading " + URL, file=sys.stderr)
    urlretrieve(URL, archive)
data = archive.read_bytes()
if sha256(data).hexdigest() != SHA256:
    raise RuntimeError("CUDA compatibility archive checksum mismatch")
if data[:8] != b"!<arch>\n":
    raise RuntimeError("Expected a Debian ar archive")
offset = 8
while offset < len(data):
    header = data[offset : offset + 60]
    name = header[:16].decode().strip().rstrip("/")
    size = int(header[48:58].decode().strip())
    offset += 60
    if name.startswith("data.tar"):
        with tarfile.open(fileobj=BytesIO(data[offset : offset + size])) as package:
            package.extractall(destination, filter="data")
        break
    offset += size + size % 2
else:
    raise RuntimeError("No data archive in compatibility package")
compat = destination / "usr/local/cuda-13.4/compat"
if not (compat / "libcuda.so.1").exists():
    raise RuntimeError("Expected libcuda.so.1 missing after extraction")
print(compat)
