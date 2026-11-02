"""Download and verify the versioned public OpenMX operator benchmarks.

This is an I/O/CPU preparation command, not a training or model-specific loader.
Existing archives and extracted files are verified and reused, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
import zlib

RECORD = "https://zenodo.org/records/7553640"
FILES = {
    "Monolayer_graphene_dataset.zip": "1262ddf10d8c0f31d511d6e9c4551abd",
    "Monolayer_MoS2_dataset.zip": "314e0144dd37650e63e2983d86449a43",
    "configs.zip": "abe4dcccf84523b71f14b27880133dd9",
    "README.md": "22ffadf620ee336594fb9960f97a184c",
}


def digest(path, algorithm="md5"):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def download(path, expected):
    if path.is_file():
        if digest(path) != expected:
            raise ValueError(f"Existing file has incorrect MD5; preserved: {path}")
        return
    partial = path.with_name(path.name + ".partial")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(
                f"{RECORD}/files/{path.name}?download=1",
                headers={"Range": f"bytes={offset}-"} if offset else {},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with partial.open(mode) as stream:
                    shutil.copyfileobj(response, stream, length=8 * 1024**2)
            if digest(partial) != expected:
                raise ValueError(f"Downloaded MD5 mismatch: {partial}")
            os.replace(partial, path)
            return
        except (OSError, ValueError) as exc:
            print(f"Download attempt {attempt + 1}: {exc}", flush=True)
            if attempt == 4:
                raise
            time.sleep(10)


def extract_verified(archive, root):
    """Check CRCs, prevent traversal/symlinks, and preserve existing files."""
    root = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink rejected: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.exists():
                crc = 0
                with destination.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                        crc = zlib.crc32(chunk, crc)
                if destination.stat().st_size != member.file_size or crc != member.CRC:
                    raise ValueError(
                        f"Existing extracted file differs; preserved: {destination}"
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".extracting")
            with bundle.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024**2)
            os.replace(partial, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, expected in FILES.items():
        path = args.root / name
        print(f"Preparing {path}", flush=True)
        download(path, expected)
        if path.suffix == ".zip":
            extract_verified(path, args.root)
        records[name] = {
            "md5": expected,
            "bytes": path.stat().st_size,
            "url": f"{RECORD}/files/{name}?download=1",
        }
    result = {
        "source": RECORD,
        "doi": "10.5281/zenodo.7553640",
        "files": records,
        "source_basis": "openmx",
        "source_cartesian_axes": "xyz",
        "note": "HDF5 is in the author's OpenMX order; DeepH-E3 Rotate.openmx2wiki_H_full applies the same single conversion as MANDALA. Coordinates are cyclically permuted once with that conversion.",
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
