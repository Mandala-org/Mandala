"""Pin existing SiO2/Zn continuations to their original, verified source.

The live recipes unexpectedly still use the retired head. Their checkpoints,
recipes, code and runtime contracts must remain unchanged. New silicon uses
the merged source. Run on ROSI before restarting silicon; no training is run.
"""

from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
SOURCE_COMMIT = "e3f18d0"
CHECKPOINTS = Path(
    "/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/publication_retrain_v2"
)
FROZEN = ROOT / ".cache/publication_retrain_v2/legacy_continuation_source"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if not FROZEN.exists():
    FROZEN.mkdir(parents=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "archive", SOURCE_COMMIT], cwd=ROOT, stdout=archive, check=True
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as source:
            source.extractall(FROZEN)

report = {
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", SOURCE_COMMIT], cwd=ROOT, text=True
    ).strip(),
    "frozen_source": str(FROZEN),
    "materials": {},
}
for material in ("sio2", "zncusnses"):
    recipe_path = Path(f"scripts/reproduce_examples/{material}_v2.json")
    recipe = json.loads((FROZEN / recipe_path).read_text())
    assert recipe["config"]["separate_shifted_self"] is True
    contract_path = CHECKPOINTS / f"{material}.contract.json"
    contract = json.loads(contract_path.read_text())
    assert digest(FROZEN / recipe_path) == contract["recipe_sha256"]
    differences = [
        name
        for name, expected in contract["source_sha256"].items()
        if digest(FROZEN / name) != expected
    ]
    assert not differences, differences
    actual_paths = {
        str(path.relative_to(FROZEN))
        for directory in ("src", "scripts")
        for path in (FROZEN / directory).rglob("*.py")
    }
    actual_paths.update(
        str(path.relative_to(FROZEN))
        for path in (FROZEN / recipe_path.parent).glob("*.sh")
    )
    assert actual_paths == set(contract["source_sha256"])
    checkpoint = CHECKPOINTS / material / "latest_checkpoint.pt"
    assert checkpoint.is_file()
    wrapper = ROOT / f"scripts/reproduce_examples/train_{material}_v2.sh"
    wrapper.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
# Preserve this already-running legacy chain's immutable source and recipe.
root=$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)
frozen="$root/.cache/publication_retrain_v2/legacy_continuation_source"
checkpoint={checkpoint}
if [[ -f "$checkpoint" ]]; then
    test -f "$frozen/scripts/reproduce_examples/{material}_v2.json"
    cd "$frozen"
    export PYTHONPATH="$frozen/src:$frozen${{PYTHONPATH:+:$PYTHONPATH}}"
    exec bash scripts/reproduce_examples/train_{material}_v2.sh "$@" --action continue
fi
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/mandala-publication-mpl
cd "$root"
exec python -u scripts/reproduce_examples/train.py --recipe scripts/reproduce_examples/{material}_v2.json "$@"
""")
    report["materials"][material] = {
        "source_files_verified": len(actual_paths),
        "source_mismatches": differences,
        "recipe_sha256": contract["recipe_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_modified": False,
        "legacy_separate_shifted_self": True,
    }
(RUN / "legacy_continuation_verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
