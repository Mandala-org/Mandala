These compact JSON fixtures are emitted by the pinned upstream implementations,
not MANDALA forwards. They retain the complete default stage count (five DeepH
layers; three DeepH-E3/MACE-H layers), with reduced channel widths. Each records
the commit, Python-source hashes, execution versions, supplied feature tensors,
resolved configuration, RNG seed, predictions, loss, and gradient/Adam-update
probes for every parameter over three fresh training steps.

Spinful/complex-output fixtures are also included for DeepH-E3 and MACE-H;
regenerate those with the same command plus `--spinful`.

Regenerate from the repository root with the recorded environment:

```bash
PYTHONPATH=src python scripts/generate_reference_fixtures.py \
  --deeph-root /path/to/DeepH-pack \
  --deeph-e3-root /path/to/DeepH-E3 \
  --mace-h-root /path/to/MACE-H
```

The generator rejects a different commit. It never calls MANDALA's model
forward. MANDALA's configuration resolver and geometry helpers prepare the
explicit supplied inputs; consequently these certify model mathematics, not an
independent end-to-end preprocessing oracle. Geometry, codecs and masking have
separate focused tests. Numerical tolerance is declared in the replay test;
cross-version or CPU/GPU bitwise equality is not claimed.

MACE-H uses externally supplied synthetic nontrivial means/scales to probe its
intentional identity-gradient affine rule. No statistics are fitted to a dataset,
and no validation/test population is consulted. Population-fitting behavior is
tested separately. Probes include three coordinates and two global moments per
parameter. Optional direct-source tests compare all parameter coordinates.
