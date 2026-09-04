# HamGNN SiO2 Stage-1 data audit

## Objective

Validate the released graph bundle, freeze a protocol-matched split, and establish the observed-support Hamiltonian cutoff before model training.

## Acceptance decision

- Block packing, periodic shifts, inverse edges, Hermiticity, and AO-irrep round trip: **PASS**
- Published structure-count agreement (663 expected): **PARTIAL/INCONCLUSIVE** (630 found)
- Training-only cutoff criteria: **PASS**

## Results

- Structures: 630
- Split: {'train': 504, 'validation': 63, 'test': 63} (`protocol-matched, split-not-identical`)
- Selected RH: 6.5 Å
- Omitted-tail matrix-element MAE floor: 0.0971841 meV
- Retained offsite squared Frobenius mass: 0.999990346
- Source SHA256: `775142e9337e3cef839711690d812ef3a2ea4a8225887e5a784cb962acb3c873`

## Shortcomings and threats to validity

The cutoff criteria can only assess blocks present in the released sparse graph. The exact published split was not present in the inspected release, and the released structure count differs from the publication specification.

## Decision

Do not start baseline training unless all structural validations pass and the cutoff criteria find a practical radius. Preserve the split and source fingerprint for all subsequent comparisons.
