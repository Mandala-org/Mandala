# Graphene/MoS₂ literature comparability audit

Audited 2026-10-29. This is a historical source/metric audit, not an ablation
result or current job-status report. Pilot observations discussed during that
audit were unfinished validation results, not published-test reproductions.
See the [dated investigation summary](mandala_accuracy_investigation.md) for
subsequent evidence and its provenance. No checkpoints, datasets, or cluster
jobs were processed for this literature audit.

## Published numbers and their meaning

All numbers below are Hamiltonian MAEs in **meV**, not band-energy errors. A four-number MoS₂ tuple follows **Mo–Mo / Mo–S / S–Mo / S–S** order.

| Source / experiment | Graphene | MoS₂ | Reduction / frame / projection status |
|---|---:|---:|---|
| DeepH original main text | 2.1 | 1.3 / 1.0 / 0.7 / 0.8 | Local-frame transformed matrix `H′`; orbital-combination mean, resolved by species pair. Confirm projection convention before comparing global MAE. |
| DeepH row reproduced in DeepH-E3 Table 1 | 2.1 | 1.3 / 1.0 / 0.8 / 0.7 | **S–Mo and S–S differ in ordering from original main text.** Preserve source-specific values. |
| DeepH-E3 original Table 1 | 0.40 | 0.51 / 0.46 / 0.45 / 0.37 | Test results; atom-pair-resolved matrix MAE. No single MoS₂ overall number in this table. |
| DeepH-E3 retrained in MACE-H, v2 Table S4 | 0.27 | Mo–Mo 0.51; Mo–S 0.44; S–S 0.36; overall 0.45 | A separate baseline experiment; three species categories. Exact overall weighting and table-specific projection status unverified. |
| MACE-H, v2 Table S4 | 0.21 | Mo–Mo 0.42; Mo–S 0.39; S–S 0.32; overall 0.39 | Same qualifications; **0.39 is overall, 0.32 is S–S**. |

Sources: [DeepH original author manuscript](https://arxiv.org/html/2104.03786v2), [DeepH journal main text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499279/), [DeepH-E3 Table 1](https://www.nature.com/articles/s41467-023-38468-8.pdf), [MACE-H v2, Tables S4/S10](https://arxiv.org/html/2508.15108v2). MACE-H v2 is dated 2026-04-01; its dataset summary is **S10**, whereas earlier versions used different supplementary numbering. The associated [journal article](https://www.nature.com/articles/s41524-026-02020-1) was found, but its full text could not be fetched reliably in this audit; the numerical claims above deliberately identify the accessible v2 source.

## Dataset, split and training contracts

| Contract | Verified evidence | What remains unknown |
|---|---|---|
| Public DeepH-E3 monolayers | Graphene: 450 structures, 72 atoms, 270/90/90. MoS₂: 500, 75 atoms, 300/100/100. No SOC. OpenMX PBE Hamiltonians; C `s2p2d1`, Mo `s3p2d2`, S `s2p2d1`. | Exact paper split-name/hash manifests. |
| Sampling | Graphene: equilibrated trajectory subsampling; MoS₂: discard initial unequilibrated half. | Independence of nearby MD samples; do not claim temporal holdout. |
| DeepH-E3 published training | Hamiltonian MSE; three node/edge updates, batch 1, initial LR 0.003 graphene / 0.005 MoS₂; halve LR on plateau. | Exact update counts, stopping epoch, selected-checkpoint provenance and repeated-seed spread. |
| MANDALA pilots, local declared protocol | Same archive counts/proportions; sorted-path NumPy split seed 42; paper architecture, MAE objective, LR 0.0027, batch 1, stored support, two 48-hour allocations. | Published split identity and convergence; remote manifest/source evidence must establish actual executed contract. |

The first three rows are supported by [DeepH-E3 author manuscript, Appendices A/B](https://arxiv.org/html/2210.13955v1). The archive is [Zenodo 7553640](https://zenodo.org/records/7553640). The pilot row is the local declaration in `docs/operator_benchmark_pilots.md` and `configs/pilots/`; it is not independent proof of remote execution.

The original DeepH manuscript describes 5,000 graphene MD frames, 270 training structures, 90 for hyperparameter optimization and the remainder for testing. Its 2.1 value therefore does not itself establish the identical 90-structure public test subset. [DeepH original, Results](https://arxiv.org/html/2104.03786v2).

MACE-H's monolayer settings are MSE, LR 0.006, three layers, correlation order 3, Gaussian node features and batch 1. It describes postprediction symmetrisation. Its results on other systems cannot establish causes of MANDALA's monolayer gap. [MACE-H v2, S3/S4 and Methods](https://arxiv.org/html/2508.15108v2).

## Exact metric and split hazards established from primary code

The pinned [DeepH-E3 analyzer](https://github.com/Xiaoxun-Gong/DeepH-E3/blob/036bf95b996052ef778c5c38e3c54c2ac0e2b591/deephe3/analyzer.py#L38) first averages absolute error across directed edges of one species pair **within each structure**, then across structures, and finally across its real orbital entries. It reads `H_pred`/`label` from `test_result.h5`. The [pinned kernel](https://github.com/Xiaoxun-Gong/DeepH-E3/blob/036bf95b996052ef778c5c38e3c54c2ac0e2b591/deephe3/kernel.py) records raw reconstructed outputs during testing; Hermitian completion is a separate export operation. This establishes the inspected code path, not proof of which scripts produced a historical table.

The [default config](https://github.com/Xiaoxun-Gong/DeepH-E3/blob/036bf95b996052ef778c5c38e3c54c2ac0e2b591/deephe3/default_configs/train_default.ini#L24) uses seed 42. However, the [pinned data reader](https://github.com/Xiaoxun-Gong/DeepH-E3/blob/036bf95b996052ef778c5c38e3c54c2ac0e2b591/deephe3/data.py#L148) collects folders using unsorted `os.walk`; the kernel shuffles integer positions. [MACE-H's inspected master reader](https://github.com/maurergroup/MACE-H/blob/master/maceh/data.py#L148) and [loader](https://github.com/maurergroup/MACE-H/blob/master/maceh/kernel.py#L823) have the same ordering dependency. Therefore matching proportions and seed does **not** verify matching memberships; the MACE-H master observations are not a certification of the locally pinned reference commit.

## Required comparison gate

Before assigning an architecture gap, report on one frozen validation split:

1. Raw global-frame structure-mean entry MAE, dataset-entry MAE, and four directed species-pair MAEs; include counts and denominators.
2. The same metrics after full reverse-pair Hermitian projection; separately report raw residuals. Never silently project or replace a raw metric.
3. Original DeepH local-frame diagnostics plus a common global physical evaluation: L1 error is not invariant to a general orbital rotation.
4. Input/archive hashes, explicit split names/hashes, graph-support counts, masks, and energy/basis/Cartesian provenance. eV↔Hartree is a unit conversion; OpenMX↔e3nn orbital permutation and xyz↔yzx Cartesian change must each occur exactly once.
5. Validation-only controlled contrasts for objective, optimizer/schedule and individual interchangeable architecture stages, with equal update budgets and training-only fitted statistics. Preserve the test partition until the final frozen protocol.

Historical SiOx pair-sum MAE is neither entry MAE nor a public-monolayer result. Comparing an unfinished raw MANDALA validation MAE directly with a published selected test MAE can motivate investigation, but cannot identify a causal component or establish a reproduced model ranking.
