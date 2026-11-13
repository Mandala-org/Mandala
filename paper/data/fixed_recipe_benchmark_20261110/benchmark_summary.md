# Fixed-recipe benchmark — snapshot snapshot

Snapshot UTC: 2026-11-10T07:03:11.864129+00:00

Frozen horizon: segment 2, 90 H100 hours per arm. Pending endpoints are not estimated from the freeze-decision epoch. No test scores were used to select a checkpoint.

All MAEs below are Hamiltonian errors in meV. Validation uses `val/hamiltonian_mae_elementwise` (eV × 1000), not the legacy pair-summed `val/hamiltonian_mae`. Explicit unit-suffixed columns use meV × 1, eV × 1000, or Hartree × 27211.4; selected columns and conversion factors are in JSON. Elementwise errors are averaged per structure, then over the validation epoch. Projected validation is shown only where explicitly logged. Terminal scores use `metrics.{raw,projected}.structure_mean_mae_mev`.

Epochs below count completed epochs (CSV epoch + 1); segment JSON already uses completed epochs. Values are `MAE @ completed epoch`. Nearest matches break ties toward the earlier epoch; † means the target has not yet been reached. Best validation is descriptive, not terminal checkpoint selection. Timing averages deduplicated full-epoch tqdm elapsed times, including validation and excluding startup/checkpoint overhead; no timing is inferred from untimestamped CSV rows.

## graphene

| Arm | Segment 1 epochs | Segment 2 / 90 h epochs | Last logged epoch | min/epoch (n) | Best raw val | Last raw val | Raw val ~350 | Raw val ~700 | Best projected val | Last projected val | Projected val ~350 | Projected val ~700 | Test projected | Test raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mandala_graphene | 703 | 1399 | 1399 | 3.843 (1399) | 0.2002 @ 1377 | 0.2018 @ 1399 | 1.054 @ 350 | 0.256 @ 700 | 0.1811 @ 1377 | 0.1831 @ 1399 | 0.9992 @ 350 | 0.2346 @ 700 | 0.1737 | 0.1833 |
| deeph_e3_graphene | 414 | 825 | 825 | 6.519 (825) | 0.3927 @ 811 | 0.4163 @ 825 | 0.9185 @ 350 | 0.583 @ 700 | unavailable | unavailable | unavailable | unavailable | 0.3584 | 0.3951 |
| mace_h_graphene | 363 | 727 | 727 | 7.398 (727) | 0.4811 @ 665 | 0.8026 @ 727 | 1.078 @ 350 | 0.6128 @ 700 | unavailable | unavailable | unavailable | unavailable | 0.7827 | 0.8099 |

## mos2

| Arm | Segment 1 epochs | Segment 2 / 90 h epochs | Last logged epoch | min/epoch (n) | Best raw val | Last raw val | Raw val ~175 | Raw val ~350 | Best projected val | Last projected val | Projected val ~175 | Projected val ~350 | Test projected | Test raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mandala_mos2 | 435 | 886 | 886 | 6.07 (886) | 0.5217 @ 844 | 0.7455 @ 886 | 2.924 @ 175 | 1.255 @ 350 | 0.4486 @ 874 | 0.6947 @ 886 | 2.51 @ 175 | 1.15 @ 350 | 0.6771 | 0.7225 |
| deeph_e3_mos2 | 196 | 391 | 391 | 13.72 (391) | 1.199 @ 383 | 1.474 @ 391 | 2.462 @ 175 | 1.589 @ 350 | unavailable | unavailable | unavailable | unavailable | 1.308 | 1.449 |
| mace_h_mos2 | 182 | 364 | 364 | 14.76 (364) | 1.081 @ 337 | 1.367 @ 364 | 1.794 @ 175 | 1.376 @ 350 | unavailable | unavailable | unavailable | unavailable | 1.159 | 1.323 |
