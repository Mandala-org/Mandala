# Fixed-recipe benchmark — interim snapshot

Snapshot UTC: 2026-11-10T01:16:52.923504+00:00

Frozen horizon: segment 2, 90 H100 hours per arm. Pending endpoints are not estimated from the freeze-decision epoch. No test scores were used to select a checkpoint.

All MAEs below are Hamiltonian errors in meV. Validation uses `val/hamiltonian_mae_elementwise` (eV × 1000), not the legacy pair-summed `val/hamiltonian_mae`. Explicit unit-suffixed columns use meV × 1, eV × 1000, or Hartree × 27211.4; selected columns and conversion factors are in JSON. Elementwise errors are averaged per structure, then over the validation epoch. Projected validation is shown only where explicitly logged. Terminal scores use `metrics.{raw,projected}.structure_mean_mae_mev`.

Epochs below count completed epochs (CSV epoch + 1); segment JSON already uses completed epochs. Values are `MAE @ completed epoch`. Nearest matches break ties toward the earlier epoch; † means the target has not yet been reached. Best validation is descriptive, not terminal checkpoint selection. Timing averages deduplicated full-epoch tqdm elapsed times, including validation and excluding startup/checkpoint overhead; no timing is inferred from untimestamped CSV rows.

## graphene

| Arm | Segment 1 epochs | Segment 2 / 90 h epochs | Last logged epoch | min/epoch (n) | Best raw val | Last raw val | Raw val ~350 | Raw val ~700 | Best projected val | Last projected val | Projected val ~350 | Projected val ~700 | Test projected | Test raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mandala_graphene | 703 | pending | 1357 | 3.842 (1357) | 0.2008 @ 1350 | 0.2027 @ 1357 | 1.054 @ 350 | 0.256 @ 700 | 0.1818 @ 1334 | 0.1838 @ 1357 | 0.9992 @ 350 | 0.2346 @ 700 | pending | pending |
| deeph_e3_graphene | 414 | pending | 799 | 6.518 (799) | 0.4433 @ 650 | 0.4911 @ 799 | 0.9185 @ 350 | 0.583 @ 700 | unavailable | unavailable | unavailable | unavailable | pending | pending |
| mace_h_graphene | 363 | pending | 692 | 7.399 (692) | 0.4811 @ 665 | 0.531 @ 692 | 1.078 @ 350 | 0.531 @ 692 † | unavailable | unavailable | unavailable | unavailable | pending | pending |

## mos2

| Arm | Segment 1 epochs | Segment 2 / 90 h epochs | Last logged epoch | min/epoch (n) | Best raw val | Last raw val | Raw val ~175 | Raw val ~350 | Best projected val | Last projected val | Projected val ~175 | Projected val ~350 | Test projected | Test raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mandala_mos2 | 435 | pending | 858 | 6.073 (858) | 0.5217 @ 844 | 0.6213 @ 858 | 2.924 @ 175 | 1.255 @ 350 | 0.45 @ 844 | 0.5389 @ 858 | 2.51 @ 175 | 1.15 @ 350 | pending | pending |
| deeph_e3_mos2 | 196 | pending | 379 | 13.72 (379) | 1.214 @ 376 | 1.303 @ 379 | 2.462 @ 175 | 1.589 @ 350 | unavailable | unavailable | unavailable | unavailable | pending | pending |
| mace_h_mos2 | 182 | pending | 346 | 14.76 (346) | 1.081 @ 337 | 1.19 @ 346 | 1.794 @ 175 | 1.19 @ 346 † | unavailable | unavailable | unavailable | unavailable | pending | pending |

