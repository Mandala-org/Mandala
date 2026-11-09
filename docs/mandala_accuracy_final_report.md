# MANDALA Graphene and MoS₂ accuracy investigation

**Final campaign review, 6 November 2026, 10:07 Warsaw.** The unchanged pilots
finished at **0.202 meV Graphene** and **0.476 meV MoS₂** raw validation MAE.
The early multi-meV errors substantially overstated the mature validation
error. The clearest ablation finding is that switching to MSE under the
registered optimizer/controller settings worsened validation MAE. Copy-wise
radial gates were promising but inconsistent. Neither the remaining
literature gap nor the independent effect of earlier learning-rate decay
was established.

## Completion

| Work | Final state |
|---|---|
| Original pilots | Graphene, MoS₂ and SiOx allocations completed successfully. Graphene and MoS₂ used their original two-segment budgets. |
| 42-arm long grid | All reached 200 epochs; **38 reached 500**: 18/18 Graphene and 20/24 MoS₂. All **11 restarted arms** reached 500. |
| Missing long endpoints | MoS₂ seed 43: baseline **316** epochs, MSE **304**, rollback-off **304**, MSE + rollback-off **318**. Their continuation gate declined the remaining work. |
| 14-arm V100 screen | All allocations completed, with 20–41 epochs. Coverage: **14/14 at 12**, **12/14 at 24**, **0/14 at 48**. |
| Final validation summaries | Slurm **792476** and **792477** both exited successfully. No campaign jobs remain queued or running. |

The long-grid accounting contains 82 completed and 28 superseded or redundant
cancelled allocations. There were no failed training allocations. Among the
26 short continuation dispatchers, 16 resumed training, six skipped models
already at 500, and four deferred expensive remainders. The long and V100
grids used **1,830.86** and **75.60 allocated GPU-hours**, respectively,
excluding the original pilots and certification. Allocation time is not a
measure of GPU utilization.

## Pilot results and comparability

| Pilot | Early captured raw MAE (meV) | Final validation epoch (zero-based) | Final raw MAE (meV) | Final projected MAE (meV) |
|---|---:|---:|---:|---:|
| Graphene | 2.097 (epoch 298) | 1360 | **0.202** | **0.183** |
| MoS₂ | 3.353 (epoch 202) | 904 | **0.476** | **0.410** |

These are logged validation values at the stated epochs, not scores of
independently reloaded validation-best checkpoints. MAE averages observed
directed matrix-entry errors within each structure, then across structures,
in the global physical frame. Projection averages complete reverse pairs
with the transpose of the opposite direction. The final pilots improved
roughly tenfold and sevenfold in raw MAE relative to the early captures,
without an architecture change. Improvement coincided with additional
updates and LR reductions; this campaign did not separate their effects.

Full certificates covered the original 270/90 Graphene and 300/100 MoS₂
train/validation memberships, including input hashes and target support.
No duplicate energy, orbital-basis or Cartesian conversion was detected
within the tested tolerances. The maximum eV/Hartree round-trip difference
was 9.54×10⁻⁷ eV; target cleanup changed an entry by at most 5.07×10⁻⁸ eV.
These results narrow the preprocessing hypothesis for the certified paths;
they do not test every preprocessing choice, such as alternate graph support.

For context, [DeepH](https://arxiv.org/html/2104.03786v2) reports Graphene
2.1 meV in its local frame; [DeepH-E3 Table 1](https://www.nature.com/articles/s41467-023-38468-8.pdf)
reports Graphene 0.40 and directed MoS₂ species-pair values
0.51/0.46/0.45/0.37 meV (Mo–Mo/Mo–S/S–Mo/S–S).
[MACE-H v2 Table S4](https://arxiv.org/html/2508.15108v2) reports 0.21
Graphene and 0.39 overall MoS₂, with retrained DeepH-E3 at 0.27/0.45.
These are reference points, not a ranking: published split membership is
unverified, and frame, reduction and projection conventions differ or are
not fully established. In particular, a species-pair sum is not an entrywise
MAE. See the [literature audit](mandala_accuracy_literature_audit.md).
No held-out results were used in this analysis or to choose ablations.

## Ablation findings

Numbers below are **best observed raw validation MAE through 500 epochs**,
in meV. Each minimum uses the same search horizon; it is not the epoch-500
endpoint. The endpoint appendix reports both, and projection at the epoch
selected by raw MAE.

| Treatment | Graphene 42 | Graphene 43 | Graphene 44 | MoS₂ 42 | MoS₂ 43 | MoS₂ 44 |
|---|---:|---:|---:|---:|---:|---:|
| MAE baseline | 0.359 | 0.266 | 0.283 | 0.654 | — | 0.819 |
| MSE | 1.905 | 1.738 | 1.861 | 2.164 | — | 2.059 |
| Rollback off | 0.370 | 0.273 | 0.341 | 0.583 | — | 2.457 |
| MSE + rollback off | 1.869 | 1.829 | 1.885 | 2.347 | — | 2.287 |
| AdamW decay zero | 0.570 | 0.398 | 0.794 | 0.893 | 0.892 | 1.297 |
| Copy-wise radial gates | 0.247 | 0.282 | 0.233 | 0.642 | 0.514 | 0.610 |
| Species-pair mean | — | — | — | 0.941 | 0.691 | 0.989 |
| Observed-entry weighting | — | — | — | 0.601 | 0.678 | 0.827 |

**Loss and optimization.** At epoch 200, the MSE arm had higher raw MAE than
the corresponding seed's MAE baseline in all six blocks; some MoS₂ blocks
are hardware-confounded. The clean Graphene seed-43 H100 comparison was
**0.272 MAE vs 1.764 MSE** at epoch 500, and all three Graphene seeds agree
in direction. Disabling rollback did not rescue MSE. This tests a loss
replacement with the same optimization settings, including the consequences
of changed gradient scale and scheduler response. It does not test a
separately tuned native-eV MSE recipe. Removing AdamW decay was worse in all
three Graphene blocks at 500. Rollback removal under MAE gave small
best-through-500 losses on Graphene and opposite signs on MoS₂ seeds 42/44;
there is no basis for turning it off globally.

**Architecture.** Copy-wise radial gating preserves parameter shapes but uses
a separate gate for each irrep copy. Graphene improved in two seeds and
regressed in seed 43. The MoS₂ numbers are encouraging, but seed 42 changes
GPU class relative to its baseline, seed 44 has different hardware exposure
from its baseline, and seed 43 lacks a 500-epoch baseline. Treat these as
replication candidates. No controlled depth, attention or MACE product
comparison was completed.

**Objective weighting.** Pair mean divides the legacy species-pair sum by the
number of pairs, keeping equal relative species-pair weights. Observed-entry
weighting changes the relative weights as well. The late MoS₂ seed-42
observed-entry result is modestly better than its baseline (0.601 vs 0.654);
seed 44 is effectively a numerical tie (0.827 vs 0.819) under different
hardware exposure. Pair mean is worse in both available late comparisons.
The seed-43 objective arms are promising at 200 epochs but trained on
different GPU classes from that seed's A100 baseline. No consistent
weighting improvement or explanation of the early gap is established.

**Hardware.** The registered context labels predate scheduler amendments.
The appendix reconstructs actual training GPU and CPU exposure from Slurm
accounting and segment epoch counts. For example, the restarted MoS₂
seed-42 MSE and radial-copy arms trained on B200; the corresponding baseline
trained on H100. Several seed-44 models actually trained on H100 then B200.
Conversely, the six restarted B200 models already had 500 epochs when their
H100 continuations ran, so those continuations did not add H100 training.
Do not pool these rows as hardware-matched effects or interpret the three
seeds as a hardware-independent uncertainty estimate.

## Implications for the next batch

Long baselines were valuable: Graphene seed 43 improved from **1.708** best
raw meV through 200 epochs to **0.266** through 500, with its first recorded
LR reduction after epoch 200. Short runs were useful for screening the large
MSE regressions, but early rankings did not reliably predict late accuracy.
For example, V100 copy-wise Graphene gates improved the best-through-24
score (3.525 vs 4.833) while the actual epoch-24 endpoint worsened
(5.038 vs 4.833). The short screen cannot substitute for late confirmation.

The highest-value next experiment is the already planned **earlier-LR-decay
control**: compare the existing schedule and one fixed earlier schedule
with the same MAE objective, seed, data, hardware and runtime. Report both
fixed-update and fixed-wall-time comparisons. Use the 48/200-epoch gates
for screening, then continue informative contrasts and their baselines
through late optimization before adding seeds. Keep radial gating as a
separate matched contrast. Use a common split and evaluator with a reference
model before making a published-gap or architecture-ranking claim.

## Evidence

The private run workspace `researchops/runs/mandala_accuracy_v1/` contains
`evidence/final_review_20261106/` (final validation summaries, endpoint
appendix and actual training hardware) and
`workflow/final_{accounting,segments}_20261106.json` (Slurm states and
epoch counts). The final analysis
files were checked against their remote hashes and the original frozen
registration and analyzer hashes; all 42 long-arm epoch counts were
reconciled with segment records. These are bounded operational records,
not a public reproduction package. The [earlier snapshot](mandala_accuracy_investigation.md)
and [ablation protocol](mandala_accuracy_ablation_protocol.md) document the
original design and amendments.
