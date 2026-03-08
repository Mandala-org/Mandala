# Ablation Assessment (WandB, 2026-03-04 runs)

Project: `b-brzoza/mandala-minimal-overfit-study-ablation-sh-loss-2x2`
Runs analyzed: 4 (all finished)

## Per-run outcomes

| Run | sh-mode | loss-aggregation | final/mae_H | final/mse_H |
|---|---|---|---:|---:|
| `ablation-sh-loss-2x2-20260304-190252-legacy-per_key` | legacy | per_key | 1.1217e-3 | 2.6570e-6 |
| `ablation-sh-loss-2x2-20260304-190252-aligned-per_key` | aligned | per_key | 1.2658e-3 | 3.1407e-6 |
| `ablation-sh-loss-2x2-20260304-190252-legacy-global` | legacy | global | 1.6155e-3 | 5.6240e-6 |
| `ablation-sh-loss-2x2-20260304-190252-aligned-global` | aligned | global | 1.8114e-3 | 6.8364e-6 |

## Impact of explored settings

- Best setting in this 2x2: `legacy + per_key`.
- `per_key` vs `global`:
  - Mean final `mae_H`: `1.1938e-3` vs `1.7134e-3`.
  - `per_key` is better by about **30.33%**.
- `legacy` vs `aligned` (with current paired radial settings):
  - Mean final `mae_H`: `1.3686e-3` vs `1.5386e-3`.
  - `legacy` is better by about **11.05%**.

## Key interpretation

- If target is `mae_H <= 1e-5`, current best is still ~112x too high.
- But current best `mse_H` is `2.657e-6`, which is close to the often-quoted `~1e-6` scale.
- DeepH-E3 training objective is masked global **MSE** (`deephe3/utils.py`), so comparing DeepH-E3 `~1e-6` directly to our `mae_H` is likely a metric mismatch.

## Most likely remaining discrepancy (after this ablation)

1. Metric mismatch in comparisons (MAE vs MSE, and potentially meV-vs-eV reporting) is likely the dominant apparent gap.
2. Residual gap in `mse_H` (roughly 2x–7x from 1e-6) is more likely due to pipeline differences outside this 2x2:
   - output decomposition/readout differences vs DeepH-E3 `e3TensorDecomp`,
   - symmetry handling and evaluation policy differences,
   - objective weighting/masking parity differences.
