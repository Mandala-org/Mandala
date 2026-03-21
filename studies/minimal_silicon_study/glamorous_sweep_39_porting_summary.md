**Reference Config**
Source run: `glamorous-sweep-39`
W&B run id: `wynxo0rr`
URL: `https://wandb.ai/b-brzoza/mandala-minimal-overfit-observables-stages/runs/wynxo0rr?nw=nwuserbbrzoza`

Reference command copied from W&B:
```bash
/data/home2/brzoza73/casus/mandala/studies/minimal_overfit_observables/overfit_observables_minimal.py --adaptive-log-interval --log-data --log-model --log-per-irrep-metrics --log-per-irrep-images --benchmark --generate-video --apply-cutoff-to-targets --require-exact-edge-match --checkpoint-dir=studies/minimal_overfit_observables/checkpoints --cutoff-radius=7 --data-path=data/small/H2O/original/H2O.matrix --device=cuda --dtype=float32 --e3layernorm=True --edge-encoder-use-sh-tensor-square=False --enable-energy=True --enable-forces=True --enable-num-electrons=True --grad-clip=1 --head-e3mlp-layers=2 --head-use-node-embeddings-for-self-edges=True --head-use-tensor-square=True --hidden-dim=32 --hidden-irreps=32x0e+32x1e+32x1o+16x2e+16x2o+16x3e+16x3o+8x4e --info-path=data/small/H2O/original/H2O.info.out --l-max=4 --log-interval=100 --loss-coef-forces=1.6022539475211734e-08 --loss-coef-observables=4.330794270449572e-09 --lr=0.00969419794083762 --lr-factor=0.2 --lr-patience=400 --matrix-targets=hamiltonian,overlap,density --n-radial=64 --num-epochs=16000 --num-layers=2 --radial-embedding-scale=none --separate-shifted-self=True --symmetrize-preds=True --train-on-energy=True --train-on-forces=True --train-on-num-electrons=True --training-unit=ev
```

**Assumed Silicon Sweep**
Prepared as:
`sweeps/minimal_silicon_glamorous_sweep_39_narrow_assumed.yaml`

This is an assumption-based template for `studies/minimal_silicon_study/train_silicon_minimal.py`:
- same observables/forces-oriented interface as the reference run,
- silicon-specific dataset defaults:
  - `data-path=/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A`
  - `train-temps=2700`
  - `val-temp=2700`
  - `n-snapshots-per-temp=100`
  - `val-n-snapshots=10`

Sweep choices follow the requested rule:
- log-uniform values:
  - `lr`: `1e-3` to `1e-2`
  - `loss-coef-observables`: `1e-9` to `1e-8`
  - `loss-coef-forces`: `1e-8` to `1e-7`
- integer values:
  - `n-radial`: `[64, 128]`
  - `num-layers`: `[2, 3]`
  - `head-e3mlp-layers`: `[2, 3]`
  - `log-interval`: `[100, 200]`
  - `lr-patience`: `[400, 800]`
- booleans:
  - all set explicitly to the same value as the source run

**Missing Or Different In `train_silicon_minimal.py`**
The current silicon script is not yet compatible with the reference observables run. Gaps:

1. Missing CLI arguments entirely
- `--dtype`
- `--matrix-targets`
- `--enable-energy`
- `--enable-forces`
- `--enable-num-electrons`
- `--train-on-energy`
- `--train-on-forces`
- `--train-on-num-electrons`
- `--loss-coef-observables`
- `--loss-coef-forces`
- `--head-e3mlp-layers`
- `--head-use-node-embeddings-for-self-edges`
- `--head-use-tensor-square`
- `--symmetrize-preds`
- `--radial-embedding-scale`

2. Boolean parsing style differs
- Current silicon script still uses `action="store_true"` for:
  - `--adaptive-log-interval`
  - `--benchmark`
  - `--separate-shifted-self`
  - `--normalize-blocks`
  - `--distance-magnitude-normalization`
  - `--edge-encoder-use-sh-tensor-square`
  - `--head-mlp-for-scalars`
  - `--train-on-irrep-parts`
  - `--apply-cutoff-to-targets`
  - `--require-exact-edge-match`
  - logging/video flags
- The assumed sweep uses explicit booleans everywhere, so the silicon script would need `parse_bool`-style arguments for parity.

3. Data interface differs
- Reference observables run uses one explicit matrix file plus one explicit info file:
  - `--data-path`
  - `--info-path`
- Silicon script uses dataset discovery:
  - root `--data-path`
  - temperature split args
  - per-snapshot `info.dat` discovery
- So `--info-path` is not a meaningful silicon argument in the current design.

4. Training targets differ
- Current silicon script only trains on the Hamiltonian loss.
- `overlap` and `density` are loaded and passed through preprocessing, but they are not predicted as independent outputs and are not trained.
- There is no multi-target forward path for `H/S/D`.

5. Observable training is missing
- No energy computation
- No number-of-electrons computation
- No force computation
- No force loss
- No observable loss weighting

6. Observable metrics/logging are missing
- No `mae_S` / `mse_S`
- No `mae_D` / `mse_D`
- No force metrics
- No energy / electron-count metrics
- No per-irrep metrics for `S` or `D`
- No `S`/`D` videos or irrep image sets

7. Architecture differs
- Silicon script instantiates `MinimalNetwork` with:
  - `head_mlp_for_scalars`
  - `edge_encoder_use_sh_tensor_square`
  - `separate_shifted_self`
- But it hardcodes:
  - `magnitude_factorization=False`
  - `head_use_tensor_square=False`
- There is currently no head E3MLP depth argument in the silicon script.
- There is currently no node-embedding-to-self-edge head option in the silicon script.

8. Prediction symmetrization behavior differs
- Silicon `predict_sample()` always returns:
  - `pred_matrix` raw
  - `pred_metrics = (pred_matrix + pred_matrix.transpose()) * 0.5`
- There is no `--symmetrize-preds` switch.

9. Dtype handling differs
- Current silicon script accepts no `--dtype`.
- Internally it hardcodes `torch.float32` in several places:
  - mapper
  - dataset config
  - graph config

10. Metric naming differs
- Silicon script currently logs Hamiltonian-focused metrics only.
- It uses `compute_detailed_metrics(pred_eval, target_eval_matrix, target_overlap)` which is Hamiltonian-specific and overlap-assisted.
- The observables-stage run expects multi-target metrics and force/observable losses.

**Bottom Line**
The narrow silicon sweep template is ready as a target configuration, but `train_silicon_minimal.py` must first be extended in four major areas:
- multi-target `H/S/D` prediction,
- observables computation (`energy`, `num electrons`, `forces`),
- training on observables/forces,
- CLI/config parity with explicit booleans and the missing options above.
