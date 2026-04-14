# Silicon Study Logging / Visualization Inventory

Source audited:
- `studies/minimal_silicon_study/train_silicon_minimal.py`
- `studies/minimal_overfit_study/detailed_logging.py`
- shared plotting helpers in `studies/minimal_silicon_e3mlp_sweep/common.py`

This inventory tracks the study logging / visualization surface and the current status in main code.

## WandB Project / Run Organization

1. Dedicated WandB project for the main-code silicon port.
   Status: implemented.
   Main path: `scripts/train_silicon.py`
   Default project: `mandala-silicon-main-study-port`

2. Stable run directory rooted at `<checkpoint_dir>/<run_name>`.
   Status: implemented.
   Main path: `scripts/train_silicon.py`

## Console Logging

3. Config banner with device, hidden irreps, cutoff, LR, epochs, clipping, logging toggles, checkpoint/frame dirs.
   Status: implemented.
   Main path: `scripts/train_silicon.py`, `src/net/silicon_study_logging.py`

4. First-sample snapshot summary.
   Status: implemented.

5. Orbital configuration printout.
   Status: implemented.

6. Mapper summary printout.
   Status: implemented.

7. First-graph summary.
   Status: implemented.

8. Initial evaluation metrics banner.
   Status: implemented.
   Main path: `src/net/artifacts.py`

9. Per-epoch metrics banner with LR and detailed Hamiltonian diagnostics.
   Status: implemented.

10. Optional per-irrep console metrics.
    Status: implemented.

11. Final evaluation banner and final Hamiltonian diagnostics.
    Status: implemented.

12. Final study-complete summary with best/final checkpoint locations.
    Status: implemented.

13. Console message for cutoff application to GT matrices.
    Status: implemented.
    Notes: emitted from `scripts/train_silicon.py` using edge counts captured during dataset preprocessing.

14. Console message for strict edge-alignment checks passing.
    Status: implemented.
    Notes: emitted once on the first successful aligned artifact evaluation when `require_exact_edge_match=True`.

## WandB Scalar Metrics: Train / Val / Initial / Final

15. Study-style Hamiltonian diagnostics:
    `mae_H`, `mse_H`, `mae_H_mod`, `mse_H_mod`, `mu_H`, `correction_mae`, `correction_mse`.
    Status: implemented.

16. `initial/*` prefixed metrics.
    Status: implemented.

17. `final/*` prefixed metrics.
    Status: implemented.

18. Matrix basic metrics aliases:
    `mae_H`, `mse_H`, `mae_S`, `mse_S`, `mae_D`, `mse_D`.
    Status: implemented.

19. `val/loss` study-style alias in the artifact callback.
    Status: implemented.

20. Lightning train/val total-loss metrics:
    `train/loss_total`, `val/loss_total`.
    Status: already present and kept.

21. Study-style block-loss aliases:
    `train/loss_block_hamiltonian`, `train/loss_block_overlap`, `train/loss_block_density`, `train/loss_block_total`.
    Status: implemented.
    Main path: `src/net/e3gnn.py`

22. Weighted observable loss metrics:
    `train/loss_energy_weighted`, `train/loss_num_electrons_weighted`, `train/loss_forces_weighted`.
    Status: implemented.

23. Observable MAEs:
    `train/energy_mae`, `val/energy_mae`, `train/num_electrons_mae`, `val/num_electrons_mae`.
    Status: implemented.

24. Force MAE / MSE from the Lightning training path.
    Status: implemented in core model logging.
    Main path: `src/net/e3gnn.py`

25. Force MAE / MSE mirrored into the artifact callback’s study-style initial/final payloads.
    Status: implemented.

26. Pre-correction electron-count MAE:
    `num_electrons_mae_pre_correction`.
    Status: implemented.
    Notes: backed by the new `rescale_density_to_num_electrons` metrics-only correction path.

27. Per-irrep WandB metrics with matrix prefixes (`H_`, `S_`, `D_`) for:
    `l1_elem`, `l2_elem`, `l1_block`, `l1_block_rel`, `l2_block`, `l2_block_rel`, `l1_full_rel`, `l2_full_rel`.
    Status: implemented.

28. Learning-rate logging.
    Status: already present and kept.

29. Grad-norm logging.
    Status: implemented.
    Main path: `src/net/e3gnn.py`

30. Removal of low-value edge-count / edge-ratio metrics that were not part of the study.
    Status: implemented.
    Main path: `src/net/e3gnn.py`

## Checkpoints / Restartability

31. Latest checkpoint written during training.
    Status: implemented.

32. Best checkpoint written by monitored validation score.
    Status: implemented.

33. Final checkpoint written at fit end.
    Status: implemented.

34. WandB summary entries pointing to latest / best / final checkpoints.
    Status: implemented.

35. Resume support from explicit checkpoint file or run directory.
    Status: implemented.
    Main path: `scripts/train_silicon.py`, `src/net/e3gnn.py`

## Artifacts / Media

36. Per-epoch frame images under `frames/<matrix>/frame_epoch_*.png`.
    Status: implemented.

37. `video_max_atoms` cropping for training-video frames.
    Status: implemented.
    Main path: `src/net/artifacts.py`

38. Final DOS comparison plot saved locally and uploaded to WandB.
    Status: implemented.

39. Final DOS scalar diagnostics added to final WandB metrics.
    Status: implemented.

40. Final distance-error curve PNG per matrix.
    Status: implemented.

41. Final distance-error curve JSON per matrix.
    Status: implemented.

42. Final per-irrep Hamiltonian images.
    Status: implemented.

43. Final per-irrep overlap images.
    Status: implemented.

44. Final per-irrep density images.
    Status: implemented.

45. Final training-progress videos uploaded to WandB.
    Status: implemented as MP4.

## Logging Schedule / Evaluation Behavior

46. Adaptive logging schedule:
    epochs 1-10 every epoch, 11-100 every 10, afterwards every `log_interval`.
    Status: implemented.

47. Validation-split evaluation for artifacts and study-style summary logging.
    Status: implemented.

48. Train-split fallback when validation is unavailable.
    Status: implemented.

## Cleanup / Differences vs Pre-Port Main Code

49. Removed study-irrelevant edge partition WandB metrics.
    Status: implemented.

50. Kept core Lightning metrics that are still useful even if not study-authentic:
    matrix MAE/MSE, total losses, per-irrep losses, LR, and optional regularization metrics.
    Status: intentional.

## Net Result

Implemented now:
- the study-style console banners that are practical in the Lightning path
- the study-style WandB Hamiltonian / matrix / per-irrep metrics
- the useful checkpoint / restart surface
- the full artifact stack requested here: DOS, distance curves, per-irrep H/S/D images, frame sequences, MP4 videos, WandB uploads
- removal of the noisy edge-count / edge-ratio metrics

Still missing:
- none from the originally identified silicon-study logging / visualization gap list
