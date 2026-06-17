# ZnCuSnSeS Radial Support Strategy

## Context

Current best ZnCuSnSeS Hamiltonian model:

- Checkpoint:
  `/data/home2/brzoza73/casus/mandala/checkpoints/ZnCuSnSeS_hamiltonian_restart_hkgmr1y6_47h/zncusnses-restart-hkgmr1y6-47h-seed43-scale1/best_model.pt`

Main observation from the dataset:

- Different atom-pair families have visibly different support radii.
- Different atom-pair families also taper off with different decay shapes.
- The target Hamiltonian block magnitudes span many orders of magnitude.

Reference plot:

- [hamiltonian_block_magnitude_vs_distance_all_pairs.png](/home/bartek/casus/mandala/eval_outputs/block_mag_vs_distance_interesting/hamiltonian_block_magnitude_vs_distance_all_pairs.png)

## Plot Comments

### 1. Block magnitude vs distance

The magnitude-vs-distance plot suggests that atom-pair families are not just rescaled copies of one another.

- Some pairs remain large further out.
- Some pairs fall off smoothly.
- Some pairs look close to compactly supported, with a sharp collapse over a short radial window.
- The same global cutoff/radial basis is therefore a poor inductive bias for all pairs simultaneously.

This strongly suggests that the model should not treat radial behavior as globally shared with only a mild pair-type correction.

### 2. Absolute error vs edge length

The absolute MAE appears to broadly decrease with distance.

- This is not surprising because far-away blocks are usually smaller.
- It means the model is not catastrophically wrong in absolute units on long-range interactions.
- However, this can be misleading if the far-away blocks are tiny and numerous.

Takeaway:

- Absolute error alone understates the difficulty of long-range prediction.

### 3. Relative error vs edge length

Relative error becomes much worse at long distances.

- This is the clearest sign that the model does not capture the long-tail / taper structure correctly.
- The model is often "close in absolute terms" but still very wrong compared to the true tiny value.
- This is exactly the regime where support modeling, pair-conditioned radial behavior, or distance-aware normalization become valuable.

Takeaway:

- The current model is under-biased for the long-range decay law.

### 4. Absolute error vs block magnitude

Absolute error increases with target block magnitude.

- This is expected and not necessarily bad.
- The model is spending most of its absolute error budget on the large blocks.
- That is usually acceptable physically, but we need to make sure tiny far-away blocks are not either ignored completely or overweighted in relative metrics.

### 5. Relative error vs block magnitude

Relative error is worst on the smallest-magnitude blocks.

- This is the complementary view of the radial relative-error plot.
- Tiny blocks are hard to predict proportionally.
- If there are many such blocks, they can distort loss design or give a misleading impression of failure.

Takeaway:

- We need a principled way to model support and to weight distance / magnitude regimes.

### 6. Per-pair validation loss screenshot

The pairwise loss plots suggest non-uniform pair difficulty.

- Some pair families sit persistently higher than others.
- Some pair families improve quickly and then plateau.
- A few pair families look relatively flat, which may indicate either:
  - genuinely easy behavior, or
  - a regime where the metric is dominated by a near-constant magnitude band.

The most important conclusion is not the exact ordering, but that pair families do not behave homogeneously enough for a single shared radial law to be ideal.

## Current Model Mismatch

What the current model already has:

- Pair awareness through `edge_type_idx` and the edge-type embedding.
- Pair-specific output heads in split mode.

What it does not have explicitly:

- Pair-specific distance normalization.
- Pair-conditioned radial basis behavior.
- Pair-specific smooth support / taper envelopes.
- Pair-aware distance-dependent loss normalization.

Important code fact:

- The current radial embedding is one global soft one-hot basis over `[0, cutoff_radius]`.
- This basis is shared across all pair types.
- It also does not encode an explicit pair-dependent cutoff envelope.

Relevant files:

- [src/data/graph_features.py](/home/bartek/casus/mandala/src/data/graph_features.py)
- [src/net/encoders.py](/home/bartek/casus/mandala/src/net/encoders.py)
- [src/net/heads.py](/home/bartek/casus/mandala/src/net/heads.py)

## Goals

We want a model family that:

- Predicts pair families with different support radii more naturally.
- Handles many orders of magnitude in block size without becoming numerically unstable.
- Does not over-focus on unimportant tiny far-away blocks.
- Still preserves good accuracy on the large chemically important short-/mid-range blocks.
- Stays close enough to the current best architecture that a narrow sweep is still meaningful.

## Selected Directions

### Highest-priority implementation direction

- Pair-conditioned radial MLP.

Rationale:

- It directly targets the main observed deficiency.
- It is more flexible than a hard-coded envelope.
- It should be implementable without fully reconstructing the model.

### Strong companion idea

- Pair-specific distance normalization.

Rationale:

- It can put different pair families into a more comparable radial coordinate system.
- It should make pair-conditioned radial learning easier.

### Worth keeping on the roadmap

- Shell-aware or basis-aware decomposition.
- Support prediction depending only on atom-pair and distance.
- Distance-envelope factorization, but only after empirical study of the decay shape.

## Open Questions

### 1. What envelope family should we use, if any?

We should not hard-code this before checking the data.

Possible families to compare:

- Exponential: `exp(-a r)`
- Gaussian-like: `exp(-a r^2)`
- Soft support: `sigmoid((R - r) / tau)`
- Piecewise log-linear
- Learned monotone spline / flexible monotone function

### 2. How do we handle many orders of magnitude?

Likely answer:

- Analyze and fit in log-magnitude space.
- Treat the envelope as fundamentally multiplicative.
- Compare behavior in both raw magnitude and `log10(magnitude + eps)`.

### 3. How do we avoid overweighting weak far-away blocks?

Promising approaches:

- Pair-balanced loss.
- Distance-binned loss normalization.
- Magnitude-aware weighting with a floor.
- A two-head setup:
  - support/activity prediction
  - value prediction conditional on support

## Recommended Implementation Sequence

### Phase 1: Preliminary empirical study

Do this before changing the model.

Objectives:

- Characterize the radial decay family for each atom pair.
- Determine whether atom pair alone is enough, or whether orbital/block family matters.
- Determine whether the taper looks smooth, sharp, or multi-regime.
- Quantify where current model error mass sits by pair, distance, and magnitude.

### Phase 2: First model variants

Implement as options:

- Pair-conditioned radial MLP.
- Pair-specific distance normalization.

Optional third variant if Phase 1 strongly supports it:

- Support/gating branch using only pair type and distance.

### Phase 3: Fresh narrow sweep

Keep the architecture close to the best-so-far model.

Sweep mostly over:

- new radial variants
- small support/gating options
- a narrow set of existing hyperparameters around the current best recipe

Do not start with a broad from-scratch architecture search.

## Preliminary Study Plan

### Study questions

1. For each atom pair, what functional family best describes block magnitude vs distance?
2. Is the decay law sufficiently well described by atom pair alone?
3. Are there distinct radial regimes:
   - strong short-range
   - medium-range taper
   - near-zero tail
4. Which regimes dominate:
   - target mass
   - absolute model error
   - relative model error

### Data products to compute

For each Hamiltonian block edge:

- atom pair key
- distance
- block magnitude
- log block magnitude
- model absolute block error
- model relative block error
- optionally orbital/block family identifier

Suggested block magnitude:

- Use the same magnitude definition as the current reference plot so comparisons stay direct.

### Fits to compare

Per atom pair, fit several simple candidate curves:

- raw-space exponential
- raw-space Gaussian-like
- log-space linear / piecewise linear
- soft cutoff sigmoid
- soft cutoff times exponential tail

Important:

- Evaluate goodness of fit in both:
  - raw magnitude space
  - log magnitude space

### Required visual artifact

Produce an artifact analogous to the current reference plot, but with fitted overlays.

Suggested primary artifact:

- `hamiltonian_block_magnitude_vs_distance_all_pairs_with_fits.png`

Contents:

- same scatter points as the current plot
- overlaid fitted curves for each atom pair
- clear legend labeling the pair family and fit family

Suggested companion artifacts:

- one per-pair panel figure with larger readable overlays
- residual-vs-distance plots per pair
- fit-quality summary table in markdown or CSV

### Required agreement view

The overlay plot should make it easy to judge:

- whether one family clearly underfits sharp taper behavior
- whether some pairs need different functional families
- whether the tails are smooth enough for a simple envelope

### Error-stratification artifacts

Produce:

- absolute error vs distance by pair
- relative error vs distance by pair
- block magnitude vs distance by pair with model predictions overlaid

This is needed to decide whether the future loss should equalize by:

- pair
- distance bin
- magnitude bin
- or a combination

### Decision rule after the study

If atom-pair-only fits are already good:

- prioritize pair-conditioned radial MLP
- optionally pair-specific distance normalization

If atom-pair fits are not enough and different block families behave differently:

- escalate to shell-aware / basis-aware decomposition

If the data shows a very clear support boundary:

- implement a support/gating branch or explicit smooth envelope option

## Most Useful Next Checks

1. Build the preliminary study script and artifacts.
2. Fit several simple radial families per pair and compare them visually.
3. Check whether pair-only structure is enough before adding shell-aware complexity.
4. Use the study results to decide whether the first new sweep should include:
   - only pair-conditioned radial MLP
   - pair-conditioned radial MLP + pair distance normalization
   - or also a support-only gating branch
