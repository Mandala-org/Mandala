# Critical evaluation of the modularity refactor (branch `modular-studio`)

Scope: everything on the branch relative to `3ec49ae`. That covers four pieces of work:

- the typed stage registry, blocks and pipeline;
- the decomposed MANDALA, DeepH, DeepH-E3 and MACE-H models;
- the envelope moved into the pipeline;
- the objective expressions, cost model and MANDALA Studio.

Sources:

- my own review;
- three independent read-only critics (framework design; numerics and performance; Studio);
- targeted tests and short probe scripts.

Findings marked **fixed** were corrected during this evaluation, and each fix has a test.

## Verdict

The refactor reaches its main goal. Every supported model is a recipe of small registered stages. The reference layers are readable library blocks, and every choice can be swapped from YAML or the Studio.

The equivalence evidence is strong:

- The decomposed blocks reproduce the old monoliths: seeded initialization bit for bit, outputs and all gradients to 1e-12.
- All six pinned upstream oracles still pass.
- The raw-training oracle compares every parameter and Adam update at its original tolerance.

The weak spots:

- Some mechanisms exist only to reproduce reference implementations (`initialization_order`).
- Some stages overlap.
- A set of non-default options of the old reference layers was dropped.
- Speed is unmeasured on GPU.
- E3GNN still owns several model decisions outside the pipeline.

None of these blocks use. Several should be addressed before the API is called stable.

## What was achieved (evidence)

| Goal | Status | Evidence |
| --- | --- | --- |
| Interchangeable parts | Achieved | ~60 registered transforms, all with complete port/option schemas (no pass-through options left); 8 library blocks; flat MANDALA config lowers to 4 blocks of stages |
| Reference models as recipes | Achieved | `configs/reference/{deeph,deeph_e3,mace_h}.yaml` import `block_libraries`; hybrids mix them |
| Numerical equivalence | Achieved | `tests/physics/test_reference_decomposition.py` (24 layer variants, init rtol=0, outputs/grads 1e-12); `test_reference_fixtures.py` (6 upstream oracles); raw-training oracle (all parameter values, original tolerance); MANDALA monolith fixtures (≤2e-13) |
| Ablation through recipes | Achieved, limited | `enabled: false` on stages and block instances; only single-output stages |
| Envelope inside the pipeline | Achieved | `pair_envelope` (static, cached) + `envelope_scale`; flat switches lower to them for any recipe; equivalence test against post-hoc scaling |
| Natural abstractions | Mostly | see design findings below |
| No big efficiency penalty | **Unverified** | CPU profiles only; GPU A/B batch prepared (now with an uncached arm to isolate dispatch cost) |
| Studio | Achieved | typed wiring, inferred vs explicit values, blocks/library, costs, expression plugins, YAML, walkthrough |

## Fixed during this evaluation

- **Multi-graph batches used the union envelope** (`e3gnn.py`).
  - What went wrong: the loss of each graph read `edge_envelope` (and auxiliary outputs for `output_error`) from the whole batch, indexed with per-graph partitions. Every graph after the first silently got another graph's envelope.
  - Fix: each graph now carries its slice of the pipeline outputs.
  - Test: `test_batched_graphs_read_their_own_envelope_slice`.
- **Envelope consistency was keyed on a name.**
  - What went wrong: the model only checked that an output `edge_envelope` existed. A recipe with `pair_envelope` but without `envelope_scale` would divide unscaled predictions by the envelope.
  - Fix: E3GNN now derives the scaled operators from the compiled graph. It rejects envelope modes whose operators are not produced by `envelope_scale`, and `with_envelope` also looks inside blocks.
- **Building the static-only pipeline consumed random draws.** The dataset's lazy construction could therefore shift model initialization. The random state is now restored (tested).
- **Bypass details.**
  - Disabled stages skipped option validation; they now validate.
  - `enabled: null` silently enabled a stage; it now raises.
  - Disabling the skip in the DeepH node update or the MACE product dropped the update. Both now use `residual` (bypass passes the update).
  - Bit-for-bit equivalence was re-checked after the change.
- **`irrep_linear.degrees_of` silently lost to an explicit `irreps_out`.** Conflicting values now raise.
- **Benchmark batch.**
  - MACE-H could not run, because its output statistics were unfitted.
  - Only the MANDALA model was A/B-tested.
  - The A/B was confounded, since only the modular tree cached static stages.
  - All three are fixed: the reference models are added, there is an uncached arm, and parameter counts are compared.
- **Studio.**
  - Wires could not be selected or deleted.
  - A node deletion took several undos and could corrupt variadic inputs.
  - Keyboard shortcuts acted behind dialogs.
  - The guide ignored Escape.
  - The bypass toggle was offered where bypass is invalid.
  - Disabled block instances could be expanded.
  - Auxiliary outputs such as `edge_envelope` were flagged as invalid targets.
  - The walkthrough made several inaccurate claims.

## Remaining findings (most important first)

### Correctness and robustness

1. **Per-graph outputs depend on module state.**
   - `E3GNN` keeps the last forward's outputs in `_last_outputs`. `_pipeline_outputs` falls back to it for unsplit inputs.
   - A new per-graph code path that forgets to attach its slice would silently read union values.
   - Graph-domain and message-support outputs are not sliced.
   - The tensors stay alive until the next forward.
   - Recommendation: pass outputs explicitly to the loss and clear the state after the step.
2. **`pair_envelope` species order.**
   - The inferred default is the orbital configuration's alphabetical order, while `edge_type_idx` follows `species.order`.
   - The flat lowering passes the order explicitly. A hand-written stage must use `${species.order}`; this is documented but not enforced.
   - Recommendation: make the option required in recipes.
3. **The static cache ignores file contents.**
   - The fingerprint hashes options, so a `pair_envelope` table refitted in place is not detected by an on-disk sample cache.
   - A fingerprint mismatch silently falls back to per-step recomputation, which is a performance cliff with no warning.
4. **MACE density normalization is now two independent switches** (`graph_norm`, `degree_scale`). Setting only `graph_norm: false` yields an unnormalized sum without warning. The expansion output is never degree-scaled inside the block.
5. **`normalize_target` changes the loss slightly.**
   - The loss now compares `physical / clamp(env, eps)`, where it used to compare the raw prediction.
   - The two are identical wherever the envelope exceeds `eps`.
   - Below `eps` the gradient of an edge is scaled by `env / eps`, and it is zero where the envelope vanishes. Previously the raw prediction there was trained against `target / eps`.
6. **Coverage gaps.** There are no tests for:
   - spinful/complex blocks through `envelope_scale`;
   - forces with a cached `pair_envelope`;
   - a batched multi-graph run of a reference recipe.

   The upstream fixture test widened the moment tolerance for Adam-updated parameters. Measured, the old bound is exceeded by 3% (sum) on one oracle (MACE-H spinful). That is consistent with the lr/eps amplification of round-off-level gradients. Initial parameters and gradients still match far inside the old tolerance.

### Design

7. **`initialization_order` is a reproduction device, not an architectural concept.**
   - It exists so that a seeded MANDALA model starts from exactly the reference weights; DeepH redraws its message weights after the normalization.
   - Mechanically it rebuilds every parameterized stage once more, doubling construction time.
   - Cleaner alternative: a two-phase compile (infer contracts, then build once in the declared order), or moving reference-parity seeding into a test-only weight-loading path and dropping the key from recipes.
8. **Overlapping stages.**
   - `self_connection` = `species_tensor_product` + `add`.
   - `endpoint_concat` overlaps with `gather_rows` + `concat`.
   - `scale_features`, `envelope_scale` and `constant_scale` are all multiplications.
   - `equivariant_convolution` (MANDALA) and `factorized_tensor_product` + `radial_channel_scale` (DeepH-E3) are two convolutions with different radial-MLP conventions. The MANDALA radial MLP keeps its historical quirk: no activation after a hidden layer whose width equals the output width. This was kept for checkpoint equivalence.
   - Recommendation: one primitive per operation, with blocks as sugar.
9. **`residual` defaults to `add_if_matching`.** It silently drops the skip when irreps differ, which is a hidden ablation kept for legacy equivalence. Recipes should prefer `add` (fail loudly) or `project`.
10. **Inconsistent vocabulary.**
    - Endpoint direction has three spellings: `gather_rows.endpoint`, `indexed_aggregation.receiver` and `endpoint_concat.order`.
    - Output size is named `out_features` / `width` / `irreps_out` / `number` / `n_gaussians`.
    - `simplify` has different defaults in `concat` and `endpoint_concat`.
    - There are two activation stages with different activation sets.
    - Port kinds do not match the representation names.
11. **Hidden coupling in `irrep_linear.degrees_of`** (a MACE-H projection rule living in a generic stage). It is now checked, but it is still a special case.
12. **Bypass semantics depend on input order.** The passed input is the first one, or the transform's declared `bypass` port. Blocks and multi-output stages cannot be bypassed. A named "update" port would be clearer.
13. **Still outside the pipeline**, in `E3GNN`:
    - target normalization and loss weighting;
    - Hermitian projection;
    - block-vector to shell conversion;
    - geometry derivation, driven by slot-name sets;
    - edge-role metadata from `separate_shifted_self`;
    - about 15 cross-setting validity checks;
    - selection of readouts in the `heads` view by name suffix.
14. **Block port contracts are heuristic.** `block_ports` takes the port of the first inner consumer of each input.

### Lost capabilities

15. **Non-default options of the old reference monoliths are no longer expressible.**
    - DeepH-E3:
      - gate after aggregation (`nonlin=True`)
      - `only_ij`
      - `init_edge`
      - `concat=False`
      - the convolution's internal norm
      - disabling the block norm or the residual
      - mean aggregation
      - custom gate activations
    - DeepH:
      - disabling the distance envelope
      - mean aggregation
      - other activations
    - These options fail loudly; none is used by a shipped recipe. Most are cheap to restore as block options (`$aggregation`) or `enabled` toggles.
16. **Old checkpoints of DeepH, DeepH-E3 and MACE-H do not load.** Only MANDALA monolith checkpoints are translated (`legacy_weights.py`). A name map exists in `tests/reference_parameters.py` and could become a translator.

### Efficiency (unverified)

17. **Dispatch overhead.**
    - DeepH-E3 grew from about 13 to about 75 stages, and MACE-H similarly.
    - Python dispatch is roughly 5–15 µs per stage, about 1 ms per forward. That is negligible for large graphs on GPU and visible for small ones.
    - `radial_channel_scale` replaces per-irrep slicing with one gather (roughly neutral).
    - `normalize_target` adds two elementwise passes.
18. **Provisional cost constants.**
    - They come from small CPU structures (8–32 atoms). Elementwise and scatter stages have no operation count, so their fit is overhead-only and poor (median relative errors up to ×100).
    - Studio times are labelled "(cpu)" and are good for relative comparisons only.
    - The GPU batch refits them.

### Studio

19. **Accessibility and missing affordances.**
    - Term operator ports are labelled only in tooltips.
    - Inventory items are not keyboard-operable.
    - There is no per-input disconnect button.
    - A library block cannot be re-added pristine once a same-named block is in the recipe.
    - Recipes that import a library are inlined on import, so an exported recipe no longer follows library updates.

## Recommended next steps

1. Run the GPU benchmark batch. It now includes the reference models and an uncached arm, and refits the cost constants.
2. Pass pipeline outputs to the loss explicitly instead of keeping them on the model (finding 1).
3. Replace `initialization_order` by a two-phase compile, or by test-only weight loading (finding 7).
4. Consolidate overlapping stages and option vocabulary, before recipes depend on them (findings 8–10).
5. Restore the cheap dropped options as block options (finding 15).
6. Add a checkpoint translator for the reference models (finding 16).
