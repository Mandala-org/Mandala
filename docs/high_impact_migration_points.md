# High-Impact Migration Points

This note captures the first-pass, highest-impact architectural and training differences identified while comparing:

- `src/`
- `studies/minimal_overfit_study`
- `studies/minimal_silicon_study`
- `studies/minimal_overfit_observables`
- `external/DeepH-E3`

It is intentionally limited to the top tier and excludes logging/misc.

1. Message-block order
   Recommendation: hard-wire `node -> edge`.
   `src` currently updates edges before nodes; both studies and DeepH-E3 update nodes before edges.

2. Node update kernel
   Recommendation: discard the simplified study node update.
   Keep the SH-aware, DeepH-style node message construction rather than the study simplification `aggregate edge -> Linear -> Gate -> e3LayerNorm`.

3. Edge encoder family
   Recommendation: do not migrate the full study edge encoder wholesale.
   If exposed at all, keep a named switch between the existing Mandala encoder and a DeepH-E3-style path.

4. Onsite vs offsite readout
   Recommendation: hard-wire zero-shift self edges to use node embeddings and non-self edges to use edge embeddings.
   This matches the physical split better and is closer to DeepH-E3.

5. Shifted-self periodic blocks
   Recommendation: keep as an option, default `false`.
   Same-atom, non-zero-shift edges can be treated separately from both true diagonal and ordinary off-diagonal blocks.

6. Self-edge SH semantics and legacy geometry modes
   Recommendation: hard-wire zeroing of non-scalar SH on zero-shift self edges, and discard legacy SH / radial-scaling modes.
   The aligned convention is the `src` / DeepH-E3-style one.

7. Hamiltonian target symmetrization
   Recommendation: hard-wire symmetrization of Hamiltonian targets in data prep.
   The studies do this explicitly; `src` currently only symmetrizes predictions at loss time.

8. Force supervision
   Recommendation: keep as an option, default `false`.
   If enabled, recompute edge features from grad-enabled positions instead of reusing cached/precomputed edge features.

9. Training target space
   Recommendation: simplify main code to matrix-space supervision only, and keep irrep-part loss decomposition as an option default `false`.
   The studies train in matrix space and only optionally decompose the loss by irrep.

10. Silicon normalization strategies
    Recommendation: keep per-class block normalization and fitted distance-magnitude normalization as options, both default `false`.
    They are meaningful training knobs but add complexity and should not be hard-wired without stronger evidence.

Additional bug noted during comparison:

- `src/net/e3gnn.py` expects `compute_graph_features()` to return `index_gnn_cutoff`, but `src/data/graph_features.py` no longer returns it. The on-the-fly edge-feature path is therefore broken and must be fixed.
