# Modularity overhaul and MANDALA Studio — implementation plan

Branch: `modular-studio`. Backward compatibility of recipes, checkpoints and
private APIs is not a goal; numerical equivalence of the default MANDALA model
is (verified by a weight-mapped comparison against the pre-refactor model).

## Goals

1. Every architectural choice of every supported model (MANDALA, DeepH,
   DeepH-E3, MACE-H, hybrids) is a stage of the typed pipeline or an option of
   one, replaceable from a recipe.
2. One uniform, self-describing transform registry that a GUI can enumerate:
   ports, options (with key/advanced flags and inferred defaults), category,
   parameter count and an asymptotic cost model.
3. Reusable, named composite blocks in recipes (the same abstraction as GUI
   grouping), so the paper model is a short, readable recipe.
4. Losses are a symbolic expression over registered objective terms.
5. No measurable performance penalty: same kernels, static geometry still
   cached, Python dispatch per stage only.

## Part A — framework

### A1. Registry (`net/registry.py`)

```python
@register_transform(
    "equivariant_convolution", category="convolution",
    summary="Gated tensor product of features with edge harmonics, weighted by a radial MLP",
    inputs=[Port("features", "edge", "irreps"), Port("harmonics", "edge", "irreps"),
            Port("radial", "edge", "invariant"), Port("pair_type", "edge", "index", optional=True)],
    options=[Option("irreps_out", "irreps", key=True, infer="same_as:features"),
             Option("tensor_product", "choice", "separate_weight", choices=[...], key=True),
             Option("radial_layers", "int_list", [64, 64]), ...],
    cost=cost.tensor_product(...),
)
def build(options, inputs, mapper): ...
```

* `TransformInfo` records factory, category, summary, ports, options, cost,
  variadic flag. `register_transform(name)` without metadata still works for
  plugins; metadata is optional and introspected where absent.
* Options are completed from the schema before the factory runs; the compiled
  pipeline stores `resolved_options` per stage and which values were given
  explicitly. Inference rules (`same_as:<port>`, layout variables) run in the
  wrapper, so the GUI sees inferred values.
* All `if name == ...` built-ins of `build_transform` move into registered
  factories (`net/transforms/builtin.py`). Re-registration requires
  `override=True`.
* Factories may refine an input contract (`TransformResult.refined_inputs`);
  the compiler's name-based special cases (`local_coordinate_message`,
  `spherical_harmonics`) are removed.
* `transform_catalog()` returns JSON metadata for every registered transform.

### A2. Composite blocks and introspection

* `pipeline.blocks`: named subgraphs with declared `inputs`, `outputs`,
  `options` (defaults) and `stages`; option values substitute `${name}` inside
  the block. A stage whose `transform` names a block is expanded at compile
  time into stages `<stage>/<inner>`; nested blocks allowed.
* `FeaturePipeline.groups` keeps the composite tree (for summaries, parameter
  aggregation and GUI collapse state).
* `net/inspection.py: analyze_pipeline(spec, system)` compiles a recipe for a
  declared species/basis system and returns per-stage output specs, resolved
  options (explicit vs inferred), parameter counts, cost estimates and
  per-stage errors (compilation continues after an error by poisoning
  downstream slots), cached per (transform, options, input specs).

### A3. Fine-grained MANDALA stages

The monolithic `NodeEncoder`, `EdgeEncoder`, `NodeUpdateBlock`,
`EdgeUpdateBlock`, `MessageBlock` and `DeepHead` are replaced by stages:

| Stage | Replaces |
| --- | --- |
| `embedding` (+ stage initialization) | node/pair type embeddings |
| `soft_one_hot_basis` | Gaussian radial embedding computed outside the pipeline |
| `spherical_harmonics` | edge harmonics computed outside the pipeline |
| `tensor_square`, `concat`, `fc_tensor_product` | rich edge encoder |
| `e3mlp` (variant, layers, activate_last, routing) | lin_pre, lin_post, refine, trunks |
| `endpoint_concat` (`combine`: concat/sum/tensor_product) | node[src], node[dst], edge assembly |
| `equivariant_convolution` | `EquiConv` (TP, gate, radial MLP, pair-conditioned radial) |
| `scatter_aggregation` (sum/mean), `projected_attention` | message aggregation |
| `self_connection` | species/pair one-hot self-connection |
| `irrep_nonlinearity` | `make_nonlinearity` |
| `layer_norm`, `irrep_dropout`, `add`, `residual` | the rest of an update block |
| `pair_projection` (routed by role × species pair) | DeepHead projections and log scales |

Routing (`route: none | species | pair | role | role_pair`) gives any
per-entity layer separate parameters per category, computed on index subsets;
DeepHead's diag/shifted-self/off-diagonal trunks are `e3mlp` with
`route: role`. `edge_role` becomes a built-in input slot.

The flat `Config` lowers (`default_graph`) into these stages, grouped into
composite blocks `node_encoder`, `edge_encoder`, `node_update`, `edge_update`
and `readout`, and `configs/reference/mandala_paper.yaml` gains an explicit
recipe form. A weight-mapping function converts pre-refactor checkpoints and is
used by the equivalence test (identical outputs to float64 round-off).

### A4. Geometry inside the pipeline

`edge_length_emb` and `edge_sh` become stage outputs. Parameter-free stages
whose inputs are all static geometry are marked `static`; the dataset
precomputes them (`FeaturePipeline.evaluate_static`) when
`precompute_edge_features` is set, and the forward pass skips static stages
whose outputs are present and whose geometry does not require gradients.
`E3GNN.forward` no longer names slots: geometry requirements are declared by
input specs.

### A5. Objective registry (`net/objectives.py`)

* `register_objective_term(name, reads=..., options=...)`: a term maps an
  `ObjectiveContext` (predicted/target operators, observables, raw outputs,
  batch, model) to a scalar.
* Built-in terms: `matrix_error(operator, kind, reduction)`, `band_energy`,
  `electron_count`, `forces`, `stress`, `spectral`, `l1`, `l2`,
  `output_error(output, target)`.
* `objective.terms` + `objective.expression` (safe AST: numbers, term names,
  `+ - * /`, unary minus, `sqrt log exp abs min max`). The flat config lowers to
  the same form, so the trainer has one total-loss path; `_graph_step` keeps
  metric logging and computes only the terms the expression references.

### A6. Smaller items

* Pipeline outputs may be any domain; non-operator outputs are returned as raw
  tensors and are available to `output_error`.
* `node_enc`/`edge_enc`/`mp_blocks`/`heads` accessors are removed in favour of
  `pipeline.stages_with_role(role)`.

## Part B — MANDALA Studio (`tools/studio`)

* Backend (`src/studio/`, Flask): `GET /api/catalog`, `POST /api/analyze`,
  `POST /api/import` (YAML recipe or flat config → graph), `POST /api/export`,
  `GET /api/presets`, `POST /api/expression/check`, `POST /api/plugin`
  (generates `plugins/<name>.py` from an expression block). It uses only the
  public framework functions above, so the GUI shows exactly what the trainer
  would build.
* Frontend (React + TypeScript + Vite + `@xyflow/react` + `elkjs`), styled
  like a shader graph editor:
  * inputs (geometry/chemistry) at the top, operator targets at the bottom,
    the loss expression graph below them; auto layout (ELK layered) with
    top-to-bottom or left-to-right orientation;
  * typed ports coloured by domain, labelled with irreps; connection
    validation on drag (domain, representation, support, invariance,
    frame, index ports) plus authoritative backend validation after each edit;
  * inferred values shown italic/dimmed, explicit values solid, with a reset
    button; key options inline, advanced options collapsed;
  * parameter count per block (log-scaled bar + number), estimated time per
    block from the cost model; blocks above 10 % of forward time scaled by
    area proportionally to their share;
  * select → "Group into block" creates a named composite with collapse /
    expand; composites export as recipe `blocks`;
  * expression blocks: typed equivariant DSL, live type check, "export as
    plugin" writes a registered Python transform;
  * YAML import/export, presets, undo/redo, system panel (species, basis,
    reference system size for cost estimates).

## Part C — performance and verification

* `scripts/benchmarks/pipeline_benchmark.py`: forward/backward wall time,
  peak memory, per-stage timing (CUDA-synchronized hooks) for the paper model
  and the reference recipes on a fixed synthetic structure set; writes JSON
  and fits per-transform cost constants (`studio/cost_constants.json`).
* `scripts/benchmarks/run_pipeline_benchmarks.sh`: frozen zero-argument
  launcher (1 GPU, 4 CPUs) that runs the benchmark on this branch and on the
  pre-refactor commit for an A/B comparison.
* Targeted tests: registry/catalog, composite expansion, inference, routing,
  weight-mapped equivalence of the decomposed MANDALA model, equivariance of
  new stages, objective expressions, static-stage caching, Studio API.
* Critic subagents review framework modularity, correctness and Studio UX;
  findings are fixed before hand-off.

## Behaviour differences from the monolithic implementation

Every configuration reproduces the monolithic model to round-off from mapped
weights, with one deliberate exception: the monolithic update blocks added the
residual whenever input and output had the same *width*, even when their
irreps differed (a non-equivariant sum). The ``residual`` stage adds only when
the irreps are equal (``add_if_matching``). Checkpoints of configurations with
equal widths but different irreps (e.g. ``hidden_base_dim: 8`` with
``hidden_irreps: 5x0e+1x1o`` and residuals on) therefore give different
predictions.

The DeepH, DeepH-E3 and MACE-H layers are library blocks
(`src/net/block_library`). They reproduce the former monolithic layers
(seeded initialization exactly; outputs and gradients to round-off) for every
option the shipped recipes use. Non-default options of the old monoliths
that have no stage equivalent are no longer accepted. These are, for DeepH-E3:
gating after aggregation (`nonlin`), `only_ij`, `init_edge`, `concat=False`,
the convolution norm, mean aggregation and custom gate activations. For DeepH:
disabling the distance envelope, mean aggregation and other activations.
Checkpoints of these three models from before the decomposition are not
translated.

The Hamiltonian envelope is computed by pipeline stages (`pair_envelope`,
`envelope_scale`); models predict physical blocks. With `normalize_target` the
loss compares prediction and target, both divided by the envelope clamped at
`hamiltonian_envelope_eps`. This equals the former comparison wherever the
envelope exceeds that value.

See `refactor_evaluation.md` for a critical evaluation of the whole refactor.
