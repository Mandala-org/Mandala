# Framework requests from MANDALA Studio

Resolved (November 2026):

1. **Pass-through options.** Every transform with declared ports or options
   now has a complete schema and rejects unknown options; the former
   pass-through transforms (DeepH, DeepH-E3 and MACE-H layers) are blocks of
   fully described stages.
2. **Objective term inputs.** `register_objective_term(operators=(...))` names
   the options that select predicted operators; the catalog lists them and the
   editor draws one port per name.
3. **Port metadata for block instances.** `net.blocks.block_ports` derives each
   block's port contract; analyses report it under `blocks` and the catalog
   under `block_library`, and the editor uses the compiler's contract.
4. **Cost model constants.** `src/net/cost_constants.json` holds per-transform
   constants fitted from CPU stage profiles; the GPU benchmark batch
   (`scripts/benchmarks/run_pipeline_benchmarks.sh`) refits them on the GPU.

Open:

- A disabled block instance is shown as a single bypassed node; its inner
  stages are not listed.
