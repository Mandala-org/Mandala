# MANDALA Studio

A visual editor and inspector for MANDALA model recipes. Models are drawn as
blocks connected by typed wires: structure inputs at the top, the predicted
operators below the network, and the training objective as an expression graph
at the very bottom. Every edit is compiled by the framework itself, so the
editor shows exactly what training would build: output irreps, parameter
counts, estimated forward time and every contract violation.

## Install and run

The backend runs in the MANDALA environment; the frontend is built once with
Node.js 18 or newer.

```bash
source mandala-venv/bin/activate
cd tools/studio/web && npm install && npm run build && cd ../../..
PYTHONPATH=src:. python -m studio --port 8765
```

Open <http://127.0.0.1:8765>. Plugins in `plugins/` are registered at start-up;
relative `plugins:` paths in recipes resolve against the repository root.

For frontend development, run the backend as above and, in a second shell:

```bash
cd tools/studio/web && npm run dev
```

The Vite server on port 5173 proxies `/api` to the backend.

## Working with the editor

| Action | How |
| --- | --- |
| Open a recipe | **Open**: a blank recipe, reference and hybrid recipes, pilots, examples, or pasted/loaded YAML |
| Add a block | Drag from the inventory onto the canvas, double-click it, or focus it and press Enter. Inputs named like an existing value of a fitting type (`edge_index`, `node_batch`, `harmonics`, or the port name on its entity: `lengths` → `edge_lengths`) are connected at once; new stages and inputs are placed next to what they connect to |
| Connect | Drag from an output port to an input port; incompatible wires are refused with the reason |
| Edit settings | Select a block; key settings are shown, the rest under *more settings*. Renaming a stage also renames an output named after it |
| Wire from the inspector | Each input of the selected stage is a list of the values it accepts (no cycles, matching entity and representation); choose one, or ✕ to disconnect |
| Replace a stage | **replace with** in the inspector swaps the transformation or block and keeps the stage id and output slots, so later stages stay connected; inputs are matched by port name, then position. Options carry over, also under another name (`number` → `n_gaussians`, `end` → `stop`), and the notice lists what was dropped |
| Ablate | Untick **enabled**: the stage or block passes through the input with the type of its output (**passes through** shows it; the recipe records it as `bypass:`); a stage with no such input cannot be bypassed |
| Recipe parameters | Select the background: the recipe inspector edits the shared `parameters:` used as `${parameters.name}` |
| Group | Select stages (Shift-drag a box, Ctrl/Cmd-click) and press **Group** or Ctrl+G |
| Add a layer | Select a layer's stages (for example `node_update_1` and `edge_update_1`) and press Ctrl+Shift+D (**Insert copy after**): the copies read the layer's outputs where the originals read values of the same type, later stages read the copies, and the graph is laid out again |
| Duplicate | Ctrl+D (**Duplicate**) copies the selection beside it; wires between the copied stages go to the copies, other inputs are shared |
| Expand a block | **⊕ expand** on a block instance; **⊖** on its frame collapses it |
| Dissolve a block | Inspector of a block instance: **Dissolve into stages** |
| Loss | Select **ℒ loss** to edit the expression or add terms; select an operation to change or wrap it |
| New transformation | **ƒ new**: define a typed expression; **Create** writes and registers `plugins/<name>.py` |
| Compare | **Compare** tab: use the current recipe or a preset as baseline; switch baselines from its header; click a row to show the stage |
| Export | **Export** writes the recipe: the model sections first, then training and run settings, then the editor layout and chemical system under `studio:` (ignored by training). Click the recipe name in the toolbar to rename it; the name is the file name |
| Find a stage | **⌕ Find stage** (Ctrl+F or /): type a stage, block or transformation name; Enter expands the blocks around it and selects it |
| Undo / redo | Ctrl+Z / Ctrl+Shift+Z; the notice names the change. Restoring recovered work keeps your newer edits one undo away |
| Layout / direction | **Layout** (L) re-runs the layered layout; ↓ / → switch top-down and left-right |
| Shared inputs | **↪ pins** draws structure inputs and structure-only features used by three or more stages as a pin beside each use; double-click a pin to go to the stage that computes it |
| Routine inputs | **⊘ routine** (on by default) hides inputs a model rarely changes: `edge_index` with its lattice shifts, `message_edge_index`, batch membership and partitions. The ports stay wired; the inspector still lists and changes them |
| Port order | The layout draws each stage's ports in the order that untangles the wires (labels move with them). The order is only a drawing: inputs stay positional in the recipe |
| Chemical system | **⚛** sets species shells (they fix the readout irreps) and the reference size for cost estimates |

### Reading the graph

* Wire colours give the entity set: node (blue), edge (orange), index
  relations (grey, dashed), operators and loss (pink). Thick wires carry
  non-scalar irreps.
* Port shapes give the representation: diamond for irreps, circle for
  invariant scalars, square for indices. Hover a port for its full contract.
* Option values are white when set in the recipe, cyan italic when inferred
  from the wiring, grey when the framework default applies. Output irreps are
  labelled the same way. Only recipe values are exported.
* The bar under each block is its weight count on a log scale (full bar = 10M);
  the ring is its share of the estimated forward time. Blocks above 10 % of the
  forward time are drawn with area proportional to their share.
* **static** marks parameter-free stages that depend on the structure only;
  they are computed once per structure and cached.
* Zoomed out, blocks show only their names, sized to the block. A recipe that
  does not fit on screen at a readable size opens at its inputs; the minimap and
  the fit-view button show the whole graph.
* A dot after the recipe name marks unsaved changes. Opening another recipe or
  closing the page asks before discarding them, and unsaved work is kept in the
  browser: after a crash or reload the editor offers to restore it.
* Totals shown as **≥** in yellow leave out stages that failed to compile.
* While a wire is drawn, ports that accept it glow and the others fade.
* An edge relation and its lattice shifts are one variable: a port marked
  **+ shifts** (`edge_roles`, `periodic_displacement`) reads `edge_index` and
  the compiler passes `edge_shift` with it.

### Checks

Connections are checked immediately for entity set (node, edge, relation,
bond neighbourhood, graph), representation family (irreps, invariant, index,
geometry, local bond frame, decoded blocks, metadata), cycles, and target
compatibility (operators must be decoded edge blocks). The **Problems** tab
lists unconnected required inputs, unknown transformations, missing required
options, block interface mismatches, duplicate stage ids, mixed edge supports
and angular bases, unused outputs, stages that reach no target, unconnected
targets, objective terms reading operators the model does not predict, and
loss expressions that reference undefined or leave unused terms. The compiler
adds every contract error of the transformations themselves, for all stages at
once.

## Expression language

Each line assigns one value. Inputs declare an entity (node, edge, graph) and
irreps. Only equivariant operations type-check:

| Expression | Meaning |
| --- | --- |
| `linear(x, "8x0e+4x1o")` | equivariant linear map |
| `tp(x, y, "4x0e+2x1o")` | fully connected tensor product |
| `norm(x)` | norm of every irrep copy (invariant) |
| `scalars(x)` | the `0e` channels of `x` |
| `concat(a, b, ...)` | channel concatenation |
| `mlp(s, [64, 64], width)` | MLP on invariant features |
| `silu sigmoid tanh relu gelu softplus exp log sqrt abs` | elementwise, invariants only |
| `x * s`, `gate(x, s)` | scale by an invariant with one channel or one per irrep copy |
| `a + b`, `a - b` | equal irreps; adding a number needs an invariant |
| `param("alpha", 0.1)` | learned scalar |

The generated plugin is plain PyTorch/e3nn code with a registered factory that
checks its input contracts; edit it freely afterwards.

## Tests

```bash
PYTHONPATH=src:. python -m pytest -q tools/studio/tests
```
