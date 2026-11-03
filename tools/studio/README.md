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
| Open a recipe | **Open**: reference and hybrid recipes, pilots, examples, or pasted/loaded YAML |
| Add a block | Drag from the inventory onto the canvas, or double-click it |
| Connect | Drag from an output port to an input port; incompatible wires are refused with the reason |
| Edit settings | Select a block; key settings are shown, the rest under *more settings* |
| Group | Select stages (Shift-drag a box, Ctrl/Cmd-click) and press **Group** or Ctrl+G |
| Expand a block | **⊕ expand** on a block instance; **⊖** on its frame collapses it |
| Dissolve a block | Inspector of a block instance: **Dissolve into stages** |
| Loss | Select **ℒ loss** to edit the expression or add terms; select an operation to change or wrap it |
| New transformation | **ƒ new**: define a typed expression; **Create** writes and registers `plugins/<name>.py` |
| Compare | **Compare** tab: use the current recipe or a preset as baseline |
| Export | **Export YAML** writes the recipe, including the editor layout under `studio:` |
| Undo / redo | Ctrl+Z / Ctrl+Shift+Z |
| Layout / direction | **Layout** (L) re-runs the layered layout; ↓ / → switch top-down and left-right |
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
