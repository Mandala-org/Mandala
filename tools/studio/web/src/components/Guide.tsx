import { useEffect, useRef, useState } from "react";
import { useStudio } from "../store";

const SEEN = "studio.guideSeen";

export function guideSeen(): boolean {
  try {
    return localStorage.getItem(SEEN) === "1";
  } catch {
    return true;
  }
}

function markSeen() {
  try {
    localStorage.setItem(SEEN, "1");
  } catch {
    /* private window: the guide simply opens again next time */
  }
}

// The step survives actions that load a preset or open another dialog.
let lastStep = 0;

interface Step {
  title: string;
  body: React.ReactNode;
  action?: { label: string; run: () => void };
}

function usePreset() {
  const { presets, loadPreset } = useStudio();
  return (name: string) => {
    const preset = presets.find((p) => p.name === name);
    if (preset) void loadPreset(preset).then(() => useStudio.getState().setModal("guide"));
  };
}

/** Opens another dialog; closing it brings the guide back at the same step. */
function openFromGuide(modal: "system" | "expression") {
  useStudio.setState({ returnToGuide: true });
  useStudio.getState().setModal(modal);
}

export function GuideModal() {
  const s = useStudio();
  const preset = usePreset();
  const [index, setStep] = useState(lastStep);
  const setIndex = (value: number) => {
    lastStep = value;
    setStep(value);
  };
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      else if (e.key === "ArrowRight" && index < steps.length - 1) setIndex(index + 1);
      else if (e.key === "ArrowLeft" && index > 0) setIndex(index - 1);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });
  const primary = useRef<HTMLButtonElement>(null);
  useEffect(() => primary.current?.focus(), [index]);
  // The guide counts as seen once it is closed, not when it first shows.
  const close = () => {
    lastStep = 0;
    markSeen();
    s.setModal(null);
  };
  const steps: Step[] = [
    {
      title: "A model is a recipe of typed stages",
      body: (
        <>
          <p>
            Every box is one registered transformation (a tensor product, an MLP, an aggregation…) or a <b>block</b> of them. Structure inputs sit at the top, the predicted
            operators and the loss at the bottom. The graph is compiled by the same code the trainer uses, so the stages, irreps and weights shown are the ones training builds.
          </p>
          <p className="muted">
            Wires are coloured by entity (nodes, edges, bond neighbourhoods) and labelled with their irreps. Structure inputs and structure-only features that many stages read
            appear as small <b>↪ pins</b> beside each use instead of long wires; double-click a pin to go to the stage that computes it. Zoomed out, boxes show only their names.
          </p>
        </>
      ),
    },
    {
      title: "Open one of the shipped models",
      body: (
        <p>
          <b>Open</b> lists the MANDALA paper model, DeepH, DeepH-E3, MACE-H and the hybrid recipes, or takes any YAML recipe. All of them are built from the same stage
          library, so their parts can be mixed. Try DeepH-E3: its node and edge updates come from the <span className="mono">deeph_e3</span> block library and are copied into the recipe
          (under <b>Blocks in this recipe</b>), where you can edit them.
        </p>
      ),
      action: { label: "Open DeepH-E3", run: () => preset("deeph_e3") },
    },
    {
      title: "Look inside blocks",
      body: (
        <p>
          Click <b>⊕ expand</b> on a block to see its stages; editing a stage inside a block changes every instance. <b>⌕ Find stage</b> (Ctrl+F or /) jumps to any stage, opening the
          blocks around it. Select a few stages and press <b>Ctrl+G</b> to turn them into a new named block. The inventory's <b>Block library</b> holds the DeepH, DeepH-E3 and MACE layers not yet in the recipe, ready to drop into any model.
        </p>
      ),
    },
    {
      title: "Wire it: types are checked as you drag",
      body: (
        <p>
          Drag from an output port to an input port: ports that accept the value light up, and an incompatible connection is refused with the reason. Or choose what each
          input reads from the lists in the inspector, and use <b>replace with</b> to swap a stage while keeping its wiring. Output irreps are inferred from the wiring:{" "}
          <i>inferred</i> values are cyan italic, values set in the recipe are white, defaults are grey, and <b>reset</b> returns a value to inference. The <b>Problems</b> tab lists everything the
          compiler rejects; click an entry to go to it.
        </p>
      ),
    },
    {
      title: "Ablate without rewiring",
      body: (
        <p>
          Untick <b>enabled</b> in the inspector to bypass a stage or a whole block: it passes one input through unchanged, the one with the type of its output (the
          inspector shows which, and refuses when no input fits). Only stages with a single output can be bypassed. Use{" "}
          the <b>Compare</b> tab to set a baseline and see what changed in weights and estimated time.
        </p>
      ),
      action: {
        label: "Show the Compare tab",
        run: () => {
          s.setPanel("diff");
          if (!s.showSide) s.togglePanels("side");
        },
      },
    },
    {
      title: "Weights and time at a glance",
      body: (
        <p>
          Each box shows its parameter count and estimated forward time for the reference system (the <b>⚛</b> button sets species, basis and size). Boxes that take more than 10%
          of the forward time grow with their share. Times come from per-transform cost constants; the label says which device they were measured on (the shipped ones are a provisional CPU fit),
          and without constants they are relative.
        </p>
      ),
      action: { label: "Set the system", run: () => openFromGuide("system") },
    },
    {
      title: "The loss is a graph too",
      body: (
        <p>
          Objective terms (matrix error, band energy, forces, …) are combined by the expression below with numbers, <span className="mono">+ − × ÷</span>{" "}
          and functions. A term that reads a predicted operator (matrix error) has an input port wired to a target. Click a term to change its options; select
          the <b>ℒ loss</b> node to add terms.
        </p>
      ),
    },
    {
      title: "Write your own block from an expression",
      body: (
        <p>
          <b>ƒ new</b> in the inventory types an equivariant expression (tensor products, norms, gates of named inputs), checks it as you type and saves it as a Python plugin under{" "}
          <span className="mono">plugins/</span>, which appears in the inventory at once.
        </p>
      ),
      action: { label: "Open the expression editor", run: () => openFromGuide("expression") },
    },
    {
      title: "Save the recipe",
      body: (
        <p>
          <b>Export</b> writes a self-contained recipe (library blocks included) that <span className="mono">scripts/train.py</span> trains as is; the layout and
          expanded blocks are kept under <span className="mono">studio:</span>. Undo/redo with Ctrl+Z / Ctrl+Shift+Z, add a layer by selecting one and pressing Ctrl+Shift+D (insert a copy after it), auto-layout with L, and flip the flow direction with ↓/→.
        </p>
      ),
    },
  ];
  const step = steps[index];
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal guide" role="dialog" aria-modal="true" aria-labelledby="guide-title">
        <div className="modal-head">
          <span>
            Getting started · {index + 1}/{steps.length} <span className="muted small">(← → to move)</span>
          </span>
          <button className="link" onClick={close} aria-label="Close the guide">
            ✕
          </button>
        </div>
        <h3 id="guide-title">{step.title}</h3>
        <div className="guide-body">{step.body}</div>
        <div className="guide-dots">
          {steps.map((_, i) => (
            <button key={i} className={`dot ${i === index ? "on" : ""}`} onClick={() => setIndex(i)} title={steps[i].title} aria-label={`Step ${i + 1}: ${steps[i].title}`} aria-current={i === index ? "step" : undefined} />
          ))}
        </div>
        <div className="button-row">
          {step.action && (
            <button
              className="button"
              onClick={() => {
                step.action!.run();
              }}
            >
              {step.action.label}
            </button>
          )}
          <span style={{ flex: 1 }} />
          <button className="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>
            Back
          </button>
          {index < steps.length - 1 ? (
            <button ref={primary} className="button primary" onClick={() => setIndex(index + 1)} title="Next (→)">
              Next
            </button>
          ) : (
            <button ref={primary} className="button primary" onClick={close}>
              Start building
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
