import { useStudio, withStudioState } from "../store";
import { failedStages, formatCount, formatSeconds, plural } from "../graph/model";
import { api } from "../api";
import { useGraph } from "../graph/context";

export function Toolbar() {
  const s = useStudio();
  const { problems } = useGraph();
  const errors = problems.filter((p) => p.level === "error").length;
  const warnings = problems.length - errors;
  const failed = failedStages(s.analysis);
  const atLeast = failed ? "≥ " : "";
  const exportYaml = async () => {
    if (!s.document) return;
    try {
      const { yaml } = await api.exportYaml(withStudioState(s.document, s.expanded, s.orientation, s.system));
      s.setModal("yaml", yaml);
    } catch (error) {
      s.notify((error as Error).message, "error");
    }
  };
  return (
    <header className="toolbar">
      <button className={`button icon ${s.showInventory ? "on" : ""}`} onClick={() => s.togglePanels("inventory")} title="Show or hide the block inventory" aria-label="Show or hide the block inventory">
        ◧
      </button>
      <div className="brand">
        <span className="brand-mark">◈</span>
        <span className="brand-name">
          MANDALA <span className="brand-sub">Studio</span>
        </span>
      </div>
      <div className="recipe-name" title="Recipe name (also the file name on export); click to rename">
        <input
          key={s.name}
          className="recipe-name-input mono"
          defaultValue={s.name}
          aria-label="Recipe name"
          size={Math.max(8, Math.min(s.name.length, 28))}
          onBlur={(e) => s.rename(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
            if (e.key === "Escape") {
              (e.target as HTMLInputElement).value = s.name;
              (e.target as HTMLInputElement).blur();
            }
          }}
        />
        {s.document && s.document !== s.saved && (
          <span className="unsaved" title="Unsaved changes: Export writes them as YAML">
            ●
          </span>
        )}
        {!s.document?.pipeline && s.document && (
          <span className="chip" title="Written as training settings: the graph shows the stages the framework builds from them">
            settings
          </span>
        )}
      </div>
      <div className="toolbar-group">
        <button className="button" onClick={() => s.setModal("open")}>
          Open
        </button>
        <button className="button" onClick={exportYaml} disabled={!s.document} title="Show the recipe as YAML to copy or download">
          Export
        </button>
      </div>
      <div className="toolbar-group">
        <button className="button icon" onClick={s.undo} disabled={!s.past.length} title="Undo (Ctrl+Z)" aria-label="Undo">
          ↶
        </button>
        <button className="button icon" onClick={s.redo} disabled={!s.future.length} title="Redo (Ctrl+Shift+Z)" aria-label="Redo">
          ↷
        </button>
      </div>
      <div className="toolbar-group">
        <button
          className="button"
          disabled={!s.selection.some((id) => id.startsWith("stage:"))}
          onClick={() => (window as unknown as { studioGroup?: () => void }).studioGroup?.()}
          title="Group the selected stages into a named block (Ctrl+G); select stages first"
          aria-label="Group"
        >
          <span className="narrow-only">⧉</span>
          <span className="roomy-only">Group</span>
        </button>
        <button className="button" onClick={s.requestLayout} title="Auto layout (L)" aria-label="Layout">
          <span className="narrow-only">⟲</span>
          <span className="roomy-only">Layout</span>
        </button>
        <button
          className={`button ${s.compactWires ? "on" : ""}`}
          aria-pressed={s.compactWires}
          onClick={() => s.setCompactWires(!s.compactWires)}
          title={
            s.compactWires
              ? "Structure inputs and structure-only features used by 3 or more stages are drawn as pins beside each use. Click to draw them as wires from one source."
              : "Every value is drawn as a wire from its source. Click to draw widely shared structure features as pins beside each use."
          }
        >
          ↪<span className="roomy-only"> pins</span>
        </button>
        <button
          className={`button ${s.hideRoutine ? "on" : ""}`}
          aria-pressed={s.hideRoutine}
          onClick={() => s.setHideRoutine(!s.hideRoutine)}
          title={
            s.hideRoutine
              ? "Routine inputs (edge_index with its shifts, batch membership, partitions) are hidden; the ports stay wired. Click to show them."
              : "Click to hide routine inputs (edge_index with its shifts, batch membership, partitions)"
          }
        >
          ⊘<span className="roomy-only"> routine</span>
        </button>
        <div className="segmented" title="Flow direction">
          <button className={s.orientation === "TB" ? "on" : ""} onClick={() => s.setOrientation("TB")} title="Flow top to bottom" aria-label="Flow top to bottom" aria-pressed={s.orientation === "TB"}>
            ↓
          </button>
          <button className={s.orientation === "LR" ? "on" : ""} onClick={() => s.setOrientation("LR")} title="Flow left to right" aria-label="Flow left to right" aria-pressed={s.orientation === "LR"}>
            →
          </button>
        </div>
      </div>
      <div className="toolbar-group">
        <button className="button" onClick={() => s.setModal("system")} title={`Chemical system: ${s.system.name} (species, basis and reference size)`} aria-label="Chemical system">
          ⚛<span className="roomy-only"> {s.system.name}</span>
        </button>
        <button className="button icon" onClick={() => s.setModal("guide")} title="Getting started: a short tour of the Studio" aria-label="Open the getting-started guide">
          ?
        </button>
      </div>
      <div className="toolbar-spacer" />
      <div className={`totals ${s.analyzing || s.dirtySinceAnalysis ? "updating" : ""} ${failed ? "incomplete" : ""}`} title={failed ? `${plural(failed, "stage")} failed to compile; their weights and time are not counted` : undefined}>
        <span className={`analysis-dot ${s.analyzing ? "busy" : s.dirtySinceAnalysis ? "stale" : "ok"}`} title={s.analyzing ? "analysing…" : `analysed in ${(s.analysis?.elapsed ?? 0).toFixed(2)} s`} />
        <span title="Weights (trainable parameters)">
          <b>
            {atLeast}
            {formatCount(s.analysis?.totals?.parameters ?? 0)}
          </b>
          <span className="wide-only"> weights</span>
        </span>
        <span
          title={
            s.catalog?.cost_calibrated
              ? `Estimated forward time for ${s.system.size.atoms} atoms`
              : "Uncalibrated estimate: compare blocks by their share; absolute times need measured cost constants"
          }
        >
          <b>
            {atLeast}
            {formatSeconds(s.analysis?.totals?.seconds ?? 0)}
          </b>
          <span className="wide-only"> per forward</span>
          {s.catalog?.cost_calibrated ? <span className="muted small wide-only"> ({s.catalog.cost_calibrated})</span> : <span className="muted small wide-only"> (relative)</span>}
        </span>
        <button
          className={`button small ${errors ? "danger" : warnings ? "warn" : ""}`}
          onClick={() => {
            s.setPanel("problems");
            if (!s.showSide) s.togglePanels("side");
          }}
          title={`Show the problems list: ${plural(errors, "error")}, ${plural(warnings, "warning")}`}
        >
          {errors || warnings ? (
            <>
              <span className="roomy-only">{`${plural(errors, "error")} · ${plural(warnings, "warning")}`}</span>
              <span className="narrow-only">{`✕ ${errors} · ⚠ ${warnings}`}</span>
            </>
          ) : (
            "✓ no problems"
          )}
        </button>
        <button className={`button icon ${s.showSide ? "on" : ""}`} onClick={() => s.togglePanels("side")} title="Show or hide the inspector" aria-label="Show or hide the inspector">
          ◨
        </button>
      </div>
    </header>
  );
}
