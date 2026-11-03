import { useStudio, withStudioState } from "../store";
import { formatCount, formatSeconds } from "../graph/model";
import { api } from "../api";
import { useGraph } from "../graph/context";

export function Toolbar() {
  const s = useStudio();
  const { problems } = useGraph();
  const errors = problems.filter((p) => p.level === "error").length;
  const warnings = problems.length - errors;
  const exportYaml = async () => {
    if (!s.document) return;
    try {
      const { yaml } = await api.exportYaml(withStudioState(s.document, s.expanded, s.orientation));
      s.setModal("yaml", yaml);
    } catch (error) {
      s.notify((error as Error).message, "error");
    }
  };
  return (
    <header className="toolbar">
      <button className={`button icon ${s.showInventory ? "on" : ""}`} onClick={() => s.togglePanels("inventory")} title="Show or hide the block inventory">
        ◧
      </button>
      <div className="brand">
        <span className="brand-mark">◈</span> MANDALA <span className="brand-sub">Studio</span>
      </div>
      <div className="recipe-name" title="Current recipe">
        {s.name}
        {!s.document?.pipeline && s.document && <span className="chip">flat</span>}
      </div>
      <div className="toolbar-group">
        <button className="button" onClick={() => s.setModal("open")}>
          Open
        </button>
        <button className="button" onClick={exportYaml} disabled={!s.document}>
          Export YAML
        </button>
      </div>
      <div className="toolbar-group">
        <button className="button icon" onClick={s.undo} disabled={!s.past.length} title="Undo (Ctrl+Z)">
          ↶
        </button>
        <button className="button icon" onClick={s.redo} disabled={!s.future.length} title="Redo (Ctrl+Shift+Z)">
          ↷
        </button>
      </div>
      <div className="toolbar-group">
        <button className="button" onClick={() => (window as unknown as { studioGroup?: () => void }).studioGroup?.()} title="Group the selected stages into a named block (Ctrl+G)">
          Group
        </button>
        <button className="button" onClick={s.requestLayout} title="Auto layout (L)">
          Layout
        </button>
        <div className="segmented" title="Flow direction">
          <button className={s.orientation === "TB" ? "on" : ""} onClick={() => s.setOrientation("TB")}>
            ↓
          </button>
          <button className={s.orientation === "LR" ? "on" : ""} onClick={() => s.setOrientation("LR")}>
            →
          </button>
        </div>
      </div>
      <div className="toolbar-group">
        <button className="button" onClick={() => s.setModal("system")} title="Species, basis and reference system size">
          ⚛ {s.system.name}
        </button>
        <button className="button icon" onClick={() => s.setModal("guide")} title="Getting started: a short tour of the Studio">
          ?
        </button>
      </div>
      <div className="toolbar-spacer" />
      <div className="totals">
        <span className={`analysis-dot ${s.analyzing ? "busy" : s.dirtySinceAnalysis ? "stale" : "ok"}`} title={s.analyzing ? "analysing…" : `analysed in ${(s.analysis?.elapsed ?? 0).toFixed(2)} s`} />
        <span title="Weights">
          <b>{formatCount(s.analysis?.totals.parameters ?? 0)}</b> weights
        </span>
        <span
          title={
            s.catalog?.cost_calibrated
              ? `Estimated forward time for ${s.system.size.atoms} atoms`
              : "Uncalibrated estimate: compare blocks by their share; absolute times need measured cost constants"
          }
        >
          <b>{formatSeconds(s.analysis?.totals.seconds ?? 0)}</b> / forward
          {s.catalog?.cost_calibrated ? <span className="muted small"> ({s.catalog.cost_calibrated})</span> : <span className="muted small"> (relative)</span>}
        </span>
        <button className={`button small ${errors ? "danger" : warnings ? "warn" : ""}`} onClick={() => { s.setPanel("problems"); if (!s.showSide) s.togglePanels("side"); }}>
          {errors} errors · {warnings} warnings
        </button>
        <button className={`button icon ${s.showSide ? "on" : ""}`} onClick={() => s.togglePanels("side")} title="Show or hide the inspector">
          ◨
        </button>
      </div>
    </header>
  );
}
