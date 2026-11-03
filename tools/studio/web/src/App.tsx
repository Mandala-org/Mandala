import { useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useStudio } from "./store";
import { GraphProvider, useGraph } from "./graph/context";
import { Canvas } from "./components/Canvas";
import { Inventory } from "./components/Inventory";
import { Inspector } from "./components/Inspector";
import { Toolbar } from "./components/Toolbar";
import { DiffPanel } from "./components/Diff";
import { ProblemsPanel } from "./components/Problems";
import { Modals } from "./components/Modals";

function SidePanel() {
  const { panel, setPanel, baseline } = useStudio();
  const { problems } = useGraph();
  return (
    <aside className="side">
      <div className="tabs">
        <button className={panel === "inspector" ? "on" : ""} onClick={() => setPanel("inspector")}>
          Inspector
        </button>
        <button className={panel === "problems" ? "on" : ""} onClick={() => setPanel("problems")}>
          Problems <span className="count">{problems.length}</span>
        </button>
        <button className={panel === "diff" ? "on" : ""} onClick={() => setPanel("diff")}>
          Compare {baseline ? <span className="count">●</span> : null}
        </button>
      </div>
      <div className="side-content">
        {panel === "inspector" && <Inspector />}
        {panel === "problems" && <ProblemsPanel />}
        {panel === "diff" && <DiffPanel />}
      </div>
    </aside>
  );
}

function Banner() {
  const { document, analysis, analyzing } = useStudio();
  if (!document) return null;
  if (!document.pipeline && analysis?.pipeline)
    return <div className="banner">Flat configuration: the graph shows the framework's lowering into stages. The first edit turns it into an explicit pipeline recipe.</div>;
  if (!analysis && analyzing) return <div className="banner">Analysing…</div>;
  return null;
}

export default function App() {
  const init = useStudio((s) => s.init);
  const toast = useStudio((s) => s.toast);
  const showInventory = useStudio((s) => s.showInventory);
  const showSide = useStudio((s) => s.showSide);
  useEffect(() => {
    init().catch((error) => useStudio.getState().notify(`Cannot reach the Studio server: ${error.message}`, "error"));
  }, [init]);
  return (
    <ReactFlowProvider>
      <GraphProvider>
        <div className="app">
          <Toolbar />
          <div className="workspace" style={{ gridTemplateColumns: `${showInventory ? "250px" : "0px"} minmax(0, 1fr) ${showSide ? "340px" : "0px"}` }}>
            {showInventory ? <Inventory /> : <div />}
            <main className="main">
              <Banner />
              <Canvas />
            </main>
            {showSide ? <SidePanel /> : <div />}
          </div>
          {toast && <div className={`toast toast-${toast.level}`}>{toast.message}</div>}
          <Modals />
        </div>
      </GraphProvider>
    </ReactFlowProvider>
  );
}
