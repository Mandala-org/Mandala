import { Component, useEffect, type ErrorInfo, type ReactNode } from "react";
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

/** A rendering failure shows a recovery panel instead of a blank page. */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Studio view failed", error, info.componentStack);
  }
  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const { past } = useStudio.getState();
    return (
      <div className="crash">
        <h3>The editor view failed</h3>
        <p className="mono small">{error.message}</p>
        <p className="muted">Your recipe is kept in this browser. Undo the last change, or try to show the view again.</p>
        <div className="button-row">
          {past.length > 0 && (
            <button
              className="button primary"
              onClick={() => {
                useStudio.getState().undo();
                this.setState({ error: null });
              }}
            >
              Undo the last change
            </button>
          )}
          <button className="button" onClick={() => this.setState({ error: null })}>
            Show the view again
          </button>
          <button className="button" onClick={() => window.location.reload()}>
            Reload the page
          </button>
        </div>
      </div>
    );
  }
}

function Recovered() {
  const recovered = useStudio((s) => s.recovered);
  const restore = useStudio((s) => s.restoreAutosave);
  const discard = useStudio((s) => s.discardAutosave);
  if (!recovered) return null;
  return (
    <div className="banner banner-recovered" role="status">
      Unsaved changes to <b>{recovered.name}</b> from {new Date(recovered.time).toLocaleString()} were kept in this browser.
      <button className="button small primary" onClick={restore}>
        Restore them
      </button>
      <button className="button small" onClick={discard}>
        Discard
      </button>
    </div>
  );
}

function Banner() {
  const { document, analysis, analyzing } = useStudio();
  if (!document) return null;
  if (!document.pipeline && analysis?.pipeline)
    return <div className="banner">This recipe is written as training settings; the graph shows the stages the framework builds from them. Your first edit turns it into a stage recipe.</div>;
  if (!analysis && analyzing) return <div className="banner">Analysing…</div>;
  return null;
}

export default function App() {
  const init = useStudio((s) => s.init);
  const toast = useStudio((s) => s.toast);
  const showInventory = useStudio((s) => s.showInventory);
  const showSide = useStudio((s) => s.showSide);
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      const { document, saved } = useStudio.getState();
      if (document && document !== saved) event.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, []);
  useEffect(() => {
    init().catch((error) => useStudio.getState().notify(`Cannot reach the Studio server: ${error.message}`, "error"));
  }, [init]);
  return (
    <ErrorBoundary>
      <ReactFlowProvider>
        <GraphProvider>
          <div className="app">
            <Toolbar />
            <div className="workspace" style={{ gridTemplateColumns: `${showInventory ? "250px" : "0px"} minmax(0, 1fr) ${showSide ? "340px" : "0px"}` }}>
              {showInventory ? <Inventory /> : <div />}
              <main className="main">
                <Recovered />
                <Banner />
                <ErrorBoundary>
                  <Canvas />
                </ErrorBoundary>
              </main>
              {showSide ? (
                <ErrorBoundary>
                  <SidePanel />
                </ErrorBoundary>
              ) : (
                <div />
              )}
            </div>
            {toast && <div className={`toast toast-${toast.level}`}>{toast.message}</div>}
            <Modals />
          </div>
        </GraphProvider>
      </ReactFlowProvider>
    </ErrorBoundary>
  );
}
