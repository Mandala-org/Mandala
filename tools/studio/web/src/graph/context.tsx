import { createContext, useContext, useMemo, type ReactNode } from "react";
import { currentPipeline, useStudio } from "../store";
import { buildGraph, type BuiltGraph } from "./build";
import { lint, type Problem } from "./validate";
import { useDiff } from "../components/Diff";

type Graph = BuiltGraph & { problems: Problem[] };

const GraphContext = createContext<Graph | null>(null);

/** Builds the displayed graph and its problems once for the whole UI. */
export function GraphProvider({ children }: { children: ReactNode }) {
  const document = useStudio((s) => s.document);
  const analysis = useStudio((s) => s.analysis);
  const catalog = useStudio((s) => s.catalog);
  const expanded = useStudio((s) => s.expanded);
  const orientation = useStudio((s) => s.orientation);
  const visibleInputs = useStudio((s) => s.visibleInputs);
  const diff = useDiff();
  const value = useMemo(() => {
    const pipeline = currentPipeline({ document, analysis });
    const objective = document?.objective?.terms ? document.objective : analysis?.objective.objective ?? null;
    const lowered = !document?.objective?.terms && Boolean(analysis?.objective.lowered);
    const args = { pipeline, analysis, catalog, expanded, orientation, objective, objectiveLowered: lowered, visibleInputs, diff: diff.marks };
    const preliminary = buildGraph({ ...args, problems: [] });
    const problems = document ? lint(document, pipeline, preliminary.graph, analysis, catalog) : [];
    // The compiler is the authority: every error it reports is listed, attached
    // to its stage (or the enclosing block) when that is on the canvas.
    for (const error of analysis?.errors ?? []) {
      if (/produced by a failing stage/.test(error.message)) continue;
      const known = preliminary.graph.stages.find((s) => s.key === error.stage || error.stage.startsWith(`${s.key}/`));
      const message = known ? error.message : `${error.stage}: ${error.message}`;
      if (!problems.some((p) => p.key === (known?.key ?? null) && p.message === message)) {
        problems.push({ level: "error", key: known?.key ?? null, message });
      }
    }
    return { ...buildGraph({ ...args, problems }), problems };
  }, [document, analysis, catalog, expanded, orientation, visibleInputs, diff.marks]);
  return <GraphContext.Provider value={value}>{children}</GraphContext.Provider>;
}

export function useGraph(): Graph {
  const graph = useContext(GraphContext);
  if (!graph) throw new Error("useGraph outside GraphProvider");
  return graph;
}
