import { createContext, useContext, useMemo, type ReactNode } from "react";
import { currentPipeline, useStudio } from "../store";
import { buildGraph, type BuiltGraph } from "./build";
import { lint, type Problem } from "./validate";
import { UNCONNECTED, compilerMessage } from "./model";
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
  const compactWires = useStudio((s) => s.compactWires);
  const hideRoutine = useStudio((s) => s.hideRoutine);
  const diff = useDiff();
  const value = useMemo(() => {
    const pipeline = currentPipeline({ document, analysis });
    const objective = document?.objective?.terms ? document.objective : analysis?.objective?.objective ?? null;
    const lowered = !document?.objective?.terms && Boolean(analysis?.objective?.lowered);
    const args = { pipeline, analysis, catalog, expanded, orientation, objective, objectiveLowered: lowered, visibleInputs, compactWires, hideRoutine, diff: diff.marks };
    const preliminary = buildGraph({ ...args, problems: [] });
    const problems = document ? lint(document, pipeline, preliminary.graph, analysis, catalog) : [];
    // The compiler is the authority: every error it reports is listed, attached
    // to its stage (or the enclosing block) when that is on the canvas.
    for (const error of analysis?.errors ?? []) {
      // Unconnected inputs are reported once by the editor, per port.
      if (UNCONNECTED.test(error.message)) continue;
      const known = preliminary.graph.stages.find((s) => s.key === error.stage || error.stage.startsWith(`${s.key}/`));
      const key = known?.key ?? null;
      // One message per cause: the stage is named once, and a missing option
      // the editor already reports is not repeated.
      const text = compilerMessage(error.stage, error.message);
      const inner = known && known.key !== error.stage ? `${error.stage.slice(known.key.length + 1)}: ${text}` : text;
      const message = known ? inner : `${error.stage}: ${text}`;
      const option = /requires option '([^']+)'/.exec(text)?.[1];
      if (option && problems.some((p) => p.key === key && p.message === `Option ${option} is required`)) continue;
      if (!problems.some((p) => p.key === key && p.message === message)) {
        problems.push({ level: "error", key, message });
      }
    }
    return { ...buildGraph({ ...args, problems }), problems };
  }, [document, analysis, catalog, expanded, orientation, visibleInputs, compactWires, hideRoutine, diff.marks]);
  return <GraphContext.Provider value={value}>{children}</GraphContext.Provider>;
}

export function useGraph(): Graph {
  const graph = useContext(GraphContext);
  if (!graph) throw new Error("useGraph outside GraphProvider");
  return graph;
}
