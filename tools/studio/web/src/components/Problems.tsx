import { useReactFlow } from "@xyflow/react";
import { useStudio } from "../store";
import { useGraph } from "../graph/context";

export function ProblemsPanel() {
  const { problems, nodes } = useGraph();
  const setSelection = useStudio((s) => s.setSelection);
  const flow = useReactFlow();
  const focus = (key: string | null) => {
    if (!key) return;
    const id = nodes.find((n) => n.id === `stage:${key}` || n.id === key)?.id;
    if (!id) return;
    flow.fitView({ nodes: [{ id }], duration: 400, maxZoom: 1.2 });
    flow.setNodes((all) => all.map((n) => ({ ...n, selected: n.id === id })));
    setSelection([id]);
  };
  if (!problems.length) return <div className="panel-body"><p className="muted">No problems: every stage compiles and every check passes.</p></div>;
  const sorted = [...problems].sort((a, b) => (a.level === b.level ? 0 : a.level === "error" ? -1 : 1));
  return (
    <div className="panel-body">
      {sorted.map((p, i) => (
        <button key={i} className={`problem problem-${p.level} clickable`} onClick={() => focus(p.key)}>
          {p.key && <span className="problem-key mono">{p.key}</span>}
          {p.message}
        </button>
      ))}
    </div>
  );
}
