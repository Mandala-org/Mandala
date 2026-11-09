import { useStudio } from "../store";
import { useGraph } from "../graph/context";

export function ProblemsPanel() {
  const { problems, nodes } = useGraph();
  const reveal = useStudio((s) => s.reveal);
  // A problem leads to its node; a stage inside a collapsed block is revealed.
  const focus = (key: string | null) => {
    if (!key) return;
    const id = nodes.find((n) => n.id === `stage:${key}` || n.id === key)?.id;
    if (id && !id.startsWith("stage:")) useStudio.setState({ focus: id, focusSelects: true });
    else reveal(key);
  };
  if (!problems.length)
    return (
      <div className="panel-body">
        <p className="muted">No problems: every stage compiles and every check passes.</p>
      </div>
    );
  const sorted = [...problems].sort((a, b) => (a.level === b.level ? 0 : a.level === "error" ? -1 : 1));
  return (
    <div className="panel-body">
      {sorted.map((p, i) => (
        <button key={i} className={`problem problem-${p.level} clickable`} onClick={() => focus(p.key)} title={p.key ? "Show it on the canvas" : undefined}>
          {p.key && <span className="problem-key mono">{p.key}</span>}
          {p.message}
        </button>
      ))}
    </div>
  );
}
