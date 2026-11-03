import { useMemo, useState } from "react";
import { currentPipeline, useStudio } from "../store";
import { addStage } from "../graph/edit";
import { CATEGORY_COLORS } from "./nodes";

const ORDER = [
  "radial_basis",
  "angular_basis",
  "geometry",
  "frame",
  "routing",
  "embedding",
  "encoder",
  "convolution",
  "product",
  "message",
  "aggregation",
  "nonlinearity",
  "normalization",
  "mlp",
  "linear",
  "combine",
  "residual",
  "readout",
  "output",
  "envelope",
  "statistics",
  "custom",
];

export function Inventory() {
  const { catalog, document, analysis, setModal, edit, showInput } = useStudio();
  const add = (name: string, kind: string) => {
    if (kind === "input") showInput(name);
    else edit((doc) => addStage(doc, name, catalog).document, `Added ${name}`);
  };
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set(["Inputs"]));
  const pipeline = currentPipeline({ document, analysis });
  const groups = useMemo(() => {
    const q = query.toLowerCase();
    const match = (...texts: string[]) => !q || texts.some((t) => t.toLowerCase().includes(q));
    const result: { title: string; color: string; items: { name: string; summary: string; kind: "transform" | "input" | "block"; detail?: string }[] }[] = [];
    const blocks = Object.entries(pipeline?.blocks ?? {}).filter(([name, def]) => match(name, def.summary ?? ""));
    if (blocks.length) result.push({ title: "Blocks in this recipe", color: CATEGORY_COLORS.block, items: blocks.map(([name, def]) => ({ name, summary: def.summary ?? `${def.stages.length} stages`, kind: "block", detail: def.inputs.join(", ") })) });
    const library = Object.entries(catalog?.block_library ?? {}).filter(([name, def]) => !pipeline?.blocks?.[name] && match(name, def.summary ?? "", def.library));
    if (library.length) result.push({ title: "Block library", color: CATEGORY_COLORS.block, items: library.map(([name, def]) => ({ name, summary: `${def.library}: ${def.summary ?? `${def.stages.length} stages`}`, kind: "block", detail: def.inputs.join(", ") })) });
    const transforms = catalog?.transforms ?? [];
    const categories = [...new Set(transforms.map((t) => t.category))].sort((a, b) => (ORDER.indexOf(a) + 100) % 100 - (ORDER.indexOf(b) + 100) % 100);
    for (const category of categories) {
      const items = transforms.filter((t) => t.category === category && match(t.name, t.summary, category));
      if (items.length) result.push({ title: category.replace("_", " "), color: CATEGORY_COLORS[category] ?? CATEGORY_COLORS.custom, items: items.map((t) => ({ name: t.name, summary: t.summary, kind: "transform", detail: t.inputs.map((p) => p.name).join(", ") })) });
    }
    const inputs = Object.entries(catalog?.inputs ?? {}).filter(([name]) => match(name, "input"));
    if (inputs.length) result.push({ title: "Inputs", color: CATEGORY_COLORS.input, items: inputs.map(([name, spec]) => ({ name, summary: `${spec.domain} ${spec.irreps ?? spec.representation}`, kind: "input" })) });
    return result;
  }, [catalog, pipeline, query]);

  const toggle = (title: string) => {
    const next = new Set(collapsed);
    if (next.has(title)) next.delete(title);
    else next.add(title);
    setCollapsed(next);
  };

  return (
    <aside className="inventory">
      <div className="inventory-head">
        <input className="search" placeholder="Search blocks…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <button className="button small accent" onClick={() => setModal("expression")} title="Define a new transformation from an expression">
          ƒ new
        </button>
      </div>
      <div className="inventory-list">
        {groups.map((group) => (
          <div key={group.title} className="inventory-group">
            <button className="inventory-group-title" onClick={() => toggle(group.title)}>
              <span className="swatch" style={{ background: group.color }} />
              {group.title}
              <span className="count">{group.items.length}</span>
              <span className="caret">{collapsed.has(group.title) && !query ? "▸" : "▾"}</span>
            </button>
            {(!collapsed.has(group.title) || query) &&
              group.items.map((item) => (
                <div
                  key={item.name}
                  className="inventory-item"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData(item.kind === "input" ? "application/mandala-input" : "application/mandala-transform", item.name);
                    e.dataTransfer.effectAllowed = "copy";
                  }}
                  onDoubleClick={() => add(item.name, item.kind)}
                  title={`${item.summary}${item.detail ? `\ninputs: ${item.detail}` : ""}\nDrag onto the canvas or double-click to add`}
                >
                  <div className="inventory-name">{item.name}</div>
                  <div className="inventory-summary">{item.summary}</div>
                </div>
              ))}
          </div>
        ))}
      </div>
    </aside>
  );
}
