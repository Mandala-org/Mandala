import { useEffect, useState } from "react";
import { currentPipeline, useStudio } from "../store";
import type { OptionInfo, ObjectiveSpec, RecipeDocument } from "../types";
import { formatCount, formatSeconds, metricsOf, recordsOf, slotSpec, portsOf, displayedInputs, outputNamesOf, type VisibleStage } from "../graph/model";
import { setEnabled, dissolveInstance, removeStage, renameStage, setOption, setRole } from "../graph/edit";
import { parseExpression, printExpression, getAt, replaceAt, type Expr } from "../loss/expression";
import type { Problem } from "../graph/validate";
import { editableObjective } from "../loss/objective";
import { useGraph } from "../graph/context";
import { WeightsBar } from "./nodes";

const IRREPS = /^\s*(\d+x\d+[eo])(\s*\+\s*\d+x\d+[eo])*\s*$/;

function parseValue(option: OptionInfo, text: string): unknown {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  if (trimmed.startsWith("${")) return trimmed;
  switch (option.type) {
    case "int": {
      const v = Number(trimmed);
      if (!Number.isInteger(v)) throw new Error(`${option.name} must be an integer`);
      return v;
    }
    case "float": {
      const v = Number(trimmed);
      if (Number.isNaN(v)) throw new Error(`${option.name} must be a number`);
      return v;
    }
    case "irreps":
      if (!IRREPS.test(trimmed)) throw new Error(`${trimmed} is not an irreps string like 8x0e+4x1o`);
      return trimmed.replace(/\s+/g, "");
    case "int_list":
    case "float_list":
    case "list": {
      const value = trimmed.startsWith("[") ? JSON.parse(trimmed) : trimmed.split(",").map((s) => s.trim()).filter(Boolean).map((s) => (Number.isNaN(Number(s)) ? s : Number(s)));
      if (!Array.isArray(value)) throw new Error(`${option.name} must be a list`);
      return value;
    }
    case "mapping":
      return JSON.parse(trimmed);
    default:
      return trimmed;
  }
}

function show(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function OptionField({ option, given, resolved, state, onChange }: { option: OptionInfo; given: unknown; resolved: unknown; state: "explicit" | "inferred" | "default"; onChange: (value: unknown) => void }) {
  const [text, setText] = useState(show(given));
  const [error, setError] = useState<string | null>(null);
  useEffect(() => setText(show(given)), [given]);
  const commit = (value: string) => {
    try {
      const parsed = parseValue(option, value);
      setError(null);
      if (JSON.stringify(parsed) !== JSON.stringify(given)) onChange(parsed);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const placeholder = state === "inferred" ? `${show(resolved)} (inferred)` : show(resolved ?? option.default);
  let input: React.ReactNode;
  if (option.type === "choice" && option.choices.length) {
    input = (
      <select value={given === undefined ? "" : show(given)} onChange={(e) => onChange(e.target.value === "" ? undefined : option.choices.find((c) => show(c) === e.target.value))}>
        <option value="">{`default (${show(resolved ?? option.default)})`}</option>
        {option.choices.map((c) => (
          <option key={show(c)} value={show(c)}>
            {show(c)}
          </option>
        ))}
      </select>
    );
  } else if (option.type === "bool") {
    input = (
      <select value={given === undefined ? "" : String(given)} onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value === "true")}>
        <option value="">{`default (${show(resolved ?? option.default)})`}</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  } else {
    input = (
      <input
        className={option.type === "irreps" ? "mono" : ""}
        value={text}
        placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && commit((e.target as HTMLInputElement).value)}
      />
    );
  }
  return (
    <div className={`option-field state-${state}`}>
      <div className="option-field-head">
        <span className="option-field-name" title={option.doc}>
          {option.name}
          {option.required && <span className="required">*</span>}
        </span>
        <span className={`state-chip state-${state}`} title={state === "inferred" ? option.infer_doc || "inferred from the wiring" : state === "default" ? "framework default" : "set in this recipe"}>
          {state === "inferred" ? `inferred${option.infer_doc ? `: ${option.infer_doc}` : ""}` : state}
        </span>
        {given !== undefined && (
          <button className="link" onClick={() => onChange(undefined)} title={option.inferred ? "Remove the value so it is inferred" : "Reset to the default"}>
            reset
          </button>
        )}
      </div>
      {input}
      {state !== "explicit" && option.type !== "choice" && option.type !== "bool" && resolved !== undefined && <div className="resolved mono">= {show(resolved)}</div>}
      {given !== undefined && typeof given === "string" && given.includes("$") && resolved !== undefined && <div className="resolved mono">= {show(resolved)}</div>}
      {error && <div className="field-error">{error}</div>}
    </div>
  );
}

function ProblemsList({ problems }: { problems: Problem[] }) {
  if (!problems.length) return null;
  return (
    <div className="problems-inline">
      {problems.map((p, i) => (
        <div key={i} className={`problem problem-${p.level}`}>
          {p.message}
        </div>
      ))}
    </div>
  );
}

function StageInspector({ visible }: { visible: VisibleStage }) {
  const { document, analysis, catalog, edit, toggleExpanded, expanded } = useStudio();
  const { problems } = useGraph();
  const pipeline = currentPipeline({ document, analysis });
  const [advanced, setAdvanced] = useState(false);
  const [renaming, setRenaming] = useState(visible.localId);
  useEffect(() => setRenaming(visible.localId), [visible.localId]);
  const records = recordsOf(analysis, visible.key, visible.isBlock);
  const metrics = metricsOf(records);
  const record = visible.isBlock ? undefined : records[0];
  const transform = catalog?.transforms.find((t) => t.name === visible.stage.transform);
  const def = pipeline?.blocks?.[visible.stage.transform];
  const given = visible.stage.options ?? {};
  const total = analysis?.totals.seconds ?? 0;
  const ports = portsOf(visible.stage, pipeline, catalog);
  const outNames = outputNamesOf(visible.stage, pipeline, catalog);
  const update = (name: string, value: unknown) => edit((doc) => setOption(doc, visible, name, value));
  const options = transform?.options ?? [];
  const key = options.filter((o) => o.key || o.required || o.name in given);
  const rest = options.filter((o) => !key.includes(o));
  const state = (name: string): "explicit" | "inferred" | "default" => (name in given ? "explicit" : record?.inferred.includes(name) ? "inferred" : "default");
  return (
    <div className="panel-body">
      <div className="inspector-title">
        <input
          className="title-input"
          value={renaming}
          onChange={(e) => setRenaming(e.target.value)}
          onBlur={() => renaming !== visible.localId && edit((doc) => renameStage(doc, visible, renaming))}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
        />
        <div className="muted">
          {visible.isBlock ? "block " : ""}
          {visible.stage.transform}
          {visible.instance ? ` · inside ${visible.instance}` : ""}
        </div>
        {(transform?.summary || def?.summary) && <p className="summary">{transform?.summary ?? def?.summary}</p>}
      </div>
      {visible.owner.kind === "block" && (
        <div className="notice">Editing stage {visible.localId} of block {visible.owner.block}: the change applies to every instance of the block.</div>
      )}
      <ProblemsList problems={[...problems.filter((p) => p.key === visible.key), ...metrics.errors.map((m) => ({ level: "error" as const, key: visible.key, message: m }))]} />
      <div className="metrics-grid">
        <div>
          <div className="metric-label">weights</div>
          <div className="metric-value">{metrics.parameters.toLocaleString()}</div>
          <WeightsBar parameters={metrics.parameters} />
        </div>
        <div>
          <div className="metric-label">{useStudio.getState().catalog?.cost_calibrated ? `est. forward time (${useStudio.getState().catalog?.cost_calibrated})` : "est. forward time (relative)"}</div>
          <div className="metric-value">{formatSeconds(metrics.seconds)}</div>
          <div className="muted">{total ? `${((metrics.seconds / total) * 100).toFixed(1)}% of the model` : ""}</div>
        </div>
      </div>
      {metrics.static && <div className="chip">static: evaluated once per structure and cached</div>}
      {(() => {
        const bypassable = visible.stage.outputs.length === 1 && visible.stage.inputs.length > 0;
        const reason = bypassable
          ? "A disabled stage passes one input through unchanged (the first, or the update of a residual or self-connection): an ablation without rewiring"
          : "Only a stage with one output and at least one input can be bypassed";
        return (
          <label className="toggle-row" title={reason}>
            <input
              type="checkbox"
              disabled={!bypassable && visible.stage.enabled !== false}
              checked={visible.stage.enabled !== false}
              onChange={(e) => edit((doc) => setEnabled(doc, visible, e.target.checked), e.target.checked ? `Enabled ${visible.localId}` : `Bypassed ${visible.localId}`)}
            />
            enabled{bypassable ? "" : " (cannot be bypassed: several outputs)"}
          </label>
        );
      })()}
      <h4>Inputs</h4>
      {displayedInputs(visible.stage, ports).map(({ port, index }) => {
        const slot = visible.inputs[index];
        const spec = slot ? slotSpec(analysis, catalog, slot) : undefined;
        return (
          <div key={index} className="port-row">
            <span className="port-row-name">
              {port.name}
              {port.optional ? "?" : ""}
              {port.variadic ? "…" : ""}
            </span>
            <span className="port-row-slot mono">{slot || "—"}</span>
            {spec && <span className="port-row-spec mono">{spec.domain} {spec.irreps ?? spec.representation}</span>}
          </div>
        );
      })}
      <h4>Outputs</h4>
      {visible.outputs.map((slot, i) => {
        const spec = slotSpec(analysis, catalog, slot);
        return (
          <div key={slot} className="port-row">
            <span className="port-row-name">{outNames[i]}</span>
            <span className="port-row-slot mono">{slot}</span>
            {spec && (
              <span className={`port-row-spec mono ${"irreps_out" in given ? "" : "inferred-text"}`}>
                {spec.domain} {spec.irreps ?? spec.representation}
              </span>
            )}
          </div>
        );
      })}
      {visible.isBlock && def && (
        <>
          <h4>Block</h4>
          <div className="button-row">
            <button className="button" onClick={() => toggleExpanded(visible.key)}>
              {expanded.has(visible.key) ? "Collapse" : "Expand"}
            </button>
            {visible.instance === null && (
              <button className="button" onClick={() => edit((doc) => dissolveInstance(doc, visible.localId), `Dissolved ${visible.localId}`)}>
                Dissolve into stages
              </button>
            )}
          </div>
          <div className="muted small">
            inputs: {def.inputs.join(", ")} · outputs: {def.outputs.join(", ")} · {def.stages.length} stages
          </div>
          {Object.entries(def.options ?? {}).map(([name, fallback]) => (
            <OptionField
              key={name}
              option={{ name, type: typeof fallback === "number" ? (Number.isInteger(fallback) ? "int" : "float") : name.includes("irreps") ? "irreps" : "str", default: fallback, required: false, key: true, choices: [], inferred: fallback === null, infer_doc: fallback === null ? "inner stages infer it" : "", doc: "" }}
              given={given[name]}
              resolved={analysis?.groups[visible.key]?.options[name]}
              state={name in given ? "explicit" : fallback === null ? "inferred" : "default"}
              onChange={(value) => update(name, value)}
            />
          ))}
        </>
      )}
      {transform && (
        <>
          <h4>Settings</h4>
          {key.map((option) => (
            <OptionField key={option.name} option={option} given={given[option.name]} resolved={record?.options[option.name]} state={state(option.name)} onChange={(v) => update(option.name, v)} />
          ))}
          {rest.length > 0 && (
            <button className="link" onClick={() => setAdvanced(!advanced)}>
              {advanced ? "▾" : "▸"} {rest.length} more settings
            </button>
          )}
          {advanced &&
            rest.map((option) => (
              <OptionField key={option.name} option={option} given={given[option.name]} resolved={record?.options[option.name]} state={state(option.name)} onChange={(v) => update(option.name, v)} />
            ))}
          {!transform.described && <p className="muted small">This transformation has no option schema; options are passed through as written.</p>}
        </>
      )}
      <label className="field">
        <span>role</span>
        <select value={visible.stage.role ?? ""} onChange={(e) => edit((doc) => setRole(doc, visible, e.target.value))}>
          {["", "geometry", "encoder", "message", "readout", "feature"].map((r) => (
            <option key={r} value={r}>
              {r || "(none)"}
            </option>
          ))}
        </select>
      </label>
      <div className="button-row">
        <button className="button danger" onClick={() => edit((doc) => removeStage(doc, visible), `Removed ${visible.key}`)}>
          Delete stage
        </button>
      </div>
    </div>
  );
}

function objectiveOf(document: RecipeDocument | null, fallback: ObjectiveSpec | null): ObjectiveSpec {
  return editableObjective(document ?? {}, fallback);
}

function LossInspector() {
  const { document, analysis, catalog, edit } = useStudio();
  const { problems } = useGraph();
  const objective = objectiveOf(document, analysis?.objective.objective ?? null);
  const [text, setText] = useState(objective.expression ?? "");
  const [error, setError] = useState<string | null>(null);
  const [termType, setTermType] = useState(catalog?.objective_terms[0]?.name ?? "");
  useEffect(() => setText(objective.expression ?? ""), [objective.expression]);
  const apply = () => {
    try {
      const tree = parseExpression(text);
      setError(null);
      edit((doc) => ({ ...doc, objective: { ...objectiveOf(doc, analysis?.objective.objective ?? null), expression: printExpression(tree) } }));
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const addTerm = () => {
    const info = catalog?.objective_terms.find((t) => t.name === termType);
    if (!info) return;
    edit((doc) => {
      const next = objectiveOf(doc, analysis?.objective.objective ?? null);
      let label = info.name;
      let n = 1;
      while (label in next.terms) label = `${info.name}_${n++}`;
      const term: Record<string, unknown> = { term: info.name };
      // The first operator port reads the first target; others start unconnected.
      const first = Object.keys(doc.pipeline?.outputs ?? {})[0];
      if (first && info.operators[0]) term[info.operators[0]] = first;
      next.terms[label] = term;
      next.expression = next.expression ? `${next.expression} + 1.0 * ${label}` : `1.0 * ${label}`;
      return { ...doc, objective: next };
    }, `Added term ${termType}`);
  };
  return (
    <div className="panel-body">
      <div className="inspector-title">
        <div className="title-static">ℒ loss</div>
        <p className="summary">The training objective: an expression over named terms. Terms are evaluated per batch; their inputs are the predicted operators wired into them.</p>
      </div>
      {analysis?.objective.lowered && !document?.objective?.terms && <div className="notice">Derived from the flat training settings. Editing it writes an explicit objective section.</div>}
      <ProblemsList problems={problems.filter((p) => p.key === "loss")} />
      {analysis?.objective.errors.map((e) => (
        <div key={e} className="problem problem-error">
          {e}
        </div>
      ))}
      <label className="field">
        <span>expression</span>
        <textarea className="mono" rows={3} value={text} onChange={(e) => setText(e.target.value)} onBlur={apply} />
      </label>
      {error && <div className="field-error">{error}</div>}
      <h4>Add a term</h4>
      <div className="button-row">
        <select value={termType} onChange={(e) => setTermType(e.target.value)}>
          {catalog?.objective_terms.map((t) => (
            <option key={t.name} value={t.name}>
              {t.name}
            </option>
          ))}
        </select>
        <button className="button" onClick={addTerm} disabled={!catalog?.objective_terms.length}>
          Add
        </button>
      </div>
      <p className="muted small">Functions: sqrt log exp abs min max. Select an operation node to change it.</p>
    </div>
  );
}

function TermInspector({ label }: { label: string }) {
  const { document, analysis, catalog, edit } = useStudio();
  const { problems } = useGraph();
  const objective = objectiveOf(document, analysis?.objective.objective ?? null);
  const term = objective.terms[label];
  if (!term) return null;
  const info = catalog?.objective_terms.find((t) => t.name === term.term);
  const setTerm = (name: string, value: unknown) =>
    edit((doc) => {
      const next = objectiveOf(doc, analysis?.objective.objective ?? null);
      const updated = { ...next.terms[label] };
      if (value === undefined) delete updated[name];
      else updated[name] = value;
      next.terms[label] = updated;
      return { ...doc, objective: next };
    });
  return (
    <div className="panel-body">
      <div className="inspector-title">
        <div className="title-static">{label}</div>
        <div className="muted">objective term {String(term.term)}</div>
        {info?.summary && <p className="summary">{info.summary}</p>}
      </div>
      <ProblemsList problems={problems.filter((p) => p.key === `term:${label}`)} />
      {info?.options.map((option) => (
        <OptionField key={option.name} option={option} given={term[option.name]} resolved={term[option.name] ?? option.default} state={option.name in term ? "explicit" : "default"} onChange={(v) => setTerm(option.name, v)} />
      ))}
    </div>
  );
}

function ExprInspector({ path }: { path: number[] }) {
  const { document, analysis, edit } = useStudio();
  const objective = objectiveOf(document, analysis?.objective.objective ?? null);
  let node: Expr | null = null;
  try {
    node = getAt(parseExpression(objective.expression ?? ""), path);
  } catch {
    node = null;
  }
  if (!node) return null;
  const replace = (make: (old: Expr) => Expr) =>
    edit((doc) => {
      const next = objectiveOf(doc, analysis?.objective.objective ?? null);
      const tree = parseExpression(next.expression ?? "");
      const updated = replaceAt(tree, path, make(getAt(tree, path)));
      next.expression = updated ? printExpression(updated) : "";
      return { ...doc, objective: next };
    });
  return (
    <div className="panel-body">
      <div className="inspector-title">
        <div className="title-static">{node.type === "num" ? "constant" : node.type === "call" ? node.fn : "operation"}</div>
        <div className="muted mono">{printExpression(node)}</div>
      </div>
      {node.type === "num" && (
        <label className="field">
          <span>value</span>
          <input type="number" step="any" defaultValue={node.value} onBlur={(e) => replace(() => ({ type: "num", value: Number(e.target.value) }))} />
        </label>
      )}
      {node.type === "bin" && (
        <div className="button-row">
          {(["+", "-", "*", "/"] as const).map((op) => (
            <button key={op} className={`button ${node && node.type === "bin" && node.op === op ? "primary" : ""}`} onClick={() => replace((old) => ({ ...(old as Extract<Expr, { type: "bin" }>), op }))}>
              {op}
            </button>
          ))}
        </div>
      )}
      <h4>Wrap</h4>
      <div className="button-row">
        {["sqrt", "log", "exp", "abs"].map((fn) => (
          <button key={fn} className="button small" onClick={() => replace((old) => ({ type: "call", fn, args: [old] }))}>
            {fn}(·)
          </button>
        ))}
        <button className="button small" onClick={() => replace((old) => ({ type: "bin", op: "*", left: { type: "num", value: 1 }, right: old }))}>
          w × (·)
        </button>
      </div>
    </div>
  );
}

function RecipeInspector() {
  const { document, analysis, name, system } = useStudio();
  const { problems } = useGraph();
  const errors = problems.filter((p) => p.level === "error").length + (analysis?.errors.length ?? 0);
  return (
    <div className="panel-body">
      <div className="inspector-title">
        <div className="title-static">{name}</div>
        <div className="muted">{document?.pipeline ? "explicit pipeline" : "flat configuration (lowered by the framework)"}</div>
      </div>
      <div className="metrics-grid">
        <div>
          <div className="metric-label">weights</div>
          <div className="metric-value">{formatCount(analysis?.totals.parameters ?? 0)}</div>
        </div>
        <div>
          <div className="metric-label">{useStudio.getState().catalog?.cost_calibrated ? `est. forward time (${useStudio.getState().catalog?.cost_calibrated})` : "est. forward time (relative)"}</div>
          <div className="metric-value">{formatSeconds(analysis?.totals.seconds ?? 0)}</div>
          <div className="muted small">
            {system.size.atoms} atoms × {system.size.neighbors} neighbours
          </div>
        </div>
      </div>
      <div className="muted small">
        {analysis?.stages.length ?? 0} stages · {errors} errors · {problems.filter((p) => p.level === "warning").length} warnings
      </div>
      <h4>System</h4>
      <div className="mono small">
        {Object.entries(system.shells ?? {}).map(([s, l]) => (
          <div key={s}>
            {s}: {l.map((x) => "spdfg"[x]).join(" ")}
          </div>
        ))}
      </div>
      {(document?.plugins?.length ?? 0) > 0 && (
        <>
          <h4>Plugins</h4>
          {document!.plugins!.map((p) => (
            <div key={p} className="mono small">
              {p}
            </div>
          ))}
        </>
      )}
      <h4>Legend</h4>
      <div className="legend">
        <div><span className="legend-swatch" style={{ background: "#5cc8ff" }} /> node features</div>
        <div><span className="legend-swatch" style={{ background: "#ffb547" }} /> edge features</div>
        <div><span className="legend-swatch" style={{ background: "#8a9bb0" }} /> index relations</div>
        <div><span className="legend-swatch" style={{ background: "#ff6b9d" }} /> operators and loss</div>
        <div><span className="legend-port diamond" /> irreps port · <span className="legend-port circle" /> invariant · <span className="legend-port square" /> index</div>
        <div><span className="option-explicit">explicit</span> · <span className="option-inferred">inferred</span> · <span className="option-default">default</span></div>
        <div className="muted small">Blocks above 10% of the forward time are drawn larger, in proportion to their share.</div>
      </div>
    </div>
  );
}

export function Inspector() {
  const { selection } = useStudio();
  const built = useGraph();
  if (selection.length !== 1) return <RecipeInspector />;
  const id = selection[0];
  if (id.startsWith("stage:")) {
    const visible = built.graph.stages.find((s) => `stage:${s.key}` === id);
    return visible ? <StageInspector visible={visible} /> : <RecipeInspector />;
  }
  if (id.startsWith("frame:")) {
    const key = id.slice(6);
    const instance = built.graph.stages.find((s) => s.instance === key);
    return (
      <div className="panel-body">
        <div className="inspector-title">
          <div className="title-static">{key}</div>
          <div className="muted">expanded block instance {instance?.owner.block}</div>
        </div>
        <button className="button" onClick={() => useStudio.getState().toggleExpanded(key)}>
          Collapse
        </button>
      </div>
    );
  }
  if (id === "loss") return <LossInspector />;
  if (id.startsWith("term:")) return <TermInspector label={id.slice(5)} />;
  if (id.startsWith("expr:")) {
    const raw = id.slice(5);
    return <ExprInspector path={raw === "root" ? [] : raw.split(".").map(Number)} />;
  }
  if (id.startsWith("input:") || id.startsWith("target:")) {
    const node = built.nodes.find((n) => n.id === id);
    const data = node?.data as { slot: string; spec?: { [k: string]: unknown } } | undefined;
    return (
      <div className="panel-body">
        <div className="inspector-title">
          <div className="title-static">{id.startsWith("input:") ? data?.slot : `target ${id.slice(7)}`}</div>
          <div className="muted">{id.startsWith("input:") ? "input slot provided by every graph batch" : `predicted operator, read from ${data?.slot || "—"}`}</div>
        </div>
        {data?.spec && (
          <div className="spec-table">
            {Object.entries(data.spec).map(([k, v]) => (
              <div key={k}>
                <span className="muted">{k}</span> <span className="mono">{String(v)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }
  return <RecipeInspector />;
}
