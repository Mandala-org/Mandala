import { useEffect, useState } from "react";
import { currentPipeline, useStudio } from "../store";
import type { FeatureSpecJson, OptionInfo, ObjectiveSpec, RecipeDocument } from "../types";
import { alreadyReported, failedStages, formatCount, formatSeconds, metricsOf, plural, recordsOf, slotSpec, portsOf, displayedInputs, outputNamesOf, type VisibleStage } from "../graph/model";
import { connect, setEnabled, setTarget, dissolveInstance, disconnect, removeStage, renameStage, replaceStage, replacementPlan, setOption, setRole } from "../graph/edit";
import { parseExpression, printExpression, getAt, replaceAt, type Expr } from "../loss/expression";
import { compatibleSlots, type Problem } from "../graph/validate";
import { editableObjective } from "../loss/objective";
import { useGraph } from "../graph/context";
import { irrepsSummary, WeightsBar } from "./nodes";

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

export function OptionField({ option, given, resolved, state, onChange, fixed = false }: { option: OptionInfo; given: unknown; resolved: unknown; state: "explicit" | "inferred" | "default"; onChange: (value: unknown) => void; fixed?: boolean }) {
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
  // Values the recipe defines by itself: the species of the chemical system.
  const hint = /shells/.test(option.name) ? "${species.shells}" : /species|elements/.test(option.name) ? "${species.order}" : "";
  const placeholder = state === "inferred" ? `${show(resolved)} (inferred)` : show(resolved ?? option.default) || (hint ? `${hint} (the recipe's species)` : "");
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
        onKeyDown={(e) => {
          // Enter commits and leaves the field, so Ctrl+Z undoes the edit, not the typing.
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
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
          {state === "inferred" ? `inferred${option.infer_doc ? `: ${option.infer_doc}` : ""}` : state === "default" && option.required ? "required" : state}
        </span>
        {fixed && option.doc && <span className="muted small">{option.doc}</span>}
        {given !== undefined && !fixed && (
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

function sameType(a?: FeatureSpecJson, b?: FeatureSpecJson) {
  return Boolean(a && b && a.domain === b.domain && a.representation === b.representation && (a.irreps ?? a.width) === (b.irreps ?? b.width));
}

/**
 * The input a bypassed stage should pass through: one with the type of the
 * stage's output, so downstream stages keep receiving what they expect.
 * ``choice`` is null when the default input already matches.
 */
function bypassChoice(names: string[], inputSpecs: (FeatureSpecJson | undefined)[], output: FeatureSpecJson | undefined, fallback: number): { choice: string | number | null; reason?: string } {
  if (!output || sameType(inputSpecs[fallback], output)) return { choice: null };
  const index = inputSpecs.findIndex((spec) => sameType(spec, output));
  if (index < 0) {
    return { choice: null, reason: `No input has the type of the output (${output.domain} ${output.irreps ?? output.representation}); bypassing would change what later stages receive` };
  }
  const name = names[index];
  return { choice: names.indexOf(name) === names.lastIndexOf(name) ? name : index };
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

/** Swap the transformation of a stage, keeping its wiring (same category first). */
function ReplaceWith({ visible }: { visible: VisibleStage }) {
  const { document, analysis, catalog, edit } = useStudio();
  const pipeline = currentPipeline({ document, analysis });
  if (!catalog) return null;
  const current = catalog.transforms.find((t) => t.name === visible.stage.transform);
  const category = current?.category;
  const categories = [...new Set(catalog.transforms.map((t) => t.category))].sort((a, b) => (a === category ? -1 : b === category ? 1 : a.localeCompare(b)));
  const blocks = Object.keys(pipeline?.blocks ?? {}).filter((name) => name !== visible.owner.block);
  const library = Object.keys(catalog.block_library ?? {}).filter((name) => !pipeline?.blocks?.[name]);
  return (
    <label className="field replace-with" title="Keeps the stage id and its output slots, so later stages stay connected; inputs are matched by port name, then position">
      <span>replace with</span>
      <select
        value=""
        onChange={(e) => {
          const next = e.target.value;
          if (!next || !pipeline) return;
          const resolved = analysis?.stages.find((s) => s.id === visible.key)?.options ?? {};
          const plan = replacementPlan(visible.stage, next, pipeline, catalog, resolved);
          const parts = [
            plan.kept.length ? `kept ${plan.kept.join(", ")}` : "",
            plan.mapped.length ? `carried ${plan.mapped.map(([a, b]) => `${a} → ${b}`).join(", ")}` : "",
            plan.dropped.length ? `dropped ${plan.dropped.join(", ")} (check the defaults)` : "",
          ].filter(Boolean);
          edit((doc) => replaceStage(doc, visible, next, catalog, resolved), `Replaced ${visible.stage.transform} with ${next} in ${visible.localId}${parts.length ? `: ${parts.join("; ")}` : ""}`);
        }}
      >
        <option value="">another transformation or block…</option>
        {categories.map((c) => (
          <optgroup key={c} label={c.replace("_", " ")}>
            {catalog.transforms
              .filter((t) => t.category === c && t.name !== visible.stage.transform)
              .map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
          </optgroup>
        ))}
        {blocks.length > 0 && (
          <optgroup label="blocks in this recipe">
            {blocks.filter((b) => b !== visible.stage.transform).map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </optgroup>
        )}
        {library.length > 0 && (
          <optgroup label="block library">
            {library.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </optgroup>
        )}
      </select>
    </label>
  );
}

function StageInspector({ visible }: { visible: VisibleStage }) {
  const { document, analysis, catalog, edit, toggleExpanded, expanded } = useStudio();
  const { problems, graph } = useGraph();
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
          onBlur={() => {
            if (renaming === visible.localId) return;
            const ok = edit((doc) => renameStage(doc, visible, renaming), `Renamed ${visible.localId} to ${renaming}`);
            if (ok) useStudio.setState({ pendingSelection: [`stage:${visible.instance ? `${visible.instance}/` : ""}${renaming}`] });
          }}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
        />
        <div className="muted">
          {visible.isBlock ? "block " : ""}
          {visible.stage.transform}
          {visible.instance ? ` · inside ${visible.instance}` : ""}
        </div>
        {(transform?.summary || def?.summary) && <p className="summary">{transform?.summary ?? def?.summary}</p>}
      </div>
      <ReplaceWith visible={visible} />
      {visible.owner.kind === "block" && (
        <div className="notice">Editing stage {visible.localId} of block {visible.owner.block}: the change applies to every instance of the block.</div>
      )}
      <ProblemsList
        problems={(() => {
          // Each cause once: compiler messages the editor's own checks already cover are left out.
          const own = problems.filter((p) => p.key === visible.key);
          const known = own.map((p) => p.message);
          return [...own, ...metrics.errors.filter((m) => !alreadyReported(known, m)).map((m) => ({ level: "error" as const, key: visible.key, message: m }))];
        })()}
      />
      <div className="metrics-grid">
        <div>
          <div className="metric-label">weights</div>
          <div className="metric-value">{metrics.parameters.toLocaleString()}</div>
          <WeightsBar parameters={metrics.parameters} label={false} />
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
          ? "A disabled stage passes one input through unchanged, one with the type of its output: an ablation without rewiring"
          : "Only a stage with one output and at least one input can be bypassed";
        const names = def ? def.inputs : visible.stage.inputs.map((_, i) => displayedInputs(visible.stage, ports).find((p) => p.index === i)?.port.name ?? String(i));
        const fallback = transform?.bypass ?? 0;
        const disabled = visible.stage.enabled === false;
        const chosen = visible.stage.bypass;
        const passed = typeof chosen === "number" ? chosen : typeof chosen === "string" ? names.indexOf(chosen) : fallback;
        const disable = () => {
          const specs = visible.inputs.map((slot) => (slot ? slotSpec(analysis, catalog, slot) : undefined));
          const { choice, reason: refused } = bypassChoice(names, specs, slotSpec(analysis, catalog, visible.outputs[0]), fallback);
          if (refused) {
            useStudio.getState().notify(`${visible.localId} cannot be bypassed: ${refused}`, "error");
            return;
          }
          const passedName = choice === null ? names[fallback] : typeof choice === "number" ? names[choice] : choice;
          edit((doc) => setEnabled(doc, visible, false, choice), `Bypassed ${visible.localId}: it passes ${passedName} through`);
        };
        return (
          <>
            <label className="toggle-row" title={reason}>
              <input
                type="checkbox"
                aria-label={`${visible.localId} enabled`}
                disabled={!bypassable && !disabled}
                checked={!disabled}
                onChange={(e) => (e.target.checked ? edit((doc) => setEnabled(doc, visible, true), `Enabled ${visible.localId}`) : disable())}
              />
              {bypassable ? "enabled (untick to bypass for an ablation)" : "enabled (cannot be bypassed: several outputs or no input)"}
            </label>
            {disabled && (
              <label className="field">
                <span>passes through</span>
                <select
                  value={String(passed)}
                  onChange={(e) => {
                    const i = Number(e.target.value);
                    const unique = names.indexOf(names[i]) === names.lastIndexOf(names[i]);
                    edit((doc) => setEnabled(doc, visible, false, i === fallback ? null : unique ? names[i] : i), `${visible.localId} passes ${names[i]} through`);
                  }}
                >
                  {names.map((name, i) => {
                    // Only inputs of the passed value's type keep later stages working.
                    const specOf = (j: number) => (visible.inputs[j] ? slotSpec(analysis, catalog, visible.inputs[j]) : undefined);
                    if (i !== passed && !sameType(specOf(i), specOf(passed))) return null;
                    return (
                      <option key={i} value={i}>
                        {name} ({visible.inputs[i] || "not connected"})
                      </option>
                    );
                  })}
                </select>
              </label>
            )}
          </>
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
            <select
              className="port-row-slot mono"
              value={slot || ""}
              aria-label={`Input ${port.name}`}
              title={port.doc || `What ${port.name} reads`}
              onChange={(e) => {
                const next = e.target.value;
                if (!next) edit((doc) => disconnect(doc, visible, index, catalog), `Disconnected ${port.name}`);
                else edit((doc) => connect(doc, graph, next, visible, index, catalog), `${visible.localId}.${port.name} reads ${next}`);
              }}
            >
              <option value="">{port.optional ? "— none (optional) —" : "— not connected —"}</option>
              {slot && !compatibleSlots(graph, analysis, catalog, visible, port, index).some((c) => c.slot === slot) && <option value={slot}>{slot} (incompatible)</option>}
              {compatibleSlots(graph, analysis, catalog, visible, port, index).map((c) => (
                <option key={c.slot} value={c.slot}>
                  {c.slot}
                  {c.spec ? ` · ${c.spec.domain} ${c.spec.irreps ? irrepsSummary(c.spec.irreps) ?? c.spec.irreps : c.spec.representation}` : ""}
                  {c.producer ? "" : " (input)"}
                </option>
              ))}
            </select>
            {slot && (
              <button className="link port-row-disconnect" title={`Disconnect ${port.name}`} aria-label={`Disconnect ${port.name}`} onClick={() => edit((doc) => disconnect(doc, visible, index, catalog), `Disconnected ${port.name}`)}>
                ✕
              </button>
            )}
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
          {Object.entries(def.options ?? {}).map(([name, fallback]) => {
            // "Inferred" unless an inner stage reports that it needs the value.
            const needed = !(name in given) && metrics.errors.some((m) => m.includes(`'${name}'`) && /requires|required/.test(m));
            return (
              <OptionField
                key={name}
                option={{ name, type: typeof fallback === "number" ? (Number.isInteger(fallback) ? "int" : "float") : name.includes("irreps") ? "irreps" : "str", default: fallback, required: needed, key: true, choices: [], inferred: fallback === null && !needed, infer_doc: fallback === null && !needed ? "inner stages infer it" : "", doc: "" }}
                given={given[name]}
                resolved={analysis?.groups[visible.key]?.options[name]}
                state={name in given ? "explicit" : fallback === null && !needed ? "inferred" : "default"}
                onChange={(value) => update(name, value)}
              />
            );
          })}
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
      {(advanced || visible.stage.role) && (
        <label className="field" title="The stage's part of the model. Readout stages stay trainable in head-only fine-tuning (freeze_backbone_train_heads_only) and form the per-operator heads; other roles only group stages in summaries">
          <span>role</span>
          <select value={visible.stage.role ?? ""} onChange={(e) => edit((doc) => setRole(doc, visible, e.target.value))}>
            {["", "geometry", "encoder", "message", "readout", "feature"].map((r) => (
              <option key={r} value={r}>
                {r || "(none)"}
              </option>
            ))}
          </select>
        </label>
      )}
      <div className="button-row">
        <button className="button" onClick={() => (window as unknown as { studioDuplicate?: () => void }).studioDuplicate?.()} title="Copy this stage beside the original (Ctrl+D)">
          Duplicate
        </button>
        <button
          className="button"
          onClick={() => (window as unknown as { studioInsertAfter?: () => void }).studioInsertAfter?.()}
          title="Insert a copy after this stage as the next layer (Ctrl+Shift+D): the copy reads this stage's output, and later stages read the copy"
        >
          Insert copy after
        </button>
        <button className="button danger" onClick={() => edit((doc) => removeStage(doc, visible), `Removed ${visible.key}`)} title={`Delete this ${visible.isBlock ? "block instance" : "stage"} (Delete key)`}>
          {visible.isBlock ? "Delete block instance" : "Delete stage"}
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
  const objective = objectiveOf(document, analysis?.objective?.objective ?? null);
  const [text, setText] = useState(objective.expression ?? "");
  const [error, setError] = useState<string | null>(null);
  const [termType, setTermType] = useState(catalog?.objective_terms[0]?.name ?? "");
  useEffect(() => setText(objective.expression ?? ""), [objective.expression]);
  const apply = () => {
    try {
      const tree = parseExpression(text);
      setError(null);
      edit((doc) => ({ ...doc, objective: { ...objectiveOf(doc, analysis?.objective?.objective ?? null), expression: printExpression(tree) } }));
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const addTerm = () => {
    const info = catalog?.objective_terms.find((t) => t.name === termType);
    if (!info) return;
    edit((doc) => {
      const next = objectiveOf(doc, analysis?.objective?.objective ?? null);
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
      {analysis?.objective?.lowered && !document?.objective?.terms && <div className="notice">Derived from the training settings. Editing it writes the objective into the recipe.</div>}
      <ProblemsList problems={problems.filter((p) => p.key === "loss")} />
      {(analysis?.objective?.errors ?? []).map((e) => (
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
  const objective = objectiveOf(document, analysis?.objective?.objective ?? null);
  const term = objective.terms[label];
  if (!term) return null;
  const info = catalog?.objective_terms.find((t) => t.name === term.term);
  const setTerm = (name: string, value: unknown) =>
    edit((doc) => {
      const next = objectiveOf(doc, analysis?.objective?.objective ?? null);
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
      {(info?.requires?.length ?? 0) > 0 &&
        (() => {
          const settings = info!.settings ?? [];
          const values = settings.map((name) => (document as Record<string, unknown> | null)?.[name]);
          const on = settings.length > 0 && values.every((v) => v === true);
          return (
            <div className={`notice ${values.some((v) => v === false) ? "notice-error" : ""}`}>
              Reads {info!.requires!.join(", ")}, computed from the predicted operators during training.{" "}
              {settings.length === 0
                ? ""
                : on
                  ? `${settings.join(" and ")} ${settings.length > 1 ? "are" : "is"} on in this recipe.`
                  : `Needs ${settings.map((name, i) => `${name}: true${values[i] === undefined ? " (not set; check the default)" : values[i] === false ? " (set to false)" : ""}`).join(", ")}.`}
            </div>
          );
        })()}
      {info && !info.operators.length && !info.options.length && !(info.requires?.length) && <p className="muted small">This term has no inputs or settings.</p>}
      <ProblemsList problems={problems.filter((p) => p.key === `term:${label}`)} />
      {info?.options.map((option) => (
        <OptionField key={option.name} option={option} given={term[option.name]} resolved={term[option.name] ?? option.default} state={option.name in term ? "explicit" : "default"} onChange={(v) => setTerm(option.name, v)} />
      ))}
    </div>
  );
}

function ExprInspector({ path }: { path: number[] }) {
  const { document, analysis, edit } = useStudio();
  const objective = objectiveOf(document, analysis?.objective?.objective ?? null);
  let node: Expr | null = null;
  try {
    node = getAt(parseExpression(objective.expression ?? ""), path);
  } catch {
    node = null;
  }
  if (!node) return null;
  const replace = (make: (old: Expr) => Expr) =>
    edit((doc) => {
      const next = objectiveOf(doc, analysis?.objective?.objective ?? null);
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
          <input
            type="number"
            step="any"
            defaultValue={node.value}
            onBlur={(e) => Number(e.target.value) !== (node as { value: number }).value && replace(() => ({ type: "num", value: Number(e.target.value) }))}
            onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          />
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

/** The recipe's shared values (``parameters:``), referenced as ${parameters.name}. */
function ParametersSection() {
  const { document, edit } = useStudio();
  const parameters = document?.parameters as Record<string, unknown> | undefined;
  if (!parameters || typeof parameters !== "object" || !Object.keys(parameters).length) return null;
  // How often the recipe refers to a parameter (stage and block options).
  const text = JSON.stringify(document?.pipeline ?? {});
  const uses = (key: string) => text.split(`\${parameters.${key}}`).length - 1;
  const typeOf = (value: unknown): OptionInfo["type"] =>
    typeof value === "boolean" ? "bool" : typeof value === "number" ? (Number.isInteger(value) ? "int" : "float") : typeof value === "string" && IRREPS.test(value) ? "irreps" : Array.isArray(value) ? "list" : "str";
  return (
    <>
      <h4>Parameters</h4>
      <p className="muted small">Shared values the stages use as ${"{parameters.name}"}; a change here applies everywhere they are used.</p>
      {Object.entries(parameters).map(([key, value]) => (
        <OptionField
          key={key}
          option={{ name: key, type: typeOf(value), default: value, required: false, key: true, choices: [], inferred: false, infer_doc: "", doc: `used ${uses(key)} time${uses(key) === 1 ? "" : "s"} in the recipe` }}
          given={value}
          resolved={value}
          state="explicit"
          fixed
          onChange={(next) => {
            if (next === undefined) return;
            edit((doc) => ({ ...doc, parameters: { ...(doc.parameters as Record<string, unknown>), [key]: next } }), `parameters.${key} = ${typeof next === "string" ? next : JSON.stringify(next)}`);
          }}
        />
      ))}
    </>
  );
}

function RecipeInspector() {
  const { document, analysis, name, system } = useStudio();
  const { problems } = useGraph();
  const errors = problems.filter((p) => p.level === "error").length;
  const failed = failedStages(analysis);
  return (
    <div className="panel-body">
      <div className="inspector-title">
        <div className="title-static">{name}</div>
        <div className="muted">{document?.pipeline ? "stage recipe" : "training settings (the framework builds the stages)"}</div>
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
        {plural(analysis?.stages?.length ?? 0, "stage")} · {plural(errors, "error")} · {plural(problems.filter((p) => p.level === "warning").length, "warning")}
      </div>
      {failed > 0 && <div className="notice">{plural(failed, "stage")} failed to compile: the weights and time above leave them out, so they are lower bounds.</div>}
      <ParametersSection />
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

/** The type of a value, readably: only the fields that say something. */
function SpecTable({ spec }: { spec?: FeatureSpecJson }) {
  if (!spec) return null;
  const rows: [string, string][] = [
    ["entity", spec.domain],
    ["representation", spec.representation.replace("_", " ")],
    ...(spec.irreps ? ([["irreps", spec.irreps]] as [string, string][]) : []),
    ...(!spec.irreps && spec.width ? ([["width", String(spec.width)]] as [string, string][]) : []),
    ...(spec.support && spec.support !== "output" ? ([["support", spec.support]] as [string, string][]) : []),
    ...(spec.frame && !["global", "none"].includes(spec.frame) ? ([["frame", spec.frame]] as [string, string][]) : []),
  ];
  return (
    <div className="spec-table">
      {rows.map(([k, v]) => (
        <div key={k}>
          <span className="muted">{k}</span> <span className="mono">{v}</span>
        </div>
      ))}
    </div>
  );
}

/** A predicted operator: which decoded edge blocks it reads. */
function TargetInspector({ operator }: { operator: string }) {
  const { document, analysis, catalog, edit } = useStudio();
  const { graph } = useGraph();
  const pipeline = currentPipeline({ document, analysis });
  const slot = pipeline?.outputs?.[operator] ?? "";
  const spec = slot ? slotSpec(analysis, catalog, slot) : undefined;
  const candidates = graph.stages
    .filter((s) => s.instance === null)
    .flatMap((s) => s.outputs)
    .filter((o) => {
      const t = slotSpec(analysis, catalog, o);
      return t && t.domain === "edge" && ["matrix", "block_vectors"].includes(t.representation);
    });
  return (
    <div className="panel-body">
      <div className="inspector-title">
        <div className="title-static">target {operator}</div>
        <div className="muted">predicted operator: decoded edge blocks the loss compares with the reference</div>
      </div>
      <label className="field">
        <span>reads</span>
        <select value={slot} onChange={(e) => edit((doc) => setTarget(doc, operator, e.target.value), e.target.value ? `${operator} reads ${e.target.value}` : `${operator} disconnected`)}>
          <option value="">— not connected —</option>
          {slot && !candidates.includes(slot) && <option value={slot}>{slot} (incompatible)</option>}
          {candidates.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>
      {!candidates.length && <p className="muted small">No stage produces decoded edge blocks yet: end the model with a readout that decodes to operator blocks.</p>}
      <SpecTable spec={spec} />
    </div>
  );
}

function SelectionInspector({ ids }: { ids: string[] }) {
  const { edit } = useStudio();
  const built = useGraph();
  const stages = ids.map((id) => built.graph.stages.find((s) => `stage:${s.key}` === id)).filter((s): s is VisibleStage => Boolean(s));
  const call = (name: "studioGroup" | "studioDuplicate" | "studioInsertAfter") => (window as unknown as Record<string, (() => void) | undefined>)[name]?.();
  return (
    <div className="panel-body">
      <div className="inspector-title">
        <div className="title-static">{plural(ids.length, "item")} selected</div>
        <div className="muted mono small">{stages.map((s) => s.localId).join(", ") || ids.join(", ")}</div>
      </div>
      {stages.length > 0 && (
        <div className="button-row wrap">
          <button className="button" onClick={() => call("studioInsertAfter")} title="Copy the selection as the next layer (Ctrl+Shift+D)">
            Insert copy after
          </button>
          <button className="button" onClick={() => call("studioDuplicate")} title="Copy the selection beside it (Ctrl+D)">
            Duplicate
          </button>
          <button className="button" onClick={() => call("studioGroup")} title="Group into a named block (Ctrl+G)">
            Group
          </button>
          <button className="button danger" onClick={() => edit((doc) => stages.reduce((current, stage) => removeStage(current, stage), doc), `Deleted ${plural(stages.length, "stage")}`)}>
            Delete
          </button>
        </div>
      )}
      <p className="muted small">Ctrl/Shift-click adds or removes items; Shift-drag selects a box.</p>
    </div>
  );
}

export function Inspector() {
  const { selection } = useStudio();
  const built = useGraph();
  if (selection.length > 1) return <SelectionInspector ids={selection} />;
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
  if (id.startsWith("ref:")) {
    const node = built.nodes.find((n) => n.id === id);
    const data = node?.data as { slot: string; producer: string | null; uses: number; spec?: { [k: string]: unknown } } | undefined;
    if (!data) return <RecipeInspector />;
    return (
      <div className="panel-body">
        <div className="inspector-title">
          <div className="title-static">{data.slot}</div>
          <div className="muted">
            {data.producer ? `computed by ${data.producer.slice(6)}` : "structure input"} · used by {data.uses} stages
          </div>
          <p className="summary">Shown beside each use instead of as wires from one source (toolbar: ↪ pins).{data.producer ? " Double-click the pin to go to the stage that computes it." : ""}</p>
        </div>
        <SpecTable spec={data.spec as FeatureSpecJson | undefined} />
      </div>
    );
  }
  if (id.startsWith("target:")) return <TargetInspector operator={id.slice(7)} />;
  if (id.startsWith("input:")) {
    const node = built.nodes.find((n) => n.id === id);
    const data = node?.data as { slot: string; spec?: FeatureSpecJson } | undefined;
    return (
      <div className="panel-body">
        <div className="inspector-title">
          <div className="title-static">{data?.slot}</div>
          <div className="muted">input provided by every structure in a batch</div>
        </div>
        <SpecTable spec={data?.spec} />
      </div>
    );
  }
  return <RecipeInspector />;
}
