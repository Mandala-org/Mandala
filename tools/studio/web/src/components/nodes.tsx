import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useStudio } from "../store";
import { alreadyReported, formatCount, formatSeconds } from "../graph/model";
import { DOMAIN_COLORS, specColor, type ExprNodeData, type FrameNodeData, type InputNodeData, type RefNodeData, type StageNodeData, type TargetNodeData, type TermNodeData } from "../graph/build";
import type { FeatureSpecJson, PortInfo } from "../types";
import { connectionProblem, type Problem } from "../graph/validate";

export const CATEGORY_COLORS: Record<string, string> = {
  input: "#5cc8ff",
  radial_basis: "#4dd0c8",
  angular_basis: "#4dd0c8",
  geometry: "#4dd0c8",
  frame: "#4dd0c8",
  routing: "#8a9bb0",
  embedding: "#a5d66b",
  encoder: "#a5d66b",
  convolution: "#ff8a5c",
  product: "#ff8a5c",
  message: "#ff8a5c",
  aggregation: "#f2c94c",
  nonlinearity: "#c792ea",
  normalization: "#9fa8da",
  mlp: "#82aaff",
  linear: "#82aaff",
  residual: "#9fa8da",
  combine: "#8a9bb0",
  readout: "#ff6b9d",
  output: "#ff6b9d",
  envelope: "#4dd0c8",
  statistics: "#9fa8da",
  block: "#e8e8e8",
  custom: "#e0b4ff",
};

function useOrientation() {
  return useStudio((s) => s.orientation);
}

function portStyle(color: string, kind: string, filled: boolean): React.CSSProperties {
  const shape: React.CSSProperties =
    kind === "index" || kind === "object"
      ? { borderRadius: 2 }
      : kind === "irreps" || kind === "channel" || kind === "local"
        ? { borderRadius: 2, transform: "rotate(45deg)" }
        : { borderRadius: "50%" };
  return { background: filled ? color : "#14181f", borderColor: color, ...shape };
}

function spread(count: number, index: number) {
  return `${((index + 1) / (count + 1)) * 100}%`;
}

function describe(spec?: FeatureSpecJson) {
  if (!spec) return "not analysed";
  const parts = [spec.domain, spec.representation];
  if (spec.irreps) parts.push(spec.irreps);
  else if (spec.width) parts.push(`width ${spec.width}`);
  if (spec.support && spec.support !== "output") parts.push(`support ${spec.support}`);
  if (spec.frame !== "global" && spec.frame !== "none") parts.push(`${spec.frame} frame`);
  return parts.join(" · ");
}

function shortIrreps(spec?: FeatureSpecJson) {
  if (!spec) return "";
  if (spec.irreps) return irrepsSummary(spec.irreps) ?? spec.irreps;
  if (spec.representation === "matrix") return `blocks ${spec.width ?? ""}`;
  if (spec.representation === "block_vectors") return "pair vectors";
  if (spec.representation === "index") return spec.width ? `index /${spec.width}` : "index";
  return spec.representation;
}

function ProblemBadge({ problems }: { problems: Problem[] }) {
  if (!problems.length) return null;
  const errors = problems.filter((p) => p.level === "error").length;
  return (
    <span className={`badge ${errors ? "badge-error" : "badge-warning"}`} title={problems.map((p) => p.message).join("\n")}>
      {errors ? "✕" : "⚠"} {problems.length}
    </span>
  );
}

export function WeightsBar({ parameters, label = true }: { parameters: number; label?: boolean }) {
  const fraction = Math.min(1, Math.log10(parameters + 1) / 7);
  return (
    <div className="weights" title={`${parameters.toLocaleString()} weights (log scale, full bar = 10M)`}>
      <div className="weights-track">
        {[0.2, 0.4, 0.6, 0.8].map((t) => (
          <span key={t} className="weights-tick" style={{ left: `${t * 100}%` }} />
        ))}
        <div className="weights-fill" style={{ width: `${fraction * 100}%` }} />
      </div>
      {label && <span className="weights-label">{parameters ? formatCount(parameters) : "—"}</span>}
    </div>
  );
}

function TimeRing({ share }: { share: number }) {
  const r = 7;
  const c = 2 * Math.PI * r;
  return (
    <svg width="18" height="18" className="time-ring">
      <circle cx="9" cy="9" r={r} stroke="#2a313d" strokeWidth="3" fill="none" />
      <circle cx="9" cy="9" r={r} stroke={share > 0.1 ? "#ff8a5c" : "#7d8a9c"} strokeWidth="3" fill="none" strokeDasharray={`${c * share} ${c}`} transform="rotate(-90 9 9)" />
    </svg>
  );
}

const IRREP_TERM = /^(\d+)x(\d+)[eo]$/;

/** Long irreps as their first term and total dimension: 128x0e+… (dim 704). */
export function irrepsSummary(value: string): string | null {
  const terms = value.split("+").map((t) => t.trim());
  if (terms.length <= 2 || !terms.every((t) => IRREP_TERM.test(t))) return null;
  const dim = terms.reduce((total, t) => {
    const [, mul, l] = IRREP_TERM.exec(t)!;
    return total + Number(mul) * (2 * Number(l) + 1);
  }, 0);
  return `${terms[0]}+… (dim ${dim})`;
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return irrepsSummary(value) ?? (value.length > 26 ? `${value.slice(0, 24)}…` : value);
  if (Array.isArray(value)) return `[${value.join(", ")}]`.slice(0, 26);
  if (typeof value === "object") return "{…}";
  return String(value);
}

/** Where a handle is drawn along its side: its place in the layout's order. */
function place(order: number[] | undefined, handle: number, fallback: number) {
  const at = order ? order.indexOf(handle) : -1;
  return at >= 0 ? at : fallback;
}

function InputHandles({ ports, count, order }: { ports: { port: PortInfo; index: number; slot: string; spec?: FeatureSpecJson }[]; count: number; order?: number[] }) {
  const orientation = useOrientation();
  const drag = useStudio((s) => s.drag);
  return (
    <>
      {ports.map(({ port, index, slot, spec }, j) => {
        const i = place(order, index, j);
        const color = spec ? specColor(spec) : port.domain && !Array.isArray(port.domain) ? DOMAIN_COLORS[port.domain] ?? "#6b7686" : "#6b7686";
        const position = orientation === "TB" ? Position.Top : Position.Left;
        const style = orientation === "TB" ? { left: spread(count, i) } : { top: `${40 + i * 22}px` };
        return (
          <Handle
            key={`in-${index}`}
            id={`in-${index}`}
            type="target"
            position={position}
            className={`port ${drag ? (connectionProblem(drag.spec, port) || (!port.domain && spec && drag.spec && spec.domain !== drag.spec.domain) ? "port-no" : "port-yes") : ""}`}
            style={{ ...style, ...portStyle(color, port.kind, Boolean(slot)) }}
            title={`${port.name}${port.with_shift ? " (an edge relation; its lattice shifts come with it)" : ""}${port.optional ? " (optional)" : ""}${port.variadic ? " (repeatable)" : ""}\naccepts: ${port.domain ? [port.domain].flat().join("/") : "any entity"}, ${port.kind}\n${slot ? `${slot}: ${describe(spec)}` : "not connected"}${port.doc ? `\n${port.doc}` : ""}`}
          />
        );
      })}
    </>
  );
}

export const StageNode = memo(function StageNode({ data, selected }: NodeProps) {
  const d = data as StageNodeData;
  const orientation = useOrientation();
  const toggle = useStudio((s) => s.toggleExpanded);
  const color = CATEGORY_COLORS[d.category] ?? CATEGORY_COLORS.custom;
  const errors = d.problems.some((p) => p.level === "error") || d.metrics.errors.length > 0;
  const portCount = d.ports.length;
  return (
    <div
      className={`stage ${selected ? "selected" : ""} ${errors ? "has-error" : ""} ${d.diff ? `diff-${d.diff}` : ""} ${d.isBlock ? "is-block" : ""} ${d.visible.stage.enabled === false ? "bypassed" : ""}`}
      style={{ "--accent": color, fontSize: `${12 * d.scale}px` } as React.CSSProperties}
    >
      <InputHandles ports={d.ports} count={portCount} order={d.inputOrder} />
      <div className="stage-far" style={{ fontSize: Math.min((d.size.width * 0.86) / (0.62 * Math.max(d.visible.localId.length, 6)), d.size.height * 0.36) }}>
        {d.visible.localId}
      </div>
      {orientation === "TB" ? (
        <div className="port-labels top">
          {d.ports.map(({ port, index, slot }, j) => (
            <span key={index} className={`port-label ${slot ? "" : "open"}`} style={{ left: spread(portCount, place(d.inputOrder, index, j)), maxWidth: `calc(100% / ${portCount + 1})` }} title={port.name}>
              {port.name}
              {port.with_shift ? " + shifts" : ""}
            </span>
          ))}
        </div>
      ) : (
        <div className="port-labels side-in">
          {d.ports.map(({ port, index, slot }, j) => (
            <span key={index} className={`port-label ${slot ? "" : "open"}`} style={{ top: 40 + place(d.inputOrder, index, j) * 22 }} title={port.name}>
              {port.name}
            </span>
          ))}
        </div>
      )}
      <div className="stage-header">
        <span className="stage-dot" />
        <span className="stage-title" title={d.summary}>
          {d.visible.localId}
        </span>
        {d.diff && <span className={`badge badge-${d.diff}`}>{d.diff === "added" ? "+" : "~"}</span>}
        {d.visible.stage.enabled === false && <span className="badge badge-bypassed" title="Disabled: passes one input through unchanged (the first, or the update of a residual or self-connection)">bypassed</span>}
        <ProblemBadge problems={[...d.problems, ...d.metrics.errors.filter((m) => !alreadyReported(d.problems.map((p) => p.message), m)).map((m) => ({ level: "error" as const, key: d.visible.key, message: m }))]} />
      </div>
      <div className="stage-transform">
        {d.isBlock ? "◆ " : ""}
        {d.visible.stage.transform}
        {d.metrics.static && <span className="chip" title="Parameter-free, computed once per structure and cached">static</span>}
      </div>
      <div className="stage-options">
        {d.keyOptions.map((o) => (
          <div key={o.name} className={`option option-${o.state}`} title={`${o.name} = ${typeof o.value === "string" ? o.value : JSON.stringify(o.value)}\n${o.state === "inferred" ? "inferred from the wiring" : o.state === "default" ? "default value" : "set in the recipe"}`}>
            <span className="option-name">{o.name}</span>
            <span className="option-value">{formatValue(o.value)}</span>
          </div>
        ))}
      </div>
      <div className="stage-metrics">
        <WeightsBar parameters={d.metrics.parameters} />
        <span className="time" title={`estimated forward time; ${(d.share * 100).toFixed(1)}% of the model`}>
          <TimeRing share={d.share} />
          {formatSeconds(d.metrics.seconds)}
        </span>
      </div>
      {d.isBlock && d.visible.stage.enabled !== false && (
        <button className="expand" onClick={() => toggle(d.visible.key)} title="Show the stages inside this block">
          ⊕ expand {d.blockInstances > 1 ? `· ${d.blockInstances} instances` : ""}
        </button>
      )}
      {d.outputs.map((o, j) => {
        const i = place(d.outputOrder, j, j);
        const position = orientation === "TB" ? Position.Bottom : Position.Right;
        const style = orientation === "TB" ? { left: spread(d.outputs.length, i) } : { top: `${40 + i * 22}px` };
        return (
          <Handle
            key={`out-${i}`}
            id={`out-${i}`}
            type="source"
            position={position}
            className="port"
            style={{ ...style, ...portStyle(specColor(o.spec), o.spec?.representation === "index" ? "index" : o.spec?.invariant ? "invariant" : "irreps", true) }}
            title={`${o.name} → ${o.slot}\n${describe(o.spec)}`}
          />
        );
      })}
      <div className={`port-labels ${orientation === "TB" ? "bottom" : "side-out"}`}>
        {d.outputs.map((o, j) => {
          const i = place(d.outputOrder, j, j);
          return (
          <span
            key={j}
            className={`port-label out ${o.manual ? "manual" : "inferred"}`}
            style={orientation === "TB" ? { left: spread(d.outputs.length, i), maxWidth: `calc(100% / ${d.outputs.length + 1})` } : { top: 40 + i * 22 }}
            title={`${o.slot}: ${describe(o.spec)}${o.manual ? "" : "\n(irreps inferred)"}`}
          >
            {shortIrreps(o.spec) || o.name}
          </span>
          );
        })}
      </div>
    </div>
  );
});

export const FrameNode = memo(function FrameNode({ data, selected }: NodeProps) {
  const d = data as FrameNodeData;
  const toggle = useStudio((s) => s.toggleExpanded);
  return (
    <div className={`frame ${selected ? "selected" : ""}`}>
      <div className="frame-header">
        <button className="collapse" onClick={() => toggle(d.key)} title="Collapse the block">
          ⊖
        </button>
        <span className="frame-title">◆ {d.block}</span>
        <span className="frame-instance">{d.key}</span>
        <span className="frame-metrics">
          {formatCount(d.metrics.parameters)} w · {formatSeconds(d.metrics.seconds)}
        </span>
        {d.instances > 1 && (
          <span className="chip chip-warn" title="Editing the inner stages changes the block definition for every instance">
            shared by {d.instances}
          </span>
        )}
      </div>
    </div>
  );
});

export const InputNode = memo(function InputNode({ data, selected }: NodeProps) {
  const d = data as InputNodeData;
  const orientation = useOrientation();
  const color = specColor(d.spec);
  return (
    <div className={`io input ${selected ? "selected" : ""}`} style={{ "--accent": color } as React.CSSProperties} title={describe(d.spec)}>
      <div className="io-name">{d.slot}</div>
      <div className="io-spec">{d.spec ? `${d.spec.domain} · ${shortIrreps(d.spec) || d.spec.representation}` : "input"}</div>
      <Handle id="out-0" type="source" position={orientation === "TB" ? Position.Bottom : Position.Right} className="port" style={portStyle(color, d.spec?.representation === "index" ? "index" : d.spec?.invariant ? "invariant" : "irreps", true)} />
    </div>
  );
});

function portKind(spec?: FeatureSpecJson) {
  return spec?.representation === "index" || spec?.representation === "object" ? "index" : spec?.invariant ? "invariant" : "irreps";
}

export const RefNode = memo(function RefNode({ data, selected }: NodeProps) {
  const d = data as RefNodeData;
  const orientation = useOrientation();
  const color = specColor(d.spec);
  return (
    <div
      className={`ref ${selected ? "selected" : ""}`}
      style={{ "--accent": color } as React.CSSProperties}
      title={`${d.slot}: ${describe(d.spec)}\nused by ${d.uses} stages; drawn beside each use${d.producer ? "\nDouble-click to go to the stage that computes it" : " (structure input)"}`}
    >
      <span className="ref-arrow">{d.producer ? "↪" : "▸"}</span>
      <span className="ref-name">{d.slot}</span>
      <Handle id="out-0" type="source" position={orientation === "TB" ? Position.Bottom : Position.Right} className="port" style={portStyle(color, portKind(d.spec), true)} />
    </div>
  );
});

export const TargetNode = memo(function TargetNode({ data, selected }: NodeProps) {
  const d = data as TargetNodeData;
  const orientation = useOrientation();
  const error = d.problems.some((p) => p.level === "error");
  return (
    <div className={`io target ${selected ? "selected" : ""} ${error ? "has-error" : ""}`} title={d.problems.map((p) => p.message).join("\n") || describe(d.spec)}>
      <Handle id="in-0" type="target" position={orientation === "TB" ? Position.Top : Position.Left} className="port" style={portStyle(DOMAIN_COLORS.operator, "matrix", Boolean(d.slot))} />
      <div className="io-name">◉ {d.operator}</div>
      <div className="io-spec">{d.slot ? `${d.slot} · ${shortIrreps(d.spec)}` : "not connected"}</div>
      <Handle id="out-0" type="source" position={orientation === "TB" ? Position.Bottom : Position.Right} className="port" style={portStyle(DOMAIN_COLORS.operator, "matrix", true)} />
    </div>
  );
});

export const TermNode = memo(function TermNode({ data, selected }: NodeProps) {
  const d = data as TermNodeData;
  const orientation = useOrientation();
  const error = d.problems.some((p) => p.level === "error");
  const options = Object.entries(d.term).filter(([k]) => k !== "term" && !d.operators.includes(k));
  const vertical = orientation === "TB";
  const read = d.operators.map((port) => d.term[port]).filter(Boolean);
  return (
    <div className={`term ${selected ? "selected" : ""} ${error ? "has-error" : ""} ${d.lowered ? "lowered" : ""}`} title={d.summary}>
      {d.operators.map((port, index) => {
        const offset = `${((index + 1) * 100) / (d.operators.length + 1)}%`;
        return (
          <Handle
            key={port}
            id={`in-${index}`}
            type="target"
            title={port}
            position={vertical ? Position.Top : Position.Left}
            className="port"
            style={{ ...portStyle(DOMAIN_COLORS.operator, "matrix", Boolean(d.term[port])), ...(vertical ? { left: offset } : { top: offset }) }}
          />
        );
      })}
      {d.operators.length > 1 && vertical && (
        <div className="term-ports">
          {d.operators.map((port) => (
            <span key={port}>{port}</span>
          ))}
        </div>
      )}
      <div className="term-label">{d.label}</div>
      <div className="term-kind">
        {String(d.term.term)}
        {read.length ? `(${read.map(String).join(", ")})` : ""}
      </div>
      {options.length > 0 && <div className="term-options">{options.map(([k, v]) => `${k}=${formatValue(v)}`).join(" · ")}</div>}
      {d.requires.length > 0 && <div className="term-options">needs {d.requires.join(", ")}</div>}
      <Handle id="out-0" type="source" position={orientation === "TB" ? Position.Bottom : Position.Right} className="port" style={portStyle(DOMAIN_COLORS.loss, "invariant", true)} />
    </div>
  );
});

const SYMBOLS: Record<string, string> = { "+": "+", "-": "−", "*": "×", "/": "÷" };

export const ExprNode = memo(function ExprNode({ data, selected }: NodeProps) {
  const d = data as ExprNodeData;
  const orientation = useOrientation();
  const e = d.expr;
  const label = e.type === "num" ? String(e.value) : e.type === "bin" ? SYMBOLS[e.op] : e.type === "neg" ? "−x" : e.type === "call" ? e.fn : "";
  return (
    <div className={`expr ${e.type === "num" ? "const" : "op"} ${selected ? "selected" : ""}`}>
      {Array.from({ length: d.arity }, (_, i) => (
        <Handle key={i} id={`in-${i}`} type="target" position={orientation === "TB" ? Position.Top : Position.Left} className="port" style={{ ...(orientation === "TB" ? { left: spread(d.arity, i) } : { top: spread(d.arity, i) }), ...portStyle(DOMAIN_COLORS.loss, "invariant", true) }} />
      ))}
      <span>{label}</span>
      <Handle id="out-0" type="source" position={orientation === "TB" ? Position.Bottom : Position.Right} className="port" style={portStyle(DOMAIN_COLORS.loss, "invariant", true)} />
    </div>
  );
});

export const LossNode = memo(function LossNode({ data, selected }: NodeProps) {
  const d = data as { expression: string; problems: Problem[] };
  const orientation = useOrientation();
  const error = d.problems.some((p) => p.level === "error");
  return (
    <div className={`loss ${selected ? "selected" : ""} ${error ? "has-error" : ""}`} title={d.expression}>
      <Handle id="in-0" type="target" position={orientation === "TB" ? Position.Top : Position.Left} className="port" style={portStyle(DOMAIN_COLORS.loss, "invariant", true)} />
      <span className="loss-symbol">ℒ</span>
      <span className="loss-text">loss</span>
    </div>
  );
});

export const nodeTypes = {
  stage: StageNode,
  frame: FrameNode,
  slot: InputNode,
  ref: RefNode,
  target: TargetNode,
  term: TermNode,
  expr: ExprNode,
  loss: LossNode,
};
