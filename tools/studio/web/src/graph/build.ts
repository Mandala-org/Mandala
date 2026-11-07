// Recipe document + analysis -> React Flow nodes and edges (with sizes for layout).

import type { Edge, Node } from "@xyflow/react";
import type { Analysis, Catalog, FeatureSpecJson, ObjectiveSpec, Pipeline, PortInfo } from "../types";
import { countInstances, displayedInputs, isRoutine, metricsOf, outputNamesOf, portsOf, recordsOf, slotSpec, visibleGraph, type StageMetrics, type VisibleGraph, type VisibleStage } from "./model";
import { childrenOf, parseExpression, type Expr } from "../loss/expression";
import type { Problem } from "./validate";

export const DOMAIN_COLORS: Record<string, string> = {
  node: "#5cc8ff",
  edge: "#ffb547",
  graph: "#c792ea",
  relation: "#8a9bb0",
  bond_neighbor: "#7fdb9a",
  operator: "#ff6b9d",
  loss: "#ff6b9d",
};

export function specColor(spec?: FeatureSpecJson): string {
  if (!spec) return "#6b7686";
  if (spec.representation === "matrix" || spec.representation === "block_vectors") return DOMAIN_COLORS.operator;
  return DOMAIN_COLORS[spec.domain] ?? "#6b7686";
}

export interface StageNodeData extends Record<string, unknown> {
  visible: VisibleStage;
  ports: { port: PortInfo; index: number; slot: string; spec?: FeatureSpecJson }[];
  outputs: { name: string; slot: string; spec?: FeatureSpecJson; manual: boolean }[];
  metrics: StageMetrics;
  share: number;
  scale: number;
  keyOptions: { name: string; value: unknown; state: "explicit" | "inferred" | "default" }[];
  category: string;
  summary: string;
  problems: Problem[];
  diff?: "added" | "changed";
  isBlock: boolean;
  blockInstances: number;
  size: { width: number; height: number };
  /** Handle numbers in drawing order, as the layout placed them (default: as listed). */
  inputOrder?: number[];
  outputOrder?: number[];
}

export interface FrameNodeData extends Record<string, unknown> {
  key: string;
  block: string;
  instances: number;
  metrics: StageMetrics;
  summary: string;
}

export interface InputNodeData extends Record<string, unknown> {
  slot: string;
  spec?: FeatureSpecJson;
  declared: boolean;
}

/** A pin standing for a widely shared slot next to one of its uses. */
export interface RefNodeData extends Record<string, unknown> {
  slot: string;
  spec?: FeatureSpecJson;
  producer: string | null; // node id of the producing stage, null for inputs
  uses: number;
}

export interface TargetNodeData extends Record<string, unknown> {
  operator: string;
  slot: string;
  spec?: FeatureSpecJson;
  problems: Problem[];
}

export interface TermNodeData extends Record<string, unknown> {
  label: string;
  term: Record<string, unknown>;
  operators: string[];
  summary: string;
  problems: Problem[];
  lowered: boolean;
  requires: string[];
}

export interface ExprNodeData extends Record<string, unknown> {
  path: number[];
  expr: Expr;
  arity: number;
}

const BASE_WIDTH = 236;

function stageSize(data: StageNodeData, orientation: "TB" | "LR") {
  const ports = Math.max(data.ports.length, data.outputs.length);
  const width = orientation === "TB" ? Math.max(BASE_WIDTH, ports * 62 + 24) : BASE_WIDTH;
  const portRows = orientation === "LR" ? Math.max(ports, 1) * 22 : 0;
  const height = 40 + 34 + data.keyOptions.length * 19 + portRows + (data.isBlock ? 18 : 0);
  return { width: width * data.scale, height: height * data.scale };
}

export interface BuiltGraph {
  nodes: Node[];
  edges: Edge[];
  graph: VisibleGraph;
  sizes: Map<string, { width: number; height: number }>;
}

function optionState(name: string, explicit: string[], inferred: string[]): "explicit" | "inferred" | "default" {
  if (explicit.includes(name)) return "explicit";
  if (inferred.includes(name)) return "inferred";
  return "default";
}

export function buildGraph(args: {
  pipeline: Pipeline | null;
  analysis: Analysis | null;
  catalog: Catalog | null;
  expanded: Set<string>;
  orientation: "TB" | "LR";
  problems: Problem[];
  objective: ObjectiveSpec | null;
  objectiveLowered: boolean;
  visibleInputs: Set<string>;
  compactWires?: boolean;
  hideRoutine?: boolean;
  diff: Map<string, "added" | "changed">;
}): BuiltGraph {
  const { pipeline, analysis, catalog, expanded, orientation, problems, objective } = args;
  const graph = visibleGraph(pipeline, expanded);
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const sizes = new Map<string, { width: number; height: number }>();
  const total = analysis?.totals?.seconds ?? 0;
  const incomplete = (analysis?.stages ?? []).some((s) => s.error);
  const problemsOf = (key: string) => problems.filter((p) => p.key === key);
  const nodeIdOf = new Map<string, string>(); // global slot -> producing node id
  const handleOf = new Map<string, string>();

  for (const frame of graph.frames) {
    const records = recordsOf(analysis, frame.key, true);
    const data: FrameNodeData = {
      key: frame.key,
      block: frame.block,
      instances: frame.instances,
      metrics: metricsOf(records),
      summary: pipeline?.blocks?.[frame.block]?.summary ?? "",
    };
    nodes.push({
      id: `frame:${frame.key}`,
      type: "frame",
      position: { x: 0, y: 0 },
      data,
      parentId: frame.parent ? `frame:${frame.parent}` : undefined,
      selectable: true,
      style: { zIndex: -1 },
    });
  }

  for (const visible of graph.stages) {
    const ports = portsOf(visible.stage, pipeline, catalog);
    const records = recordsOf(analysis, visible.key, visible.isBlock);
    const metrics = metricsOf(records);
    const share = total > 0 ? metrics.seconds / total : 0;
    // Shares of an incomplete total (failed stages count as zero) mislead.
    // Structure-only stages are computed once per structure and cached, so
    // their share of a forward pass does not make them large.
    const scale = share > 0.1 && !incomplete && !metrics.static ? Math.min(2.4, Math.sqrt(share / 0.1)) : 1;
    const record = !visible.isBlock ? records[0] : undefined;
    const explicit = record?.explicit ?? Object.keys(visible.stage.options ?? {});
    const inferred = record?.inferred ?? [];
    const transform = catalog?.transforms.find((t) => t.name === visible.stage.transform);
    const keyOptions: StageNodeData["keyOptions"] = [];
    if (transform) {
      for (const option of transform.options) {
        if (!option.key && !explicit.includes(option.name)) continue;
        if (keyOptions.length >= 4) break;
        const value = record ? record.options[option.name] : (visible.stage.options ?? {})[option.name] ?? option.default;
        keyOptions.push({ name: option.name, value, state: optionState(option.name, explicit, inferred) });
      }
    } else {
      // Block instances show their options as resolved for this instance.
      const resolved = analysis?.groups?.[visible.key]?.options ?? {};
      for (const [name, value] of Object.entries(visible.stage.options ?? {}).slice(0, 4)) {
        keyOptions.push({ name, value: resolved[name] ?? value, state: "explicit" });
      }
    }
    const outNames = outputNamesOf(visible.stage, pipeline, catalog);
    const irrepsExplicit = explicit.includes("irreps_out");
    const data: StageNodeData = {
      visible,
      ports: displayedInputs(visible.stage, ports).map(({ port, index }) => {
        const slot = visible.inputs[index] ?? "";
        return { port, index, slot, spec: slot ? slotSpec(analysis, catalog, slot) : undefined };
      }),
      outputs: visible.outputs.map((slot, i) => ({ name: outNames[i], slot, spec: slotSpec(analysis, catalog, slot), manual: irrepsExplicit })),
      metrics,
      share,
      scale,
      keyOptions,
      category: transform?.category ?? (visible.isBlock ? "block" : "custom"),
      summary: transform?.summary ?? pipeline?.blocks?.[visible.stage.transform]?.summary ?? "",
      problems: problemsOf(visible.key),
      diff: args.diff.get(visible.key),
      isBlock: visible.isBlock,
      blockInstances: visible.isBlock && pipeline ? countInstances(pipeline, visible.stage.transform) : 0,
      size: { width: 0, height: 0 },
    };
    const id = `stage:${visible.key}`;
    const size = stageSize(data, orientation);
    data.size = size;
    sizes.set(id, size);
    nodes.push({
      id,
      type: "stage",
      position: { x: 0, y: 0 },
      data,
      parentId: visible.instance ? `frame:${visible.instance}` : undefined,
      style: { width: size.width, height: size.height },
      ariaLabel: `${visible.isBlock ? "Block" : "Stage"} ${visible.localId}: ${visible.stage.transform}${visible.stage.enabled === false ? ", bypassed" : ""}${data.problems.length ? `, ${data.problems.length} problems` : ""}`,
    });
    visible.outputs.forEach((slot, i) => {
      nodeIdOf.set(slot, id);
      handleOf.set(slot, `out-${i}`);
    });
  }

  // Structure inputs and structure-only (static) features read in many places
  // are drawn as a pin beside each use instead of wires across the graph.
  const uses = new Map<string, number>();
  for (const stage of graph.stages) for (const slot of stage.inputs) if (slot) uses.set(slot, (uses.get(slot) ?? 0) + 1);
  const pinned = new Set<string>();
  if (args.compactWires) {
    for (const [slot, count] of uses) {
      if (count < 3) continue;
      const producer = graph.producers.get(slot);
      const shared = producer ? metricsOf(recordsOf(analysis, producer.key, false)).static : true;
      if (shared) pinned.add(slot);
    }
  }

  // Routine structure inputs (edge relations, batch membership) can be left
  // undrawn: the ports stay wired, only the inputs and their wires are hidden.
  const hidden = (slot: string) => Boolean(args.hideRoutine) && !graph.producers.has(slot) && isRoutine(slot, slotSpec(analysis, catalog, slot));

  // Input slots read by visible stages, plus inputs the user placed.
  const used = new Set<string>(args.visibleInputs);
  for (const stage of graph.stages) for (const slot of stage.inputs) if (slot && !graph.producers.has(slot) && !pinned.has(slot) && !hidden(slot)) used.add(slot);
  for (const slot of Object.values(pipeline?.outputs ?? {})) if (slot && !graph.producers.has(slot)) used.add(slot);
  for (const slot of used) {
    const spec = slotSpec(analysis, catalog, slot);
    const id = `input:${slot}`;
    const data: InputNodeData = { slot, spec, declared: Boolean(pipeline?.inputs?.[slot]) };
    sizes.set(id, { width: 170, height: 46 });
    nodes.push({ id, type: "slot", position: { x: 0, y: 0 }, data, style: { width: 170, height: 46 }, ariaLabel: `Input ${slot}` });
    nodeIdOf.set(slot, id);
    handleOf.set(slot, "out-0");
  }

  const edgeFor = (slot: string, target: string, targetHandle: string) => {
    const source = nodeIdOf.get(slot);
    if (!source) return;
    const spec = slotSpec(analysis, catalog, slot);
    const color = specColor(spec);
    edges.push({
      id: `${source}|${handleOf.get(slot)}->${target}|${targetHandle}`,
      source,
      sourceHandle: handleOf.get(slot),
      target,
      targetHandle,
      data: { slot, spec },
      className: spec && (spec.representation === "index" || spec.representation === "object") ? "wire-index" : spec && !spec.invariant && spec.irreps ? "wire-irreps" : "wire-scalar",
      style: { stroke: color, strokeWidth: spec && !spec.invariant && spec.irreps ? 2.4 : 1.5 },
    });
  };

  for (const visible of graph.stages) {
    const target = `stage:${visible.key}`;
    visible.inputs.forEach((slot, index) => {
      if (!slot || hidden(slot)) return;
      if (!pinned.has(slot)) {
        edgeFor(slot, target, `in-${index}`);
        return;
      }
      const spec = slotSpec(analysis, catalog, slot);
      const producer = graph.producers.get(slot);
      const id = `ref:${slot}>${visible.key}#${index}`;
      const data: RefNodeData = { slot, spec, producer: producer ? `stage:${producer.key}` : null, uses: uses.get(slot) ?? 0 };
      const size = { width: Math.min(200, Math.max(84, 40 + 7.4 * slot.length)), height: 26 };
      sizes.set(id, size);
      nodes.push({ id, type: "ref", position: { x: 0, y: 0 }, data, style: size, parentId: visible.instance ? `frame:${visible.instance}` : undefined });
      edges.push({
        id: `${id}->${target}|in-${index}`,
        source: id,
        sourceHandle: "out-0",
        target,
        targetHandle: `in-${index}`,
        data: { slot, spec },
        className: "wire-pin",
        style: { stroke: specColor(spec), strokeWidth: 1.4 },
      });
    });
  }

  // Targets.
  for (const [operator, slot] of Object.entries(pipeline?.outputs ?? {})) {
    const id = `target:${operator}`;
    const data: TargetNodeData = { operator, slot, spec: slot ? slotSpec(analysis, catalog, slot) : undefined, problems: problemsOf(id) };
    sizes.set(id, { width: 190, height: 52 });
    nodes.push({ id, type: "target", position: { x: 0, y: 0 }, data, style: { width: 190, height: 52 }, ariaLabel: `Predicted operator ${operator}${slot ? `, from ${slot}` : ", not connected"}` });
    if (slot) edgeFor(slot, id, "in-0");
  }

  // Objective terms and the loss expression.
  if (objective?.terms) {
    for (const [label, term] of Object.entries(objective.terms)) {
      const info = catalog?.objective_terms.find((t) => t.name === term.term);
      const id = `term:${label}`;
      const data: TermNodeData = {
        label,
        term,
        operators: info?.operators ?? (term.operator ? ["operator"] : []),
        summary: info?.summary ?? "",
        problems: problemsOf(id),
        lowered: args.objectiveLowered,
        requires: info?.requires ?? [],
      };
      const size = { width: 200, height: 64 };
      sizes.set(id, size);
      nodes.push({ id, type: "term", position: { x: 0, y: 0 }, data, style: size, ariaLabel: `Loss term ${label}: ${String(term.term)}` });
      data.operators.forEach((port, index) => {
        const operator = term[port] as string | undefined;
        if (!operator || pipeline?.outputs?.[operator] === undefined) return;
        edges.push({
          id: `target:${operator}->${id}:${port}`,
          source: `target:${operator}`,
          sourceHandle: "out-0",
          target: id,
          targetHandle: `in-${index}`,
          style: { stroke: DOMAIN_COLORS.operator, strokeWidth: 1.6, strokeDasharray: "5 4" },
        });
      });
    }
    let tree: Expr | null = null;
    try {
      tree = objective.expression ? parseExpression(objective.expression) : null;
    } catch {
      tree = null;
    }
    const lossSize = { width: 150, height: 56 };
    sizes.set("loss", lossSize);
    nodes.push({ id: "loss", type: "loss", position: { x: 0, y: 0 }, data: { expression: objective.expression ?? "", problems: problemsOf("loss") }, style: lossSize, ariaLabel: `Loss: ${objective.expression ?? "no expression"}` });
    if (tree) {
      const place = (expr: Expr, path: number[], parent: string, handle: string) => {
        if (expr.type === "name") {
          const id = `term:${expr.name}`;
          if (nodes.some((n) => n.id === id)) {
            edges.push({ id: `${id}->${parent}|${handle}|${path.join(".")}`, source: id, sourceHandle: "out-0", target: parent, targetHandle: handle, style: { stroke: DOMAIN_COLORS.loss, strokeWidth: 1.8 } });
          }
          return;
        }
        const id = `expr:${path.join(".") || "root"}`;
        const children = childrenOf(expr);
        const data: ExprNodeData = { path, expr, arity: children.length };
        const size = expr.type === "num" ? { width: 86, height: 38 } : { width: 64, height: 64 };
        sizes.set(id, size);
        nodes.push({ id, type: "expr", position: { x: 0, y: 0 }, data, style: size });
        edges.push({ id: `${id}->${parent}|${handle}`, source: id, sourceHandle: "out-0", target: parent, targetHandle: handle, style: { stroke: DOMAIN_COLORS.loss, strokeWidth: 1.8 } });
        children.forEach((child, i) => place(child, [...path, i], id, `in-${i}`));
      };
      place(tree, [], "loss", "in-0");
    }
  }

  return { nodes, edges, graph, sizes };
}
