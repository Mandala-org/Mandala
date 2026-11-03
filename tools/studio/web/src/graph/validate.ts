// Instant structural checks. The backend compile is authoritative; these give
// immediate feedback while wiring and catch problems the compiler reports only
// one at a time.

import type { Analysis, Catalog, FeatureSpecJson, Pipeline, PortInfo, RecipeDocument } from "../types";
import { displayedInputs, portsOf, slotSpec, type VisibleGraph } from "./model";
import { parseExpression, labelsOf } from "../loss/expression";

const KIND_LABEL: Record<string, string> = {
  irreps: "an irrep feature",
  invariant: "an invariant (0e) feature",
  index: "an integer index",
  geometry: "Cartesian geometry",
  object: "graph metadata",
  local: "bond-frame local components",
  matrix: "decoded orbital blocks",
  channel: "channel-first irreps",
  block_vectors: "per species-pair vectors",
};

export function portDomains(port: PortInfo): string[] | null {
  if (port.domain === null) return null;
  return Array.isArray(port.domain) ? port.domain : [port.domain];
}

function kindMatches(spec: FeatureSpecJson, kind: string): boolean {
  const irrep = spec.irreps !== null && (spec.representation === "scalar" || spec.representation === "irreps") && spec.frame === "global";
  switch (kind) {
    case "irreps":
      return irrep;
    case "invariant":
      return irrep && spec.invariant;
    case "index":
      return spec.representation === "index";
    case "geometry":
      return spec.representation === "geometry";
    case "object":
      return spec.representation === "object";
    case "local":
      return spec.representation === "local_components" && spec.frame === "bond";
    case "matrix":
      return spec.representation === "matrix";
    case "channel":
      return spec.representation === "channel_irreps";
    case "block_vectors":
      return spec.representation === "block_vectors";
    default:
      return true;
  }
}

/** Why ``spec`` cannot feed ``port``, or null when it can. */
export function connectionProblem(spec: FeatureSpecJson | undefined, port: PortInfo): string | null {
  if (!spec) return null;
  const domains = portDomains(port);
  if (domains && !domains.includes(spec.domain)) {
    return `${port.name} takes ${domains.join("/")} features; this slot is indexed by ${spec.domain}`;
  }
  if (!kindMatches(spec, port.kind)) {
    const got = spec.irreps ? `${spec.representation} ${spec.irreps}` : spec.representation;
    return `${port.name} needs ${KIND_LABEL[port.kind] ?? port.kind}; this slot is ${got}${spec.frame !== "global" ? ` in the ${spec.frame} frame` : ""}`;
  }
  return null;
}

export interface Problem {
  level: "error" | "warning";
  key: string | null;
  message: string;
}

export function lint(document: RecipeDocument, pipeline: Pipeline | null | undefined, graph: VisibleGraph, analysis: Analysis | null, catalog: Catalog | null): Problem[] {
  const problems: Problem[] = [];
  if (!pipeline) return problems;
  const consumers = new Map<string, string[]>();
  for (const stage of graph.stages) {
    for (const slot of stage.inputs) {
      if (!slot) continue;
      consumers.set(slot, [...(consumers.get(slot) ?? []), stage.key]);
    }
  }
  const targets = new Set(Object.values(pipeline.outputs ?? {}));
  const blocks = pipeline.blocks ?? {};

  // Duplicate ids per owner.
  const owners: [string, { id: string }[]][] = [["recipe", pipeline.stages], ...Object.entries(blocks).map(([n, b]) => [`block ${n}`, b.stages] as [string, { id: string }[]])];
  for (const [owner, stages] of owners) {
    const seen = new Set<string>();
    for (const stage of stages) {
      if (seen.has(stage.id)) problems.push({ level: "error", key: stage.id, message: `Duplicate stage id ${stage.id} in ${owner}` });
      seen.add(stage.id);
    }
  }

  for (const stage of graph.stages) {
    const transform = catalog?.transforms.find((t) => t.name === stage.stage.transform);
    const def = blocks[stage.stage.transform];
    if (!transform && !def) {
      problems.push({ level: "error", key: stage.key, message: `Unknown transformation ${stage.stage.transform} (missing plugin?)` });
      continue;
    }
    if (def && stage.stage.inputs.length !== def.inputs.length) {
      problems.push({ level: "error", key: stage.key, message: `Block ${stage.stage.transform} takes ${def.inputs.length} inputs; the instance wires ${stage.stage.inputs.length}` });
    }
    if (def && stage.stage.outputs.length !== def.outputs.length) {
      problems.push({ level: "error", key: stage.key, message: `Block ${stage.stage.transform} has ${def.outputs.length} outputs; the instance declares ${stage.stage.outputs.length}` });
    }
    const ports = portsOf(stage.stage, pipeline, catalog);
    for (const { port, index } of displayedInputs(stage.stage, ports)) {
      const slot = stage.inputs[index];
      if (!slot) {
        if (!port.optional && !port.variadic) problems.push({ level: "error", key: stage.key, message: `Input ${port.name} is not connected` });
        continue;
      }
      if (!graph.producers.has(slot) && !slotSpec(analysis, catalog, slot)) {
        problems.push({ level: "error", key: stage.key, message: `Input ${port.name} reads ${slot}, which nothing produces` });
        continue;
      }
      const problem = connectionProblem(slotSpec(analysis, catalog, slot), port);
      if (problem) problems.push({ level: "error", key: stage.key, message: problem });
    }
    // Entity sets of edge-scoped inputs should agree.
    const supports = new Set(
      stage.inputs
        .map((slot) => slotSpec(analysis, catalog, slot))
        .filter((spec): spec is FeatureSpecJson => Boolean(spec) && spec!.domain === "edge")
        .map((spec) => spec.support ?? "output"),
    );
    if (supports.size > 1) problems.push({ level: "warning", key: stage.key, message: `Edge inputs live on different supports: ${[...supports].join(", ")}` });
    const bases = new Set(
      stage.inputs
        .map((slot) => slotSpec(analysis, catalog, slot))
        .filter((spec): spec is FeatureSpecJson => Boolean(spec?.irreps) && !spec!.invariant && Boolean(spec!.angular_basis))
        .map((spec) => spec.angular_basis!),
    );
    if (bases.size > 1) problems.push({ level: "warning", key: stage.key, message: `Inputs use different angular bases (${[...bases].join(", ")}); insert irrep_cartesian_basis` });
    if (transform) {
      const given = stage.stage.options ?? {};
      for (const option of transform.options) {
        if (option.required && !(option.name in given)) {
          problems.push({ level: "error", key: stage.key, message: `Option ${option.name} is required` });
        }
      }
    }
    stage.outputs.forEach((slot) => {
      if (!consumers.has(slot) && !targets.has(slot)) {
        problems.push({ level: "warning", key: stage.key, message: `Output ${slot} is never used` });
      }
    });
  }

  // Stages that do not reach any target.
  const live = new Set<string>();
  const stack = [...targets].map((slot) => graph.producers.get(slot)?.key).filter(Boolean) as string[];
  const byKey = new Map(graph.stages.map((s) => [s.key, s]));
  while (stack.length) {
    const key = stack.pop()!;
    if (live.has(key)) continue;
    live.add(key);
    for (const slot of byKey.get(key)?.inputs ?? []) {
      const producer = graph.producers.get(slot);
      if (producer) stack.push(producer.key);
    }
  }
  for (const stage of graph.stages) {
    if (!live.has(stage.key)) problems.push({ level: "warning", key: stage.key, message: "Does not contribute to any target" });
  }
  for (const [operator, slot] of Object.entries(pipeline.outputs ?? {})) {
    if (!slot) problems.push({ level: "error", key: `target:${operator}`, message: `Target ${operator} is not connected` });
    else if (!graph.producers.has(slot) && !slotSpec(analysis, catalog, slot)) problems.push({ level: "error", key: `target:${operator}`, message: `Target ${operator} reads ${slot}, which nothing produces` });
    else {
      const spec = slotSpec(analysis, catalog, slot);
      // Other outputs are auxiliary (e.g. edge_envelope); only operators read
      // by objective terms must be decoded blocks.
      const terms = Object.values((document.objective ?? analysis?.objective.objective ?? null)?.terms ?? {});
      const readByTerm = terms.some((term) =>
        (catalog?.objective_terms.find((t) => t.name === term.term)?.operators ?? []).some((port) => term[port] === operator),
      );
      if (readByTerm && spec && !(spec.domain === "edge" && ["matrix", "block_vectors"].includes(spec.representation))) {
        problems.push({ level: "error", key: `target:${operator}`, message: `Operator ${operator} must be decoded edge blocks (matrix or block vectors); ${slot} is a ${spec.domain} ${spec.representation} feature` });
      }
    }
  }

  // Objective.
  const objective = document.objective ?? analysis?.objective.objective ?? null;
  if (objective?.terms) {
    for (const [label, term] of Object.entries(objective.terms)) {
      const info = catalog?.objective_terms.find((t) => t.name === term.term);
      if (catalog && catalog.objective_terms.length && !info) problems.push({ level: "error", key: `term:${label}`, message: `Unknown objective term ${String(term.term)}` });
      for (const port of info?.operators ?? []) {
        const operator = term[port] as string | undefined;
        if (!operator) problems.push({ level: "warning", key: `term:${label}`, message: `Term ${label}: operator port ${port} is not connected` });
        if (operator && !(operator in (pipeline.outputs ?? {}))) {
          problems.push({ level: "error", key: `term:${label}`, message: `Term ${label} reads operator ${operator} (${port}), which the model does not predict` });
        }
      }
    }
    if (objective.expression) {
      try {
        const labels = labelsOf(parseExpression(objective.expression));
        for (const label of labels) if (!(label in objective.terms)) problems.push({ level: "error", key: "loss", message: `Loss expression uses undefined term ${label}` });
        for (const label of Object.keys(objective.terms)) if (!labels.has(label)) problems.push({ level: "warning", key: `term:${label}`, message: `Term ${label} is not used by the loss expression` });
      } catch (error) {
        problems.push({ level: "error", key: "loss", message: `Loss expression: ${(error as Error).message}` });
      }
    } else {
      problems.push({ level: "error", key: "loss", message: "The loss expression is empty" });
    }
  }
  return problems;
}

/** True if wiring ``source`` (a stage key) into ``target`` would create a cycle. */
export function createsCycle(graph: VisibleGraph, source: string, target: string): boolean {
  if (source === target) return true;
  const consumers = new Map<string, string[]>();
  for (const stage of graph.stages) {
    for (const slot of stage.inputs) {
      const producer = graph.producers.get(slot);
      if (producer) consumers.set(producer.key, [...(consumers.get(producer.key) ?? []), stage.key]);
    }
  }
  const stack = [target];
  const seen = new Set<string>();
  while (stack.length) {
    const key = stack.pop()!;
    if (key === source) return true;
    if (seen.has(key)) continue;
    seen.add(key);
    stack.push(...(consumers.get(key) ?? []));
  }
  return false;
}
