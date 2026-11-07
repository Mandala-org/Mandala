// The visible stage graph of a recipe: top-level stages, with expanded block
// instances replaced by their inner stages (recursively). Slot names are the
// framework's global names after block expansion (``instance/local``).

import type { BlockPorts, Analysis, BlockDef, Catalog, FeatureSpecJson, Pipeline, PortInfo, Stage, StageAnalysis } from "../types";

export interface Owner {
  kind: "top" | "block";
  block?: string;
}

export interface VisibleStage {
  key: string; // global stage path, the framework's stage id for non-block stages
  localId: string;
  stage: Stage;
  owner: Owner;
  instance: string | null; // expanded instance path containing this stage
  isBlock: boolean;
  inputs: string[]; // global slot names
  outputs: string[];
}

export interface Frame {
  key: string; // instance path
  block: string;
  parent: string | null;
  instances: number;
}

export interface VisibleGraph {
  stages: VisibleStage[];
  frames: Frame[];
  producers: Map<string, { key: string; index: number }>;
}

export function countInstances(pipeline: Pipeline, block: string): number {
  let count = 0;
  const visit = (stages: Stage[]) => {
    for (const stage of stages) {
      if (stage.transform === block) count += 1;
    }
  };
  visit(pipeline.stages);
  for (const def of Object.values(pipeline.blocks ?? {})) visit(def.stages);
  return count;
}

export function visibleGraph(pipeline: Pipeline | null | undefined, expanded: Set<string>): VisibleGraph {
  const result: VisibleGraph = { stages: [], frames: [], producers: new Map() };
  if (!pipeline) return result;
  const blocks = pipeline.blocks ?? {};

  const walk = (
    stages: Stage[],
    scope: (local: string) => string,
    prefix: string,
    instance: string | null,
    owner: Owner,
    depth: number,
  ) => {
    for (const stage of stages) {
      const key = prefix ? `${prefix}/${stage.id}` : stage.id;
      const inputs = (stage.inputs ?? []).map((s) => (s === "" ? "" : scope(s)));
      const outputs = (stage.outputs ?? []).map((s) => scope(s));
      const def: BlockDef | undefined = blocks[stage.transform];
      // A bypassed instance has no inner stages to show.
      if (def && expanded.has(key) && depth < 12 && stage.enabled !== false) {
        result.frames.push({ key, block: stage.transform, parent: instance, instances: countInstances(pipeline, stage.transform) });
        const inner = (local: string) => {
          const i = def.inputs.indexOf(local);
          if (i >= 0) return inputs[i] ?? "";
          const o = def.outputs.indexOf(local);
          if (o >= 0) return outputs[o] ?? "";
          return `${key}/${local}`;
        };
        walk(def.stages, inner, key, key, { kind: "block", block: stage.transform }, depth + 1);
        continue;
      }
      const visible: VisibleStage = {
        key,
        localId: stage.id,
        stage,
        owner,
        instance,
        isBlock: Boolean(def),
        inputs,
        outputs,
      };
      result.stages.push(visible);
      outputs.forEach((slot, index) => result.producers.set(slot, { key, index }));
    }
  };
  walk(pipeline.stages, (s) => s, "", null, { kind: "top" }, 0);
  return result;
}

// Analysis records of a visible stage (one record, or all records of a block instance).
export function recordsOf(analysis: Analysis | null, key: string, isBlock: boolean): StageAnalysis[] {
  if (!analysis) return [];
  if (!isBlock) return analysis.stages.filter((s) => s.id === key);
  return analysis.stages.filter((s) => s.id === key || s.id.startsWith(`${key}/`));
}

// Unconnected inputs are reported by the editor's own checks, per port.
export const UNCONNECTED = /input\(s\) .* not connected|reads undefined slots: \[''\]|produced by a failing stage|outputs reference undefined slots \[''\]/;

/** A compiler message without the stage name it starts with (shown beside it). */
export function compilerMessage(stage: string, message: string): string {
  // "Stage radial (soft_one_hot_basis): ...", "Stage radial: ..." or "radial: ...".
  const escaped = stage.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return message.replace(new RegExp(`^(Stage )?${escaped}( \\([^)]*\\))?: `), "");
}

export interface StageMetrics {
  parameters: number;
  seconds: number;
  errors: string[];
  static: boolean;
  analyzed: boolean;
}

export function metricsOf(records: StageAnalysis[]): StageMetrics {
  return {
    parameters: records.reduce((t, r) => t + (r.parameters ?? 0), 0),
    seconds: records.reduce((t, r) => t + (r.seconds ?? 0), 0),
    errors: records
      .filter((r) => r.error && !UNCONNECTED.test(r.error))
      .map((r) => (r.id.includes("/") ? `${r.id}: ${compilerMessage(r.id, r.error!)}` : compilerMessage(r.id, r.error!))),
    static: records.length > 0 && records.every((r) => r.static),
    analyzed: records.length > 0,
  };
}

/**
 * The last known spec of every slot of the open recipe. A disconnected input
 * makes everything downstream fail to compile; what the slots carried before
 * still says what an input should read.
 */
export const slotMemory: { specs: Record<string, FeatureSpecJson> } = { specs: {} };

/** Block port contracts reported by the last analysis (net.blocks.block_ports). */
export const blockPortMemory: { ports: Record<string, BlockPorts> } = { ports: {} };

// Port descriptions of a stage: catalog ports for transforms; for block
// instances the compiler's contract, or (while an edit awaits analysis) the
// ports of the inner consumers.
export function portsOf(stage: Stage, pipeline: Pipeline | null | undefined, catalog: Catalog | null, depth = 0): PortInfo[] {
  const transform = catalog?.transforms.find((t) => t.name === stage.transform);
  if (transform) return transform.inputs;
  const def = pipeline?.blocks?.[stage.transform];
  if (!def || depth > 8) return [];
  const known = blockPortMemory.ports[stage.transform] ?? catalog?.block_library?.[stage.transform]?.ports;
  if (known && known.inputs.map((p) => p.name).join("|") === def.inputs.join("|")) return known.inputs;
  return def.inputs.map((name) => {
    for (const inner of def.stages) {
      const index = inner.inputs.indexOf(name);
      if (index >= 0) {
        const ports = portsOf(inner, pipeline, catalog, depth + 1);
        const port = ports[Math.min(index, ports.length - 1)];
        if (port) return { ...port, name, optional: false, variadic: false };
      }
    }
    return { name, domain: null, kind: "any", optional: false, variadic: false, doc: "" };
  });
}

export function outputNamesOf(stage: Stage, pipeline: Pipeline | null | undefined, catalog: Catalog | null): string[] {
  const transform = catalog?.transforms.find((t) => t.name === stage.transform);
  if (transform) {
    return stage.outputs.map((_, i) => transform.outputs[i] ?? `output_${i}`);
  }
  const def = pipeline?.blocks?.[stage.transform];
  return stage.outputs.map((_, i) => def?.outputs[i] ?? `output_${i}`);
}

// The ports drawn for a stage: every connected input plus, for a variadic last
// port, one free slot to connect another value.
export function displayedInputs(stage: Stage, ports: PortInfo[]): { port: PortInfo; index: number }[] {
  const shown: { port: PortInfo; index: number }[] = [];
  const count = Math.max(stage.inputs.length, ports.filter((p) => !p.optional && !p.variadic).length);
  for (let i = 0; i < count; i++) {
    shown.push({ port: ports[Math.min(i, ports.length - 1)] ?? anyPort(i), index: i });
  }
  const last = ports[ports.length - 1];
  if (last?.variadic) {
    shown.push({ port: last, index: shown.length });
  } else {
    for (let i = count; i < ports.length; i++) shown.push({ port: ports[i], index: i });
  }
  return shown;
}

function anyPort(i: number): PortInfo {
  return { name: `input_${i}`, domain: null, kind: "any", optional: false, variadic: false, doc: "" };
}

export function slotSpec(analysis: Analysis | null, catalog: Catalog | null, slot: string): FeatureSpecJson | undefined {
  return analysis?.slots[slot] ?? analysis?.inputs[slot] ?? catalog?.inputs[slot];
}

export function isInputSlot(analysis: Analysis | null, catalog: Catalog | null, pipeline: Pipeline | null | undefined, slot: string): boolean {
  return Boolean(pipeline?.inputs?.[slot] || catalog?.inputs[slot] || analysis?.inputs[slot]);
}

export function formatCount(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}G`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return `${Math.round(value)}`;
}

export function formatSeconds(value: number): string {
  if (value <= 0) return "0 s";
  if (value >= 1) return `${value.toFixed(2)} s`;
  if (value >= 1e-3) return `${(value * 1e3).toFixed(1)} ms`;
  return `${(value * 1e6).toFixed(0)} µs`;
}

/** Stages that failed to compile: totals count them as zero, so they are lower bounds. */
export function failedStages(analysis: Analysis | null): number {
  return analysis?.stages?.filter((s) => s.error).length ?? 0;
}

export function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** Values a new top-level stage can read: structure inputs and top-level slots. */
export function availableSlots(analysis: Analysis | null, catalog: Catalog | null): Record<string, FeatureSpecJson> {
  const result: Record<string, FeatureSpecJson> = { ...(catalog?.inputs ?? {}), ...(analysis?.inputs ?? {}) };
  for (const [slot, spec] of Object.entries(analysis?.slots ?? {})) if (!slot.includes("/")) result[slot] = spec;
  for (const slot of Object.keys(result)) if (isShift(slot)) delete result[slot];
  return result;
}

/** Whether a key press goes to a text field (shortcuts leave it alone). */
export function typingInto(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  if (target.closest("textarea, [contenteditable='true']")) return true;
  return target instanceof HTMLInputElement && !["checkbox", "radio", "button", "range", "color", "file"].includes(target.type);
}

/** Whether a compiler message repeats a problem the editor already reports. */
export function alreadyReported(messages: string[], message: string): boolean {
  const option = /requires option '([^']+)'/.exec(message)?.[1];
  return messages.some((m) => m === message || message.endsWith(m) || (option !== undefined && m === `Option ${option} is required`));
}

/** Edge relations and their lattice shifts: one variable in recipes and the editor. */
export const RELATION_SHIFTS: Record<string, string> = { edge_index: "edge_shift", message_edge_index: "message_shift" };
const SHIFTS = new Set(Object.values(RELATION_SHIFTS));

/** A shift slot, which travels with its relation and is never wired by hand. */
export function isShift(slot: string): boolean {
  return SHIFTS.has(slot);
}

/** Why a port that reads a relation with its shifts cannot read ``slot``. */
export function shiftProblem(slot: string, port: PortInfo): string | null {
  if (!port.with_shift || slot in RELATION_SHIFTS) return null;
  return `${port.name} reads an edge relation with its lattice shifts: ${Object.keys(RELATION_SHIFTS).join(" or ")}`;
}

/** Inputs a model rarely changes: relations, batch membership, partitions. */
export function isRoutine(slot: string, spec: FeatureSpecJson | undefined): boolean {
  return spec?.domain === "relation" || ["node_batch", "edge_batch", "selfloop_edge"].includes(slot);
}
