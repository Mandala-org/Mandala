// Pure edits of a recipe document. Every function returns a new document or
// throws an Error whose message is shown to the user.

import type { BlockDef, Catalog, FeatureSpecJson, Pipeline, RecipeDocument, Stage } from "../types";
import { connectionProblem } from "./validate";
import { countInstances, displayedInputs, portsOf, type VisibleGraph, type VisibleStage } from "./model";

const clone = <T,>(value: T): T => structuredClone(value);

export function stageList(pipeline: Pipeline, owner: VisibleStage["owner"]): Stage[] {
  if (owner.kind === "top") return pipeline.stages;
  const block = pipeline.blocks?.[owner.block!];
  if (!block) throw new Error(`Block ${owner.block} does not exist`);
  return block.stages;
}

function findStage(pipeline: Pipeline, visible: VisibleStage): Stage {
  const stage = stageList(pipeline, visible.owner).find((s) => s.id === visible.localId);
  if (!stage) throw new Error(`Stage ${visible.key} no longer exists`);
  return stage;
}

/** Stable topological order: every stage after the producers of its inputs. */
export function topologicalOrder(stages: Stage[]): Stage[] {
  const producer = new Map<string, number>();
  stages.forEach((stage, index) => stage.outputs.forEach((slot) => producer.set(slot, index)));
  const order: Stage[] = [];
  const state = new Map<number, number>();
  const visit = (index: number) => {
    if (state.get(index) === 2) return;
    if (state.get(index) === 1) throw new Error(`Stage ${stages[index].id} is part of a cycle`);
    state.set(index, 1);
    for (const slot of stages[index].inputs) {
      const p = producer.get(slot);
      if (p !== undefined && p !== index) visit(p);
    }
    state.set(index, 2);
    order.push(stages[index]);
  };
  stages.forEach((_, index) => visit(index));
  return order;
}

function reorder(pipeline: Pipeline, owner: VisibleStage["owner"]) {
  const ordered = topologicalOrder(stageList(pipeline, owner));
  if (owner.kind === "top") pipeline.stages = ordered;
  else pipeline.blocks![owner.block!].stages = ordered;
}

/** Express a global slot in the namespace of ``target``'s owner. */
function localSlot(pipeline: Pipeline, graph: VisibleGraph, slot: string, target: VisibleStage): string {
  if (target.instance === null) {
    if (slot.includes("/")) {
      // A value computed inside an expanded block instance: export it.
      const producer = graph.stages.find((s) => s.outputs.includes(slot));
      if (!producer || producer.instance === null) throw new Error(`Cannot read ${slot} here`);
      return exportFromInstance(pipeline, graph, producer, slot);
    }
    return slot;
  }
  const instance = target.instance;
  const frame = graph.stages.find((s) => s.instance === instance);
  const def = pipeline.blocks![target.owner.block!];
  if (slot.startsWith(`${instance}/`) && !slot.slice(instance.length + 1).includes("/")) {
    return slot.slice(instance.length + 1);
  }
  const instanceStage = findInstance(pipeline, instance);
  const inputIndex = instanceStage.inputs.indexOf(slot);
  if (inputIndex >= 0) return def.inputs[inputIndex];
  const outputIndex = instanceStage.outputs.indexOf(slot);
  if (outputIndex >= 0) return def.outputs[outputIndex];
  if (countInstances(pipeline, target.owner.block!) > 1 || !frame) {
    throw new Error(`Block ${target.owner.block} is used by several instances; add the value to its inputs explicitly`);
  }
  // Extend the block interface with a new input.
  let name = slot.replace(/\W+/g, "_");
  while (def.inputs.includes(name) || def.outputs.includes(name)) name += "_in";
  def.inputs.push(name);
  instanceStage.inputs.push(slot);
  return name;
}

function findInstance(pipeline: Pipeline, path: string): Stage {
  const parts = path.split("/");
  let stages = pipeline.stages;
  let stage: Stage | undefined;
  for (const part of parts) {
    stage = stages.find((s) => s.id === part);
    if (!stage) throw new Error(`Block instance ${path} not found`);
    stages = pipeline.blocks?.[stage.transform]?.stages ?? [];
  }
  return stage!;
}

function exportFromInstance(pipeline: Pipeline, graph: VisibleGraph, producer: VisibleStage, slot: string): string {
  const instance = producer.instance!;
  const block = producer.owner.block!;
  if (countInstances(pipeline, block) > 1) throw new Error(`Block ${block} is used by several instances; add an output to it explicitly`);
  if (instance.includes("/")) throw new Error("Values inside nested blocks are exported one level at a time");
  const def = pipeline.blocks![block];
  const local = slot.slice(instance.length + 1);
  const stage = findInstance(pipeline, instance);
  def.outputs.push(local);
  const name = `${instance}_${local}`.replace(/\W+/g, "_");
  stage.outputs.push(name);
  void graph;
  return name;
}

export function connect(document: RecipeDocument, graph: VisibleGraph, slot: string, target: VisibleStage, index: number, catalog: Catalog | null): RecipeDocument {
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  const local = localSlot(pipeline, graph, slot, target);
  const stage = findStage(pipeline, target);
  const ports = portsOf(stage, pipeline, catalog);
  while (stage.inputs.length < index) stage.inputs.push("");
  stage.inputs[index] = local;
  void ports;
  reorder(pipeline, target.owner);
  return doc;
}

export function disconnect(document: RecipeDocument, target: VisibleStage, index: number, catalog: Catalog | null): RecipeDocument {
  const doc = clone(document);
  const stage = findStage(doc.pipeline!, target);
  const ports = portsOf(stage, doc.pipeline, catalog);
  const port = displayedInputs(stage, ports).find((p) => p.index === index)?.port;
  if (port?.variadic || (port?.optional && index === stage.inputs.length - 1)) stage.inputs.splice(index, 1);
  else stage.inputs[index] = "";
  while (stage.inputs.length && stage.inputs[stage.inputs.length - 1] === "" && (ports[stage.inputs.length - 1]?.optional ?? false)) stage.inputs.pop();
  return doc;
}

export function setTarget(document: RecipeDocument, operator: string, slot: string): RecipeDocument {
  const doc = clone(document);
  if (slot.includes("/")) throw new Error("Targets read top-level slots; collapse or export the value from its block");
  doc.pipeline!.outputs = { ...doc.pipeline!.outputs, [operator]: slot };
  return doc;
}

export function uniqueId(pipeline: Pipeline, base: string): string {
  const taken = new Set([...pipeline.stages.map((s) => s.id), ...Object.keys(pipeline.blocks ?? {})]);
  const allSlots = new Set(pipeline.stages.flatMap((s) => s.outputs));
  let candidate = base;
  let n = 1;
  while (taken.has(candidate) || allSlots.has(candidate)) candidate = `${base}_${n++}`;
  return candidate;
}

/**
 * Adds a stage at the top level. Inputs named like an available value of a
 * fitting type (``available``: slot -> spec) are connected to it.
 */
export function addStage(document: RecipeDocument, transform: string, catalog: Catalog | null, available: Record<string, FeatureSpecJson> = {}): { document: RecipeDocument; id: string; connected: string[] } {
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  let def = pipeline.blocks?.[transform];
  const info = catalog?.transforms.find((t) => t.name === transform);
  const library = catalog?.block_library?.[transform];
  if (!def && !info && library) {
    // A library block is copied into the recipe, where it can be edited.
    const { ports: _ports, ...definition } = clone(library);
    pipeline.blocks = { ...(pipeline.blocks ?? {}), [transform]: definition };
    def = definition;
  }
  if (!def && !info) throw new Error(`Unknown transformation ${transform}`);
  const id = uniqueId(pipeline, transform);
  const outputs = def ? def.outputs : info!.outputs;
  const required = def ? def.inputs.length : info!.inputs.filter((p) => !p.optional && !p.variadic).length + (info!.inputs.some((p) => p.variadic) ? 1 : 0);
  const stage: Stage = {
    id,
    transform,
    inputs: Array(required).fill(""),
    outputs: outputs.length === 1 ? [id] : outputs.map((name) => `${id}_${name}`),
  };
  // Required options stay unset: the stage reports them until they are given.
  const connected: string[] = [];
  const ports = portsOf(stage, pipeline, catalog);
  stage.inputs = stage.inputs.map((slot, i) => {
    const port = ports[Math.min(i, ports.length - 1)];
    if (!port) return slot;
    // The port's own name, or its name on the port's entity (lengths -> edge_lengths).
    const names = [port.name, ...(typeof port.domain === "string" ? [`${port.domain}_${port.name}`] : [])];
    const found = names.find((name) => available[name] && !connectionProblem(available[name], port));
    if (!found) return slot;
    connected.push(found);
    return found;
  });
  pipeline.stages.push(stage);
  return { document: doc, id, connected };
}

/**
 * Copies stages of one level (the recipe or one block definition). Wires
 * between copied stages go to the copies; other inputs are shared with the
 * originals. Returns the visible keys of the copies.
 */
export function duplicateStages(document: RecipeDocument, targets: VisibleStage[]): { document: RecipeDocument; keys: string[] } {
  if (!targets.length) throw new Error("Select the stages to duplicate");
  const owner = targets[0].owner;
  if (targets.some((t) => t.owner.kind !== owner.kind || t.owner.block !== owner.block || t.instance !== targets[0].instance)) {
    throw new Error("Duplicate stages of one block or of the top level at a time");
  }
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  const stages = stageList(pipeline, owner);
  const taken = new Set([...stages.map((s) => s.id), ...stages.flatMap((s) => s.outputs), ...Object.keys(pipeline.blocks ?? {})]);
  if (owner.kind === "block") {
    const def = pipeline.blocks![owner.block!];
    for (const name of [...def.inputs, ...def.outputs]) taken.add(name);
  }
  const fresh = (base: string) => {
    const stem = base.replace(/_(copy|\d+)$/, "");
    let n = 1;
    let candidate = `${stem}_${n}`;
    while (taken.has(candidate)) candidate = `${stem}_${++n}`;
    taken.add(candidate);
    return candidate;
  };
  const renamed = new Map<string, string>();
  const originals = targets.map((t) => findStage(pipeline, t));
  const copies = originals.map((stage) => {
    const copy = clone(stage);
    copy.id = fresh(stage.id);
    copy.outputs = stage.outputs.map((slot) => {
      const name = slot === stage.id ? copy.id : fresh(slot);
      renamed.set(slot, name);
      return name;
    });
    return copy;
  });
  for (const copy of copies) copy.inputs = copy.inputs.map((slot) => renamed.get(slot) ?? slot);
  stages.push(...copies);
  reorder(pipeline, owner);
  const prefix = targets[0].instance ? `${targets[0].instance}/` : "";
  return { document: doc, keys: copies.map((c) => `${prefix}${c.id}`) };
}

/** Option names that mean the same quantity in different transformations. */
const OPTION_ALIASES = [
  ["number", "n_gaussians", "num_basis"], // basis functions of a radial basis
  ["end", "stop", "r_max"], // where a radial basis ends (Angstrom)
];

export interface ReplacementPlan {
  options: Record<string, unknown>;
  kept: string[];
  mapped: [string, string][];
  dropped: string[];
}

/**
 * Options a stage keeps when its transformation is replaced: those the new
 * one knows, and those it knows under another name (with the value the stage
 * resolved to, so a 128-function basis stays 128 functions).
 */
export function replacementPlan(stage: Stage, transform: string, pipeline: Pipeline, catalog: Catalog | null, resolved: Record<string, unknown> = {}): ReplacementPlan {
  const info = catalog?.transforms.find((t) => t.name === transform);
  const def = pipeline.blocks?.[transform] ?? catalog?.block_library?.[transform];
  const known = new Set(info ? info.options.map((o) => o.name) : Object.keys(def?.options ?? {}));
  const given = stage.options ?? {};
  const plan: ReplacementPlan = { options: {}, kept: [], mapped: [], dropped: [] };
  for (const [name, value] of Object.entries(given)) {
    if (known.has(name)) {
      plan.options[name] = value;
      plan.kept.push(name);
      continue;
    }
    const target = OPTION_ALIASES.find((group) => group.includes(name))?.find((alias) => known.has(alias));
    if (target && !(target in plan.options)) {
      plan.options[target] = value;
      plan.mapped.push([name, target]);
    } else plan.dropped.push(name);
  }
  for (const [name, value] of Object.entries(resolved)) {
    if (name in given || value === undefined || value === null) continue;
    const target = OPTION_ALIASES.find((group) => group.includes(name))?.find((alias) => known.has(alias) && alias !== name);
    if (target && !(target in plan.options)) {
      plan.options[target] = value;
      plan.mapped.push([name, target]);
    }
  }
  return plan;
}

/**
 * Swaps a stage's transformation (or block) in place. Its id and output slots
 * stay, so later stages keep reading it; inputs are kept by port name, then by
 * position; options the new transformation knows are kept.
 */
export function replaceStage(document: RecipeDocument, target: VisibleStage, transform: string, catalog: Catalog | null, resolved: Record<string, unknown> = {}): RecipeDocument {
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  const stage = findStage(pipeline, target);
  const previous = stage.transform;
  const before = portsOf(stage, pipeline, catalog).map((p) => p.name);
  let def = pipeline.blocks?.[transform];
  const info = catalog?.transforms.find((t) => t.name === transform);
  const library = catalog?.block_library?.[transform];
  if (!def && !info && library) {
    const { ports: _ports, ...definition } = clone(library);
    pipeline.blocks = { ...(pipeline.blocks ?? {}), [transform]: definition };
    def = definition;
  }
  if (!def && !info) throw new Error(`Unknown transformation ${transform}`);
  if (target.owner.kind === "block" && transform === target.owner.block) throw new Error("A block cannot contain itself");
  const names = def ? def.inputs : info!.inputs.map((p) => p.name);
  const variadic = Boolean(info?.inputs[info.inputs.length - 1]?.variadic);
  const taken = new Set<number>();
  const inputs = names.map((name, i) => {
    const byName = before.indexOf(name);
    if (byName >= 0 && byName < stage.inputs.length && !taken.has(byName)) {
      taken.add(byName);
      return stage.inputs[byName];
    }
    if (i < stage.inputs.length && !taken.has(i)) {
      taken.add(i);
      return stage.inputs[i];
    }
    return "";
  });
  if (variadic) for (let i = names.length; i < stage.inputs.length; i++) if (!taken.has(i)) inputs.push(stage.inputs[i]);
  const optional = (i: number) => Boolean(info?.inputs[Math.min(i, info.inputs.length - 1)]?.optional || (variadic && i >= names.length - 1));
  while (inputs.length && inputs[inputs.length - 1] === "" && optional(inputs.length - 1)) inputs.pop();
  const outputNames = def ? def.outputs : info!.outputs;
  const outputs = outputNames.map((name, i) => stage.outputs[i] ?? `${stage.id}_${name}`);
  const { options } = replacementPlan(stage, transform, pipeline, catalog, resolved);
  stage.transform = transform;
  stage.inputs = inputs;
  stage.outputs = outputs;
  if (Object.keys(options).length) stage.options = options;
  else delete stage.options;
  delete stage.bypass;
  // A library block copied in for the replaced instance goes with it.
  if (pipeline.blocks?.[previous]?.library && countInstances(pipeline, previous) === 0) {
    const { [previous]: _gone, ...blocks } = pipeline.blocks;
    pipeline.blocks = blocks;
  }
  reorder(pipeline, target.owner);
  return doc;
}

function sameType(a?: FeatureSpecJson, b?: FeatureSpecJson) {
  return Boolean(a && b && a.domain === b.domain && a.representation === b.representation && (a.irreps ?? a.width) === (b.irreps ?? b.width));
}

/**
 * Copies top-level stages as the next layer. Where an original reads a value
 * from outside the selection that one of the selection's outputs replaces (the
 * same type: node features in, node features out), its copy reads that output.
 * Later stages and targets reading the originals' outputs then read the copies'.
 */
export function insertAfter(document: RecipeDocument, targets: VisibleStage[], specOf: (slot: string) => FeatureSpecJson | undefined): { document: RecipeDocument; keys: string[]; chained: [string, string][]; repointed: string[] } {
  if (targets.some((t) => t.owner.kind !== "top")) throw new Error("Insert a copy after works on top-level stages; collapse the block or select its instance");
  const { document: doc, keys } = duplicateStages(document, targets);
  const pipeline = doc.pipeline!;
  const originals = targets.map((t) => pipeline.stages.find((s) => s.id === t.localId)!);
  const copies = keys.map((key) => pipeline.stages.find((s) => s.id === key)!);
  const produced = new Set(originals.flatMap((s) => s.outputs));
  const outputs = originals.flatMap((s) => s.outputs);
  const chained: [string, string][] = [];
  // Each value entering the selection from outside, replaced by the selection
  // output of its type (the reading stage's own output preferred).
  const replacement = new Map<string, string>();
  originals.forEach((stage) => {
    for (const slot of stage.inputs) {
      if (!slot || produced.has(slot) || replacement.has(slot)) continue;
      const own = stage.outputs.find((o) => sameType(specOf(o), specOf(slot)));
      const any = own ?? outputs.find((o) => sameType(specOf(o), specOf(slot)));
      if (any) {
        replacement.set(slot, any);
        chained.push([slot, any]);
      }
    }
  });
  copies.forEach((copy, i) => {
    copy.inputs = copy.inputs.map((slot, j) => (originals[i].inputs[j] && replacement.has(originals[i].inputs[j]) ? replacement.get(originals[i].inputs[j])! : slot));
  });
  // Downstream readers of the originals now read the copies.
  const renamed = new Map<string, string>();
  originals.forEach((stage, i) => stage.outputs.forEach((slot, j) => renamed.set(slot, copies[i].outputs[j])));
  const inside = new Set([...originals, ...copies]);
  const repointed: string[] = [];
  for (const stage of pipeline.stages) {
    if (inside.has(stage)) continue;
    const before = stage.inputs.join("|");
    stage.inputs = stage.inputs.map((slot) => renamed.get(slot) ?? slot);
    if (stage.inputs.join("|") !== before) repointed.push(stage.id);
  }
  for (const [operator, slot] of Object.entries(pipeline.outputs)) {
    if (renamed.has(slot)) {
      pipeline.outputs[operator] = renamed.get(slot)!;
      repointed.push(`target ${operator}`);
    }
  }
  reorder(pipeline, { kind: "top" });
  return { document: doc, keys, chained, repointed };
}

export function removeStage(document: RecipeDocument, target: VisibleStage): RecipeDocument {
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  const list = stageList(pipeline, target.owner);
  const index = list.findIndex((s) => s.id === target.localId);
  if (index < 0) return doc;
  const [removed] = list.splice(index, 1);
  // A library block copied in for this instance goes with its last instance.
  if (pipeline.blocks?.[removed.transform]?.library && countInstances(pipeline, removed.transform) === 0) {
    const { [removed.transform]: _gone, ...blocks } = pipeline.blocks;
    pipeline.blocks = blocks;
  }
  const dangling = new Set(removed.outputs);
  for (const stage of list) stage.inputs = stage.inputs.map((s) => (dangling.has(s) ? "" : s));
  if (target.owner.kind === "top") {
    for (const [operator, slot] of Object.entries(pipeline.outputs)) if (dangling.has(slot)) pipeline.outputs[operator] = "";
  }
  return doc;
}

/**
 * Renames a stage. An output named after the stage (the convention for a
 * single output) is renamed with it, and its readers follow.
 */
export function renameStage(document: RecipeDocument, target: VisibleStage, id: string): RecipeDocument {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(id)) throw new Error("Stage ids are identifiers (letters, digits, underscores)");
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  const list = stageList(pipeline, target.owner);
  if (list.some((s) => s.id === id)) throw new Error(`A stage named ${id} already exists`);
  const stage = findStage(pipeline, target);
  const old = stage.id;
  stage.id = id;
  if (stage.outputs.includes(old) && !list.some((s) => s.outputs.includes(id))) {
    stage.outputs = stage.outputs.map((slot) => (slot === old ? id : slot));
    for (const other of list) other.inputs = other.inputs.map((slot) => (slot === old ? id : slot));
    if (target.owner.kind === "top") {
      for (const [operator, slot] of Object.entries(pipeline.outputs)) if (slot === old) pipeline.outputs[operator] = id;
    } else {
      const def = pipeline.blocks![target.owner.block!];
      def.outputs = def.outputs.map((slot) => (slot === old ? id : slot));
    }
  }
  return doc;
}

export function setOption(document: RecipeDocument, target: VisibleStage, name: string, value: unknown): RecipeDocument {
  const doc = clone(document);
  const stage = findStage(doc.pipeline!, target);
  const options = { ...(stage.options ?? {}) };
  if (value === undefined) delete options[name];
  else options[name] = value;
  if (Object.keys(options).length) stage.options = options;
  else delete stage.options;
  return doc;
}

/** Enables a stage, or bypasses it passing ``bypass`` (default input when null). */
export function setEnabled(document: RecipeDocument, target: VisibleStage, enabled: boolean, bypass: string | number | null = null): RecipeDocument {
  const doc = clone(document);
  const stage = findStage(doc.pipeline!, target);
  if (enabled) {
    delete stage.enabled;
    delete stage.bypass;
  } else {
    stage.enabled = false;
    if (bypass === null) delete stage.bypass;
    else stage.bypass = bypass;
  }
  return doc;
}

export function setRole(document: RecipeDocument, target: VisibleStage, role: string): RecipeDocument {
  const doc = clone(document);
  const stage = findStage(doc.pipeline!, target);
  if (role) stage.role = role;
  else delete stage.role;
  return doc;
}

/** Turn top-level stages into a named block and one instance of it. */
export function groupStages(document: RecipeDocument, keys: string[], name: string): RecipeDocument {
  if (!/^[a-z][a-z0-9_]*$/.test(name)) throw new Error("Block names use lowercase letters, digits and underscores");
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  pipeline.blocks = pipeline.blocks ?? {};
  if (pipeline.blocks[name]) throw new Error(`A block named ${name} already exists`);
  const selected = pipeline.stages.filter((s) => keys.includes(s.id));
  if (selected.length < 1) throw new Error("Select top-level stages to group");
  const rest = pipeline.stages.filter((s) => !keys.includes(s.id));
  const internal = new Set(selected.flatMap((s) => s.outputs));
  const inputs: string[] = [];
  for (const stage of selected) for (const slot of stage.inputs) if (slot && !internal.has(slot) && !inputs.includes(slot)) inputs.push(slot);
  const used = new Set([...rest.flatMap((s) => s.inputs), ...Object.values(pipeline.outputs)]);
  const outputs = [...internal].filter((slot) => used.has(slot));
  if (!outputs.length) throw new Error("The selection produces nothing used outside it");
  const roles = new Set(selected.map((s) => s.role ?? ""));
  const def: BlockDef = {
    inputs,
    outputs,
    // Inner stages keep their roles (e.g. readout, which drives head-only training).
    stages: clone(selected),
  };
  const instanceId = uniqueId(pipeline, `${name}_0`);
  const instance: Stage = { id: instanceId, transform: name, inputs, outputs };
  if (roles.size === 1 && [...roles][0]) instance.role = [...roles][0];
  pipeline.blocks[name] = def;
  pipeline.stages = topologicalOrder([...rest, instance]);
  return doc;
}

function substitute(value: unknown, parameters: Record<string, unknown>): unknown {
  if (typeof value === "string") {
    const whole = value.match(/^\$([A-Za-z_]\w*)$/);
    if (whole) return parameters[whole[1]] ?? undefined;
    return value.replace(/\$\(([A-Za-z_]\w*)\)/g, (_, n) => String(parameters[n]));
  }
  if (Array.isArray(value)) return value.map((v) => substitute(v, parameters)).filter((v) => v !== undefined);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      const s = substitute(v, parameters);
      if (s !== undefined) out[k] = s;
    }
    return out;
  }
  return value;
}

/** Replace a top-level block instance by its stages. */
export function dissolveInstance(document: RecipeDocument, id: string): RecipeDocument {
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  const index = pipeline.stages.findIndex((s) => s.id === id);
  const instance = pipeline.stages[index];
  const def = pipeline.blocks?.[instance?.transform];
  if (!def) throw new Error("Only block instances can be dissolved");
  const parameters = { ...(def.options ?? {}), ...(instance.options ?? {}) };
  const rename = (local: string) => {
    const i = def.inputs.indexOf(local);
    if (i >= 0) return instance.inputs[i];
    const o = def.outputs.indexOf(local);
    if (o >= 0) return instance.outputs[o];
    return `${id}_${local}`;
  };
  const inner: Stage[] = def.stages.map((stage) => {
    const copy = clone(stage);
    copy.id = `${id}_${stage.id}`;
    copy.inputs = stage.inputs.map(rename);
    copy.outputs = stage.outputs.map(rename);
    if (copy.options) copy.options = substitute(copy.options, parameters) as Record<string, unknown>;
    if (instance.role && !copy.role) copy.role = instance.role;
    return copy;
  });
  pipeline.stages.splice(index, 1, ...inner);
  if (countInstances(pipeline, instance.transform) === 0) delete pipeline.blocks![instance.transform];
  return doc;
}

export function materialize(document: RecipeDocument, pipeline: Pipeline): RecipeDocument {
  const doc = clone(document);
  doc.pipeline = clone(pipeline);
  return doc;
}
