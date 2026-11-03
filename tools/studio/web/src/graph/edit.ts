// Pure edits of a recipe document. Every function returns a new document or
// throws an Error whose message is shown to the user.

import type { BlockDef, Catalog, Pipeline, RecipeDocument, Stage } from "../types";
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

export function addStage(document: RecipeDocument, transform: string, catalog: Catalog | null): { document: RecipeDocument; id: string } {
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
  if (info) {
    const options: Record<string, unknown> = {};
    for (const option of info.options) if (option.required) options[option.name] = option.choices[0] ?? (option.type === "irreps" ? "8x0e" : option.type.includes("int") ? 8 : "");
    if (Object.keys(options).length) stage.options = options;
  }
  pipeline.stages.push(stage);
  return { document: doc, id };
}

export function removeStage(document: RecipeDocument, target: VisibleStage): RecipeDocument {
  const doc = clone(document);
  const pipeline = doc.pipeline!;
  const list = stageList(pipeline, target.owner);
  const index = list.findIndex((s) => s.id === target.localId);
  if (index < 0) return doc;
  const [removed] = list.splice(index, 1);
  const dangling = new Set(removed.outputs);
  for (const stage of list) stage.inputs = stage.inputs.map((s) => (dangling.has(s) ? "" : s));
  if (target.owner.kind === "top") {
    for (const [operator, slot] of Object.entries(pipeline.outputs)) if (dangling.has(slot)) pipeline.outputs[operator] = "";
  }
  return doc;
}

export function renameStage(document: RecipeDocument, target: VisibleStage, id: string): RecipeDocument {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(id)) throw new Error("Stage ids are identifiers (letters, digits, underscores)");
  const doc = clone(document);
  const list = stageList(doc.pipeline!, target.owner);
  if (list.some((s) => s.id === id)) throw new Error(`A stage named ${id} already exists`);
  findStage(doc.pipeline!, target).id = id;
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

export function setEnabled(document: RecipeDocument, target: VisibleStage, enabled: boolean): RecipeDocument {
  const doc = clone(document);
  const stage = findStage(doc.pipeline!, target);
  if (enabled) delete stage.enabled;
  else stage.enabled = false;
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
