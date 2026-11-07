// Layered layout with ELK: inputs first, the loss last, expanded blocks as
// compound nodes laid out with their contents. Every handle is an ELK port on
// its side of the box. ELK may reorder a stage's ports along that side (and
// the structure inputs in the first layer) to untangle the wires; the order
// it chose is returned, and the boxes draw their ports in that order. Port
// order is only a drawing: inputs stay positional in the recipe.

import ELK, { type ElkNode, type ElkExtendedEdge, type ElkPort } from "elkjs/lib/elk.bundled.js";
import type { Edge, Node } from "@xyflow/react";
import type { StageNodeData, TermNodeData, ExprNodeData } from "./build";

const elk = new ELK();

type Orientation = "TB" | "LR";

/** Handle offsets along a side, as drawn by the node components (nodes.tsx). */
export function spreadFraction(count: number, index: number) {
  return (index + 1) / (count + 1);
}

function handles(node: Node, width: number, height: number, orientation: Orientation): { id: string; x: number; y: number; input: boolean }[] {
  const tb = orientation === "TB";
  const along = (input: boolean, fraction: number, row: number) =>
    tb ? { x: fraction * width, y: input ? 0 : height } : { x: input ? 0 : width, y: row >= 0 ? row : fraction * height };
  const result: { id: string; x: number; y: number; input: boolean }[] = [];
  const add = (id: string, input: boolean, fraction: number, row = -1) => result.push({ id, input, ...along(input, fraction, row) });
  switch (node.type) {
    case "stage": {
      const d = node.data as StageNodeData;
      d.ports.forEach(({ index }, i) => add(`in-${index}`, true, spreadFraction(d.ports.length, i), 40 + i * 22));
      d.outputs.forEach((_, i) => add(`out-${i}`, false, spreadFraction(d.outputs.length, i), 40 + i * 22));
      break;
    }
    case "slot":
    case "ref":
      add("out-0", false, 0.5);
      break;
    case "target":
      add("in-0", true, 0.5);
      add("out-0", false, 0.5);
      break;
    case "term": {
      const d = node.data as TermNodeData;
      d.operators.forEach((_, i) => add(`in-${i}`, true, spreadFraction(d.operators.length, i)));
      add("out-0", false, 0.5);
      break;
    }
    case "expr": {
      const d = node.data as ExprNodeData;
      for (let i = 0; i < d.arity; i++) add(`in-${i}`, true, spreadFraction(d.arity, i));
      add("out-0", false, 0.5);
      break;
    }
    case "loss":
      add("in-0", true, 0.5);
      break;
  }
  return result;
}

const portId = (node: string, handle: string) => `${node}::${handle}`;

/** A laid-out node; for a stage, the order its input and output handles are drawn in. */
export interface LaidOut {
  x: number;
  y: number;
  width?: number;
  height?: number;
  inputOrder?: number[];
  outputOrder?: number[];
}

export async function layoutGraph(
  nodes: Node[],
  edges: Edge[],
  sizes: Map<string, { width: number; height: number }>,
  orientation: Orientation,
): Promise<Map<string, LaidOut>> {
  const direction = orientation === "TB" ? "DOWN" : "RIGHT";
  const elkNodes = new Map<string, ElkNode>();
  const ports = new Set<string>();
  const root: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": direction,
      "elk.hierarchyHandling": "INCLUDE_CHILDREN",
      "elk.layered.spacing.nodeNodeBetweenLayers": "64",
      "elk.spacing.nodeNode": "36",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.edgeRouting": "SPLINES",
    },
    children: [],
    edges: [],
  };
  for (const node of nodes) {
    const size = sizes.get(node.id);
    const width = size?.width ?? 200;
    const height = size?.height ?? 80;
    const elkNode: ElkNode = { id: node.id, width, height, layoutOptions: {}, children: [] };
    if (node.type === "frame") {
      elkNode.layoutOptions = {
        "elk.padding": "[top=46,left=18,bottom=18,right=18]",
        "elk.direction": direction,
      };
      delete elkNode.width;
      delete elkNode.height;
    } else {
      const nodePorts: ElkPort[] = handles(node, width, height, orientation).map((h) => {
        const id = portId(node.id, h.id);
        ports.add(id);
        const side = orientation === "TB" ? (h.input ? "NORTH" : "SOUTH") : h.input ? "WEST" : "EAST";
        return { id, x: h.x, y: h.y, width: 1, height: 1, layoutOptions: { "elk.port.side": side } };
      });
      if (nodePorts.length) {
        elkNode.ports = nodePorts;
        elkNode.layoutOptions!["elk.portConstraints"] = node.type === "stage" ? "FIXED_SIDE" : "FIXED_POS";
      }
    }
    if (node.type === "slot") elkNode.layoutOptions!["elk.layered.layering.layerConstraint"] = "FIRST";
    if (node.id === "loss") elkNode.layoutOptions!["elk.layered.layering.layerConstraint"] = "LAST";
    elkNodes.set(node.id, elkNode);
  }
  for (const node of nodes) {
    const elkNode = elkNodes.get(node.id)!;
    const parent = node.parentId ? elkNodes.get(node.parentId) : root;
    (parent ?? root).children!.push(elkNode);
  }
  const end = (node: string, handle: string | null | undefined) => {
    const id = portId(node, handle ?? "");
    return ports.has(id) ? id : node;
  };
  const elkEdges: ElkExtendedEdge[] = edges.map((edge) => ({
    id: edge.id,
    sources: [end(edge.source, edge.sourceHandle)],
    targets: [end(edge.target, edge.targetHandle)],
  }));
  root.edges = elkEdges;
  const result = await elk.layout(root);
  const positions = new Map<string, LaidOut>();
  const along = (port: ElkPort) => (orientation === "TB" ? port.x ?? 0 : port.y ?? 0);
  // Handle numbers (in-3, out-0) in the order ELK placed them along the side.
  const order = (child: ElkNode, kind: "in" | "out") =>
    (child.ports ?? [])
      .filter((port) => port.id.split("::")[1]?.startsWith(`${kind}-`))
      .sort((a, b) => along(a) - along(b))
      .map((port) => Number(port.id.split("::")[1].slice(kind.length + 1)));
  const visit = (node: ElkNode) => {
    for (const child of node.children ?? []) {
      const laid: LaidOut = { x: child.x ?? 0, y: child.y ?? 0, width: child.width, height: child.height };
      if (child.ports?.length) {
        laid.inputOrder = order(child, "in");
        laid.outputOrder = order(child, "out");
      }
      positions.set(child.id, laid);
      visit(child);
    }
  };
  visit(result);
  return positions;
}
