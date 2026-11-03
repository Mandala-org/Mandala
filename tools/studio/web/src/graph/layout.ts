// Layered layout with ELK: inputs first, the loss last, expanded blocks as
// compound nodes laid out with their contents.

import ELK, { type ElkNode, type ElkExtendedEdge } from "elkjs/lib/elk.bundled.js";
import type { Edge, Node } from "@xyflow/react";

const elk = new ELK();

export async function layoutGraph(
  nodes: Node[],
  edges: Edge[],
  sizes: Map<string, { width: number; height: number }>,
  orientation: "TB" | "LR",
): Promise<Map<string, { x: number; y: number; width?: number; height?: number }>> {
  const direction = orientation === "TB" ? "DOWN" : "RIGHT";
  const elkNodes = new Map<string, ElkNode>();
  const root: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": direction,
      "elk.hierarchyHandling": "INCLUDE_CHILDREN",
      "elk.layered.spacing.nodeNodeBetweenLayers": "64",
      "elk.spacing.nodeNode": "36",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.crossingMinimization.semiInteractive": "true",
      "elk.edgeRouting": "SPLINES",
    },
    children: [],
    edges: [],
  };
  for (const node of nodes) {
    const size = sizes.get(node.id);
    const elkNode: ElkNode = {
      id: node.id,
      width: size?.width ?? 200,
      height: size?.height ?? 80,
      layoutOptions: {},
      children: [],
    };
    if (node.type === "frame") {
      elkNode.layoutOptions = {
        "elk.padding": "[top=46,left=18,bottom=18,right=18]",
        "elk.direction": direction,
      };
      delete elkNode.width;
      delete elkNode.height;
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
  const elkEdges: ElkExtendedEdge[] = edges.map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] }));
  root.edges = elkEdges;
  const result = await elk.layout(root);
  const positions = new Map<string, { x: number; y: number; width?: number; height?: number }>();
  const visit = (node: ElkNode) => {
    for (const child of node.children ?? []) {
      positions.set(child.id, { x: child.x ?? 0, y: child.y ?? 0, width: child.width, height: child.height });
      visit(child);
    }
  };
  visit(result);
  return positions;
}
