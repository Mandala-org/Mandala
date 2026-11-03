import { useCallback, useEffect, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type OnConnectEnd,
} from "@xyflow/react";
import { currentPipeline, layoutMemory, useStudio } from "../store";
import type { RecipeDocument } from "../types";
import { specColor, type BuiltGraph } from "../graph/build";
import { useGraph } from "../graph/context";
import { layoutGraph } from "../graph/layout";
import { connectionProblem, createsCycle } from "../graph/validate";
import { connect, disconnect, removeStage, setTarget, addStage } from "../graph/edit";
import { nodeTypes } from "./nodes";
import { editableObjective } from "../loss/objective";
import { portsOf, slotSpec, displayedInputs } from "../graph/model";
import { parseExpression, printExpression, removeLabel, replaceAt } from "../loss/expression";

function signature(built: BuiltGraph) {
  return built.nodes
    .map((n) => {
      const s = built.sizes.get(n.id);
      return `${n.id}:${n.parentId ?? ""}:${Math.round((s?.width ?? 0) / 40)}x${Math.round((s?.height ?? 0) / 40)}`;
    })
    .sort()
    .join("|") + "#" + built.edges.map((e) => `${e.source}>${e.target}`).sort().join("|");
}

export function Canvas() {
  const document = useStudio((s) => s.document);
  const analysis = useStudio((s) => s.analysis);
  const catalog = useStudio((s) => s.catalog);
  const orientation = useStudio((s) => s.orientation);
  const layoutRequest = useStudio((s) => s.layoutRequest);
  const edit = useStudio((s) => s.edit);
  const notify = useStudio((s) => s.notify);
  const setSelection = useStudio((s) => s.setSelection);
  const built = useGraph();
  const [positions, setPositions] = useState<Map<string, { x: number; y: number; width?: number; height?: number }>>(new Map());
  const [nodes, setNodes] = useState<Node[]>([]);
  const [selectedEdges, setSelectedEdges] = useState<Set<string>>(new Set());
  const lastSignature = useRef("");
  const lastRequest = useRef(-1);
  const rejection = useRef<string | null>(null);
  const pending = useRef<Map<string, { x: number; y: number }>>(new Map());
  const fitPending = useRef(false);
  const requestedLayout = useRef(false);
  const layoutRun = useRef(0);
  const flow = useReactFlow();
  const pipeline = currentPipeline({ document, analysis });

  // Re-run the layered layout when the structure changes or on request.
  // A requested layout (load, expand, orientation) places every node; after an
  // edit, nodes keep their positions and only new nodes take layout positions.
  useEffect(() => {
    const sig = signature(built);
    if (layoutRequest !== lastRequest.current) requestedLayout.current = true;
    const requested = requestedLayout.current;
    if (sig === lastSignature.current && layoutRequest === lastRequest.current) return;
    lastSignature.current = sig;
    lastRequest.current = layoutRequest;
    if (!built.nodes.length) return;
    const run = ++layoutRun.current;
    layoutGraph(built.nodes, built.edges, built.sizes, orientation).then((laid) => {
      // Only the newest layout is applied; older results are discarded.
      if (run !== layoutRun.current) return;
      requestedLayout.current = false;
      const restore = requested ? layoutMemory.restore : null;
      layoutMemory.restore = null;
      setPositions((previous) => {
        const next = new Map(laid);
        let added = 0;
        for (const node of built.nodes) {
          if (node.parentId || node.type === "frame") continue;
          let placed = pending.current.get(node.id.replace(/^stage:/, "")) ?? restore?.[node.id] ?? (!requested ? previous.get(node.id) : undefined);
          if (!placed && !requested && previous.size && node.type === "stage") {
            // A stage added without a drop position appears in the middle of the view.
            const bounds = window.document.querySelector(".canvas")?.getBoundingClientRect();
            const center = bounds ? flow.screenToFlowPosition({ x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2 }) : { x: 0, y: 0 };
            placed = { x: center.x - 110 + 30 * added, y: center.y - 50 + 30 * added };
            added += 1;
          }
          if (placed) next.set(node.id, { ...next.get(node.id), x: placed.x, y: placed.y });
        }
        pending.current.clear();
        return next;
      });
      if (requested) fitPending.current = true;
    });
  }, [built, orientation, layoutRequest]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    layoutMemory.positions = Object.fromEntries(
      [...positions].filter(([id]) => !built.nodes.find((n) => n.id === id)?.parentId).map(([id, p]) => [id, { x: Math.round(p.x), y: Math.round(p.y) }]),
    );
  }, [positions, built]);

  useEffect(() => {
    setNodes((current) => {
      const selected = new Set(current.filter((n) => n.selected).map((n) => n.id));
      return built.nodes.map((node) => {
        const position = positions.get(node.id);
        const style = { ...(node.style ?? {}) };
        if (node.type === "frame" && position?.width) {
          style.width = position.width;
          style.height = position.height;
        }
        return { ...node, position: position ? { x: position.x, y: position.y } : node.position, style, selected: selected.has(node.id) };
      });
    });
    if (fitPending.current && positions.size) {
      fitPending.current = false;
      setTimeout(() => flow.fitView({ padding: 0.08, duration: 300 }), 80);
    }
  }, [built, positions]); // eslint-disable-line react-hooks/exhaustive-deps

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setNodes((current) => applyNodeChanges(changes, current));
    const moved = changes.filter((c) => c.type === "position" && c.position && !c.dragging);
    if (moved.length) {
      setPositions((previous) => {
        const next = new Map(previous);
        for (const change of moved) if (change.type === "position" && change.position) next.set(change.id, { ...next.get(change.id), ...change.position });
        return next;
      });
    }
  }, []);

  const visibleOf = useCallback((nodeId: string) => built.graph.stages.find((s) => `stage:${s.key}` === nodeId), [built]);

  const sourceSlot = useCallback(
    (connection: Connection | Edge): string | null => {
      const node = built.nodes.find((n) => n.id === connection.source);
      if (!node) return null;
      if (node.type === "slot") return (node.data as { slot: string }).slot;
      if (node.type === "target") return `operator:${(node.data as { operator: string }).operator}`;
      const visible = visibleOf(node.id);
      const index = Number((connection.sourceHandle ?? "out-0").split("-")[1]);
      return visible?.outputs[index] ?? null;
    },
    [built, visibleOf],
  );

  const isValidConnection = useCallback(
    (connection: Connection | Edge) => {
      rejection.current = null;
      const slot = sourceSlot(connection);
      if (!slot) return false;
      const target = built.nodes.find((n) => n.id === connection.target);
      if (!target) return false;
      if (target.type === "term") {
        if (!slot.startsWith("operator:")) {
          rejection.current = "Objective terms read predicted operators: connect a target";
          return false;
        }
        return true;
      }
      if (slot.startsWith("operator:")) {
        rejection.current = "Targets feed objective terms only";
        return false;
      }
      const spec = slotSpec(analysis, catalog, slot);
      if (target.type === "target") {
        if (spec && !(spec.domain === "edge" && ["matrix", "block_vectors"].includes(spec.representation))) {
          rejection.current = `A target needs decoded edge blocks (matrix or block vectors); this slot is a ${spec.domain} ${spec.representation} feature`;
          return false;
        }
        return true;
      }
      if (target.type !== "stage") return false;
      const visible = visibleOf(target.id)!;
      const index = Number((connection.targetHandle ?? "in-0").split("-")[1]);
      const ports = portsOf(visible.stage, pipeline, catalog);
      const port = displayedInputs(visible.stage, ports).find((p) => p.index === index)?.port;
      if (!port) return false;
      const problem = connectionProblem(spec, port);
      if (problem) {
        rejection.current = problem;
        return false;
      }
      const producer = built.graph.producers.get(slot);
      if (producer && createsCycle(built.graph, producer.key, visible.key)) {
        rejection.current = "This connection would create a cycle";
        return false;
      }
      return true;
    },
    [analysis, built, catalog, pipeline, sourceSlot, visibleOf],
  );

  const onConnectEnd: OnConnectEnd = useCallback(
    (_, state) => {
      if (state.toNode && !state.isValid && rejection.current) notify(rejection.current, "error");
    },
    [notify],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const slot = sourceSlot(connection);
      const target = built.nodes.find((n) => n.id === connection.target);
      if (!slot || !target) return;
      if (target.type === "term") {
        const { label, operators } = target.data as { label: string; operators: string[] };
        const port = operators[Number((connection.targetHandle ?? "in-0").split("-")[1])];
        const operator = slot.slice("operator:".length);
        if (!port) return;
        edit((doc) => {
          const objective = editableObjective(doc, analysis?.objective.objective);
          objective.terms[label] = { ...objective.terms[label], [port]: operator };
          return { ...doc, objective };
        }, `${label} reads ${operator}`);
        return;
      }
      if (target.type === "target") {
        edit((doc) => setTarget(doc, (target.data as { operator: string }).operator, slot));
        return;
      }
      const visible = visibleOf(target.id)!;
      const index = Number((connection.targetHandle ?? "in-0").split("-")[1]);
      edit((doc) => connect(doc, built.graph, slot, visible, index, catalog));
    },
    [analysis, built, catalog, edit, sourceSlot, visibleOf],
  );

  // Each deleted element becomes a document change; one deletion is one edit.
  const edgeRemoval = useCallback(
    (edge: Edge): ((doc: RecipeDocument) => RecipeDocument) | null => {
      const target = built.nodes.find((n) => n.id === edge.target);
      const index = Number((edge.targetHandle ?? "in-0").split("-")[1]);
      if (target?.type === "stage") {
        const visible = visibleOf(target.id)!;
        return (doc) => disconnect(doc, visible, index, catalog);
      }
      if (target?.type === "target") {
        const operator = (target.data as { operator: string }).operator;
        return (doc) => setTarget({ ...doc, pipeline: { ...doc.pipeline!, outputs: { ...doc.pipeline!.outputs } } }, operator, "");
      }
      if (target?.type === "term") {
        const { label, operators } = target.data as { label: string; operators: string[] };
        const port = operators[index];
        if (!port) return null;
        return (doc) => {
          const objective = editableObjective(doc, analysis?.objective.objective);
          const { [port]: _removed, ...rest } = objective.terms[label];
          objective.terms[label] = rest;
          return { ...doc, objective };
        };
      }
      return null;
    },
    [analysis, built, catalog, visibleOf],
  );

  const nodeRemoval = useCallback(
    (node: Node): ((doc: RecipeDocument) => RecipeDocument) | null => {
      if (node.type === "stage") {
        const visible = visibleOf(node.id);
        return visible ? (doc) => removeStage(doc, visible) : null;
      }
      if (node.type === "term") {
        const label = (node.data as { label: string }).label;
        return (doc) => {
          const objective = editableObjective(doc, analysis?.objective.objective);
          delete objective.terms[label];
          const tree = objective.expression ? removeLabel(parseExpression(objective.expression), label) : null;
          objective.expression = tree ? printExpression(tree) : "";
          return { ...doc, objective };
        };
      }
      if (node.type === "expr") {
        const path = (node.data as { path: number[] }).path;
        return (doc) => {
          const objective = editableObjective(doc, analysis?.objective.objective);
          const tree = replaceAt(parseExpression(objective.expression ?? ""), path, null);
          objective.expression = tree ? printExpression(tree) : "";
          return { ...doc, objective };
        };
      }
      if (node.type === "target") {
        const operator = (node.data as { operator: string }).operator;
        return (doc) => {
          const outputs = { ...doc.pipeline!.outputs };
          delete outputs[operator];
          return { ...doc, pipeline: { ...doc.pipeline!, outputs } };
        };
      }
      return null;
    },
    [analysis, visibleOf],
  );

  const onDelete = useCallback(
    ({ nodes: deletedNodes, edges: deletedEdges }: { nodes: Node[]; edges: Edge[] }) => {
      const gone = new Set(deletedNodes.map((n) => n.id));
      const handle = (edge: Edge) => Number((edge.targetHandle ?? "in-0").split("-")[1]);
      // Wires of deleted nodes go with them; later ports first, so that
      // removing a variadic input does not shift the ones still to remove.
      const changes = [
        ...deletedEdges
          .filter((e) => !gone.has(e.source) && !gone.has(e.target))
          .sort((a, b) => handle(b) - handle(a))
          .map(edgeRemoval),
        ...deletedNodes.map(nodeRemoval),
      ].filter((change): change is (doc: RecipeDocument) => RecipeDocument => change !== null);
      if (!changes.length) return;
      const what = deletedNodes.length ? `${deletedNodes.length} node(s)` : `${deletedEdges.length} wire(s)`;
      edit((doc) => changes.reduce((current, change) => change(current), doc), `Deleted ${what}`);
      setSelectedEdges(new Set());
    },
    [edgeRemoval, edit, nodeRemoval],
  );

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setSelectedEdges((previous) => {
      const next = new Set(previous);
      for (const change of changes) {
        if (change.type !== "select") continue;
        if (change.selected) next.add(change.id);
        else next.delete(change.id);
      }
      return next;
    });
  }, []);
  const edges = built.edges.map((e) => (selectedEdges.has(e.id) ? { ...e, selected: true } : e));
  const modalOpen = useStudio((st) => st.modal !== null);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const transform = event.dataTransfer.getData("application/mandala-transform");
      const input = event.dataTransfer.getData("application/mandala-input");
      const position = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      if (input) {
        useStudio.getState().showInput(input);
        setPositions((p) => new Map(p).set(`input:${input}`, position));
        return;
      }
      if (transform) {
        let id = "";
        const ok = edit((doc) => {
          const result = addStage(doc, transform, catalog);
          id = result.id;
          return result.document;
        });
        if (ok) {
          pending.current.set(id, position);
          setPositions((p) => new Map(p).set(`stage:${id}`, position));
        }
      }
    },
    [catalog, edit, flow],
  );

  // Keyboard shortcuts.
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("input, textarea, select") || useStudio.getState().modal) return;
      const mod = event.ctrlKey || event.metaKey;
      if (mod && event.key.toLowerCase() === "z" && !event.shiftKey) {
        event.preventDefault();
        useStudio.getState().undo();
      } else if (mod && (event.key.toLowerCase() === "y" || (event.key.toLowerCase() === "z" && event.shiftKey))) {
        event.preventDefault();
        useStudio.getState().redo();
      } else if (mod && event.key.toLowerCase() === "g") {
        event.preventDefault();
        groupSelection();
      } else if (event.key.toLowerCase() === "l" && !mod) {
        useStudio.getState().requestLayout();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const groupSelection = useCallback(() => {
    const keys = nodes.filter((n) => n.selected && n.type === "stage").map((n) => visibleOf(n.id)).filter((v) => v && v.instance === null).map((v) => v!.localId);
    if (keys.length < 1) {
      notify("Select top-level stages to group: Shift-drag a box or Ctrl-click", "error");
      return;
    }
    useStudio.getState().askGroup(keys);
  }, [nodes, notify, visibleOf]);

  useEffect(() => {
    (window as unknown as { studioGroup: () => void }).studioGroup = groupSelection;
  }, [groupSelection]);

  return (
    <div className="canvas" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        isValidConnection={isValidConnection}
        onDelete={onDelete}
        onSelectionChange={({ nodes: selected }) => setSelection(selected.map((n) => n.id).sort())}
        onNodeClick={(event, node) => {
          // Ctrl/Cmd/Shift-click toggles a node in the selection.
          if (!(event.ctrlKey || event.metaKey || event.shiftKey)) return;
          const previous = new Set(useStudio.getState().selection);
          if (previous.has(node.id) && event.detail === 1) previous.delete(node.id);
          else previous.add(node.id);
          setNodes((all) => all.map((n) => ({ ...n, selected: previous.has(n.id) })));
          setSelection([...previous].sort());
        }}
        deleteKeyCode={modalOpen ? null : ["Delete", "Backspace"]}
        multiSelectionKeyCode={["Meta", "Control"]}
        selectionKeyCode="Shift"
        minZoom={0.05}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{ type: "default" }}
        connectionLineStyle={{ stroke: "#e6e9ef", strokeWidth: 2 }}
        fitView
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="#2a3140" />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable style={{ width: 150, height: 100 }} nodeColor={(n) => (n.type === "stage" ? "#3a4456" : n.type === "frame" ? "#1e2430" : specColor())} maskColor="rgba(10,12,16,0.7)" />
      </ReactFlow>
    </div>
  );
}
