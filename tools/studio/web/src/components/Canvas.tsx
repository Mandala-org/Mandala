import { useCallback, useEffect, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  ControlButton,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
  useStore,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type OnConnectEnd,
} from "@xyflow/react";
import { currentPipeline, layoutMemory, preferences, useStudio } from "../store";
import type { RecipeDocument } from "../types";
import { specColor, type BuiltGraph } from "../graph/build";
import { useGraph } from "../graph/context";
import { layoutGraph, type LaidOut } from "../graph/layout";
import { connectionProblem, createsCycle, expectedDomain } from "../graph/validate";
import { connect, disconnect, duplicateStages, insertAfter, removeStage, setTarget, addStage } from "../graph/edit";
import { nodeTypes } from "./nodes";
import { Finder } from "./Finder";
import { editableObjective } from "../loss/objective";
import { availableSlots, portsOf, shiftProblem, slotSpec, displayedInputs, typingInto } from "../graph/model";
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

/** The free position nearest ``center`` for a box of ``size`` (spiral search). */
function freeSpot(center: { x: number; y: number }, size: { width: number; height: number }, boxes: { x: number; y: number; width: number; height: number }[], skip: number) {
  const gap = 24;
  const free = (x: number, y: number) => boxes.every((b) => x + size.width + gap <= b.x || b.x + b.width + gap <= x || y + size.height + gap <= b.y || b.y + b.height + gap <= y);
  let found = 0;
  for (let ring = 0; ring < 40; ring++) {
    for (let i = -ring; i <= ring; i++) {
      for (const [dx, dy] of [[i, -ring], [i, ring], [-ring, i], [ring, i]]) {
        const x = center.x - size.width / 2 + dx * 60;
        const y = center.y - size.height / 2 + dy * 60;
        if (free(x, y) && found++ >= skip) return { x, y };
      }
    }
  }
  return { x: center.x - size.width / 2, y: center.y - size.height / 2 };
}

/** Far zoom levels show block names only (styles: `.canvas.lod-far`). */
function ZoomDetail({ target }: { target: React.RefObject<HTMLDivElement> }) {
  const far = useStore((s) => s.transform[2] < 0.6);
  useEffect(() => {
    target.current?.classList.toggle("lod-far", far);
  }, [far, target]);
  return null;
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
  const [positions, setPositions] = useState<Map<string, LaidOut>>(new Map());
  const [nodes, setNodes] = useState<Node[]>([]);
  const [selectedEdges, setSelectedEdges] = useState<Set<string>>(new Set());
  const lastSignature = useRef("");
  const lastRequest = useRef(-1);
  const rejection = useRef<string | null>(null);
  const pending = useRef<Map<string, { x: number; y: number }>>(new Map());
  const placedBoxes = useRef<Map<string, { x: number; y: number; width: number; height: number }>>(new Map());
  const fitPending = useRef(false);
  const requestedLayout = useRef(false);
  const layoutRun = useRef(0);
  const flow = useReactFlow();
  const container = useRef<HTMLDivElement>(null);
  const [showMap, setShowMap] = useState(preferences.get("studio.minimap") !== "0");
  const toggleMap = () => {
    preferences.set("studio.minimap", showMap ? "0" : "1");
    setShowMap(!showMap);
  };
  const pipeline = currentPipeline({ document, analysis });

  // A whole recipe rarely fits on screen at a readable size: show it whole when
  // it does, otherwise its beginning (the inputs) at a readable zoom; the
  // minimap and "fit view" give the overview.
  const showRecipe = useCallback(() => {
    const bounds = container.current?.getBoundingClientRect();
    // Laid-out positions and known sizes: new nodes are not measured yet.
    const placed = [...placedBoxes.current.values()];
    if (!bounds || !placed.length) return;
    const left = Math.min(...placed.map((b) => b.x));
    const upper = Math.min(...placed.map((b) => b.y));
    const box = { x: left, y: upper, width: Math.max(...placed.map((b) => b.x + b.width)) - left, height: Math.max(...placed.map((b) => b.y + b.height)) - upper };
    // The minimap covers the bottom-right corner: a whole-recipe view keeps clear of it.
    const mapWidth = window.document.querySelector(".react-flow__minimap") ? 170 : 0;
    const fit = Math.min((bounds.width - mapWidth) / (box.width * 1.08), bounds.height / (box.height * 1.08));
    // Above the zoom where boxes show only their names (ZoomDetail).
    const readable = 0.62;
    // Room at the top for the stage finder.
    const top = 52;
    if (fit >= readable) {
      const zoom = Math.min(1, fit);
      flow.setViewport({ x: (bounds.width - mapWidth - box.width * zoom) / 2 - box.x * zoom, y: Math.max(top, (bounds.height - box.height * zoom) / 2) - box.y * zoom, zoom }, { duration: 300 });
      return;
    }
    const zoom = Math.min(0.85, Math.max(readable, bounds.width / (box.width * 1.05)));
    const across = box.width * zoom <= bounds.width;
    const x = across ? (bounds.width - box.width * zoom) / 2 - box.x * zoom : 24 - box.x * zoom;
    flow.setViewport({ x, y: top - box.y * zoom, zoom }, { duration: 300 });
  }, [flow]);

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
        const placedNew = new Map<string, { x: number; y: number }>();
        let added = 0;
        for (const node of built.nodes) {
          if (node.parentId || node.type === "frame") continue;
          let placed = pending.current.get(node.id.replace(/^stage:/, "")) ?? restore?.[node.id] ?? (!requested ? previous.get(node.id) : undefined);
          if (!placed && !requested && previous.size) {
            // A new node goes into free space next to what it connects to: an
            // input above (before) its reader, a stage below (after) its
            // producers; otherwise near the middle of the view.
            const size = built.sizes.get(node.id) ?? { width: 240, height: 120 };
            const tb = orientation === "TB";
            const at = (id: string) => placedNew.get(id) ?? previous.get(id);
            const reader = built.edges.filter((e) => e.source === node.id).map((e) => at(e.target)).find(Boolean);
            const producers = built.edges.filter((e) => e.target === node.id).map((e) => ({ p: at(e.source), size: built.sizes.get(e.source) })).filter((x) => x.p);
            let center: { x: number; y: number };
            if (reader && node.type !== "stage") {
              center = tb ? { x: reader.x + size.width / 2, y: reader.y - size.height / 2 - 50 } : { x: reader.x - size.width / 2 - 70, y: reader.y + size.height / 2 };
            } else if (producers.length) {
              const last = producers.reduce((a, b) => ((tb ? b.p!.y : b.p!.x) > (tb ? a.p!.y : a.p!.x) ? b : a));
              const extent = tb ? last.size?.height ?? 80 : last.size?.width ?? 200;
              center = tb ? { x: last.p!.x + size.width / 2, y: last.p!.y + extent + 60 + size.height / 2 } : { x: last.p!.x + extent + 80 + size.width / 2, y: last.p!.y + size.height / 2 };
            } else {
              const bounds = window.document.querySelector(".canvas")?.getBoundingClientRect();
              center = bounds ? flow.screenToFlowPosition({ x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2 }) : { x: 0, y: 0 };
            }
            const obstacles = [...previous, ...placedNew]
              .filter(([id]) => id !== node.id && built.sizes.has(id) && built.nodes.some((n) => n.id === id && !n.parentId))
              .map(([id, p]) => ({ ...p, ...built.sizes.get(id)! }));
            placed = freeSpot(center, size, obstacles, 0);
            placedNew.set(node.id, placed);
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
      // New copies become the selection once they are on the canvas.
      const copies = useStudio.getState().pendingSelection;
      const ready = copies.length > 0 && copies.every((id) => built.nodes.some((n) => n.id === id));
      if (ready) setTimeout(() => useStudio.setState({ pendingSelection: [], selection: [...copies].sort() }), 0);
      const selected = ready ? new Set(copies) : new Set(current.filter((n) => n.selected).map((n) => n.id));
      return built.nodes.map((node) => {
        const position = positions.get(node.id);
        const style = { ...(node.style ?? {}) };
        if (node.type === "frame" && position?.width) {
          style.width = position.width;
          style.height = position.height;
        }
        // Stages draw their ports in the order the layout chose.
        const data = node.type === "stage" && position?.inputOrder ? { ...node.data, inputOrder: position.inputOrder, outputOrder: position.outputOrder } : node.data;
        return { ...node, data, position: position ? { x: position.x, y: position.y } : node.position, style, selected: selected.has(node.id) };
      });
    });
    placedBoxes.current = new Map(
      built.nodes
        .filter((n) => !n.parentId && positions.has(n.id))
        .map((n) => {
          const p = positions.get(n.id)!;
          const size = built.sizes.get(n.id) ?? { width: p.width ?? 200, height: p.height ?? 80 };
          return [n.id, { x: p.x, y: p.y, width: p.width ?? size.width, height: p.height ?? size.height }];
        }),
    );
    if (fitPending.current && positions.size) {
      fitPending.current = false;
      const focus = useStudio.getState().focus;
      setTimeout(() => {
        if (focus && flow.getNode(focus.split("|")[0])) focusNode(focus, useStudio.getState().focusSelects);
        else showRecipe();
        useStudio.setState({ focus: null, focusSelects: false });
      }, 80);
    }
  }, [built, positions]); // eslint-disable-line react-hooks/exhaustive-deps

  // Brings a node into view; it becomes the selection when the store selected it.
  const focusNode = useCallback(
    (id: string, select: boolean) => {
      if (select) {
        setNodes((all) => all.map((n) => ({ ...n, selected: n.id === id })));
        useStudio.getState().setSelection([id]);
      }
      // Positions and sizes are known before React Flow measures a new node.
      // Several ids ("a|b") are shown together.
      const boxes = id.split("|").map((part) => {
        const internal = flow.getInternalNode(part);
        if (!internal) return null;
        const { x, y } = internal.internals.positionAbsolute;
        return { x, y, width: internal.measured.width ?? Number(internal.style?.width ?? 200), height: internal.measured.height ?? Number(internal.style?.height ?? 80) };
      }).filter((b): b is { x: number; y: number; width: number; height: number } => b !== null);
      const bounds = container.current?.getBoundingClientRect();
      if (!boxes.length || !bounds) return;
      const x = Math.min(...boxes.map((b) => b.x));
      const y = Math.min(...boxes.map((b) => b.y));
      const width = Math.max(...boxes.map((b) => b.x + b.width)) - x;
      const height = Math.max(...boxes.map((b) => b.y + b.height)) - y;
      // A stage is shown readable (details visible); a frame or a group as a whole.
      const whole = id.startsWith("frame:") || boxes.length > 1;
      const room = whole ? 1.15 : 2.6;
      const fitted = Math.min(1, bounds.width / (width * room), bounds.height / (height * room));
      const zoom = whole ? Math.max(0.3, fitted) : Math.max(0.75, fitted);
      flow.setCenter(x + width / 2, y + height / 2, { zoom, duration: 300 });
    },
    [flow],
  );

  // Opening another recipe starts without a selection.
  const opened = useStudio((s) => s.opened);
  useEffect(() => {
    setNodes((all) => all.map((n) => (n.selected ? { ...n, selected: false } : n)));
    setSelectedEdges(new Set());
  }, [opened]);

  const focus = useStudio((s) => s.focus);
  useEffect(() => {
    if (!focus || requestedLayout.current || fitPending.current || !flow.getNode(focus.split("|")[0])) return;
    focusNode(focus, useStudio.getState().focusSelects);
    useStudio.setState({ focus: null, focusSelects: false });
  }, [focus, nodes, flow, focusNode]);

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
      if (node.type === "slot" || node.type === "ref") return (node.data as { slot: string }).slot;
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
      const problem = shiftProblem(slot, port) ?? connectionProblem(spec, port);
      if (problem) {
        rejection.current = problem;
        return false;
      }
      // A port that accepts any entity still expects the one it reads now: a
      // node port of a block fed with edge features fails deep inside.
      const expected = expectedDomain(built.graph, analysis, catalog, visible, index, port);
      if (spec && expected && expected !== spec.domain) {
        rejection.current = `${visible.localId}.${port.name} reads ${expected} features (as it does now or on the other ${visible.stage.transform} stages); this value is ${spec.domain} features`;
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

  const onConnectStart = useCallback(
    (_: unknown, params: { nodeId: string | null; handleId: string | null; handleType: string | null }) => {
      if (params.handleType !== "source" || !params.nodeId) return;
      const slot = sourceSlot({ source: params.nodeId, sourceHandle: params.handleId, target: "", targetHandle: null });
      if (slot && !slot.startsWith("operator:")) useStudio.setState({ drag: { slot, spec: slotSpec(analysis, catalog, slot) } });
    },
    [analysis, catalog, sourceSlot],
  );

  const onConnectEnd: OnConnectEnd = useCallback(
    (_, state) => {
      useStudio.setState({ drag: null });
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
          const objective = editableObjective(doc, analysis?.objective?.objective);
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
      const replaced = visible.inputs[index];
      const port = displayedInputs(visible.stage, portsOf(visible.stage, pipeline, catalog)).find((p) => p.index === index)?.port;
      edit(
        (doc) => connect(doc, built.graph, slot, visible, index, catalog),
        replaced && replaced !== slot ? `${visible.localId}.${port?.name ?? index} now reads ${slot} instead of ${replaced} (Ctrl+Z undoes)` : undefined,
      );
    },
    [analysis, built, catalog, edit, pipeline, sourceSlot, visibleOf],
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
          const objective = editableObjective(doc, analysis?.objective?.objective);
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
          const objective = editableObjective(doc, analysis?.objective?.objective);
          delete objective.terms[label];
          const tree = objective.expression ? removeLabel(parseExpression(objective.expression), label) : null;
          objective.expression = tree ? printExpression(tree) : "";
          return { ...doc, objective };
        };
      }
      if (node.type === "expr") {
        const path = (node.data as { path: number[] }).path;
        return (doc) => {
          const objective = editableObjective(doc, analysis?.objective?.objective);
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
        let connected: string[] = [];
        const ok = edit((doc) => {
          const result = addStage(doc, transform, catalog, availableSlots(analysis, catalog));
          id = result.id;
          connected = result.connected;
          return result.document;
        });
        if (ok && connected.length) notify(`Added ${transform}, connected ${connected.join(", ")} by name`);
        if (ok) {
          pending.current.set(id, position);
          setPositions((p) => new Map(p).set(`stage:${id}`, position));
          useStudio.setState({ pendingSelection: [`stage:${id}`] });
        }
      }
    },
    [analysis, catalog, edit, flow, notify],
  );

  // Keyboard shortcuts.
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (typingInto(event.target) || useStudio.getState().modal) return;
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
      } else if (mod && event.key.toLowerCase() === "d") {
        event.preventDefault();
        if (event.shiftKey) insertAfterSelection();
        else duplicateSelection();
      } else if (event.key.toLowerCase() === "l" && !mod) {
        useStudio.getState().requestLayout();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  // Copies appear offset from the originals and become the selection.
  const duplicateSelection = useCallback(() => {
    const targets = nodes.filter((n) => n.selected && n.type === "stage").map((n) => visibleOf(n.id)!).filter(Boolean);
    if (!targets.length) {
      notify("Select the stages to duplicate", "error");
      return;
    }
    let keys: string[] = [];
    const ok = edit((doc) => {
      const result = duplicateStages(doc, targets);
      keys = result.keys;
      return result.document;
    }, `Duplicated ${targets.length} stage${targets.length > 1 ? "s" : ""}`);
    if (!ok) return;
    // Past the selection in the flow direction, keeping the copies' arrangement.
    const boxes = targets.map((t) => ({ at: positions.get(`stage:${t.key}`), size: built.sizes.get(`stage:${t.key}`) }));
    const placed = boxes.filter((b) => b.at && b.size) as { at: { x: number; y: number }; size: { width: number; height: number } }[];
    const extent = placed.length
      ? orientation === "TB"
        ? Math.max(...placed.map((b) => b.at.y + b.size.height)) - Math.min(...placed.map((b) => b.at.y))
        : Math.max(...placed.map((b) => b.at.x + b.size.width)) - Math.min(...placed.map((b) => b.at.x))
      : 0;
    targets.forEach((target, i) => {
      const at = boxes[i].at;
      if (!at || target.instance) return;
      pending.current.set(keys[i], orientation === "TB" ? { x: at.x + 40, y: at.y + extent + 80 } : { x: at.x + extent + 80, y: at.y + 40 });
    });
    useStudio.setState({ pendingSelection: keys.map((key) => `stage:${key}`) });
  }, [built, edit, nodes, notify, orientation, positions, visibleOf]);

  // Copies chained after the selection as the next layer; the graph is laid out again.
  const insertAfterSelection = useCallback(() => {
    const targets = nodes.filter((n) => n.selected && n.type === "stage").map((n) => visibleOf(n.id)!).filter(Boolean);
    if (!targets.length) {
      notify("Select the stages of a layer to insert a copy after it", "error");
      return;
    }
    let report: ReturnType<typeof insertAfter> | null = null;
    const ok = edit((doc) => {
      report = insertAfter(doc, targets, (slot) => slotSpec(analysis, catalog, slot));
      return report.document;
    });
    if (!ok || !report) return;
    const { keys, chained, repointed } = report as ReturnType<typeof insertAfter>;
    // Laid out again, with the view on the new layer.
    useStudio.setState({ pendingSelection: keys.map((key) => `stage:${key}`), focus: keys.map((key) => `stage:${key}`).join("|"), focusSelects: false });
    useStudio.getState().requestLayout();
    notify(
      `Inserted ${keys.join(", ")}` +
        (chained.length ? `; they read ${chained.map(([, to]) => to).join(", ")}` : "; no input could be chained, wire them by hand") +
        (repointed.length ? `; ${repointed.join(", ")} now read${repointed.length === 1 ? "s" : ""} the copies` : ""),
    );
  }, [analysis, catalog, edit, nodes, notify, visibleOf]);

  useEffect(() => {
    const w = window as unknown as { studioDuplicate: () => void; studioInsertAfter: () => void };
    w.studioDuplicate = duplicateSelection;
    w.studioInsertAfter = insertAfterSelection;
  }, [duplicateSelection, insertAfterSelection]);

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
    <div className="canvas" ref={container} onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <Finder />
      {built.graph.stages.length === 0 && document && (
        <div className="empty-hint" role="note">
          <b>An empty model.</b> Add stages from the inventory (◧): drag them here, double-click them, or focus one and press Enter. Inputs named like a structure input
          (<span className="mono">edge_lengths</span>, <span className="mono">edge_vectors</span>, <span className="mono">node_type_idx</span>) connect by themselves; choose the
          rest in the inspector. End with a readout to the <b>hamiltonian</b> target. Open a shipped model to start from a working one.
        </div>
      )}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onConnect={onConnect}
        onConnectStart={onConnectStart}
        onConnectEnd={onConnectEnd}
        isValidConnection={isValidConnection}
        onDelete={onDelete}
        onSelectionChange={({ nodes: selected }) => setSelection(selected.map((n) => n.id).sort())}
        onNodeDoubleClick={(_, node) => {
          // A pin leads to the stage that computes its slot.
          const producer = node.type === "ref" ? (node.data as { producer: string | null }).producer : null;
          if (producer) focusNode(producer, true);
        }}
        onNodeClick={(event, node) => {
          // A plain click shows the node in the inspector.
          if (!(event.ctrlKey || event.metaKey || event.shiftKey)) {
            const state = useStudio.getState();
            if (state.panel !== "inspector") state.setPanel("inspector");
            if (!state.showSide) state.togglePanels("side");
            return;
          }
          // Ctrl/Cmd/Shift-click toggles a node in the selection.
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
      >
        <ZoomDetail target={container} />
        <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="#2a3140" />
        <Controls showInteractive={false}>
          <ControlButton onClick={toggleMap} title={showMap ? "Hide the minimap" : "Show the minimap"} aria-label={showMap ? "Hide the minimap" : "Show the minimap"}>
            {showMap ? "▣" : "□"}
          </ControlButton>
        </Controls>
        {showMap && <MiniMap pannable zoomable style={{ width: 150, height: 100 }} nodeColor={(n) => (n.type === "stage" ? "#3a4456" : n.type === "frame" ? "#1e2430" : n.type === "ref" ? "#2a3140" : specColor())} maskColor="rgba(10,12,16,0.7)" />}
      </ReactFlow>
    </div>
  );
}
