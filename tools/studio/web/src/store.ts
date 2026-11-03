import { create } from "zustand";
import { api, type SystemChoice } from "./api";
import type { Analysis, Catalog, Pipeline, Preset, RecipeDocument } from "./types";
import { materialize } from "./graph/edit";
import { blockPortMemory } from "./graph/model";
import { guideSeen } from "./components/Guide";

export type Orientation = "TB" | "LR";

export interface Baseline {
  name: string;
  document: RecipeDocument;
  analysis: Analysis;
}

export type Panel = "inspector" | "diff" | "problems";

interface StudioState {
  catalog: Catalog | null;
  presets: Preset[];
  document: RecipeDocument | null;
  name: string;
  analysis: Analysis | null;
  analyzing: boolean;
  dirtySinceAnalysis: boolean;
  system: SystemChoice & { name: string };
  orientation: Orientation;
  expanded: Set<string>;
  selection: string[];
  past: RecipeDocument[];
  future: RecipeDocument[];
  baseline: Baseline | null;
  panel: Panel;
  layoutRequest: number;
  toast: { message: string; level: "info" | "error" } | null;
  visibleInputs: Set<string>;
  modal: null | "expression" | "yaml" | "open" | "system" | "group" | "guide";
  groupKeys: string[];
  yamlText: string;
  showInventory: boolean;
  showSide: boolean;
  togglePanels: (which: "inventory" | "side") => void;

  init: () => Promise<void>;
  loadPreset: (preset: Preset) => Promise<void>;
  loadYaml: (yaml: string, name: string) => Promise<void>;
  analyze: () => Promise<void>;
  edit: (change: (document: RecipeDocument, pipeline: Pipeline) => RecipeDocument, label?: string) => boolean;
  undo: () => void;
  redo: () => void;
  setSystem: (system: SystemChoice & { name: string }) => void;
  setOrientation: (orientation: Orientation) => void;
  toggleExpanded: (key: string) => void;
  setSelection: (ids: string[]) => void;
  setPanel: (panel: Panel) => void;
  notify: (message: string, level?: "info" | "error") => void;
  requestLayout: () => void;
  setBaseline: (baseline: Baseline | null) => void;
  setModal: (modal: StudioState["modal"], yamlText?: string) => void;
  askGroup: (keys: string[]) => void;
  showInput: (slot: string) => void;
  refreshCatalog: (transforms?: Catalog["transforms"]) => Promise<void>;
}

const DEFAULT_SYSTEM = { name: "Si s2p2d1", shells: { Si: [0, 0, 1, 1, 2] }, size: { atoms: 64, neighbors: 40 } };

/** Node positions shown on the canvas, written to ``studio.layout`` on export. */
export const layoutMemory: { positions: Record<string, { x: number; y: number }>; restore: Record<string, { x: number; y: number }> | null } = {
  positions: {},
  restore: null,
};

interface StudioSection {
  layout?: Record<string, { x: number; y: number }>;
  expanded?: string[];
  orientation?: Orientation;
}

function studioSection(document: RecipeDocument): StudioSection {
  const section = document.studio;
  return section && typeof section === "object" ? (section as StudioSection) : {};
}

/** The document with the current editor state stored under ``studio``. */
export function withStudioState(document: RecipeDocument, expanded: Set<string>, orientation: Orientation): RecipeDocument {
  return { ...document, studio: { orientation, expanded: [...expanded].sort(), layout: layoutMemory.positions } };
}

function restoredView(document: RecipeDocument, fallback: Orientation) {
  const section = studioSection(document);
  layoutMemory.restore = section.layout && Object.keys(section.layout).length ? section.layout : null;
  return {
    expanded: new Set(section.expanded ?? []),
    orientation: section.orientation === "LR" || section.orientation === "TB" ? section.orientation : fallback,
  };
}

let analysisTimer: ReturnType<typeof setTimeout> | undefined;
let analysisGeneration = 0;

export const useStudio = create<StudioState>((set, get) => ({
  catalog: null,
  presets: [],
  document: null,
  name: "untitled",
  analysis: null,
  analyzing: false,
  dirtySinceAnalysis: false,
  system: DEFAULT_SYSTEM,
  orientation: (localStorage.getItem("studio.orientation") as Orientation) || "TB",
  expanded: new Set(),
  selection: [],
  past: [],
  future: [],
  baseline: null,
  panel: "inspector",
  layoutRequest: 0,
  toast: null,
  visibleInputs: new Set(),
  modal: null,
  groupKeys: [],
  yamlText: "",
  showInventory: window.innerWidth >= 1000,
  showSide: window.innerWidth >= 760,
  togglePanels: (which) => set(which === "inventory" ? { showInventory: !get().showInventory } : { showSide: !get().showSide }),

  init: async () => {
    const [catalog, presets] = await Promise.all([api.catalog(), api.presets()]);
    set({ catalog, presets });
    const initial = presets.find((p) => p.name === "mandala_paper") ?? presets[0];
    if (initial) await get().loadPreset(initial);
    if (!guideSeen()) set({ modal: "guide" });
  },

  loadPreset: async (preset) => {
    // A preset names its system, or the backend derives one from the recipe's
    // species; the previous recipe's system never carries over.
    let system = get().system;
    if (preset.system && get().catalog?.systems[preset.system]) {
      system = { ...system, name: preset.system, shells: get().catalog!.systems[preset.system] };
    } else {
      system = { ...system, name: "from recipe", shells: undefined as unknown as SystemChoice["shells"] };
    }
    analysisGeneration += 1;
    clearTimeout(analysisTimer);
    set({ analyzing: true, system, modal: null });
    try {
      const { document, analysis } = await api.importPreset(preset.path, system);
      if (!system.shells && analysis.system?.species_shells) {
        const shells = analysis.system.species_shells;
        const named = Object.entries(get().catalog?.systems ?? {}).find(
          ([, candidate]) => JSON.stringify(candidate) === JSON.stringify(shells),
        );
        system = { ...system, name: named ? named[0] : "from recipe", shells };
      }
      // Plugins listed by the recipe register their transforms during import.
      if ((document.plugins as unknown[] | undefined)?.length) set({ catalog: await api.catalog() });
      set({
        document,
        analysis,
        name: preset.name,
        analyzing: false,
        past: [],
        future: [],
        ...restoredView(document, get().orientation),
        selection: [],
        visibleInputs: new Set(),
        layoutRequest: get().layoutRequest + 1,
        dirtySinceAnalysis: false,
      });
    } catch (error) {
      set({ analyzing: false });
      get().notify((error as Error).message, "error");
    }
  },

  loadYaml: async (yaml, name) => {
    const generation = ++analysisGeneration;
    clearTimeout(analysisTimer);
    set({ analyzing: true, modal: null });
    try {
      const { document, analysis } = await api.importYaml(yaml, get().system);
      if (generation !== analysisGeneration) return;
      if ((document.plugins as unknown[] | undefined)?.length) set({ catalog: await api.catalog() });
      set({
        dirtySinceAnalysis: false,
        document,
        analysis,
        name,
        analyzing: false,
        past: [],
        future: [],
        visibleInputs: new Set(),
        ...restoredView(document, get().orientation),
        selection: [],
        layoutRequest: get().layoutRequest + 1,
      });
    } catch (error) {
      set({ analyzing: false });
      get().notify((error as Error).message, "error");
    }
  },

  analyze: async () => {
    const document = get().document;
    if (!document) return;
    const generation = ++analysisGeneration;
    set({ analyzing: true });
    try {
      const analysis = await api.analyze(document, get().system);
      if (generation !== analysisGeneration || analysis.stale) return;
      set({ analysis, analyzing: false, dirtySinceAnalysis: false });
    } catch (error) {
      if (generation === analysisGeneration) {
        set({ analyzing: false });
        get().notify((error as Error).message, "error");
      }
    }
  },

  edit: (change, label) => {
    const { document, analysis } = get();
    if (!document) return false;
    let base = document;
    if (!base.pipeline) {
      if (!analysis?.pipeline) {
        get().notify("This recipe has no pipeline to edit", "error");
        return false;
      }
      base = materialize(base, analysis.pipeline);
      get().notify("Converted the flat configuration into an explicit pipeline");
    }
    let next: RecipeDocument;
    try {
      next = change(base, base.pipeline!);
    } catch (error) {
      get().notify((error as Error).message, "error");
      return false;
    }
    set({ document: next, past: [...get().past.slice(-99), document], future: [], dirtySinceAnalysis: true });
    if (label) get().notify(label);
    clearTimeout(analysisTimer);
    analysisTimer = setTimeout(() => get().analyze(), 250);
    return true;
  },

  undo: () => {
    const { past, document, future } = get();
    if (!past.length || !document) return;
    set({ document: past[past.length - 1], past: past.slice(0, -1), future: [document, ...future] });
    get().analyze();
  },

  redo: () => {
    const { past, document, future } = get();
    if (!future.length || !document) return;
    set({ document: future[0], past: [...past, document], future: future.slice(1) });
    get().analyze();
  },

  setSystem: (system) => {
    set({ system });
    get().analyze();
  },

  setOrientation: (orientation) => {
    localStorage.setItem("studio.orientation", orientation);
    set({ orientation, layoutRequest: get().layoutRequest + 1 });
  },

  toggleExpanded: (key) => {
    const expanded = new Set(get().expanded);
    if (expanded.has(key)) {
      for (const item of [...expanded]) if (item === key || item.startsWith(`${key}/`)) expanded.delete(item);
    } else expanded.add(key);
    set({ expanded, layoutRequest: get().layoutRequest + 1 });
  },

  setSelection: (selection) => {
    const current = get().selection;
    if (current.length === selection.length && current.every((id, i) => id === selection[i])) return;
    set({ selection });
  },
  setPanel: (panel) => set({ panel }),
  notify: (message, level = "info") => {
    set({ toast: { message, level } });
    setTimeout(() => {
      if (get().toast?.message === message) set({ toast: null });
    }, level === "error" ? 6000 : 2500);
  },
  requestLayout: () => set({ layoutRequest: get().layoutRequest + 1 }),
  setBaseline: (baseline) => set({ baseline, panel: baseline ? "diff" : get().panel }),
  setModal: (modal, yamlText) => set({ modal, yamlText: yamlText ?? get().yamlText }),
  askGroup: (groupKeys) => set({ groupKeys, modal: "group" }),
  showInput: (slot) => {
    const visibleInputs = new Set(get().visibleInputs);
    visibleInputs.add(slot);
    set({ visibleInputs });
  },
  refreshCatalog: async (transforms) => {
    if (transforms && get().catalog) {
      set({ catalog: { ...get().catalog!, transforms } });
      return;
    }
    set({ catalog: await api.catalog() });
  },
}));

// Ports of block instances follow the compiler's contracts of the last analysis.
useStudio.subscribe((state) => {
  blockPortMemory.ports = state.analysis?.blocks ?? {};
});

export function currentPipeline(state: Pick<StudioState, "document" | "analysis">): Pipeline | null {
  return state.document?.pipeline ?? state.analysis?.pipeline ?? null;
}
