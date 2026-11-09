import { create } from "zustand";
import { api, type SystemChoice } from "./api";
import type { Analysis, Catalog, FeatureSpecJson, Pipeline, Preset, RecipeDocument } from "./types";
import { materialize } from "./graph/edit";
import { blockPortMemory, slotMemory } from "./graph/model";
import { guideSeen } from "./components/Guide";

export type Orientation = "TB" | "LR";

/** Browser storage for per-viewer preferences; unavailable storage is ignored. */
export const preferences = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* private window: the preference lasts for this page only */
    }
  },
  remove(key: string) {
    try {
      localStorage.removeItem(key);
    } catch {
      /* nothing stored */
    }
  },
};

/** Unsaved work kept in this browser, offered again after a crash or reload. */
export interface Autosave {
  name: string;
  document: RecipeDocument;
  system: SystemChoice & { name: string };
  time: number;
}

const AUTOSAVE = "studio.autosave";

export function readAutosave(): Autosave | null {
  try {
    const value = JSON.parse(preferences.get(AUTOSAVE) ?? "null");
    return value && typeof value === "object" && value.document ? (value as Autosave) : null;
  } catch {
    return null;
  }
}

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
  /** The document as last opened or exported; anything else is unsaved. */
  saved: RecipeDocument | null;
  name: string;
  analysis: Analysis | null;
  analyzing: boolean;
  dirtySinceAnalysis: boolean;
  system: SystemChoice & { name: string };
  orientation: Orientation;
  /** Widely shared structure features are drawn as pins next to each use. */
  compactWires: boolean;
  /** Routine inputs (edge relations, batch membership) are not drawn. */
  hideRoutine: boolean;
  setHideRoutine: (hide: boolean) => void;
  expanded: Set<string>;
  selection: string[];
  past: RecipeDocument[];
  future: RecipeDocument[];
  /** What each undo step did, parallel to ``past`` and ``future``. */
  pastLabels: string[];
  futureLabels: string[];
  baseline: Baseline | null;
  panel: Panel;
  layoutRequest: number;
  /** Node to bring into view (after the next requested layout when one is pending). */
  focus: string | null;
  /** Whether the focused node also becomes the selection. */
  focusSelects: boolean;
  /** The value being dragged onto a port, while a wire is drawn. */
  drag: { slot: string; spec?: FeatureSpecJson } | null;
  /** Nodes to select once they appear on the canvas (new or copied stages). */
  pendingSelection: string[];
  toast: { message: string; level: "info" | "error" } | null;
  visibleInputs: Set<string>;
  modal: null | "expression" | "yaml" | "open" | "system" | "group" | "guide";
  /** A dialog opened from the guide returns to it when closed. */
  returnToGuide: boolean;
  groupKeys: string[];
  yamlText: string;
  showInventory: boolean;
  showSide: boolean;
  togglePanels: (which: "inventory" | "side") => void;

  init: () => Promise<void>;
  /** Counts recipes opened, so views can reset what belonged to the previous one. */
  opened: number;
  /** Unsaved work found at start-up, until restored or discarded. */
  recovered: Autosave | null;
  restoreAutosave: () => void;
  discardAutosave: () => void;
  /** Whether unsaved edits may be discarded (asks when there are any). */
  confirmDiscard: () => boolean;
  markSaved: () => void;
  rename: (name: string) => void;
  loadPreset: (preset: Preset) => Promise<void>;
  loadYaml: (yaml: string, name: string) => Promise<void>;
  analyze: () => Promise<void>;
  edit: (change: (document: RecipeDocument, pipeline: Pipeline) => RecipeDocument, label?: string) => boolean;
  undo: () => void;
  redo: () => void;
  setSystem: (system: SystemChoice & { name: string }) => void;
  setOrientation: (orientation: Orientation) => void;
  setCompactWires: (compact: boolean) => void;
  toggleExpanded: (key: string) => void;
  /** Expands the blocks around a stage and brings it into view, selected. */
  reveal: (key: string) => void;
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
  system?: SystemChoice & { name: string };
}

/** The chemical system an exported recipe was edited with, when valid. */
function storedSystem(document: RecipeDocument): (SystemChoice & { name: string }) | null {
  const system = studioSection(document).system;
  if (!system || typeof system !== "object" || !system.shells || typeof system.shells !== "object") return null;
  const valid = Object.values(system.shells).every((ls) => Array.isArray(ls) && ls.every((l) => Number.isInteger(l) && l >= 0 && l <= 4));
  if (!valid || !system.size?.atoms || !system.size?.neighbors) return null;
  return { name: String(system.name ?? "from recipe"), shells: system.shells, size: system.size };
}

function studioSection(document: RecipeDocument): StudioSection {
  const section = document.studio;
  return section && typeof section === "object" ? (section as StudioSection) : {};
}

/** The document with the current editor state stored under ``studio``. */
export function withStudioState(document: RecipeDocument, expanded: Set<string>, orientation: Orientation, system?: SystemChoice & { name: string }): RecipeDocument {
  return { ...document, studio: { orientation, expanded: [...expanded].sort(), layout: layoutMemory.positions, ...(system?.shells ? { system } : {}) } };
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
  saved: null,
  opened: 0,
  recovered: null,
  name: "untitled",
  analysis: null,
  analyzing: false,
  dirtySinceAnalysis: false,
  system: DEFAULT_SYSTEM,
  orientation: preferences.get("studio.orientation") === "LR" ? "LR" : "TB",
  compactWires: preferences.get("studio.compactWires") !== "0",
  hideRoutine: preferences.get("studio.hideRoutine") !== "0",
  expanded: new Set(),
  selection: [],
  past: [],
  future: [],
  pastLabels: [],
  futureLabels: [],
  baseline: null,
  panel: "inspector",
  layoutRequest: 0,
  focus: null,
  focusSelects: false,
  drag: null,
  pendingSelection: [],
  toast: null,
  visibleInputs: new Set(),
  modal: null,
  returnToGuide: false,
  groupKeys: [],
  yamlText: "",
  showInventory: window.innerWidth >= 1280,
  showSide: window.innerWidth >= 760,
  togglePanels: (which) => set(which === "inventory" ? { showInventory: !get().showInventory } : { showSide: !get().showSide }),

  init: async () => {
    const recovered = readAutosave();
    const started = analysisGeneration;
    const [catalog, presets] = await Promise.all([api.catalog(), api.presets()]);
    set({ catalog, presets });
    // A recipe the user opened while the Studio was starting stays open.
    if (analysisGeneration !== started) {
      if (recovered) set({ recovered });
      return;
    }
    // The paper model as an explicit stage recipe: what you see is the recipe.
    const initial = presets.find((p) => p.name === "mandala_paper_blocks") ?? presets.find((p) => p.name === "mandala_paper") ?? presets[0];
    if (initial) await get().loadPreset(initial);
    if (recovered) set({ recovered });
    else if (!guideSeen()) set({ modal: "guide" });
  },

  restoreAutosave: () => {
    const { recovered, document, saved, past } = get();
    if (!recovered) return;
    // Edits made since the page opened stay one undo away.
    const edited = document && document !== saved;
    if (edited && !window.confirm("Replace the changes you made since opening the page with the recovered work? Undo (Ctrl+Z) brings them back.")) return;
    set({
      document: recovered.document,
      saved: null,
      name: recovered.name,
      system: recovered.system,
      past: edited ? [...past, document!] : [],
      pastLabels: edited ? [...get().pastLabels, "changes made before restoring"] : [],
      future: [],
      futureLabels: [],
      recovered: null,
      selection: [],
      ...restoredView(recovered.document, get().orientation),
      layoutRequest: get().layoutRequest + 1,
    });
    get().analyze();
  },

  discardAutosave: () => {
    // Drops the recovered work; edits made since the offer stay saved.
    const { document, saved, name, system } = get();
    if (document && document !== saved) preferences.set(AUTOSAVE, JSON.stringify({ name, document, system, time: Date.now() } satisfies Autosave));
    else preferences.remove(AUTOSAVE);
    set({ recovered: null });
  },

  confirmDiscard: () => {
    const { document, saved, name } = get();
    if (!document || document === saved) return true;
    return window.confirm(`Discard the unsaved changes to ${name}? Export keeps them as YAML.`);
  },

  markSaved: () => set({ saved: get().document }),

  rename: (name) => {
    const clean = name.trim().replace(/\.ya?ml$/, "").replace(/[^\w.-]+/g, "_");
    if (clean) set({ name: clean });
  },

  loadPreset: async (preset) => {
    if (!get().confirmDiscard()) return;
    // A preset names its system, or the backend derives one from the recipe's
    // species; the previous recipe's system never carries over.
    const previous = get().system;
    let system = previous;
    if (preset.system && get().catalog?.systems[preset.system]) {
      system = { ...system, name: preset.system, shells: get().catalog!.systems[preset.system] };
    } else {
      system = { ...system, name: "from recipe", shells: undefined as unknown as SystemChoice["shells"] };
    }
    const generation = ++analysisGeneration;
    clearTimeout(analysisTimer);
    set({ analyzing: true, system, modal: null });
    try {
      const { document, analysis } = await api.importPreset(preset.path, system);
      // Only the most recently requested recipe is opened.
      if (generation !== analysisGeneration) return;
      if (analysis.stale) throw new Error(`Could not analyse ${preset.name}; open it again`);
      if (!system.shells && analysis.system?.species_shells) {
        const shells = analysis.system.species_shells;
        const named = Object.entries(get().catalog?.systems ?? {}).find(
          ([, candidate]) => JSON.stringify(candidate) === JSON.stringify(shells),
        );
        system = { ...system, name: named ? named[0] : "from recipe", shells };
      }
      // Plugins listed by the recipe register their transforms during import.
      if ((document.plugins as unknown[] | undefined)?.length) set({ catalog: await api.catalog() });
      const stored = storedSystem(document);
      const reanalyze = Boolean(stored && JSON.stringify(stored.shells) !== JSON.stringify(system.shells));
      if (stored) system = stored;
      set({
        document,
        saved: document,
        opened: get().opened + 1,
        baseline: null,
        analysis,
        system,
        name: preset.name,
        analyzing: false,
        past: [],
        future: [],
        pastLabels: [],
        futureLabels: [],
        ...restoredView(document, get().orientation),
        selection: [],
        visibleInputs: new Set(),
        layoutRequest: get().layoutRequest + 1,
        dirtySinceAnalysis: false,
        toast: get().toast?.level === "error" ? null : get().toast,
      });
      if (reanalyze) get().analyze();
    } catch (error) {
      set({ analyzing: false, system: previous });
      get().notify((error as Error).message, "error");
    }
  },

  loadYaml: async (yaml, name) => {
    if (!get().confirmDiscard()) return;
    const generation = ++analysisGeneration;
    clearTimeout(analysisTimer);
    set({ analyzing: true, modal: null });
    // A recipe that declares its species defines the system; otherwise the
    // current system stays.
    const declared = /^species:/m.test(yaml);
    let system = get().system;
    try {
      const { document, analysis } = await api.importYaml(yaml, declared ? { ...system, shells: undefined as unknown as SystemChoice["shells"] } : system);
      if (generation !== analysisGeneration) return;
      if (declared && analysis.system?.species_shells) {
        const shells = analysis.system.species_shells;
        const named = Object.entries(get().catalog?.systems ?? {}).find(([, candidate]) => JSON.stringify(candidate) === JSON.stringify(shells));
        system = { ...system, name: named ? named[0] : "from recipe", shells };
      }
      // A recipe exported by the Studio remembers the system it was edited with.
      const stored = storedSystem(document);
      const reanalyze = Boolean(stored && JSON.stringify(stored.shells) !== JSON.stringify(analysis.system?.species_shells));
      if (stored) system = stored;
      if ((document.plugins as unknown[] | undefined)?.length) set({ catalog: await api.catalog() });
      set({
        dirtySinceAnalysis: false,
        document,
        saved: document,
        opened: get().opened + 1,
        baseline: null,
        analysis,
        system,
        name,
        analyzing: false,
        past: [],
        future: [],
        pastLabels: [],
        futureLabels: [],
        visibleInputs: new Set(),
        ...restoredView(document, get().orientation),
        selection: [],
        layoutRequest: get().layoutRequest + 1,
        toast: get().toast?.level === "error" ? null : get().toast,
      });
      if (reanalyze) get().analyze();
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
      if (generation !== analysisGeneration) return;
      // The server skipped this request for a newer one that is not ours to
      // wait for (an import): ask again for the current document.
      if (analysis.stale) {
        clearTimeout(analysisTimer);
        analysisTimer = setTimeout(() => get().analyze(), 100);
        return;
      }
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
    let converted = false;
    if (!base.pipeline) {
      if (!analysis?.pipeline) {
        get().notify("This recipe has no pipeline to edit", "error");
        return false;
      }
      base = materialize(base, analysis.pipeline);
      converted = true;
    }
    let next: RecipeDocument;
    try {
      next = change(base, base.pipeline!);
    } catch (error) {
      get().notify((error as Error).message, "error");
      return false;
    }
    set({
      document: next,
      past: [...get().past.slice(-99), document],
      pastLabels: [...get().pastLabels.slice(-99), label ?? "edit"],
      future: [],
      futureLabels: [],
      dirtySinceAnalysis: true,
    });
    // One notice per edit: a first edit of a flat configuration also says it
    // became an explicit pipeline recipe.
    const notice = label ? (converted ? `${label}; the recipe is now written as stages` : label) : converted ? "The recipe is now written as stages" : "";
    if (notice) get().notify(notice);
    clearTimeout(analysisTimer);
    analysisTimer = setTimeout(() => get().analyze(), 250);
    return true;
  },

  undo: () => {
    const { past, document, future, pastLabels, futureLabels } = get();
    if (!past.length || !document) return;
    const label = pastLabels[pastLabels.length - 1] ?? "edit";
    set({
      document: past[past.length - 1],
      past: past.slice(0, -1),
      pastLabels: pastLabels.slice(0, -1),
      future: [document, ...future],
      futureLabels: [label, ...futureLabels],
    });
    get().notify(`Undid: ${label}`);
    get().analyze();
  },

  redo: () => {
    const { past, document, future, pastLabels, futureLabels } = get();
    if (!future.length || !document) return;
    const label = futureLabels[0] ?? "edit";
    set({
      document: future[0],
      past: [...past, document],
      pastLabels: [...pastLabels, label],
      future: future.slice(1),
      futureLabels: futureLabels.slice(1),
    });
    get().notify(`Redid: ${label}`);
    get().analyze();
  },

  setSystem: (system) => {
    set({ system });
    get().analyze();
  },

  setOrientation: (orientation) => {
    preferences.set("studio.orientation", orientation);
    set({ orientation, layoutRequest: get().layoutRequest + 1 });
  },

  setHideRoutine: (hideRoutine) => {
    preferences.set("studio.hideRoutine", hideRoutine ? "1" : "0");
    set({ hideRoutine, layoutRequest: get().layoutRequest + 1 });
  },

  setCompactWires: (compactWires) => {
    preferences.set("studio.compactWires", compactWires ? "1" : "0");
    set({ compactWires, layoutRequest: get().layoutRequest + 1 });
  },

  toggleExpanded: (key) => {
    const expanded = new Set(get().expanded);
    let focus: string;
    if (expanded.has(key)) {
      for (const item of [...expanded]) if (item === key || item.startsWith(`${key}/`)) expanded.delete(item);
      focus = `stage:${key}`;
    } else {
      expanded.add(key);
      focus = `frame:${key}`;
    }
    set({ expanded, focus, focusSelects: false, selection: [], layoutRequest: get().layoutRequest + 1 });
  },

  reveal: (key) => {
    const expanded = new Set(get().expanded);
    const parts = key.split("/");
    for (let i = 1; i < parts.length; i++) expanded.add(parts.slice(0, i).join("/"));
    const id = `stage:${key}`;
    // A stage already on the canvas is focused without moving anything; the
    // inspector shows it.
    const panels = { panel: "inspector" as const, showSide: true };
    if (expanded.size === get().expanded.size) set({ focus: id, focusSelects: true, ...panels });
    else set({ expanded, focus: id, focusSelects: true, layoutRequest: get().layoutRequest + 1, ...panels });
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
  setModal: (modal, yamlText) => {
    if (modal === null && get().returnToGuide && get().modal !== "guide") {
      set({ modal: "guide", returnToGuide: false });
      return;
    }
    set({ modal, yamlText: yamlText ?? get().yamlText, returnToGuide: modal === "guide" ? false : get().returnToGuide });
  },
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

// Ports of block instances follow the compiler's contracts of the last analysis;
// slot specs accumulate until another recipe is opened.
useStudio.subscribe((state, previous) => {
  blockPortMemory.ports = state.analysis?.blocks ?? {};
  if (state.document !== previous.document && state.document === state.saved) slotMemory.specs = {};
  if (state.analysis !== previous.analysis && state.analysis?.slots) slotMemory.specs = { ...slotMemory.specs, ...state.analysis.slots };
});

// Unsaved work is written to this browser a moment after each edit; saving or
// opening a recipe (nothing unsaved) clears it. While a recovery offer is
// pending, the earlier work is only replaced by newer edits, never cleared.
let autosaveTimer: ReturnType<typeof setTimeout> | undefined;
useStudio.subscribe((state, previous) => {
  if (state.document === previous.document && state.saved === previous.saved) return;
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => {
    const { document, saved, name, system, recovered } = useStudio.getState();
    if (document && document !== saved) preferences.set(AUTOSAVE, JSON.stringify({ name, document, system, time: Date.now() } satisfies Autosave));
    else if (!recovered) preferences.remove(AUTOSAVE);
  }, 800);
});

export function currentPipeline(state: Pick<StudioState, "document" | "analysis">): Pipeline | null {
  return state.document?.pipeline ?? state.analysis?.pipeline ?? null;
}
