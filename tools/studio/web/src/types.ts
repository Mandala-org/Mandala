// Shapes of the Studio API and of MANDALA recipe documents.

export type Domain = "node" | "edge" | "bond_neighbor" | "graph" | "relation";

export interface PortInfo {
  name: string;
  domain: string | string[] | null;
  kind: string;
  optional: boolean;
  variadic: boolean;
  doc: string;
}

export interface OptionInfo {
  name: string;
  type: string;
  default: unknown;
  required: boolean;
  key: boolean;
  choices: unknown[];
  inferred: boolean;
  infer_doc: string;
  doc: string;
}

export interface TransformInfo {
  name: string;
  category: string;
  summary: string;
  inputs: PortInfo[];
  outputs: string[];
  options: OptionInfo[];
  pure: boolean;
  described: boolean;
}

export interface ObjectiveTermInfo {
  name: string;
  summary: string;
  category: string;
  options: OptionInfo[];
  /** Options that select a predicted operator: the term's input ports. */
  operators: string[];
}

export interface FeatureSpecJson {
  domain: string;
  representation: string;
  irreps: string | null;
  frame: string;
  width: number | null;
  support: string | null;
  angular_basis: string | null;
  invariant: boolean;
  builtin?: boolean;
  used?: boolean;
}

export interface Catalog {
  transforms: TransformInfo[];
  objective_terms: ObjectiveTermInfo[];
  block_library?: Record<string, LibraryBlock>;
  inputs: Record<string, FeatureSpecJson>;
  systems: Record<string, Record<string, number[]>>;
  /** Device the cost constants were measured on, or false (relative estimates). */
  cost_calibrated?: string | false;
}

export interface StageAnalysis {
  id: string;
  transform: string;
  group: string | null;
  role: string;
  inputs: string[];
  outputs: string[];
  options: Record<string, unknown>;
  explicit: string[];
  inferred: string[];
  static: boolean;
  error: string | null;
  parameters?: number;
  operations?: number;
  seconds?: number;
}

export interface ObjectiveSpec {
  terms: Record<string, Record<string, unknown>>;
  expression?: string | null;
}

export interface Analysis {
  stale?: boolean;
  stages: StageAnalysis[];
  slots: Record<string, FeatureSpecJson>;
  inputs: Record<string, FeatureSpecJson>;
  groups: Record<string, { block: string; options: Record<string, unknown>; explicit: string[]; inputs: string[]; outputs: string[]; stages: string[] }>;
  outputs: Record<string, string>;
  /** Port contracts of the recipe's blocks, as the compiler derives them. */
  blocks?: Record<string, BlockPorts>;
  errors: { stage: string; message: string }[];
  totals: { parameters: number; seconds: number };
  pipeline: Pipeline | null;
  objective: { objective: ObjectiveSpec | null; lowered: boolean; errors: string[]; labels: string[] };
  system: { species_shells: Record<string, number[]>; atoms: number; neighbors: number };
  flat: boolean;
  elapsed: number;
}

export interface Stage {
  id: string;
  transform: string;
  inputs: string[];
  outputs: string[];
  options?: Record<string, unknown>;
  role?: string;
  initialization?: Record<string, unknown>;
  /** false bypasses the stage: its output is one of its inputs (ablation). */
  enabled?: boolean;
}

export interface BlockDef {
  inputs: string[];
  outputs: string[];
  options?: Record<string, unknown>;
  stages: Stage[];
  summary?: string;
  category?: string;
  /** The block library the definition was copied from. */
  library?: string;
  initialization_order?: string[];
}

export interface BlockPorts {
  inputs: PortInfo[];
  outputs: PortInfo[];
}

export interface LibraryBlock extends BlockDef {
  library: string;
  ports: BlockPorts;
}

export interface Pipeline {
  inputs?: Record<string, Record<string, unknown>>;
  blocks?: Record<string, BlockDef>;
  stages: Stage[];
  outputs: Record<string, string>;
  operator_layout?: unknown;
}

export interface RecipeDocument {
  pipeline?: Pipeline | null;
  objective?: ObjectiveSpec | null;
  plugins?: string[];
  [key: string]: unknown;
}

export interface Preset {
  group: string;
  name: string;
  path: string;
  kind: "pipeline" | "flat";
  summary: string;
  plugin: string | null;
  system: string | null;
}

export interface SystemSize {
  atoms: number;
  neighbors: number;
}
