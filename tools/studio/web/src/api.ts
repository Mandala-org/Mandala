import type { Analysis, Catalog, Preset, RecipeDocument, SystemSize } from "./types";

async function call<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({ error: response.statusText }));
  if (!response.ok) throw new Error(data.error ?? response.statusText);
  return data as T;
}

export interface SystemChoice {
  shells: Record<string, number[]>;
  size: SystemSize;
}

export const api = {
  catalog: () => call<Catalog>("/api/catalog"),
  presets: () => call<Preset[]>("/api/presets"),
  importPreset: (path: string, system: SystemChoice) =>
    call<{ document: RecipeDocument; analysis: Analysis }>("/api/import", {
      path,
      system: system.shells,
      size: system.size,
    }),
  importYaml: (yaml: string, system: SystemChoice) =>
    call<{ document: RecipeDocument; analysis: Analysis }>("/api/import", {
      yaml,
      system: system.shells,
      size: system.size,
    }),
  analyze: (document: RecipeDocument, system: SystemChoice) =>
    call<Analysis>("/api/analyze", { document, system: system.shells, size: system.size }),
  exportYaml: (document: RecipeDocument, header?: string) =>
    call<{ yaml: string }>("/api/export", { document, header }),
  checkExpression: (request: ExpressionRequest) => call<ExpressionCheck>("/api/expression/check", request),
  createPlugin: (request: ExpressionRequest) =>
    call<{ path: string; code: string; transforms: Catalog["transforms"] }>("/api/plugin", request),
};

export interface ExpressionInput {
  name: string;
  domain: string;
  irreps: string;
}

export interface ExpressionRequest {
  name: string;
  inputs: ExpressionInput[];
  source: string;
  outputs: string[];
  summary: string;
  category: string;
}

export interface ExpressionCheck {
  ok: boolean;
  error?: { message: string; line: number | null; column: number | null };
  types: Record<string, { kind: string; irreps?: string; domain?: string | null; invariant?: boolean; dim?: number }>;
  outputs?: Record<string, { irreps: string; domain: string }>;
  code?: string;
}
