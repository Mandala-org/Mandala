import type { ObjectiveSpec, RecipeDocument } from "../types";

/**
 * The objective to edit: the recipe's own terms, or the framework's lowering
 * of the flat settings, merged into the recipe's objective section so its
 * other keys (reduction, weighting) are kept.
 */
export function editableObjective(document: RecipeDocument, lowered: ObjectiveSpec | null | undefined): ObjectiveSpec {
  const own = (document.objective ?? {}) as Partial<ObjectiveSpec> & Record<string, unknown>;
  const source = own.terms ? own : { ...own, ...(lowered ?? { terms: {}, expression: "" }) };
  return structuredClone(source) as ObjectiveSpec;
}
