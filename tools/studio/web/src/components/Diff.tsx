import { useMemo, useState } from "react";
import { useStudio } from "../store";
import { formatCount, formatSeconds } from "../graph/model";
import type { Analysis, StageAnalysis } from "../types";
import { api } from "../api";

export interface DiffRow {
  id: string;
  status: "added" | "removed" | "changed";
  transform: string;
  details: string[];
  parameters: number;
  seconds: number;
}

function optionsDiff(before: StageAnalysis, after: StageAnalysis): string[] {
  const details: string[] = [];
  if (before.transform !== after.transform) details.push(`transform ${before.transform} → ${after.transform}`);
  const keys = new Set([...Object.keys(before.options), ...Object.keys(after.options)]);
  for (const key of keys) {
    const a = JSON.stringify(before.options[key]);
    const b = JSON.stringify(after.options[key]);
    if (a !== b) details.push(`${key}: ${a ?? "—"} → ${b ?? "—"}`);
  }
  if (before.inputs.join() !== after.inputs.join()) details.push(`inputs ${before.inputs.join(", ")} → ${after.inputs.join(", ")}`);
  if ((before.parameters ?? 0) !== (after.parameters ?? 0)) details.push(`weights ${formatCount(before.parameters ?? 0)} → ${formatCount(after.parameters ?? 0)}`);
  return details;
}

export function diffAnalyses(before: Analysis, after: Analysis): DiffRow[] {
  const old = new Map(before.stages.map((s) => [s.id, s]));
  const now = new Map(after.stages.map((s) => [s.id, s]));
  const rows: DiffRow[] = [];
  for (const stage of after.stages) {
    const previous = old.get(stage.id);
    if (!previous) rows.push({ id: stage.id, status: "added", transform: stage.transform, details: [], parameters: stage.parameters ?? 0, seconds: stage.seconds ?? 0 });
    else {
      const details = optionsDiff(previous, stage);
      if (details.length)
        rows.push({ id: stage.id, status: "changed", transform: stage.transform, details, parameters: (stage.parameters ?? 0) - (previous.parameters ?? 0), seconds: (stage.seconds ?? 0) - (previous.seconds ?? 0) });
    }
  }
  for (const stage of before.stages) {
    if (!now.has(stage.id)) rows.push({ id: stage.id, status: "removed", transform: stage.transform, details: [], parameters: -(stage.parameters ?? 0), seconds: -(stage.seconds ?? 0) });
  }
  return rows;
}

export function useDiff() {
  const baseline = useStudio((s) => s.baseline);
  const analysis = useStudio((s) => s.analysis);
  return useMemo(() => {
    const marks = new Map<string, "added" | "changed">();
    if (!baseline || !analysis) return { rows: [] as DiffRow[], marks };
    const rows = diffAnalyses(baseline.analysis, analysis);
    for (const row of rows) {
      if (row.status === "removed") continue;
      const parts = row.id.split("/");
      for (let i = 1; i <= parts.length; i++) {
        const key = parts.slice(0, i).join("/");
        const existing = marks.get(key);
        marks.set(key, i === parts.length ? row.status : existing === "added" ? "added" : "changed");
      }
    }
    return { rows, marks };
  }, [baseline, analysis]);
}

function signed(value: number, format: (v: number) => string) {
  if (!value) return "0";
  return `${value > 0 ? "+" : "−"}${format(Math.abs(value))}`;
}

export function DiffPanel() {
  const { baseline, analysis, document, name, presets, system, setBaseline, notify } = useStudio();
  const { rows } = useDiff();
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(false);

  const compareWith = async (path: string) => {
    if (!path) return;
    setLoading(true);
    try {
      const preset = presets.find((p) => p.path === path)!;
      const result = await api.importPreset(path, system);
      setBaseline({ name: preset.name, document: result.document, analysis: result.analysis });
    } catch (error) {
      notify((error as Error).message, "error");
    } finally {
      setLoading(false);
    }
  };

  if (!baseline) {
    return (
      <div className="panel-body">
        <p className="muted">
          Compare the current recipe with a baseline: differences in stages, options, weights and estimated time are listed here and marked on the graph.
        </p>
        <button className="button primary" disabled={!analysis || !document} onClick={() => analysis && document && setBaseline({ name: `${name} (snapshot)`, document: structuredClone(document), analysis })}>
          Use the current recipe as baseline
        </button>
        <label className="field">
          <span>…or compare with a preset</span>
          <select value="" onChange={(e) => compareWith(e.target.value)} disabled={loading}>
            <option value="">choose a recipe</option>
            {presets.map((p) => (
              <option key={p.path} value={p.path}>
                {p.group} / {p.name}
              </option>
            ))}
          </select>
        </label>
      </div>
    );
  }
  const totals = analysis?.totals ?? { parameters: 0, seconds: 0 };
  const shown = rows.filter((r) => !filter || r.id.includes(filter) || r.transform.includes(filter));
  const counts = { added: rows.filter((r) => r.status === "added").length, removed: rows.filter((r) => r.status === "removed").length, changed: rows.filter((r) => r.status === "changed").length };
  return (
    <div className="panel-body">
      <div className="diff-head">
        <div>
          baseline <b>{baseline.name}</b>
        </div>
        <button className="button small" onClick={() => setBaseline(null)}>
          clear
        </button>
      </div>
      <div className="diff-summary">
        <div>
          <span className="muted">weights</span> {formatCount(totals.parameters)} <span className="delta">{signed(totals.parameters - baseline.analysis.totals.parameters, formatCount)}</span>
        </div>
        <div>
          <span className="muted">time</span> {formatSeconds(totals.seconds)} <span className="delta">{signed(totals.seconds - baseline.analysis.totals.seconds, formatSeconds)}</span>
        </div>
        <div>
          <span className="diff-added">+{counts.added}</span> <span className="diff-removed">−{counts.removed}</span> <span className="diff-changed">~{counts.changed}</span> stages
        </div>
      </div>
      <input className="search" placeholder="filter stages" value={filter} onChange={(e) => setFilter(e.target.value)} />
      <div className="diff-rows">
        {shown.length === 0 && <p className="muted">No differences.</p>}
        {shown.map((row) => (
          <div key={`${row.status}:${row.id}`} className={`diff-row diff-${row.status}`}>
            <div className="diff-row-head">
              <span className="diff-status">{row.status === "added" ? "+" : row.status === "removed" ? "−" : "~"}</span>
              <span className="diff-id">{row.id}</span>
              <span className="muted">{row.transform}</span>
            </div>
            {row.details.map((d) => (
              <div key={d} className="diff-detail">
                {d}
              </div>
            ))}
            <div className="diff-detail muted">
              {signed(row.parameters, formatCount)} weights · {signed(row.seconds, formatSeconds)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
