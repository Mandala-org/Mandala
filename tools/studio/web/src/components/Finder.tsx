import { useEffect, useMemo, useRef, useState } from "react";
import { useStudio } from "../store";
import { useGraph } from "../graph/context";
import { typingInto } from "../graph/model";

interface Candidate {
  key: string; // stage path
  name: string;
  transform: string;
  inside: string | null; // enclosing block instance, when hidden in a collapsed block
}

/** Find a stage by name, transformation or slot (Ctrl+F or /); Enter reveals it. */
export function Finder() {
  const analysis = useStudio((s) => s.analysis);
  const reveal = useStudio((s) => s.reveal);
  const { graph } = useGraph();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (typingInto(event.target) || useStudio.getState().modal) return;
      if (((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") || (event.key === "/" && !event.ctrlKey && !event.metaKey)) {
        event.preventDefault();
        setOpen(true);
        setTimeout(() => input.current?.select(), 0);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const candidates = useMemo(() => {
    const result = new Map<string, Candidate>();
    for (const visible of graph.stages) {
      result.set(visible.key, { key: visible.key, name: visible.localId, transform: visible.stage.transform, inside: null });
    }
    // Stages inside collapsed blocks, from the compiled pipeline.
    for (const stage of analysis?.stages ?? []) {
      if (result.has(stage.id)) continue;
      const cut = stage.id.lastIndexOf("/");
      result.set(stage.id, { key: stage.id, name: stage.id.slice(cut + 1), transform: stage.transform, inside: cut > 0 ? stage.id.slice(0, cut) : null });
    }
    return [...result.values()];
  }, [graph, analysis]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const score = (c: Candidate) => {
      const name = c.name.toLowerCase();
      if (name === q) return 0;
      if (name.startsWith(q)) return 1;
      if (c.key.toLowerCase().includes(q)) return 2;
      if (c.transform.toLowerCase().includes(q)) return 3;
      return 9;
    };
    return candidates
      .map((c) => [score(c), c] as const)
      .filter(([s]) => s < 9)
      .sort((a, b) => a[0] - b[0] || a[1].key.length - b[1].key.length)
      .slice(0, 9)
      .map(([, c]) => c);
  }, [candidates, query]);

  useEffect(() => setActive(0), [query]);

  const pick = (candidate: Candidate | undefined) => {
    if (!candidate) return;
    reveal(candidate.key);
    setOpen(false);
    setQuery("");
  };

  if (!open) {
    return (
      <button className="finder-open" onClick={() => { setOpen(true); setTimeout(() => input.current?.focus(), 0); }} title="Find a stage (Ctrl+F or /)">
        ⌕ Find stage
      </button>
    );
  }
  return (
    <div className="finder" role="search">
      <input
        ref={input}
        autoFocus
        value={query}
        placeholder="Stage, block or transformation…"
        aria-label="Find a stage"
        onChange={(e) => setQuery(e.target.value)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            setOpen(false);
            setQuery("");
          } else if (e.key === "ArrowDown") {
            e.preventDefault();
            setActive((a) => Math.min(a + 1, matches.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((a) => Math.max(a - 1, 0));
          } else if (e.key === "Enter") pick(matches[active]);
        }}
      />
      {query.trim() && (
        <div className="finder-results" role="listbox">
          {matches.length === 0 && <div className="finder-empty">No stage matches “{query}”</div>}
          {matches.map((m, i) => (
            <button key={m.key} role="option" aria-selected={i === active} className={`finder-item ${i === active ? "on" : ""}`} onMouseDown={(e) => e.preventDefault()} onClick={() => pick(m)}>
              <span className="mono">{m.name}</span>
              <span className="muted small">
                {m.transform}
                {m.inside ? ` · in ${m.inside}` : ""}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
