import { useEffect, useMemo, useRef, useState } from "react";
import { useStudio } from "../store";
import { api, type ExpressionCheck, type ExpressionInput } from "../api";
import { groupStages } from "../graph/edit";
import { GuideModal } from "./Guide";

function Modal({ title, children, onClose, wide }: { title: string; children: React.ReactNode; onClose: () => void; wide?: boolean }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal ${wide ? "wide" : ""}`}>
        <div className="modal-head">
          <span>{title}</span>
          <button className="link" onClick={onClose}>
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function OpenModal() {
  const { presets, loadPreset, loadYaml, setModal } = useStudio();
  const [yaml, setYaml] = useState("");
  const file = useRef<HTMLInputElement>(null);
  const groups = [...new Set(presets.map((p) => p.group))];
  return (
    <Modal title="Open a recipe" onClose={() => setModal(null)} wide>
      <div className="open-grid">
        <div>
          {groups.map((group) => (
            <div key={group} className="preset-group">
              <div className="preset-group-title">{group}</div>
              {presets
                .filter((p) => p.group === group)
                .map((p) => (
                  <button key={p.path} className="preset" onClick={() => loadPreset(p)}>
                    <span className="preset-name">{p.name}</span>
                    <span className={`chip ${p.kind === "flat" ? "" : "chip-blue"}`}>{p.kind}</span>
                    {p.summary && <span className="preset-summary">{p.summary}</span>}
                  </button>
                ))}
            </div>
          ))}
        </div>
        <div className="open-yaml">
          <div className="muted">Paste a recipe or load a YAML file</div>
          <textarea className="mono" value={yaml} onChange={(e) => setYaml(e.target.value)} placeholder="pipeline:&#10;  stages: …" rows={18} />
          <div className="button-row">
            <input ref={file} type="file" accept=".yaml,.yml" style={{ display: "none" }} onChange={async (e) => {
              const f = e.target.files?.[0];
              if (f) loadYaml(await f.text(), f.name.replace(/\.ya?ml$/, ""));
            }} />
            <button className="button" onClick={() => file.current?.click()}>
              Load file…
            </button>
            <button className="button primary" disabled={!yaml.trim()} onClick={() => loadYaml(yaml, "pasted recipe")}>
              Open pasted YAML
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
}

function YamlModal() {
  const { yamlText, setModal, name, notify } = useStudio();
  const download = () => {
    const blob = new Blob([yamlText], { type: "text/yaml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.yaml`;
    a.click();
  };
  return (
    <Modal title="Recipe YAML" onClose={() => setModal(null)} wide>
      <p className="muted small">Only values set in the recipe are written; inferred and default values stay implicit.</p>
      <textarea className="mono yaml-view" readOnly value={yamlText} rows={26} />
      <div className="button-row">
        <button className="button" onClick={() => navigator.clipboard.writeText(yamlText).then(() => notify("Copied"))}>
          Copy
        </button>
        <button className="button primary" onClick={download}>
          Download {name}.yaml
        </button>
      </div>
    </Modal>
  );
}

function SystemModal() {
  const { catalog, system, setSystem, setModal } = useStudio();
  const [text, setText] = useState(
    Object.entries(system.shells ?? {})
      .map(([s, l]) => `${s}: ${l.join(" ")}`)
      .join("\n"),
  );
  const [atoms, setAtoms] = useState(system.size.atoms);
  const [neighbors, setNeighbors] = useState(system.size.neighbors);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState(system.name);
  const apply = () => {
    try {
      const shells: Record<string, number[]> = {};
      for (const line of text.split("\n").filter((l) => l.trim())) {
        const [species, rest] = line.split(":");
        const ls = (rest ?? "").trim().split(/[\s,]+/).filter(Boolean).map((v) => ("spdfg".includes(v) ? "spdfg".indexOf(v) : Number(v)));
        if (!species.trim() || !ls.length || ls.some((l) => !Number.isInteger(l) || l < 0 || l > 4)) throw new Error(`Cannot read "${line}": write "Si: 0 0 1 1 2" or "Si: s s p p d"`);
        shells[species.trim()] = ls;
      }
      setSystem({ name, shells, size: { atoms, neighbors } });
      setModal(null);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <Modal title="Chemical system" onClose={() => setModal(null)}>
      <p className="muted small">Species and their orbital shells define the operator layout (readout irreps); the reference size scales the cost estimates.</p>
      <label className="field">
        <span>preset</span>
        <select
          value={catalog?.systems[name] ? name : ""}
          onChange={(e) => {
            const shells = catalog?.systems[e.target.value];
            if (!shells) return;
            setName(e.target.value);
            setText(Object.entries(shells).map(([s, l]) => `${s}: ${l.join(" ")}`).join("\n"));
          }}
        >
          <option value="">custom</option>
          {Object.keys(catalog?.systems ?? {}).map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>species: shells (l values or s p d f)</span>
        <textarea className="mono" rows={4} value={text} onChange={(e) => {
          setText(e.target.value);
          setName("custom");
        }} />
      </label>
      <div className="field-row">
        <label className="field">
          <span>atoms</span>
          <input type="number" min={1} value={atoms} onChange={(e) => setAtoms(Number(e.target.value))} />
        </label>
        <label className="field">
          <span>neighbours per atom</span>
          <input type="number" min={1} value={neighbors} onChange={(e) => setNeighbors(Number(e.target.value))} />
        </label>
      </div>
      {error && <div className="field-error">{error}</div>}
      <div className="button-row">
        <button className="button primary" onClick={apply}>
          Apply
        </button>
      </div>
    </Modal>
  );
}

const EXAMPLE = `h = linear(x, "8x0e+4x1o")
out = h * sigmoid(mlp(norm(h), [16], 12)) + x`;

function ExpressionModal() {
  const { setModal, refreshCatalog, notify, edit, document } = useStudio();
  const [name, setName] = useState("gated_mix");
  const [summary, setSummary] = useState("Linear mix gated by an MLP of copy norms, with a skip");
  const [category, setCategory] = useState("custom");
  const [inputs, setInputs] = useState<ExpressionInput[]>([{ name: "x", domain: "node", irreps: "8x0e+4x1o" }]);
  const [source, setSource] = useState(EXAMPLE);
  const [outputs, setOutputs] = useState("out");
  const [check, setCheck] = useState<ExpressionCheck | null>(null);
  const [showCode, setShowCode] = useState(false);
  const request = useMemo(() => ({ name, inputs, source, outputs: outputs.split(",").map((o) => o.trim()).filter(Boolean), summary, category }), [name, inputs, source, outputs, summary, category]);
  useEffect(() => {
    // Replies to superseded requests are ignored, so a slow check never
    // overwrites the result for the text currently in the editor.
    let current = true;
    const timer = setTimeout(
      () =>
        api
          .checkExpression(request)
          .then((result) => current && setCheck(result))
          .catch((e) => current && setCheck({ ok: false, error: { message: e.message, line: null, column: null }, types: {} })),
      250,
    );
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [request]);
  const create = async () => {
    try {
      const result = await api.createPlugin(request);
      await refreshCatalog(result.transforms);
      if (document && !(document.plugins ?? []).includes(result.path)) edit((doc) => ({ ...doc, plugins: [...(doc.plugins ?? []), result.path] }));
      notify(`Registered ${name} from ${result.path}; drag it from the inventory`);
      setModal(null);
    } catch (e) {
      notify((e as Error).message, "error");
    }
  };
  const errorLine = check?.error?.line;
  return (
    <Modal title="New transformation from an expression" onClose={() => setModal(null)} wide>
      <div className="expr-grid">
        <div>
          <div className="field-row">
            <label className="field">
              <span>name</span>
              <input className="mono" value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <label className="field">
              <span>category</span>
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                {["custom", "nonlinearity", "message", "mlp", "linear", "combine", "residual", "normalization", "readout", "envelope"].map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
          </div>
          <label className="field">
            <span>summary</span>
            <input value={summary} onChange={(e) => setSummary(e.target.value)} />
          </label>
          <div className="field">
            <span>inputs</span>
            {inputs.map((input, i) => (
              <div key={i} className="input-row">
                <input className="mono" value={input.name} onChange={(e) => setInputs(inputs.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} />
                <select value={input.domain} onChange={(e) => setInputs(inputs.map((x, j) => (j === i ? { ...x, domain: e.target.value } : x)))}>
                  <option>node</option>
                  <option>edge</option>
                  <option>graph</option>
                </select>
                <input className="mono grow" value={input.irreps} onChange={(e) => setInputs(inputs.map((x, j) => (j === i ? { ...x, irreps: e.target.value } : x)))} />
                <button className="link" onClick={() => setInputs(inputs.filter((_, j) => j !== i))}>
                  ✕
                </button>
              </div>
            ))}
            <button className="link" onClick={() => setInputs([...inputs, { name: `y${inputs.length}`, domain: inputs[0]?.domain ?? "node", irreps: "4x0e" }])}>
              + input
            </button>
          </div>
          <label className="field">
            <span>definition (one assignment per line)</span>
            <div className="code-editor">
              <div className="gutter">
                {source.split("\n").map((_, i) => (
                  <div key={i} className={errorLine === i + 1 ? "error-line" : ""}>
                    {i + 1}
                  </div>
                ))}
              </div>
              <textarea className="mono" spellCheck={false} rows={8} value={source} onChange={(e) => setSource(e.target.value)} />
            </div>
          </label>
          <label className="field">
            <span>outputs</span>
            <input className="mono" value={outputs} onChange={(e) => setOutputs(e.target.value)} />
          </label>
        </div>
        <div className="expr-side">
          <div className={`check ${check?.ok ? "ok" : "bad"}`}>{check?.ok ? "✓ equivariant and well typed" : check?.error ? `${check.error.line ? `line ${check.error.line}: ` : ""}${check.error.message}` : "checking…"}</div>
          <div className="types">
            {inputs.map((i) => (
              <div key={`in-${i.name}`} className="type-row">
                <span className="mono">{i.name}</span>
                <span className="muted">{i.domain}</span>
                <span className="mono">{i.irreps}</span>
              </div>
            ))}
            {Object.entries(check?.types ?? {}).map(([n, t]) => (
              <div key={n} className={`type-row ${request.outputs.includes(n) ? "is-output" : ""}`}>
                <span className="mono">{n}</span>
                <span className="muted">{t.domain ?? t.kind}</span>
                <span className="mono">{t.irreps}</span>
              </div>
            ))}
          </div>
          <details className="dsl-help">
            <summary>Language</summary>
            <ul>
              <li><code>linear(x, "irreps")</code>, <code>tp(x, y, "irreps")</code></li>
              <li><code>norm(x)</code> copy norms, <code>scalars(x)</code> 0e channels</li>
              <li><code>silu sigmoid tanh relu gelu softplus exp log sqrt abs</code> on invariants</li>
              <li><code>x * s</code>, <code>gate(x, s)</code>: scale by an invariant with 1 or one-per-copy channels</li>
              <li><code>mlp(s, [64, 64], width)</code>, <code>concat(a, b)</code>, <code>param("alpha", 0.1)</code></li>
            </ul>
          </details>
          {check?.code && (
            <button className="link" onClick={() => setShowCode(!showCode)}>
              {showCode ? "▾" : "▸"} generated plugin
            </button>
          )}
          {showCode && <pre className="code-preview">{check?.code}</pre>}
          <div className="button-row">
            <button className="button primary" disabled={!check?.ok} onClick={create}>
              Create plugins/{name}.py
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
}

function GroupModal() {
  const { groupKeys, setModal, edit, document } = useStudio();
  const existing = new Set(Object.keys(document?.pipeline?.blocks ?? {}));
  let suggestion = "my_block";
  for (let n = 1; existing.has(suggestion); n++) suggestion = `my_block_${n}`;
  const [name, setName] = useState(suggestion);
  const [summary, setSummary] = useState("");
  const valid = /^[a-z][a-z0-9_]*$/.test(name) && !existing.has(name);
  const apply = () => {
    if (!valid) return;
    const ok = edit((doc) => {
      const next = groupStages(doc, groupKeys, name);
      if (summary) next.pipeline!.blocks![name].summary = summary;
      return next;
    }, `Grouped ${groupKeys.length} stages into ${name}`);
    if (ok) setModal(null);
  };
  return (
    <Modal title={`Group ${groupKeys.length} stages into a block`} onClose={() => setModal(null)}>
      <p className="muted small">
        The selection becomes a reusable block definition in <span className="mono">pipeline.blocks</span> and one instance of it. Values crossing the selection boundary become the block's inputs and outputs.
      </p>
      <div className="mono small">{groupKeys.join(", ")}</div>
      <label className="field">
        <span>block name</span>
        <input className="mono" autoFocus value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && apply()} />
      </label>
      {!valid && <div className="field-error">{existing.has(name) ? "A block with this name exists" : "Use lowercase letters, digits and underscores"}</div>}
      <label className="field">
        <span>summary (optional)</span>
        <input value={summary} onChange={(e) => setSummary(e.target.value)} />
      </label>
      <div className="button-row">
        <button className="button primary" disabled={!valid} onClick={apply}>
          Group
        </button>
      </div>
    </Modal>
  );
}

export function Modals() {
  const modal = useStudio((s) => s.modal);
  if (modal === "open") return <OpenModal />;
  if (modal === "yaml") return <YamlModal />;
  if (modal === "system") return <SystemModal />;
  if (modal === "expression") return <ExpressionModal />;
  if (modal === "group") return <GroupModal />;
  if (modal === "guide") return <GuideModal />;
  return null;
}
