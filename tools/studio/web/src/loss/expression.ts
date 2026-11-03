// The objective expression language: numbers, term labels, + - * /, unary
// minus, parentheses and sqrt log exp abs min max. Parsed into a tree that the
// editor draws as blocks and prints back to text.

export type Expr =
  | { type: "num"; value: number }
  | { type: "name"; name: string }
  | { type: "neg"; operand: Expr }
  | { type: "bin"; op: "+" | "-" | "*" | "/"; left: Expr; right: Expr }
  | { type: "call"; fn: string; args: Expr[] };

export const FUNCTIONS: Record<string, number> = { sqrt: 1, log: 1, exp: 1, abs: 1, min: 2, max: 2 };

type Token = { kind: "num" | "name" | "op" | "lp" | "rp" | "comma"; text: string; pos: number };

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  const re = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|([+\-*/])|(\()|(\))|(,))/y;
  let pos = 0;
  while (pos < text.length) {
    if (/^\s*$/.test(text.slice(pos))) break;
    re.lastIndex = pos;
    const m = re.exec(text);
    if (!m) throw new Error(`Unexpected character at ${pos + 1}: ${text.slice(pos).trim()[0]}`);
    const start = m.index + m[0].length - (m[1] ?? m[2] ?? m[3] ?? m[4] ?? m[5] ?? m[6]).length;
    if (m[1]) tokens.push({ kind: "num", text: m[1], pos: start });
    else if (m[2]) tokens.push({ kind: "name", text: m[2], pos: start });
    else if (m[3]) tokens.push({ kind: "op", text: m[3], pos: start });
    else if (m[4]) tokens.push({ kind: "lp", text: "(", pos: start });
    else if (m[5]) tokens.push({ kind: "rp", text: ")", pos: start });
    else tokens.push({ kind: "comma", text: ",", pos: start });
    pos = re.lastIndex;
  }
  return tokens;
}

export function parseExpression(text: string): Expr {
  const tokens = tokenize(text);
  let i = 0;
  const peek = () => tokens[i];
  const expect = (kind: Token["kind"]) => {
    const t = tokens[i];
    if (!t || t.kind !== kind) throw new Error(`Expected ${kind === "rp" ? ")" : kind} ${t ? `at ${t.pos + 1}` : "at the end"}`);
    i += 1;
    return t;
  };
  const primary = (): Expr => {
    const t = peek();
    if (!t) throw new Error("Unexpected end of expression");
    if (t.kind === "op" && (t.text === "-" || t.text === "+")) {
      i += 1;
      const operand = primary();
      return t.text === "-" ? { type: "neg", operand } : operand;
    }
    if (t.kind === "num") {
      i += 1;
      return { type: "num", value: Number(t.text) };
    }
    if (t.kind === "name") {
      i += 1;
      if (peek()?.kind === "lp") {
        if (!(t.text in FUNCTIONS)) throw new Error(`Unknown function ${t.text}`);
        i += 1;
        const args: Expr[] = [additive()];
        while (peek()?.kind === "comma") {
          i += 1;
          args.push(additive());
        }
        expect("rp");
        if (args.length !== FUNCTIONS[t.text]) throw new Error(`${t.text} takes ${FUNCTIONS[t.text]} argument(s)`);
        return { type: "call", fn: t.text, args };
      }
      return { type: "name", name: t.text };
    }
    if (t.kind === "lp") {
      i += 1;
      const inner = additive();
      expect("rp");
      return inner;
    }
    throw new Error(`Unexpected ${t.text} at ${t.pos + 1}`);
  };
  const multiplicative = (): Expr => {
    let left = primary();
    while (peek()?.kind === "op" && (peek()!.text === "*" || peek()!.text === "/")) {
      const op = tokens[i++].text as "*" | "/";
      left = { type: "bin", op, left, right: primary() };
    }
    return left;
  };
  const additive = (): Expr => {
    let left = multiplicative();
    while (peek()?.kind === "op" && (peek()!.text === "+" || peek()!.text === "-")) {
      const op = tokens[i++].text as "+" | "-";
      left = { type: "bin", op, left, right: multiplicative() };
    }
    return left;
  };
  if (!tokens.length) throw new Error("Empty expression");
  const tree = additive();
  if (i < tokens.length) throw new Error(`Unexpected ${tokens[i].text} at ${tokens[i].pos + 1}`);
  return tree;
}

const PRECEDENCE = { "+": 1, "-": 1, "*": 2, "/": 2 } as const;

function formatNumber(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) < 1e6) return value.toFixed(1);
  const text = String(value);
  return text.length > 10 ? value.toExponential(3).replace(/\.?0+e/, "e") : text;
}

export function printExpression(expr: Expr, parent = 0, right = false): string {
  switch (expr.type) {
    case "num":
      return formatNumber(expr.value);
    case "name":
      return expr.name;
    case "neg":
      return `-${printExpression(expr.operand, 3)}`;
    case "call":
      return `${expr.fn}(${expr.args.map((a) => printExpression(a)).join(", ")})`;
    case "bin": {
      const p = PRECEDENCE[expr.op];
      const text = `${printExpression(expr.left, p)} ${expr.op} ${printExpression(expr.right, p, true)}`;
      const wrap = p < parent || (right && p === parent && (expr.op === "-" || expr.op === "/" || parent === 2));
      return wrap ? `(${text})` : text;
    }
  }
}

export function labelsOf(expr: Expr, into = new Set<string>()): Set<string> {
  switch (expr.type) {
    case "name":
      into.add(expr.name);
      break;
    case "neg":
      labelsOf(expr.operand, into);
      break;
    case "bin":
      labelsOf(expr.left, into);
      labelsOf(expr.right, into);
      break;
    case "call":
      expr.args.forEach((a) => labelsOf(a, into));
      break;
  }
  return into;
}

/** Remove every occurrence of ``label`` (as a summand or a factor). */
export function removeLabel(expr: Expr, label: string): Expr | null {
  switch (expr.type) {
    case "name":
      return expr.name === label ? null : expr;
    case "num":
      return expr;
    case "neg": {
      const operand = removeLabel(expr.operand, label);
      return operand ? { type: "neg", operand } : null;
    }
    case "call": {
      const args = expr.args.map((a) => removeLabel(a, label));
      return args.some((a) => a === null) ? null : { ...expr, args: args as Expr[] };
    }
    case "bin": {
      const left = removeLabel(expr.left, label);
      const right = removeLabel(expr.right, label);
      if (expr.op === "+" || expr.op === "-") {
        if (!left && !right) return null;
        if (!left) return expr.op === "-" ? { type: "neg", operand: right! } : right;
        if (!right) return left;
        return { ...expr, left, right };
      }
      return left && right ? { ...expr, left, right } : null;
    }
  }
}

/** Child expressions in drawing order, with their path suffixes. */
export function childrenOf(expr: Expr): Expr[] {
  switch (expr.type) {
    case "neg":
      return [expr.operand];
    case "bin":
      return [expr.left, expr.right];
    case "call":
      return expr.args;
    default:
      return [];
  }
}

export function replaceAt(expr: Expr, path: number[], replacement: Expr | null): Expr | null {
  if (!path.length) return replacement;
  const [head, ...rest] = path;
  const children = childrenOf(expr).slice();
  const updated = replaceAt(children[head], rest, replacement);
  if (expr.type === "neg") return updated ? { type: "neg", operand: updated } : null;
  if (expr.type === "bin") {
    if (!updated) return head === 0 ? (expr.op === "-" ? { type: "neg", operand: expr.right } : expr.right) : expr.left;
    return head === 0 ? { ...expr, left: updated } : { ...expr, right: updated };
  }
  if (expr.type === "call") {
    if (!updated) return null;
    children[head] = updated;
    return { ...expr, args: children };
  }
  return expr;
}

export function getAt(expr: Expr, path: number[]): Expr {
  return path.reduce((node, index) => childrenOf(node)[index], expr);
}
