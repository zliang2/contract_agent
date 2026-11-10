// Split contract text into segments for highlighting cited spans by character offset.

export interface Span {
  start: number;
  end: number;
  active: boolean;
}

export interface Segment {
  text: string;
  start: number;
  kind: "plain" | "cited" | "active";
}

/** Cut `text` at every span boundary; each piece is "active" if inside the active span,
 * "cited" if inside any other span, else "plain". Overlapping spans are allowed, and the
 * pieces always concatenate back to `text` exactly. */
export function segments(text: string, spans: Span[]): Segment[] {
  const valid = spans
    .map((s) => ({ ...s, start: Math.max(0, s.start), end: Math.min(text.length, s.end) }))
    .filter((s) => s.start < s.end);
  const cuts = new Set([0, text.length]);
  for (const s of valid) {
    cuts.add(s.start);
    cuts.add(s.end);
  }
  const points = [...cuts].sort((a, b) => a - b);
  const out: Segment[] = [];
  for (let i = 0; i + 1 < points.length; i++) {
    const [a, b] = [points[i], points[i + 1]];
    const covering = valid.filter((s) => s.start <= a && b <= s.end);
    const kind = covering.some((s) => s.active) ? "active" : covering.length ? "cited" : "plain";
    const last = out[out.length - 1];
    if (last && last.kind === kind) last.text += text.slice(a, b);
    else out.push({ text: text.slice(a, b), start: a, kind });
  }
  return out;
}
