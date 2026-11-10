// Two-pane review UI (docs/SPEC.md §9). Left: contract picker, question, cited answer.
// Right: contract text; clicking a citation chip scrolls to and highlights its exact
// character span (start_char/end_char index into the contract's full text).

import { useEffect, useMemo, useRef, useState } from "react";
import {
  type Citation,
  type Contract,
  type ContractSummary,
  type QueryResponse,
  getContract,
  listContracts,
  query,
} from "./api";
import { segments } from "./highlight";

const ALL = "all";

interface Numbered extends Citation {
  n: number; // 1-based chip number, unique across the answer
}

export default function App() {
  const [contracts, setContracts] = useState<ContractSummary[]>([]);
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<string>(ALL);
  const [viewer, setViewer] = useState<Contract | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [active, setActive] = useState<Numbered | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    listContracts().then(setContracts, (e) => setError(String(e)));
  }, []);

  // Load the contract to view: the selected one, or the one a clicked chip cites.
  const viewId = active?.contract_id ?? (selected === ALL ? null : Number(selected));
  useEffect(() => {
    if (viewId === null || viewer?.id === viewId) return;
    getContract(viewId).then(setViewer, (e) => setError(String(e)));
  }, [viewId, viewer?.id]);

  const numbered: Numbered[][] = useMemo(() => {
    let n = 0;
    return (result?.claims ?? []).map((c) => c.citations.map((x) => ({ ...x, n: ++n })));
  }, [result]);

  const parts = useMemo(() => {
    if (!viewer) return [];
    const spans = numbered
      .flat()
      .filter((x) => x.contract_id === viewer.id)
      .map((x) => ({ start: x.start_char, end: x.end_char, active: x.n === active?.n }));
    return segments(viewer.text, spans);
  }, [viewer, numbered, active]);

  useEffect(() => {
    activeRef.current?.scrollIntoView?.({ block: "center", behavior: "smooth" });
  }, [active, viewer]);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setActive(null);
    try {
      setResult(await query(question, selected === ALL ? null : Number(selected)));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  const shown = contracts.filter((c) => c.title.toLowerCase().includes(filter.toLowerCase()));
  let firstActive = true;

  return (
    <div className="app">
      <section className="left">
        <h1>Clausewitz</h1>
        <p className="sub">Ask about a CUAD contract. Every claim cites the contract text.</p>
        <form onSubmit={ask}>
          <label htmlFor="filter">Contract</label>
          <input
            id="filter"
            placeholder="Filter titles…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <select
            aria-label="Contract"
            value={selected}
            onChange={(e) => {
              setSelected(e.target.value);
              setActive(null);
              setResult(null);
              if (e.target.value === ALL) setViewer(null);
            }}
          >
            <option value={ALL}>All contracts (search the corpus)</option>
            {shown.map((c) => (
              <option key={c.id} value={c.id}>
                {c.title} ({c.split})
              </option>
            ))}
          </select>
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            rows={3}
            placeholder="e.g. Governing Law: which state's law governs the contract?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "Answering…" : "Ask"}
          </button>
        </form>

        {error && <div className="note error">{error}</div>}

        {result && (
          <div className="answer" aria-live="polite">
            {result.refused ? (
              <div className="note refused" role="status">
                <strong>Refused.</strong> {result.answer}
                {result.dropped_claims > 0 &&
                  ` (${result.dropped_claims} generated claim${result.dropped_claims === 1 ? "" : "s"} failed verification.)`}
              </div>
            ) : (
              <>
                <p>{result.answer}</p>
                <ol className="claims">
                  {result.claims.map((c, i) => (
                    <li key={i}>
                      {c.text}{" "}
                      {numbered[i].map((x) => (
                        <button
                          key={x.n}
                          type="button"
                          className={`chip${active?.n === x.n ? " on" : ""}`}
                          aria-label={`Citation ${x.n}`}
                          title={x.quote}
                          onClick={() => setActive(x)}
                        >
                          {x.n}
                        </button>
                      ))}
                    </li>
                  ))}
                </ol>
                {result.dropped_claims > 0 && (
                  <div className="note dropped" role="status">
                    {result.dropped_claims} claim{result.dropped_claims === 1 ? " was" : "s were"}{" "}
                    dropped because no citation could be verified against the contract.
                  </div>
                )}
              </>
            )}
            <p className="trace">trace {result.trace_id}</p>
          </div>
        )}
        <p className="disclaimer">Describes what the contract says; not legal advice.</p>
      </section>

      <section className="right" aria-label="Contract text">
        {viewer ? (
          <>
            <h2>{viewer.title}</h2>
            <div className="contract">
              {parts.map((p) => {
                if (p.kind === "plain") return <span key={p.start}>{p.text}</span>;
                const isFirst = p.kind === "active" && firstActive;
                if (isFirst) firstActive = false;
                return (
                  <mark key={p.start} className={p.kind} ref={isFirst ? activeRef : undefined}>
                    {p.text}
                  </mark>
                );
              })}
            </div>
          </>
        ) : (
          <p className="empty">Pick a contract, or click a citation to open the contract it cites.</p>
        )}
      </section>
    </div>
  );
}
