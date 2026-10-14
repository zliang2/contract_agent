# Clausewitz — Technical Spec

Clausewitz answers plain-English questions about commercial contracts and cites the exact clause behind every claim. When it cannot find support for a claim, it says so instead of guessing.

**Corpus:** CUAD v1 (The Atticus Project) — 510 commercial contracts from public SEC EDGAR filings, annotated by lawyers across 41 clause types. The annotations double as the evaluation answer key.

**How to use this spec:** build it one phase at a time (see "Build plan"). Each phase ends with acceptance checks. Do not start the next phase until the current phase's checks pass and the human has reviewed the code.

---

## 1. Goals and non-goals

**Goals**
- Every claim in an answer cites a character span `(contract_id, start_char, end_char)` in the source contract.
- Unsupported claims are dropped or refused, never shown as fact.
- Retrieval and answer quality are measured on a held-out test split, and measurement runs in CI.
- Latency, token usage, tokens/sec and cost are traced for every request.
- Runs on GCP: Cloud Run for compute, Cloud SQL (PostgreSQL + pgvector) for storage.

**Non-goals (v1)**
- Legal advice, drafting or redlining.
- User-uploaded contracts (corpus is fixed to CUAD).
- User accounts or multi-tenant access control.

---

## 2. Tech stack (defaults — change only with a note in `docs/decisions.md`)

| Concern | Choice |
| --- | --- |
| Language / deps | Python 3.12, `uv` for dependency management |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + pgvector (local: `pgvector/pgvector:pg16` Docker image; prod: Cloud SQL) |
| DB driver | `psycopg` 3 (+ `pgvector` Python adapter) |
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers` (384-dim, runs on CPU) |
| BM25 | `bm25s`, index built in-app (Cloud SQL has no true BM25; `ts_rank` is not BM25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| LLM | Provider interface with at least two implementations: Anthropic (Claude API) and Google (Gemini via `google-genai` / Vertex AI). Model IDs come from env vars, never hard-coded. |
| Frontend | React + TypeScript (Vite) |
| Tests | `pytest`; `ruff` for lint/format |
| CI/CD | GitHub Actions |
| Deploy | Docker → Artifact Registry → Cloud Run; Cloud SQL Python Connector; Secret Manager for keys; Cloud Storage for the BM25 index |

Environment variables (see `.env.example`): `DATABASE_URL`, `ANTHROPIC_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GENERATE_MODEL`, `VERIFY_MODEL`, `BM25_INDEX_PATH`, `PRICE_TABLE_PATH`.

---

## 3. Repository layout

```
clausewitz/
  CLAUDE.md
  README.md
  .env.example
  docker-compose.yml          # local Postgres + pgvector
  Dockerfile
  pyproject.toml
  docs/
    SPEC.md                   # this file
    decisions.md              # every non-obvious design choice + why
    results.md                # eval results, each with command, commit, date
  clausewitz/                 # Python package
    config.py
    db.py
    schema.sql
    ingest/
      load_cuad.py
      chunk.py
      embed.py
      bm25_index.py
      load_db.py
    retrieval/
      bm25.py
      vector.py
      fusion.py
      rerank.py
    llm/
      providers.py            # Provider interface + Anthropic + Gemini
      generate.py
      verify.py
    pipeline.py               # orchestrates the 6 stages
    tracing.py
    pricing.py
    api/
      main.py
  evals/
    build_cases.py
    metrics.py
    run_eval.py
    subsets/ci_50.json
    reports/                  # JSON outputs of eval runs
  web/                        # React + TS app
  tests/
  .github/workflows/
    ci.yml
    deploy.yml
```

---

## 4. Data model

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE contracts (
  id          SERIAL PRIMARY KEY,
  title       TEXT NOT NULL,
  split       TEXT NOT NULL CHECK (split IN ('dev', 'test')),
  full_text   TEXT NOT NULL
);

CREATE TABLE chunks (
  id            SERIAL PRIMARY KEY,
  contract_id   INT NOT NULL REFERENCES contracts(id),
  section_path  TEXT,                 -- e.g. "12.3 Termination"
  start_char    INT NOT NULL,
  end_char      INT NOT NULL,
  text          TEXT NOT NULL,
  embedding     vector(384) NOT NULL
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks (contract_id);

CREATE TABLE annotations (            -- CUAD gold spans
  id           SERIAL PRIMARY KEY,
  contract_id  INT NOT NULL REFERENCES contracts(id),
  clause_type  TEXT NOT NULL,
  start_char   INT NOT NULL,
  end_char     INT NOT NULL
);
```

**Invariant (tested):** for every chunk, `contracts.full_text[start_char:end_char] == chunks.text`. Citations, verification and UI highlighting all depend on this.

---

## 5. Ingestion (offline job)

1. **Load** CUAD v1. Before writing the loader, locate the official release (The Atticus Project; GitHub `TheAtticusProject/cuad`) and inspect the actual file format — it is SQuAD-style JSON with one context per contract and one question per clause type. Do not assume field names; read them from the file.
2. **Split by contract** into dev (80%) and test (20%) with a fixed seed, before anything else. No test contract may influence tuning.
3. **Chunk at clause boundaries** using numbered sections and headings. Fall back to ~400-token windows with ~50-token overlap for long or unstructured sections. Keep `section_path`.
4. **Embed** each chunk (`bge-small-en-v1.5`, normalized vectors).
5. **Build the BM25 index** over the same chunks with `bm25s`; save to disk (prod: Cloud Storage). Keep a mapping from BM25 doc index → `chunks.id`.
6. **Load** into Postgres, one transaction per contract. Also load gold spans into `annotations`.

Run as: `uv run python -m clausewitz.ingest.load_db`

---

## 6. Query pipeline

Each question runs through six stages. Every stage is timed and traced. A question may target one contract (`contract_id`) or the whole corpus.

1. **Keyword retrieval** — BM25 top 50 (filtered to `contract_id` if given). Catches exact legal terms embeddings blur.
2. **Vector retrieval** — embed the question; pgvector top 50 by cosine distance.
3. **Fusion** — Reciprocal Rank Fusion, `k = 60`: `score = Σ 1 / (k + rank)` across both lists.
4. **Rerank** — cross-encoder scores each (question, chunk) pair; keep top 8.
5. **Generate** — the LLM receives the question and the 8 chunks, each labeled with its chunk id. It must return JSON matching this schema (validate with Pydantic; retry once on invalid JSON):
   ```json
   {
     "answer": "string",
     "claims": [
       {"text": "string", "citations": [{"chunk_id": 0, "quote": "exact text copied from the chunk"}]}
     ]
   }
   ```
6. **Verify** — for each citation:
   - **Quote check:** after whitespace and quote-character normalization, `quote` must be a substring of the cited chunk's text. Map the match back to absolute `start_char`/`end_char` in the contract.
   - **Support check:** a short LLM call (`VERIFY_MODEL`) answers whether the quote supports the claim (yes/no + one-line reason).
   - A claim survives only if it has at least one citation passing both checks. Drop the rest.
   - If no claim survives → refusal: `"I couldn't find support for this in the contract."`

**Response**
```json
{
  "answer": "string",
  "claims": [
    {"text": "string",
     "citations": [{"chunk_id": 0, "contract_id": 0, "start_char": 0, "end_char": 0, "quote": "string"}]}
  ],
  "refused": false,
  "dropped_claims": 0,
  "trace_id": "string"
}
```

The pipeline takes a `verify: bool` flag so evals can measure with the verifier off and on.

---

## 7. Evaluation

**Test cases:** each CUAD (contract, clause type) pair becomes a case. Use CUAD's question text for that clause type. Cases where the contract has no gold span for that clause type are "should refuse" cases.

**Metrics** (`evals/metrics.py`)

| Metric | Definition |
| --- | --- |
| Recall@k | Share of answerable cases where ≥1 gold span overlaps a chunk in the top k (report k = 8 after rerank; also report each retrieval mode alone) |
| Citation precision | Share of cited spans that overlap a gold span |
| Ungrounded answer rate | Share of answers containing ≥1 claim that fails verification. With the verifier OFF, run verification in audit-only mode to count failures without dropping claims |
| Refusal accuracy | On "should refuse" cases, share of answers that refuse |
| Latency | p50 / p95 end-to-end and per stage |
| Cost per query | Tokens × prices from `PRICE_TABLE_PATH` |

Because the verifier also measures ungroundedness, hand-label a random sample of 30 answers and report agreement with the automatic check in `docs/results.md`.

**Model benchmark:** run the same pipeline on the test split with three generator models (at least one Claude, one Gemini); only the model changes. Output one comparison table: accuracy (span overlap), ungrounded rate, p95 latency, cost/query.

**Run:** `uv run python -m evals.run_eval --split {dev|test} --stage {retrieval|full} [--no-verify] [--model ID] [--subset evals/subsets/ci_50.json]`. Each run writes a JSON report to `evals/reports/` containing the git commit, date, config and all metrics.

**Rule:** numbers in `README.md` and `docs/results.md` must be copied from a report file, with the command and commit that produced them. Never estimate or invent a metric.

**CI (`ci.yml`):**
- Every PR: `ruff`, `pytest`, and the retrieval-only eval on `ci_50.json` (no LLM cost). Fail if recall@8 drops more than 2 points below the value stored for `main`.
- Manual dispatch / nightly: the full eval on `ci_50.json` with the LLM (needs API secrets).

---

## 8. API

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Liveness for Cloud Run |
| `GET /contracts?split=` | List contracts (id, title) |
| `GET /contracts/{id}` | Full text for the viewer |
| `POST /query` | Body `{question, contract_id?}` → response in §6 |

The API also serves the built React app as static files (one Cloud Run service, no CORS setup).

## 9. UI

Two panes. Left: contract picker, question box, answer with numbered citation chips after each claim, and a visible note when claims were dropped or the answer was refused. Right: contract text; clicking a chip scrolls to and highlights the cited span using its character offsets.

---

## 10. Deployment and observability

- **Cloud Run:** single service from the Dockerfile. Bake the embedding and reranker models into the image. Load the BM25 index from Cloud Storage at startup. Set min instances = 1 to avoid cold-start model loading on requests.
- **Cloud SQL:** PostgreSQL 16 with the `vector` extension; connect via the Cloud SQL Python Connector.
- **Secrets:** API keys and DB credentials in Secret Manager, injected as env vars. Never commit secrets or bake them into the image.
- **deploy.yml:** on merge to `main`, build → push to Artifact Registry → deploy to Cloud Run.

**Tracing (`tracing.py`):** every request gets a `trace_id` and emits one structured JSON log line (Cloud Logging picks up stdout JSON) with:
- per-stage durations (bm25, vector, fusion, rerank, generate, verify) and total
- input/output tokens per model call
- generation tokens/sec (output tokens ÷ generation time)
- cost per request (from token counts and the price table)
- chunks retrieved, claims generated, claims dropped, refused (bool)

Build a log-based dashboard for p50/p95 latency, cost/request and refusal rate.

---

## 11. Build plan

Work one phase at a time. At the end of each phase: run the acceptance checks, commit, write a short summary of what was built and any decisions (append decisions to `docs/decisions.md`), then stop and wait for review.

### Phase 0 — Scaffold
- Repo layout from §3, `pyproject.toml`, `docker-compose.yml`, `.env.example`, `ruff` + `pytest` config, `schema.sql`.
- **Done when:** `docker compose up -d db` starts Postgres; `CREATE EXTENSION vector` succeeds; `uv run pytest` and `uv run ruff check` pass.

### Phase 1 — Ingestion
- §5 end to end.
- **Done when:** 510 contracts loaded with a fixed dev/test split; a test asserts the chunk-offset invariant for every chunk; annotations loaded; the job prints chunk count and average chunk length. The BM25 index file exists and maps to `chunks.id`.

### Phase 2 — Retrieval
- BM25, vector, RRF fusion, rerank (§6 stages 1–4). `run_eval --stage retrieval`.
- **Done when:** on the dev split, the eval prints recall@8 for BM25-only, vector-only, hybrid, and hybrid + rerank. Results recorded in `docs/results.md` with command and commit. Unit tests cover RRF.

### Phase 3 — Generation and verification
- Provider interface (Anthropic + Gemini), structured output, verifier (§6 stages 5–6), `verify` flag.
- **Done when:** unit tests cover quote normalization, substring matching and offset mapping (including a deliberately paraphrased quote that must fail); dev-split eval reports ungrounded rate with verifier off vs. on and citation precision; 30 hand-labeled answers compared with the automatic check.

### Phase 4 — Eval harness, benchmark and CI
- Full metrics (§7), JSON reports, `ci_50.json` subset, `ci.yml`, three-model benchmark on the test split.
- **Done when:** CI passes on a PR; a deliberately broken retriever makes CI fail; the benchmark table is in `docs/results.md`, copied from report files.

### Phase 5 — API and UI
- §8 and §9.
- **Done when:** `POST /query` works locally against Docker Postgres; clicking a citation chip highlights the exact span; refusals display correctly; a smoke test hits every endpoint.

### Phase 6 — Deploy and tracing
- §10.
- **Done when:** the Cloud Run URL serves the UI and answers a query; `/healthz` is green; a request's log line contains every tracing field; `README.md` covers architecture, setup, and results (copied from reports).
