# Clausewitz

Answers plain-English questions about the 510 CUAD commercial contracts and cites the exact contract text behind every claim. Each answer goes through these steps:
- hybrid retrieval (BM25 + pgvector) finds candidate chunks;
- a cross-encoder reranks them;
- an LLM answers from the top chunks and must quote the chunk behind each claim;
- a verifier drops any claim whose quote isn't in the contract or doesn't support it.

Design: [`docs/SPEC.md`](docs/SPEC.md). Every design choice and its alternatives: [`docs/decisions.md`](docs/decisions.md).

## Architecture

```
                 ┌──────────── React + TS UI (web/) ────────────┐
                 │ picker · question · answer with citation chips│
                 │ contract viewer highlights start_char:end_char│
                 └───────────────────────┬──────────────────────┘
                                         │ same origin
FastAPI (clausewitz/api) ── POST /query ─┴─ GET /contracts[/{id}] ── /healthz
   │
   ├─ 1 BM25 top 50 ───────────┐  bm25s index (built at ingestion; GCS in prod)
   ├─ 2 vector top 50 ─────────┤  bge-small-en-v1.5 → pgvector (Postgres 16)
   ├─ 3 RRF fusion (k = 60) ◄──┘
   ├─ 4 cross-encoder rerank → top 8   ms-marco-MiniLM-L-6-v2
   ├─ 5 generate: JSON {answer, claims[{text, citations[{chunk_id, quote}]}]}
   ├─ 6 verify: quote is in the chunk (normalized) → exact offsets; LLM support check;
   │            drop unsupported claims; refuse if none survive
   └─ trace: one JSON log line per request (stage timings, tokens, cost, refusals)
```

Each LLM provider sits behind one interface: Anthropic, plus Gemini via Vertex AI or AI Studio. Model IDs come from `GENERATE_MODEL` and `VERIFY_MODEL`, never from code.

| Path | What |
|---|---|
| `clausewitz/ingest/` | CUAD download and parse, dev/test split, heading-aware chunking with exact offsets, embeddings, BM25 index |
| `clausewitz/retrieval/` | BM25, vector, RRF, reranker |
| `clausewitz/llm/` | providers, generation, verifier |
| `clausewitz/pipeline.py` | the six stages, timed |
| `clausewitz/api/`, `clausewitz/tracing.py` | HTTP API, structured request logs |
| `evals/` | cases, metrics, retrieval and full evals, independent judge, CI gate, reports |
| `web/` | review UI |
| `Dockerfile`, `.github/workflows/`, `deploy/` | image, CI, eval and deploy workflows, GCP setup |

## Setup (local)

You need Docker, [uv](https://docs.astral.sh/uv/) and Node 24.

```bash
cp .env.example .env                  # add GEMINI_API_KEY and/or ANTHROPIC_API_KEY and model IDs
docker compose up -d db
uv sync
uv run python -m clausewitz.db        # apply schema
uv run python -m clausewitz.ingest.load_cuad --download
uv run python -m clausewitz.ingest.load_db  # chunks, embeds and indexes 510 contracts

uv run pytest && uv run ruff check .
(cd web && npm ci && npm test && npm run build)
```

Run the app. The API serves `web/dist` at http://localhost:8000:

```bash
uv run uvicorn clausewitz.api.main:app --reload
# or develop the UI with hot reload at http://localhost:5173 (proxies to :8000):
(cd web && npm run dev)
```

Run the app in the container:

```bash
docker build -t clausewitz .
docker run -p 8080:8080 --env-file .env \
  -e DATABASE_URL=postgresql://clausewitz:clausewitz@host.docker.internal:5432/clausewitz \
  -e BM25_INDEX_PATH=/data/bm25 -v "$PWD/indexes/bm25:/data/bm25:ro" clausewitz
```

Deploy to Cloud Run + Cloud SQL: [`docs/deploy.md`](docs/deploy.md).

## Evaluation

```bash
uv run python -m evals.run_eval --split dev --stage retrieval                # recall@k, no LLM
uv run python -m evals.run_eval --split dev --stage full --sample 50          # LLM; resumable
uv run python -m evals.run_eval --split dev --stage retrieval --subset evals/subsets/ci_50.json
```

Each run writes a JSON report to `evals/reports/`. A report holds the git commit, config, every per-case record and the metrics computed from those records. CI (`ci.yml`) runs on every PR:
- lint and tests;
- the UI tests and build;
- the DB-backed tests, including an API smoke test;
- a retrieval gate on `ci_50` that fails if hybrid + rerank recall@8 drops more than 2 points below the `main` baseline.

## Results

Every number below is copied from a report file. Commands, commits and the full tables are in [`docs/results.md`](docs/results.md).

**Retrieval, dev split** (5325 answerable cases, retrieving within the case's contract): [`2026-10-15_dev_retrieval_154c41f_merged8.json`](evals/reports/2026-10-15_dev_retrieval_154c41f_merged8.json), commit `154c41f`.

| Mode | Recall@1 | Recall@3 | Recall@8 |
|---|---|---|---|
| BM25 only | 0.491 | 0.668 | 0.805 |
| Vector only | 0.328 | 0.522 | 0.736 |
| Hybrid (RRF) | 0.446 | 0.640 | 0.811 |
| Hybrid + rerank | 0.482 | 0.698 | 0.856 |
| Random 8 chunks | — | — | 0.418 |

**CI gate baseline** (`ci_50`, 50 answerable dev cases): hybrid + rerank recall@8 = 0.920, from [`2026-11-10_dev_retrieval_5129e94_ci_50.json`](evals/reports/2026-11-10_dev_retrieval_5129e94_ci_50.json), commit `5129e94`.

**Generation and verification:** not measured yet. The dev-split run (verifier off vs on, citation precision, judge-measured ungrounded rate) is in progress on Gemini's free tier. The three-model benchmark on the test split hasn't been run. No LLM numbers will appear here until their report files exist.

## Data

[CUAD v1](https://www.atticusprojectai.org/cuad) by The Atticus Project, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Hendrycks et al., *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021. The data is downloaded at setup time and not stored in this repo.

Clausewitz describes what a contract says; it is not legal advice.
