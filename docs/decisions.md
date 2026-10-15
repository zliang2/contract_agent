# Design decisions

Each entry: what was decided, why, and alternatives considered.

## Phase 0 — Scaffold

### Dependencies added per phase, not up front
- **What:** `pyproject.toml` only lists what Phase 0 uses (psycopg, pgvector, pydantic). ML libraries (sentence-transformers, bm25s), LLM SDKs and FastAPI arrive in the phase that first uses them.
- **Why:** sentence-transformers pulls in PyTorch (~GBs). Keeping Phase 0 light makes CI fast and keeps each phase's diff honest about what it needs.
- **Alternatives:** declare everything now — simpler once, but slows every CI run and hides which phase needs what.

### Idempotent schema (`IF NOT EXISTS`) with named indexes
- **What:** `schema.sql` can be applied any number of times; indexes have explicit names.
- **Why:** ingestion, tests and Cloud Run startup can all call `init_schema()` safely. Named indexes are easy to drop/rebuild (e.g., rebuilding HNSW after bulk load).
- **Alternatives:** a migration tool (Alembic) — overkill for three fixed tables in v1.

### DB tests skip, not fail, when Postgres is down
- **What:** tests marked `db` skip if `DATABASE_URL` is unreachable; CI runs a pgvector service container so they always run there.
- **Why:** pure unit tests stay runnable on any laptop; CI still enforces the DB checks.
- **Alternatives:** always require Docker locally — stricter, but blocks quick test runs.

### Python pinned to 3.12
- **What:** `requires-python = ">=3.12,<3.13"`, `.python-version` = 3.12.
- **Why:** spec says 3.12; PyTorch wheels lag new Python releases, so 3.13+ risks install failures in Phase 1.

### Read vectors back via `Vector.to_numpy()`
- **What:** pgvector's Python adapter (0.5.x) returns `pgvector.Vector` objects, not numpy arrays; code that needs arrays calls `.to_numpy()`. `numpy` is a direct dependency because the adapter no longer installs it.
- **Why:** found when the Phase 0 round-trip test failed against real Postgres. Writes still accept numpy arrays directly.
- **Alternatives:** pin an older adapter that returned arrays — avoids one call, but pins us to stale code.

### Local Docker via Docker Desktop
- **What:** Docker Desktop on macOS for the local Postgres container.
- **Why:** this Mac runs macOS 14, where Homebrew no longer ships bottles; Colima's `lima` dependency compiled from source for too long. Docker Desktop installs prebuilt. Prod doesn't use it (Cloud SQL).
- **Alternatives:** Colima (headless, open source; workable on a supported macOS), Postgres.app + pgvector built manually.

## Phase 1 — Ingestion

### Source: Zenodo CUAD_v1.json, not the GitHub `data.zip`
- **What:** download the official CUAD v1 release from Zenodo (record 4595826, MD5-verified) and use `CUAD_v1.json`.
- **Why:** it holds all 510 contracts in one file. The GitHub `data.zip` ships the authors' own train/test split for model fine-tuning; the spec requires our own seeded split.
- **Checked before writing the loader:** 510 docs × 1 context × 41 questions; clause type = question-id suffix, always equal to the quoted name in the question; `is_impossible` always equals "no answers"; all 13,823 answer offsets reproduce their text exactly.

### `context` from the JSON is `full_text` (not the TXT files)
- **Why:** gold answer offsets are defined against the JSON `context`. Using the TXT files would require re-aligning every span. CUAD's README warns PDF→TXT conversion left inconsistent spacing, so nothing downstream may assume clean layout.

### Split: sort titles, shuffle with seed 42, first 20% = test
- **What:** 408 dev / 102 test, stored in `contracts.split`.
- **Why:** sorting first makes the split depend only on the seed and the set of titles, not file order (tested). Done before chunking/embedding so nothing about test contracts informs tuning.
- **Alternatives:** stratify by contract type (from titles) — more balanced, but titles are noisy labels; plain random is easier to defend and 102 contracts is a reasonable sample.

### Chunker: heading-aware sections, greedy packing, token windows as fallback
- **What:** paragraphs → sections (heading = "12.3 Title", "ARTICLE/SECTION n", or short ALL-CAPS line) → pack whole consecutive sections up to 400 tokens → split a single section >400 tokens into 400-token windows with 50-token overlap.
- **Why:** packing keeps clause boundaries (never cuts a clause unless it alone exceeds the budget) while avoiding tiny chunks for one-line clauses. Chunks are defined *only* as offsets and their text is sliced from `full_text`, so the offset invariant can't drift.
- **Measured on CUAD (all 510):** 18,332 chunks, mean 324 / median 385 bge tokens, max 400 (under bge's 512 limit, so nothing is silently truncated at embedding time); 93% have a section path; 31 contracts have no detectable headings and fall back to windows; 496/13,823 gold spans (3.6%) cross a chunk boundary (gold spans longer than a chunk exist; recall counts overlap, not containment).
- **Alternatives:** fixed windows only (simpler, but cuts clauses mid-sentence); one chunk per section (many chunks of a few tokens, and huge sections exceed the embedder's limit); LLM-based segmentation (costly, non-deterministic).

### Token counts use the embedding model's own tokenizer
- **What:** "400 tokens" means bge-small WordPiece tokens; the chunker takes a `token_spans` function so tests use a whitespace tokenizer and don't download a model.
- **Why:** the budget exists because of the embedder's 512-token limit, so it should be measured in its tokens.

### Section path label for packed chunks: "first … last"
- **What:** e.g. `1.1 Grant and Acceptance … 1.2 License`.
- **Why:** short enough for the UI, still tells a reader which clauses are inside.

### BM25: bm25s with English stopwords + Snowball stemming; built *after* DB insert
- **What:** index saved to `BM25_INDEX_PATH` with `chunk_mapping.json` (`chunk_ids[i]`, `contract_ids[i]` for BM25 doc `i`).
- **Why:** stemming lets "terminate" match "termination" (tested). The spec lists BM25 before the DB load, but the mapping needs `chunks.id`, which exists only after insert. Storing `contract_ids` lets Phase 2 filter to one contract without a DB round-trip.
- **Alternatives:** Postgres `ts_rank` (not BM25 — no length normalization or IDF saturation); no stemming (misses inflections common in legal text).

### Ingestion is a full rebuild
- **What:** `TRUNCATE … RESTART IDENTITY`, then insert contracts sorted by title, one transaction per contract.
- **Why:** with sorted input and a seeded split, reruns reproduce the same ids, so the BM25 mapping and any saved eval reports stay comparable. Incremental updates aren't needed: the corpus is fixed.

### CPU-only PyTorch on Linux
- **What:** `[tool.uv.sources]` pins `torch` to the PyTorch CPU index on Linux.
- **Why:** default Linux wheels bundle multi-GB CUDA libraries that CI and Cloud Run (CPU) never use. The lock has zero `nvidia-*` packages.

## Phase 2 — Retrieval

### Eval scope: retrieve within the case's contract
- **What:** every eval case passes its `contract_id`; retrieval only ranks that contract's chunks. Corpus-wide retrieval is supported (`contract_id=None`) but not what recall@k measures.
- **Why:** every CUAD question is about one specific contract ("this contract"), and the UI picks a contract before asking. Corpus-wide search would mostly measure whether retrieval finds the right *contract*, which is a different problem.
- **Consequence:** 88 of 510 contracts have ≤ 8 chunks, where top-8 is the whole contract and recall@8 is trivially 1. So reports also give an exact random-8 baseline and recall@8 restricted to contracts with > 8 chunks.
- **Alternatives:** corpus-wide cases (harder, but not the product's use case).

### BM25 contract filter: score all chunks, mask, then take top k
- **Why:** `bm25s.get_scores` returns scores for all 18k chunks in about a millisecond, so masking by contract before the top-k cut is exact and cheap. Retrieving a global top-k and filtering afterward would return fewer than k (or zero) chunks for most contracts. IDF stays corpus-wide (510 contracts), which is more stable than per-contract IDF over ~36 chunks.
- **Zero-score chunks are dropped**, so BM25 may return fewer than k; RRF handles short lists.

### Vector contract filter: exact scan via a MATERIALIZED CTE; HNSW only for corpus-wide
- **Why:** pgvector applies `WHERE` after its approximate HNSW search, so a filtered query over one contract's ~36 chunks out of 18k can return fewer than k rows or miss true neighbors. pgvector 0.8 iterative scans mitigate this but still approximate. A contract has 1–253 chunks, so exact distance is cheap and always correct (tested against brute force).
- **Alternatives:** `hnsw.iterative_scan` (approximate); per-contract partial indexes (510 indexes).

### bge query instruction prefix
- **What:** questions are embedded as `"Represent this sentence for searching relevant passages: " + question`; chunks have no prefix.
- **Why:** this is how bge-v1.5 is documented for query→passage retrieval.
- **Alternatives:** no prefix (the model card says v1.5 works without it, with slightly lower scores). Untested here — candidate for an ablation.

### RRF with k = 60, ties broken by chunk id
- **Why:** RRF uses only ranks, so BM25 scores and cosine similarities need no normalization; k = 60 is the value from Cormack et al. (2009) and the spec. Breaking ties by id makes results deterministic, which the eval relies on.
- **Alternatives:** weighted score fusion (needs score normalization and a tuned weight).

### Rerank the top 50 fused candidates, keep 8
- **Why:** the spec fixes 50 per retriever and 8 final, but not the rerank pool size. 50 bounds cross-encoder cost (about 150 ms median per question on a laptop) while covering a typical contract (36 chunks on average). The report includes recall@50 of the fused pool, which is the upper bound on what reranking can reach.

### Eval runs the pipeline once per case and reads every mode from it
- **What:** one `rerank`-mode run per case; BM25-only, vector-only and hybrid top-k are read from its per-stage rankings.
- **Why:** each stage is deterministic, so this equals running the modes separately at a quarter of the cost.

### Reports record git commit and dirty flag
- **Why:** CLAUDE.md requires every quoted number to trace to a commit. Reports from a dirty tree or with `--limit` get suffixes in the filename and are not quoted.

### Full evals run on GitHub Actions, sharded 8 ways
- **What:** `eval-retrieval.yml` (manual trigger) ingests CUAD once, dumps the database, restores it into 8 parallel jobs that each run `run_eval --shard I/8`, and merges the shard reports with `evals.merge_reports`.
- **Why:** the full dev eval (~5,400 cross-encoder reranks) is 15–25 minutes of sustained GPU load, which overheated the development laptop (a fanless MacBook Air). The repo is public, so hosted runners are free. Runners are CPU-only, so the work is split into 8 shards to keep wall-clock time reasonable. One shared dump keeps every shard on identical data and chunk ids.
- **Reports store per-case records** (the first relevant rank per mode, chunk counts, timings), and every metric is computed from those records. So merged metrics equal single-run metrics (checked: identical apart from latency), and anyone can recompute a number from the report.
- **Latency caveat:** latencies in CI reports are from CPU runners, not a laptop or Cloud Run.
- **Alternatives:** run locally in batches with cool-down pauses (slow, manual); evaluate a random sample (noisier); a cloud GPU VM (costs money, more setup).

### Phase 2 result: recorded as measured, reranker kept pending review
- **What:** the first full dev eval showed hybrid + rerank (recall@8 0.731) below hybrid without it (0.759) and BM25 alone (0.771). The pipeline still defaults to `rerank` mode, as the spec says.
- **Why not change it now:** changing the production path is a design decision for review, and any fix should be measured on dev before it is adopted. Options are listed in `docs/results.md`.

### Eval questions: "clause type: details" instead of CUAD's annotation template (spec deviation)
- **What:** `build_cases` rewrites each CUAD question from `Highlight the parts (if any) of this contract related to "X" that should be reviewed by a lawyer. Details: D` to `X: D` (all 41 match the template; tested). `--question-style cuad_template` keeps the verbatim text. Spec §7 says "use CUAD's question text"; this keeps its content and drops the shared wrapper.
- **Why:** the template is the instruction the lawyers saw while annotating, not a question a user of this system would type. The first dev run (verbatim template) showed the ms-marco reranker, trained on short search queries, lowering recall. The rewrite is applied identically to every retrieval mode and to every later stage, so the comparison stays fair. The verbatim-template results stay in `docs/results.md`.
- **Alternatives:** keep the template (measures robustness to an input users won't send); rewrite only the reranker's query (hides the issue inside one stage); hand-write new questions (no longer CUAD's wording, and hard to defend as unbiased).

### Cache the ingested database in Actions
- **What:** the ingest job caches `db.dump`, the BM25 index and CUAD JSON, keyed on a hash of the ingestion code, schema, settings and `uv.lock`.
- **Why:** ingestion takes about 42 minutes on a CPU runner and only changes when that code does. The key ensures a stale database is never reused after an ingestion change.
