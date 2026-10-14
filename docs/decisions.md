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
