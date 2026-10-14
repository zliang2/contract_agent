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
