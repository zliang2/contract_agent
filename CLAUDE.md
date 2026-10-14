# Clausewitz

Contract-review AI agent over the CUAD corpus: hybrid retrieval (BM25 + pgvector), cross-encoder reranking, cited answers, and a verifier that drops unsupported claims. Full design and build plan: `docs/SPEC.md`. Read it before doing anything.

## How to work in this repo

- Build one phase of `docs/SPEC.md` §11 at a time. When a phase's acceptance checks pass, commit, summarize what you built, and stop for review. Do not start the next phase on your own.
- Explain non-obvious choices as you make them, and append each one (what, why, alternatives considered) to `docs/decisions.md`. I need to be able to defend every choice in an interview.
- Prefer small, readable modules over clever abstractions. Add a docstring explaining what each module does and where it sits in the pipeline.
- Write tests alongside code. The chunk-offset invariant and the verifier's quote matching must always be tested.
- Never estimate, round up, or invent a metric. Every number in `README.md` or `docs/results.md` must come from an eval report file in `evals/reports/`, listed with the command and git commit that produced it.
- Never commit secrets. Keys live in `.env` locally (gitignored) and Secret Manager in prod.
- Model IDs come from env vars (`GENERATE_MODEL`, `VERIFY_MODEL`); don't hard-code them.
- If the CUAD files don't match what the spec expects, stop and tell me rather than guessing.

## Commands

- Start DB: `docker compose up -d db`
- Tests / lint: `uv run pytest` · `uv run ruff check .`
- Ingest: `uv run python -m clausewitz.ingest.load_db`
- Eval: `uv run python -m evals.run_eval --split dev --stage retrieval`
- API: `uv run uvicorn clausewitz.api.main:app --reload`
