# Clausewitz

Answers plain-English questions about CUAD commercial contracts and cites the exact clause behind every claim. Design: [`docs/SPEC.md`](docs/SPEC.md).

## Setup

```bash
cp .env.example .env
docker compose up -d db
uv sync
uv run python -m clausewitz.db     # apply schema
uv run pytest && uv run ruff check .
```

Results: see [`docs/results.md`](docs/results.md) (filled in from eval reports only).
