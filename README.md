# Clausewitz

Answers plain-English questions about CUAD commercial contracts and cites the exact clause behind every claim. Design: [`docs/SPEC.md`](docs/SPEC.md).

## Setup

```bash
cp .env.example .env
docker compose up -d db
uv sync
uv run python -m clausewitz.db     # apply schema
uv run pytest && uv run ruff check .

# Phase 1: download CUAD v1 and ingest (~5 min on a laptop)
uv run python -m clausewitz.ingest.load_cuad --download
uv run python -m clausewitz.ingest.load_db
```

Results: see [`docs/results.md`](docs/results.md) (filled in from eval reports only).

## Data

[CUAD v1](https://www.atticusprojectai.org/cuad) by The Atticus Project, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Hendrycks et al., *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021. The data is downloaded at setup time and not stored in this repo.
