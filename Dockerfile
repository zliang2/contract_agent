# One image serves the API and the built React UI (docs/SPEC.md §10).
# The embedding and reranker models are baked in, so a new instance never downloads them.
# The BM25 index is fetched from Cloud Storage at startup (BM25_INDEX_URI), and secrets
# come from Secret Manager as env vars; neither is in the image.

FROM node:24.22.2-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/hf PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY clausewitz/ clausewitz/
COPY README.md ./
RUN uv sync --locked --no-dev

# Bake the models (model IDs come from settings / the rerank module, not repeated here).
RUN python -c "from clausewitz.ingest.embed import get_model; get_model()" \
 && python -c "from clausewitz.retrieval.rerank import get_model; get_model()"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY evals/prices.json evals/prices.json
COPY --from=web /web/dist web/dist

RUN useradd --create-home --uid 10001 app && mkdir -p indexes && chown -R app /app/indexes
USER app
EXPOSE 8080
# Cloud Run sets PORT. One worker: models are loaded per process, and requests are
# I/O-bound on the LLM, which the thread pool already overlaps.
CMD ["sh", "-c", "exec uvicorn clausewitz.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
