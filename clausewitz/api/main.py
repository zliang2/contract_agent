"""HTTP API and static UI host (docs/SPEC.md §8). The outermost layer of the system.

    uv run uvicorn clausewitz.api.main:app --reload

Endpoints:
- `GET /healthz`: liveness for Cloud Run (also checks the database answers).
- `GET /contracts?split=`: contract ids and titles for the picker.
- `GET /contracts/{id}`: full text for the viewer; citation offsets index into it.
- `POST /query`: `{question, contract_id?}` → the §6 response from `pipeline.answer`.

Each query logs one JSON trace line (`clausewitz.tracing`). If `web/dist` exists (the
built React app), it is served at `/`, so the UI and API share one origin and one Cloud
Run service, with no CORS. Models and the BM25 index load at startup, not on the first
request; the BM25 index is first downloaded from Cloud Storage if `BM25_INDEX_URI` is set.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from clausewitz import pipeline, tracing
from clausewitz.config import get_settings
from clausewitz.llm.providers import QuotaExhausted

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def warm_up() -> None:
    """Fetch the BM25 index if needed, and load it and both models into memory."""
    from clausewitz.gcs import ensure_bm25_index
    from clausewitz.retrieval import bm25, rerank, vector

    ensure_bm25_index()
    bm25.get_index()
    vector.embed_query("warm up")
    rerank.rerank("warm up", [(0, "warm up")], top_n=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = ConnectionPool(
        get_settings().database_url, min_size=1, max_size=4, configure=register_vector, open=True
    )
    if app.state.warm_up:
        warm_up()
    yield
    app.state.pool.close()


app = FastAPI(title="Clausewitz", lifespan=lifespan)
app.state.warm_up = True


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    contract_id: int | None = None


@app.get("/healthz")
def healthz():
    with app.state.pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/contracts")
def list_contracts(split: Literal["dev", "test"] | None = None):
    sql = "SELECT id, title, split FROM contracts"
    params: tuple = ()
    if split:
        sql, params = sql + " WHERE split = %s", (split,)
    with app.state.pool.connection() as conn:
        rows = conn.execute(sql + " ORDER BY title", params).fetchall()
    return [{"id": i, "title": t, "split": s} for i, t, s in rows]


@app.get("/contracts/{contract_id}")
def get_contract(contract_id: int):
    with app.state.pool.connection() as conn:
        row = conn.execute(
            "SELECT id, title, split, full_text FROM contracts WHERE id = %s", (contract_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "contract not found")
    return {"id": row[0], "title": row[1], "split": row[2], "text": row[3]}


@app.post("/query")
def query(q: Query):
    with app.state.pool.connection() as conn:
        if q.contract_id is not None:
            found = conn.execute("SELECT 1 FROM contracts WHERE id = %s", (q.contract_id,))
            if found.fetchone() is None:
                raise HTTPException(404, "contract not found")
        try:
            a = pipeline.answer(q.question, q.contract_id, conn, verify="enforce")
        except QuotaExhausted as e:
            raise HTTPException(503, "LLM quota exhausted; try again later") from e
    tracing.emit(tracing.trace_record(a))
    return a.to_response()


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
