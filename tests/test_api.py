"""Smoke test: every API endpoint against the ingested DB, with a fake LLM (no paid calls).

Also checks that each query writes one trace line containing every §10 tracing field."""

import json

import pytest
from fastapi.testclient import TestClient

from clausewitz import pipeline
from clausewitz.api import main
from clausewitz.llm.providers import FakeProvider
from tests.test_pipeline_answer_db import conn, respond  # noqa: F401  (fixture)

pytestmark = pytest.mark.db


@pytest.fixture
def client(conn, monkeypatch):  # noqa: F811  (`conn` skips when the DB isn't ingested)
    monkeypatch.setattr(pipeline, "get_provider", lambda model: FakeProvider(respond))
    monkeypatch.setattr(main.app.state, "warm_up", False)
    with TestClient(main.app) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_contracts_and_contract(client):
    dev = client.get("/contracts", params={"split": "dev"}).json()
    every = client.get("/contracts").json()
    assert 0 < len(dev) < len(every) and {c["split"] for c in dev} == {"dev"}
    one = client.get(f"/contracts/{dev[0]['id']}").json()
    assert one["title"] == dev[0]["title"] and len(one["text"]) > 0
    assert client.get("/contracts/999999").status_code == 404
    assert client.get("/contracts", params={"split": "nope"}).status_code == 422


def test_query_cites_exact_span_and_traces(client, capsys):
    c = client.get("/contracts").json()[0]
    capsys.readouterr()
    body = {"question": "Governing Law: which law?", "contract_id": c["id"]}
    r = client.post("/query", json=body)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "claims", "refused", "dropped_claims", "trace_id"}
    assert not body["refused"] and body["dropped_claims"] == 1
    (x,) = body["claims"][0]["citations"]
    text = client.get(f"/contracts/{x['contract_id']}").json()["text"]
    assert " ".join(text[x["start_char"] : x["end_char"]].split()) == x["quote"]

    (line,) = [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]
    assert line["trace_id"] == body["trace_id"]
    assert set(line["duration_ms"]) >= {*main.tracing.STAGES, "total"}
    for field in ("llm_calls", "input_tokens", "output_tokens", "generation_tokens_per_s",
                  "cost_usd", "chunks_retrieved", "claims_generated", "claims_dropped",
                  "refused"):  # fmt: skip
        assert field in line
    assert line["chunks_retrieved"] == 8 and line["claims_generated"] == 2


def test_query_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "contract_id": 999999}).status_code == 404
