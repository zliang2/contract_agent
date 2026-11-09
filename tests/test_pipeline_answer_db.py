"""End-to-end answer() and full-eval run against the ingested DB with fake LLMs.

The fake generator quotes the first retrieved chunk verbatim (must verify) and adds a
paraphrased claim (must be dropped). No paid calls; the retrieval models do load."""

import json
import re

import pytest

from clausewitz import pipeline
from clausewitz.db import connect
from clausewitz.llm.generate import GeneratedAnswer
from clausewitz.llm.providers import FakeProvider
from clausewitz.llm.verify import REFUSAL

pytestmark = pytest.mark.db


def respond(system, user, schema):
    if schema is GeneratedAnswer:
        m = re.search(r'<excerpt chunk_id="(\d+)">\n(.+?)\n</excerpt>', user, re.S)
        chunk_id, text = int(m.group(1)), m.group(2)
        quote = " ".join(text.split()[:12])  # whitespace-normalized copy
        return json.dumps({"answer": "x", "claims": [
            {"text": "real", "citations": [{"chunk_id": chunk_id, "quote": quote}]},
            {"text": "fake", "citations": [{"chunk_id": chunk_id, "quote": "made up words"}]},
        ]})  # fmt: skip
    return json.dumps({"supported": True, "reason": "ok"})


@pytest.fixture
def fake_llm(monkeypatch):
    provider = FakeProvider(respond)
    monkeypatch.setattr(pipeline, "get_provider", lambda model: provider)
    import evals.full_eval as full_eval

    monkeypatch.setattr(full_eval, "get_provider", lambda model: provider)
    return provider


@pytest.fixture(scope="module")
def conn(db_url):
    with connect(db_url, vector=False) as c:
        exists = c.execute("SELECT to_regclass('chunks') IS NOT NULL").fetchone()[0]
        if not exists or not c.execute("SELECT EXISTS (SELECT 1 FROM chunks)").fetchone()[0]:
            pytest.skip("database not ingested")
    with connect(db_url) as c:
        yield c


def contract_id(conn):
    return conn.execute("SELECT id FROM contracts ORDER BY id LIMIT 1").fetchone()[0]


def test_enforce_keeps_verbatim_claim_and_drops_paraphrase(conn, fake_llm):
    cid = contract_id(conn)
    a = pipeline.answer("Governing Law: which law governs?", cid, conn, verify="enforce",
                        generate_model="g", verify_model="v")  # fmt: skip
    assert not a.refused and a.dropped_claims == 1
    resp = a.to_response()
    (c,) = resp["claims"]
    (x,) = c["citations"]
    full_text = conn.execute(
        "SELECT full_text FROM contracts WHERE id = %s", (x["contract_id"],)
    ).fetchone()[0]
    # The highlighted span is real contract text matching the quote (whitespace-normalized).
    assert " ".join(full_text[x["start_char"] : x["end_char"]].split()) == x["quote"]
    assert set(a.timings_ms) >= {"bm25", "vector", "rerank", "generate", "verify", "total"}
    assert len(resp["trace_id"]) == 32


def test_refusal_when_nothing_verifies(conn, monkeypatch):
    def unsupported(system, user, schema):
        if schema is GeneratedAnswer:
            return respond(system, user, schema)
        return json.dumps({"supported": False, "reason": "no"})

    monkeypatch.setattr(pipeline, "get_provider", lambda m: FakeProvider(unsupported))
    a = pipeline.answer("q", contract_id(conn), conn, verify="enforce",
                        generate_model="g", verify_model="v")  # fmt: skip
    assert a.refused and a.answer == REFUSAL and a.to_response()["claims"] == []


def test_full_eval_records_and_metrics(conn, fake_llm):
    from evals.build_cases import build_cases
    from evals.full_eval import full_metrics, run_full_cases

    cases = [c for c in build_cases(conn, "dev") if c.answerable][:2]
    cases += [c for c in build_cases(conn, "dev") if not c.answerable][:1]
    records = run_full_cases(conn, cases, "g", "v", "j")
    assert len(records) == 3
    assert all(len(r["claims"]) == 2 for r in records)
    assert all([c["verifier_passed"] for c in r["claims"]] == [True, False] for r in records)
    m = full_metrics(records, {"models": {}})
    assert m["conditions"]["on"]["n_answered"] == 3
    assert m["conditions"]["on"]["refusal_accuracy"] == 0.0  # the fake always answers


def test_full_eval_resumes_from_checkpoint(conn, fake_llm, tmp_path):
    from evals.build_cases import build_cases
    from evals.full_eval import run_full_cases

    cases = build_cases(conn, "dev")[:3]
    ckpt, run_id = tmp_path / "ck.jsonl", {"commit": "x"}
    first = run_full_cases(conn, cases[:2], "g", "v", "j", checkpoint=ckpt, run_id=run_id)
    n_calls = len(fake_llm.calls)
    again = run_full_cases(conn, cases, "g", "v", "j", checkpoint=ckpt, run_id=run_id)
    assert [r["clause_type"] for r in again] == [c.clause_type for c in cases]
    assert again[:2] == first
    assert len(fake_llm.calls) > n_calls  # only the third case made new calls
    assert len(ckpt.read_text().splitlines()) == 1 + 3
