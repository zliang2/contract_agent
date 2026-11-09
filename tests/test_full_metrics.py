"""full_metrics on hand-built records, so each definition is pinned down exactly."""

import pytest

from evals.full_eval import full_metrics

PRICES = {"models": {"gen": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
                     "ver": {"input_per_mtok": 0.5, "output_per_mtok": 0.5}}}  # fmt: skip


def cit(gold: bool, passed: bool, found: bool = True):
    return {"quote_found": found, "passed": passed, "overlaps_gold": gold and found,
            "start_char": 0 if found else None, "end_char": 5 if found else None}  # fmt: skip


def claim(verifier_passed, judge_supported, *cits):
    return {"text": "c", "verifier_passed": verifier_passed,
            "judge_supported": judge_supported, "citations": list(cits)}  # fmt: skip


def record(answerable, claims, gen_tokens=(1000, 100), verify_calls=1):
    calls = [{"stage": "generate", "model": "gen", "input_tokens": gen_tokens[0],
              "output_tokens": gen_tokens[1]}]  # fmt: skip
    calls += [{"stage": "verify", "model": "ver", "input_tokens": 100, "output_tokens": 10}
              ] * verify_calls  # fmt: skip
    calls += [{"stage": "judge", "model": "ver", "input_tokens": 50, "output_tokens": 10}]
    return {"answerable": answerable, "error": None, "claims": claims, "calls": calls,
            "timings_ms": {"rerank": 10.0, "generate": 100.0, "verify": 20.0}}  # fmt: skip


RECORDS = [
    # 1. answerable: one good claim, one hallucinated claim the verifier catches
    record(True, [claim(True, True, cit(True, True)), claim(False, False, cit(False, False))]),
    # 2. answerable: only a bad claim -> verifier refuses
    record(True, [claim(False, False, cit(False, False))]),
    # 3. should refuse: model invents a claim, verifier catches it
    record(False, [claim(False, False, cit(False, False, found=False))]),
    # 4. should refuse: model correctly gives no claims
    record(False, [], verify_calls=0),
]


def test_counts():
    m = full_metrics(RECORDS, PRICES)
    assert (m["n_cases"], m["n_answerable"], m["n_should_refuse"]) == (4, 2, 2)
    assert m["claims_generated"] == 4
    assert m["quote_not_found_rate"] == pytest.approx(1 / 4)


def test_verifier_off():
    off = full_metrics(RECORDS, PRICES)["conditions"]["off"]
    assert off["n_answered"] == 3
    assert off["ungrounded_rate_judge"] == pytest.approx(3 / 3)  # all answered have a bad claim
    assert off["citation_precision"] == pytest.approx(1 / 3)  # 3 located citations, 1 on gold
    assert off["refusal_accuracy"] == pytest.approx(1 / 2)
    assert off["answer_rate"] == pytest.approx(1.0)
    assert off["accuracy"] == pytest.approx(1 / 2)


def test_verifier_on():
    on = full_metrics(RECORDS, PRICES)["conditions"]["on"]
    assert on["n_answered"] == 1
    assert on["ungrounded_rate_judge"] == pytest.approx(0.0)
    assert on["ungrounded_rate_verifier"] == pytest.approx(0.0)  # zero by construction
    assert on["citation_precision"] == pytest.approx(1.0)
    assert on["refusal_accuracy"] == pytest.approx(1.0)
    assert on["answer_rate"] == pytest.approx(1 / 2)  # case 2 was refused
    assert on["accuracy"] == pytest.approx(1 / 2)
    assert on["dropped_claims_per_answer"] == pytest.approx(3 / 4)


def test_cost_counts_verify_only_when_on_and_excludes_judge():
    m = full_metrics(RECORDS, PRICES)
    gen = (1000 * 1.0 + 100 * 2.0) / 1e6
    ver = (100 * 0.5 + 10 * 0.5) / 1e6
    assert m["conditions"]["off"]["cost_per_query_usd"] == pytest.approx(gen)
    assert m["conditions"]["on"]["cost_per_query_usd"] == pytest.approx(gen + 3 * ver / 4)
    assert m["judge_cost_total_usd"] == pytest.approx(4 * (50 * 0.5 + 10 * 0.5) / 1e6)


def test_unknown_price_gives_none_not_a_guess():
    m = full_metrics(RECORDS, {"models": {}})
    assert m["conditions"]["off"]["cost_per_query_usd"] is None
