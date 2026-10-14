import pytest

from clausewitz.ingest.load_cuad import Contract, assign_splits, parse_cuad

TEXT = "This Agreement is governed by the laws of Texas."


def raw(answers, title="ACME_AGREEMENT"):
    return {
        "version": "aok_v1.0",
        "data": [
            {
                "title": title,
                "paragraphs": [
                    {
                        "context": TEXT,
                        "qas": [
                            {
                                "id": f"{title}__Governing Law",
                                "question": 'Highlight ... related to "Governing Law" ...',
                                "answers": answers,
                                "is_impossible": not answers,
                            },
                            {
                                "id": f"{title}__Audit Rights",
                                "question": 'Highlight ... related to "Audit Rights" ...',
                                "answers": [],
                                "is_impossible": True,
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_parse_reads_gold_spans_and_questions():
    contracts, questions = parse_cuad(raw([{"text": "laws of Texas", "answer_start": 34}]))
    (c,) = contracts
    assert c.full_text == TEXT
    (g,) = c.gold
    assert (g.clause_type, TEXT[g.start_char : g.end_char]) == ("Governing Law", "laws of Texas")
    assert [q.clause_type for q in questions] == ["Audit Rights", "Governing Law"]


def test_parse_rejects_bad_offsets():
    with pytest.raises(ValueError, match="offsets"):
        parse_cuad(raw([{"text": "laws of Texas", "answer_start": 30}]))


def test_split_is_80_20_deterministic_and_order_independent():
    contracts = [Contract(title=f"c{i:03d}", full_text="x") for i in range(510)]
    assign_splits(contracts)
    first = {c.title: c.split for c in contracts}
    reversed_contracts = [Contract(title=c.title, full_text="x") for c in reversed(contracts)]
    assign_splits(reversed_contracts)
    assert {c.title: c.split for c in reversed_contracts} == first
    assert sum(s == "test" for s in first.values()) == 102
