"""Verifier tests: quote normalization, substring matching, offset mapping, and the
drop/keep logic in each mode. No network: the support check uses FakeProvider."""

import json

import pytest

from clausewitz.llm.providers import FakeProvider
from clausewitz.llm.verify import (
    REFUSAL,
    Chunk,
    find_quote,
    normalize,
    normalize_with_map,
    verify,
)

CONTRACT = (
    "ARTICLE 9  TERMINATION\n\n"
    "9.1 Either party may terminate this Agreement upon ninety (90) days’ prior\n"
    "    written notice to the other party.\n\n"
    "9.2 The “Company” shall pay all fees accrued before termination."
)
CHUNK = Chunk(id=7, contract_id=3, start_char=1000, text=CONTRACT)


def test_normalize_collapses_whitespace_and_straightens_quotes():
    assert normalize("  a\n\n  b\t“c”  d’s  ") == 'a b "c" d\'s'


def test_normalize_index_map_points_into_original():
    text = "x “y”\n\nz"
    norm, index = normalize_with_map(text)
    assert len(norm) == len(index)
    for out_ch, i in zip(norm, index, strict=True):
        if out_ch not in " '\"":
            assert text[i] == out_ch


def test_exact_quote_maps_to_exact_offsets():
    quote = "Either party may terminate this Agreement"
    start, end = find_quote(CONTRACT, quote)
    assert CONTRACT[start:end] == quote


def test_quote_matches_across_line_breaks_and_curly_quotes():
    # The model copies the text with straight quotes and single spaces.
    quote = "ninety (90) days' prior written notice to the other party."
    start, end = find_quote(CONTRACT, quote)
    assert CONTRACT[start:end] == (
        "ninety (90) days’ prior\n    written notice to the other party."
    )
    start, end = find_quote(CONTRACT, 'The "Company" shall pay')
    assert CONTRACT[start:end] == "The “Company” shall pay"


@pytest.mark.parametrize(
    "quote",
    [
        "Either party can terminate this Agreement on 90 days' notice",  # paraphrase
        "either party may terminate this agreement",  # case differs
        "Either party may terminate this Agreement upon sixty (60) days",  # changed fact
        "",
        "   ",
    ],
)
def test_paraphrased_or_altered_quotes_fail(quote):
    assert find_quote(CONTRACT, quote) is None


def support(answer: bool):
    return FakeProvider(lambda s, u, schema: json.dumps({"supported": answer, "reason": "r"}))


def claim(text, *citations):
    return {"text": text, "citations": [{"chunk_id": c, "quote": q} for c, q in citations]}


GOOD = claim("Either party may terminate on 90 days' notice.",
             (7, "Either party may terminate this Agreement upon ninety (90) days"))  # fmt: skip
PARAPHRASED = claim("Either party may terminate on 90 days' notice.",
                    (7, "Either party can terminate on 90 days notice"))  # fmt: skip
WRONG_CHUNK = claim("Fees are due.", (99, "shall pay all fees"))


def test_enforce_keeps_supported_claim_with_absolute_offsets():
    v = verify([GOOD], {7: CHUNK}, mode="enforce", provider=support(True), model="m")
    (c,) = v.claims
    (cit,) = c.citations
    assert cit.passed and cit.contract_id == 3
    local = CONTRACT.index("Either party")
    assert (cit.start_char, cit.end_char) == (
        1000 + local, 1000 + local + len(GOOD["citations"][0]["quote"])
    )  # fmt: skip
    assert v.dropped_claims == 0 and not v.refused


def test_enforce_drops_paraphrase_without_calling_llm():
    provider = support(True)
    v = verify([PARAPHRASED, WRONG_CHUNK], {7: CHUNK}, "enforce", provider, "m")
    assert v.claims == [] and v.dropped_claims == 2 and v.refused
    assert provider.calls == []  # support check only runs on found quotes


def test_enforce_drops_unsupported_claim():
    v = verify([GOOD], {7: CHUNK}, mode="enforce", provider=support(False), model="m")
    assert v.refused and v.dropped_claims == 1
    assert v.all_claims[0].citations[0].supported is False


def test_claim_survives_if_any_citation_passes():
    mixed = claim("Termination needs notice.", (99, "x"),
                  (7, "written notice to the other party."))  # fmt: skip
    v = verify([mixed], {7: CHUNK}, mode="enforce", provider=support(True), model="m")
    assert len(v.claims) == 1 and v.dropped_claims == 0


def test_audit_keeps_everything_but_records_failures():
    v = verify([GOOD, PARAPHRASED], {7: CHUNK}, "audit", support(True), "m")
    assert len(v.claims) == 2 and v.dropped_claims == 0 and not v.refused
    assert [c.passed for c in v.all_claims] == [True, False]


def test_off_makes_no_llm_calls():
    v = verify([GOOD, PARAPHRASED], {7: CHUNK}, mode="off")
    assert len(v.claims) == 2 and v.llm_calls == []
    assert v.claims[0].citations[0].quote_found and v.claims[0].citations[0].supported is None


def test_invalid_verifier_json_counts_as_unsupported():
    v = verify([GOOD], {7: CHUNK}, "enforce", FakeProvider(lambda *a: "not json"), "m")
    assert v.refused and "invalid JSON" in v.all_claims[0].citations[0].reason


def test_no_claims_is_a_refusal():
    v = verify([], {7: CHUNK}, mode="enforce", provider=support(True), model="m")
    assert v.refused
    assert REFUSAL == "I couldn't find support for this in the contract."
