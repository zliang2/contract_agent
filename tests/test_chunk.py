"""Chunker tests. The offset invariant and full coverage are checked on synthetic text
and, when the CUAD file is present, on every real contract."""

from pathlib import Path

import pytest

from clausewitz.config import get_settings
from clausewitz.ingest.chunk import (
    chunk_contract,
    heading_label,
    whitespace_token_spans,
)


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


CONTRACT = (
    "DISTRIBUTION AGREEMENT\n\n"
    "This Agreement is made between Acme and Beta.\n\n"
    "1. Definitions. " + words(30, "def") + "\n\n"
    "2. Term. " + words(20, "term") + "\n\n"
    "3. Termination. " + words(250, "tm") + "\n\n"
    "ARTICLE IV Payment\n\n" + words(40, "pay") + "\n"
)


def assert_invariants(text: str, chunks) -> None:
    covered = [False] * len(text)
    for ch in chunks:
        assert 0 <= ch.start_char < ch.end_char <= len(text)
        piece = text[ch.start_char : ch.end_char]
        assert piece == piece.strip(), "chunks start and end on non-whitespace"
        for i in range(ch.start_char, ch.end_char):
            covered[i] = True
    missing = [i for i, c in enumerate(text) if not c.isspace() and not covered[i]]
    assert not missing, (
        f"text not covered by any chunk, e.g. {text[missing[0] : missing[0] + 30]!r}"
    )
    starts = [ch.start_char for ch in chunks]
    assert starts == sorted(starts)


def n_tokens(text: str, ch) -> int:
    return len(whitespace_token_spans(text[ch.start_char : ch.end_char]))


def test_invariants_on_synthetic_contract():
    chunks = chunk_contract(CONTRACT, max_tokens=100, overlap=10)
    assert_invariants(CONTRACT, chunks)
    assert all(n_tokens(CONTRACT, ch) <= 100 for ch in chunks)


def test_small_sections_are_packed_without_splitting_clauses():
    chunks = chunk_contract(CONTRACT, max_tokens=100, overlap=10)
    first = chunks[0]
    text = CONTRACT[first.start_char : first.end_char]
    assert "1. Definitions." in text and "2. Term." in text  # packed together
    assert first.section_path == "DISTRIBUTION AGREEMENT … 2. Term"


def test_long_section_is_windowed_with_overlap():
    chunks = chunk_contract(CONTRACT, max_tokens=100, overlap=10)
    windows = [ch for ch in chunks if ch.section_path == "3. Termination"]
    assert len(windows) == 3  # 252 tokens, stride 90
    for a, b in zip(windows, windows[1:], strict=False):
        assert b.start_char < a.end_char  # neighbors overlap
        shared = whitespace_token_spans(CONTRACT[b.start_char : a.end_char])
        assert len(shared) == 10


def test_text_without_headings_falls_back_to_windows():
    text = words(1000)
    chunks = chunk_contract(text, max_tokens=400, overlap=50)
    assert_invariants(text, chunks)
    assert len(chunks) == 3 and all(ch.section_path is None for ch in chunks)


def test_empty_and_whitespace_text():
    assert chunk_contract("") == []
    assert chunk_contract(" \n\n \t") == []


@pytest.mark.parametrize(
    ("paragraph", "label"),
    [
        ("12.3 Termination. Either party may terminate", "12.3 Termination"),
        ("1. Appointment. Company appoints", "1. Appointment"),
        ("ARTICLE V\nCONFIDENTIALITY", "ARTICLE V"),
        ("Section 4.2: Fees owed", "Section 4.2: Fees owed"),
        ("RECITALS", "RECITALS"),
        ("The parties agree as follows.", None),
        ("2019 was a good year for the parties", None),
        ("(a) the Company shall", None),
    ],
)
def test_heading_label(paragraph, label):
    assert heading_label(paragraph) == label


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_contract("x", max_tokens=10, overlap=10)


@pytest.mark.skipif(
    not Path(get_settings().cuad_path).exists(), reason="CUAD not downloaded (load_cuad --download)"
)
def test_invariants_on_every_cuad_contract():
    from clausewitz.ingest.load_cuad import load_cuad

    contracts, _ = load_cuad()
    for c in contracts:
        assert_invariants(c.full_text, chunk_contract(c.full_text))
