"""Ingestion step 3: split a contract into retrieval chunks at clause boundaries.

A chunk is a character span `[start, end)` of the contract's full text, and its text is
always `full_text[start:end]` — the offset invariant (docs/SPEC.md §4) holds by
construction, which citations, verification and UI highlighting depend on.

Algorithm:
1. Split the text into paragraphs (blank-line separated).
2. A paragraph whose first line looks like a heading ("12.3 Termination", "ARTICLE V",
   "SECTION 4", or a short ALL-CAPS line) starts a new section.
3. Pack whole consecutive sections into one chunk while it stays within `max_tokens`,
   so short clauses aren't stranded in tiny chunks and no clause is cut mid-way.
4. A section longer than `max_tokens` on its own is split into token windows of
   `max_tokens` with `overlap` tokens shared between neighbors.

Token counts come from a `token_spans` function (text -> character spans of tokens) so
the embedding model's own tokenizer defines "400 tokens"; tests pass a whitespace one.
"""

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass

TokenSpans = Callable[[str], list[tuple[int, int]]]

MAX_TOKENS = 400
OVERLAP = 50

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_NUMBERED = re.compile(
    r"^\s*(?:(?:ARTICLE|Article|SECTION|Section)\s+[0-9IVXLC]+(?:\.\d+)*|\d{1,2}(?:\.\d{1,2})*)"
    r"[.:)]?\s+(?=\S)"
)
_ALL_CAPS = re.compile(r"^\s*[A-Z][A-Z0-9 ,&/'()\-]{2,79}\.?\s*$")
_LABEL_END = re.compile(r"[.:]\s|\n")


@dataclass(frozen=True)
class Chunk:
    start_char: int
    end_char: int
    section_path: str | None


@dataclass(frozen=True)
class _Section:
    start: int
    end: int
    label: str | None


def whitespace_token_spans(text: str) -> list[tuple[int, int]]:
    """Whitespace tokenizer: each run of non-space characters is one token."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """Character spans of non-empty paragraphs, with surrounding whitespace trimmed."""
    spans, pos = [], 0
    for m in [*_PARAGRAPH_BREAK.finditer(text), None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            spans.append((pos + lead, pos + len(seg.rstrip())))
        if m:
            pos = m.end()
    return spans


def heading_label(paragraph: str) -> str | None:
    """Return a short section label if the paragraph starts with a heading, else None."""
    first_line = paragraph.split("\n", 1)[0]
    if num := _NUMBERED.match(first_line):
        # The label runs from the number to the end of the title ("12.3 Termination."),
        # searched after the number so "1. Appointment" isn't cut at "1.".
        end = _LABEL_END.search(paragraph, num.end())
        label = paragraph[: end.start()] if end else first_line
        return " ".join(label.split())[:80]
    if _ALL_CAPS.match(first_line) and any(ch.isalpha() for ch in first_line):
        return " ".join(first_line.split())[:80]
    return None


def _sections(text: str) -> list[_Section]:
    """Group paragraphs under the most recent heading; text before any heading has no label."""
    sections: list[_Section] = []
    for start, end in _paragraphs(text):
        label = heading_label(text[start:end])
        if label is not None or not sections:
            sections.append(_Section(start, end, label))
        else:
            prev = sections[-1]
            sections[-1] = _Section(prev.start, end, prev.label)
    return sections


def _path(labels: list[str | None]) -> str | None:
    named = [lab for lab in labels if lab]
    if not named:
        return None
    return named[0] if len(named) == 1 else f"{named[0]} … {named[-1]}"


def chunk_contract(
    text: str,
    token_spans: TokenSpans = whitespace_token_spans,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    """Chunk one contract. Returns chunks in document order."""
    if not 0 <= overlap < max_tokens:
        raise ValueError("need 0 <= overlap < max_tokens")
    tokens = token_spans(text)
    starts = [s for s, _ in tokens]

    def n_tokens(start: int, end: int) -> int:
        return bisect.bisect_left(starts, end) - bisect.bisect_left(starts, start)

    chunks: list[Chunk] = []
    pending: list[_Section] = []

    def flush() -> None:
        if pending:
            chunks.append(
                Chunk(pending[0].start, pending[-1].end, _path([s.label for s in pending]))
            )
            pending.clear()

    for sec in _sections(text):
        size = n_tokens(sec.start, sec.end)
        if size > max_tokens:
            flush()
            chunks.extend(_windows(sec, tokens, starts, max_tokens, overlap))
        elif pending and n_tokens(pending[0].start, sec.end) > max_tokens:
            flush()
            pending.append(sec)
        else:
            pending.append(sec)
    flush()
    return chunks


def _windows(
    sec: _Section,
    tokens: list[tuple[int, int]],
    starts: list[int],
    max_tokens: int,
    overlap: int,
) -> list[Chunk]:
    """Split one long section into overlapping token windows, aligned to token edges."""
    lo = bisect.bisect_left(starts, sec.start)
    hi = bisect.bisect_left(starts, sec.end)
    stride = max_tokens - overlap
    out = []
    for i in range(lo, hi, stride):
        j = min(i + max_tokens, hi)
        out.append(Chunk(tokens[i][0], min(tokens[j - 1][1], sec.end), sec.label))
        if j == hi:
            break
    return out
