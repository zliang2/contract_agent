"""Pipeline stage 6: verify every citation and drop claims without support (§6).

For each citation:
1. **Quote check** (deterministic): after normalizing whitespace and quote characters,
   the quote must be a substring of the cited chunk's text. The match is mapped back to
   absolute `start_char`/`end_char` in the contract via an index map built during
   normalization, so the UI can highlight exactly the matched characters.
2. **Support check** (LLM, `VERIFY_MODEL`): does the quote support the claim? Only run
   for citations that passed the quote check.

A claim survives if at least one citation passes both. Modes:
- "enforce": drop claims that fail (production default).
- "audit": run the same checks but keep every claim, recording the failures. Used by
  evals to measure the verifier-off condition without changing what is shown.
- "off": no checks, no LLM calls; citations get offsets only if their quote is found.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from clausewitz.llm.providers import LLMResult, Provider

Mode = Literal["enforce", "audit", "off"]

REFUSAL = "I couldn't find support for this in the contract."

_QUOTE_CHARS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
}  # fmt: skip


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text and return, for each output character, its index in `text`.

    Curly/prime quotes become straight quotes, and every whitespace run becomes one
    space. Leading/trailing whitespace is dropped. Nothing else changes: case,
    punctuation and wording must match exactly.
    """
    out: list[str] = []
    index: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            index.append(i - 1)
            pending_space = False
        out.append(_QUOTE_CHARS.get(ch, ch))
        index.append(i)
    return "".join(out), index


def normalize(text: str) -> str:
    return normalize_with_map(text)[0]


def find_quote(chunk_text: str, quote: str) -> tuple[int, int] | None:
    """Locate `quote` in `chunk_text` after normalization.

    Returns `(start, end)` offsets into the original `chunk_text` such that
    `chunk_text[start:end]` is the matched passage (first occurrence), or None.
    """
    norm_quote = normalize(quote)
    if not norm_quote:
        return None
    norm_chunk, index = normalize_with_map(chunk_text)
    pos = norm_chunk.find(norm_quote)
    if pos < 0:
        return None
    return index[pos], index[pos + len(norm_quote) - 1] + 1


class SupportVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported: bool
    reason: str


SUPPORT_SYSTEM = (
    "You check citations in answers about legal contracts. Given a claim and a quoted "
    "passage from the contract, decide whether the passage on its own supports the claim. "
    "Answer supported=true only if the passage states or directly implies the claim; "
    "partial, related or merely topical passages do not count. Give a one-line reason."
)


def support_prompt(claim: str, quote: str) -> str:
    return f"Claim:\n{claim}\n\nQuoted passage:\n{quote}"


@dataclass
class Chunk:
    id: int
    contract_id: int
    start_char: int
    text: str


@dataclass
class CitationCheck:
    chunk_id: int
    quote: str
    contract_id: int | None = None
    start_char: int | None = None  # absolute offsets in the contract, if the quote was found
    end_char: int | None = None
    quote_found: bool = False
    supported: bool | None = None  # None: support check not run
    reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.quote_found and self.supported is True


@dataclass
class ClaimCheck:
    text: str
    citations: list[CitationCheck]

    @property
    def passed(self) -> bool:
        return any(c.passed for c in self.citations)


@dataclass
class Verification:
    claims: list[ClaimCheck]  # claims to show (all of them unless mode == "enforce")
    all_claims: list[ClaimCheck]  # every generated claim with its check results
    dropped_claims: int
    refused: bool
    llm_calls: list[LLMResult] = field(default_factory=list)


def check_quote(citation: dict, chunks: dict[int, Chunk]) -> CitationCheck:
    """Quote check for one `{chunk_id, quote}` citation against the retrieved chunks."""
    check = CitationCheck(chunk_id=citation["chunk_id"], quote=citation["quote"])
    chunk = chunks.get(check.chunk_id)
    if chunk is None:  # cited a chunk that wasn't provided
        return check
    span = find_quote(chunk.text, check.quote)
    if span is not None:
        check.quote_found = True
        check.contract_id = chunk.contract_id
        check.start_char = chunk.start_char + span[0]
        check.end_char = chunk.start_char + span[1]
    return check


def verify(
    claims: list[dict],
    chunks: dict[int, Chunk],
    mode: Mode = "enforce",
    provider: Provider | None = None,
    model: str = "",
) -> Verification:
    """Check generated claims (`[{text, citations: [{chunk_id, quote}]}]`)."""
    llm_calls: list[LLMResult] = []
    checked: list[ClaimCheck] = []
    for claim in claims:
        citations = [check_quote(c, chunks) for c in claim["citations"]]
        if mode != "off":
            for c in citations:
                if not c.quote_found:
                    continue
                result = provider.complete_json(
                    model, SUPPORT_SYSTEM, support_prompt(claim["text"], c.quote),
                    SupportVerdict, max_tokens=1024,
                )  # fmt: skip
                llm_calls.append(result)
                if result.refused:
                    c.supported, c.reason = False, "verifier refused"
                    continue
                try:
                    verdict = SupportVerdict.model_validate_json(result.text)
                    c.supported, c.reason = verdict.supported, verdict.reason
                except ValueError:
                    c.supported, c.reason = False, "verifier returned invalid JSON"
        checked.append(ClaimCheck(claim["text"], citations))

    shown = [c for c in checked if c.passed] if mode == "enforce" else checked
    return Verification(
        claims=shown,
        all_claims=checked,
        dropped_claims=len(checked) - len(shown),
        refused=not shown,
        llm_calls=llm_calls,
    )
