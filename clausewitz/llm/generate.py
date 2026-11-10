"""Pipeline stage 5: generate a cited answer from the retrieved chunks.

The model sees the question and the top chunks, each labeled with its chunk id, and must
return JSON matching `GeneratedAnswer`: an answer plus claims, each citing chunk ids
with quotes copied verbatim. Output is validated with Pydantic; on invalid output the
call is retried once (docs/SPEC.md §6). Stage 6 (`verify.py`) checks every citation.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError

from clausewitz.llm.providers import LLMResult, Provider
from clausewitz.llm.verify import Chunk

MAX_OUTPUT_TOKENS = 4096


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: int
    quote: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    citations: list[Citation]


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    claims: list[Claim]


SYSTEM = (
    "You answer questions about one commercial contract using only the excerpts provided.\n"
    "\n"
    "Rules:\n"
    "- Use only the excerpts. Do not use outside knowledge or assumptions about typical "
    "contracts.\n"
    "- Split your answer into claims: short factual statements. Every claim must cite at "
    "least one excerpt by its chunk id, with a quote copied character for character from "
    "that excerpt. The quote must be the passage that supports the claim, not a paraphrase.\n"
    "- If the excerpts do not answer the question, return an answer saying the contract "
    "excerpts do not address it, and an empty claims list.\n"
    "- This is not legal advice; describe what the contract says."
)


def build_prompt(question: str, chunks: list[Chunk], titles: dict[int, str] | None = None) -> str:
    """Prompt with each chunk as an `<excerpt>`. With `titles` (corpus-wide questions, where
    excerpts come from many contracts), each excerpt also names its contract."""

    def attrs(c: Chunk) -> str:
        if titles is None:
            return f'chunk_id="{c.id}"'
        return f'chunk_id="{c.id}" contract="{titles[c.contract_id]}"'

    excerpts = "\n\n".join(f"<excerpt {attrs(c)}>\n{c.text}\n</excerpt>" for c in chunks)
    return f"Contract excerpts:\n\n{excerpts}\n\nQuestion: {question}"


@dataclass
class Generation:
    answer: GeneratedAnswer | None  # None if both attempts failed or the model refused
    attempts: list[LLMResult] = field(default_factory=list)
    error: str | None = None


def generate(
    question: str,
    chunks: list[Chunk],
    provider: Provider,
    model: str,
    titles: dict[int, str] | None = None,
) -> Generation:
    """Generate a cited answer. Retries once if the JSON doesn't validate."""
    prompt = build_prompt(question, chunks, titles)
    gen = Generation(answer=None)
    for _ in range(2):
        result = provider.complete_json(model, SYSTEM, prompt, GeneratedAnswer, MAX_OUTPUT_TOKENS)
        gen.attempts.append(result)
        if result.refused:
            gen.error = f"model refused ({result.stop_reason})"
            return gen
        try:
            gen.answer = GeneratedAnswer.model_validate_json(result.text)
            gen.error = None
            return gen
        except ValidationError as e:
            gen.error = f"invalid JSON: {e.errors()[0]['msg']}"
    return gen
