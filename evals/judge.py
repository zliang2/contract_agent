"""Independent grounding judge for evals (not part of the served pipeline).

The verifier decides what users see, so measuring groundedness with the verifier itself
would be circular: with the verifier on, every shown claim has passed it by definition.
The judge is a separate LLM call (`JUDGE_MODEL`, ideally a different model from
`VERIFY_MODEL`) with a different input: the claim plus the *full text* of every chunk it
cites, not just the quote. Its agreement with human labels is measured on a hand-labeled
sample (`evals/label_sample.py`) and reported alongside its numbers.
"""

from pydantic import BaseModel, ConfigDict

from clausewitz.llm.providers import LLMResult, Provider


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported: bool
    reason: str


SYSTEM = (
    "You grade answers about legal contracts. You get one claim and the contract text it "
    "cites. Decide whether the cited text supports the claim: supported=true only if the "
    "text states or clearly implies everything the claim asserts. Claims that add facts, "
    "numbers, conditions or conclusions not in the text are not supported. Give a one-line "
    "reason."
)


def judge_prompt(claim: str, cited_texts: list[str]) -> str:
    sources = "\n\n".join(f"<cited_text>\n{t}\n</cited_text>" for t in cited_texts)
    return f"Claim:\n{claim}\n\n{sources}"


def judge_claim(
    claim: str, cited_texts: list[str], provider: Provider, model: str
) -> tuple[bool | None, str, LLMResult]:
    """Returns (supported, reason, call). supported is None if the judge failed."""
    if not cited_texts:
        return False, "no cited text", None
    result = provider.complete_json(
        model, SYSTEM, judge_prompt(claim, cited_texts), JudgeVerdict, max_tokens=1024
    )
    if result.refused:
        return None, "judge refused", result
    try:
        v = JudgeVerdict.model_validate_json(result.text)
        return v.supported, v.reason, result
    except ValueError:
        return None, "judge returned invalid JSON", result
