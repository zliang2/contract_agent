import json

from clausewitz.llm.generate import GeneratedAnswer, build_prompt, generate
from clausewitz.llm.providers import FakeProvider, LLMResult
from clausewitz.llm.verify import Chunk

CHUNKS = [Chunk(1, 9, 0, "Governed by the laws of Texas."), Chunk(2, 9, 40, "Fees due monthly.")]
VALID = json.dumps(
    {
        "answer": "Texas law governs.",
        "claims": [
            {"text": "Texas law governs.",
             "citations": [{"chunk_id": 1, "quote": "the laws of Texas"}]}
        ],
    }
)  # fmt: skip


def scripted(*outputs):
    it = iter(outputs)
    return FakeProvider(lambda *a: next(it))


def test_prompt_labels_each_chunk_with_its_id():
    p = build_prompt("Which law governs?", CHUNKS)
    assert '<excerpt chunk_id="1">\nGoverned by the laws of Texas.\n</excerpt>' in p
    assert '<excerpt chunk_id="2">' in p and p.endswith("Question: Which law governs?")


def test_corpus_wide_prompt_names_each_excerpts_contract():
    p = build_prompt("q", CHUNKS, titles={9: "Acme Supply"})
    assert '<excerpt chunk_id="1" contract="Acme Supply">' in p
    # Without titles (single-contract questions) the prompt is unchanged.
    assert "contract=" not in build_prompt("q", CHUNKS)


def test_valid_output_parses_first_try():
    provider = scripted(VALID)
    g = generate("q", CHUNKS, provider, "m")
    assert isinstance(g.answer, GeneratedAnswer) and len(g.attempts) == 1
    assert g.answer.claims[0].citations[0].chunk_id == 1
    assert provider.calls[0]["schema"] is GeneratedAnswer


def test_invalid_json_is_retried_once():
    g = generate("q", CHUNKS, scripted("{not json", VALID), "m")
    assert g.answer is not None and len(g.attempts) == 2 and g.error is None


def test_gives_up_after_two_invalid_outputs():
    g = generate("q", CHUNKS, scripted("{", json.dumps({"answer": "x"}), VALID), "m")
    assert g.answer is None and len(g.attempts) == 2 and "invalid JSON" in g.error


def test_refusal_is_not_retried():
    class Refuser:
        def complete_json(self, *a, **k):
            return LLMResult("", "m", 10, 0, 1.0, refused=True, stop_reason="refusal")

    g = generate("q", CHUNKS, Refuser(), "m")
    assert g.answer is None and len(g.attempts) == 1 and "refused" in g.error
