"""Build eval cases: one per (contract, clause type) in a split (docs/SPEC.md §7).

The question comes from CUAD's question for that clause type: by default rewritten as
"<clause type>: <details>" (see `plain_question`), or verbatim with `cuad_template`.
A case with no gold span for its clause type is a "should refuse" case. Gold spans and
chunk spans are read from the database, so cases and retrieval always agree on offsets.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from clausewitz.ingest.load_cuad import load_cuad

QUESTION_STYLES = ("plain", "cuad_template")
_TEMPLATE = re.compile(
    r'^Highlight the parts \(if any\) of this contract related to "(.+?)" '
    r"that should be reviewed by a lawyer\. Details: (.+)$",
    re.S,
)


def plain_question(cuad_question: str) -> str:
    """Rewrite CUAD's annotation template as "<clause type>: <details>".

    The template ("Highlight the parts (if any) of this contract related to "X" that should
    be reviewed by a lawyer. Details: ...") is the instruction annotators saw. Users of
    this system ask plain questions, so evals default to the clause name plus its details.
    """
    m = _TEMPLATE.match(cuad_question)
    if not m:
        raise ValueError(f"CUAD question doesn't match the template: {cuad_question[:80]!r}")
    return " ".join(f"{m.group(1)}: {m.group(2)}".split())


@dataclass
class Case:
    contract_id: int
    title: str
    clause_type: str
    question: str
    gold: list[tuple[int, int]] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


def build_cases(
    conn, split: str, subset: str | Path | None = None, question_style: str = "plain"
) -> list[Case]:
    """All cases for `split`, optionally limited to a subset file of {title, clause_type}.

    `question_style` is "plain" (clause name + details) or "cuad_template" (verbatim).
    """
    if question_style not in QUESTION_STYLES:
        raise ValueError(f"question_style must be one of {QUESTION_STYLES}")
    _, questions = load_cuad()
    text = {
        q.clause_type: plain_question(q.question) if question_style == "plain" else q.question
        for q in questions
    }
    contracts = conn.execute(
        "SELECT id, title FROM contracts WHERE split = %s ORDER BY id", (split,)
    ).fetchall()
    gold: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for cid, ctype, s, e in conn.execute(
        "SELECT a.contract_id, a.clause_type, a.start_char, a.end_char FROM annotations a "
        "JOIN contracts c ON c.id = a.contract_id WHERE c.split = %s ORDER BY a.id",
        (split,),
    ):
        gold[(cid, ctype)].append((s, e))
    cases = [
        Case(cid, title, q.clause_type, text[q.clause_type], gold.get((cid, q.clause_type), []))
        for cid, title in contracts
        for q in questions
    ]
    if subset:
        wanted = {(x["title"], x["clause_type"]) for x in json.loads(Path(subset).read_text())}
        cases = [c for c in cases if (c.title, c.clause_type) in wanted]
    return cases
