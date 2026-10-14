"""Build eval cases: one per (contract, clause type) in a split (docs/SPEC.md §7).

The question is CUAD's own question text for that clause type. A case with no gold span
for its clause type is a "should refuse" case. Gold spans and chunk spans are read from
the database, so cases and retrieval always agree on offsets.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from clausewitz.ingest.load_cuad import load_cuad


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


def build_cases(conn, split: str, subset: str | Path | None = None) -> list[Case]:
    """All cases for `split`, optionally limited to a subset file of {title, clause_type}."""
    _, questions = load_cuad()
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
        Case(cid, title, q.clause_type, q.question, gold.get((cid, q.clause_type), []))
        for cid, title in contracts
        for q in questions
    ]
    if subset:
        wanted = {(x["title"], x["clause_type"]) for x in json.loads(Path(subset).read_text())}
        cases = [c for c in cases if (c.title, c.clause_type) in wanted]
    return cases
