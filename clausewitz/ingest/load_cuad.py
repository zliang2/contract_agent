"""Ingestion step 1–2: download CUAD v1, parse it, and split contracts into dev/test.

CUAD_v1.json is SQuAD-style: one entry per contract with a single `context` (the full
contract text) and 41 questions, one per clause type. Gold answers are character
offsets into `context`, so `context` is used verbatim as `contracts.full_text`.

Run `python -m clausewitz.ingest.load_cuad --download` to fetch and unpack the release.
"""

import argparse
import hashlib
import json
import random
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from clausewitz.config import get_settings

ZENODO_URL = "https://zenodo.org/api/records/4595826/files/CUAD_v1.zip/content"
ZENODO_MD5 = "c38f490a984420b8a62600db401fafd5"
SPLIT_SEED = 42
TEST_FRACTION = 0.2


@dataclass
class GoldSpan:
    clause_type: str
    start_char: int
    end_char: int


@dataclass
class Contract:
    title: str
    full_text: str
    split: str = ""
    gold: list[GoldSpan] = field(default_factory=list)


@dataclass
class ClauseQuestion:
    clause_type: str
    question: str


def parse_cuad(raw: dict) -> tuple[list[Contract], list[ClauseQuestion]]:
    """Turn the CUAD JSON into contracts with gold spans, plus the 41 clause questions.

    The clause type is the suffix of each question id (`<title>__<clause type>`).
    Raises if an answer's offsets don't reproduce its text — the whole pipeline relies on them.
    """
    contracts: list[Contract] = []
    questions: dict[str, str] = {}
    for doc in raw["data"]:
        (para,) = doc["paragraphs"]
        contract = Contract(title=doc["title"], full_text=para["context"])
        for qa in para["qas"]:
            prefix = doc["title"] + "__"
            if not qa["id"].startswith(prefix):
                raise ValueError(f"unexpected question id format: {qa['id']}")
            clause_type = qa["id"][len(prefix) :]
            questions.setdefault(clause_type, qa["question"])
            for ans in qa["answers"]:
                start = ans["answer_start"]
                end = start + len(ans["text"])
                if contract.full_text[start:end] != ans["text"]:
                    raise ValueError(f"answer offsets don't match text in {qa['id']}")
                contract.gold.append(GoldSpan(clause_type, start, end))
        contracts.append(contract)
    clause_questions = [ClauseQuestion(t, q) for t, q in sorted(questions.items())]
    return contracts, clause_questions


def assign_splits(contracts: list[Contract], seed: int = SPLIT_SEED) -> None:
    """Assign 80% dev / 20% test by contract, deterministically.

    Titles are sorted before shuffling so the split depends only on the seed and the
    set of titles, not on file order.
    """
    titles = sorted(c.title for c in contracts)
    random.Random(seed).shuffle(titles)
    n_test = round(len(titles) * TEST_FRACTION)
    test = set(titles[:n_test])
    for c in contracts:
        c.split = "test" if c.title in test else "dev"


def load_cuad(path: str | Path | None = None) -> tuple[list[Contract], list[ClauseQuestion]]:
    """Read CUAD from disk, parse it, and assign splits. Contracts come back sorted by title."""
    path = Path(path or get_settings().cuad_path)
    with path.open() as f:
        contracts, questions = parse_cuad(json.load(f))
    assign_splits(contracts)
    contracts.sort(key=lambda c: c.title)
    return contracts, questions


def download(dest_dir: str | Path = "data/raw") -> Path:
    """Fetch CUAD_v1.zip from Zenodo, verify its MD5, and extract only the JSON."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "CUAD_v1.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(ZENODO_URL, zip_path)
    md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if md5 != ZENODO_MD5:
        raise ValueError(f"CUAD_v1.zip checksum mismatch: {md5}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("CUAD_v1/CUAD_v1.json", dest_dir)
    return dest_dir / "CUAD_v1" / "CUAD_v1.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="fetch the release from Zenodo")
    args = ap.parse_args()
    if args.download:
        print("downloaded to", download())
    contracts, questions = load_cuad()
    n_test = sum(c.split == "test" for c in contracts)
    print(
        f"{len(contracts)} contracts ({len(contracts) - n_test} dev / {n_test} test), "
        f"{len(questions)} clause types, {sum(len(c.gold) for c in contracts)} gold spans"
    )
