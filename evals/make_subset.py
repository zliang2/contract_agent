"""Write a fixed eval subset file (e.g. `evals/subsets/ci_50.json`) for CI (docs/SPEC.md §7).

    uv run python -m evals.make_subset --split dev --n 50 --seed 0 --out evals/subsets/ci_50.json

Samples `n` answerable cases from the split with a seeded RNG and writes them as
`[{title, clause_type}]`, the format `build_cases(subset=...)` reads. Only answerable
cases are drawn because the CI gate measures retrieval recall, which is defined on
answerable cases only; 50 of them make one case worth exactly 2 recall points.
"""

import argparse
import json
from pathlib import Path

from clausewitz.db import connect
from evals.build_cases import build_cases
from evals.run_eval import sample_cases


def make_subset(conn, split: str, n: int, seed: int) -> list[dict]:
    answerable = [c for c in build_cases(conn, split) if c.answerable]
    picked = sample_cases(answerable, n, seed)
    return [{"title": c.title, "clause_type": c.clause_type} for c in picked]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with connect(vector=False) as conn:
        subset = make_subset(conn, args.split, args.n, args.seed)
    Path(args.out).write_text(json.dumps(subset, indent=1) + "\n")
    print(f"wrote {len(subset)} cases to {args.out}")


if __name__ == "__main__":
    main()
