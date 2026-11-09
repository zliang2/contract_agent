"""Hand-label a random sample of answers and measure agreement with the automatic checks.

    uv run python -m evals.label_sample draw REPORT --n 30 --seed 0   # writes a labeling file
    (edit the file: set "human_grounded": true/false for each answer)
    uv run python -m evals.label_sample score LABELS                  # prints agreement

Samples answered cases from a full-stage report, verifier-off condition (every generated
claim, so failures are visible). An answer is grounded if every claim is supported by the
contract text it cites. Agreement is computed against:
- the verifier (answer grounded iff every claim passed the verifier), and
- the independent judge (answer grounded iff no claim was judged unsupported).
"""

import argparse
import json
import random
from pathlib import Path

from clausewitz.db import connect


def draw(report_path: str, n: int, seed: int) -> Path:
    report = json.loads(Path(report_path).read_text())
    answered = [r for r in report["records"] if r["claims"]]
    picked = random.Random(seed).sample(answered, min(n, len(answered)))
    items = []
    with connect(vector=False) as conn:
        for r in picked:
            claims = []
            for c in r["claims"]:
                ids = [x["chunk_id"] for x in c["citations"]]
                texts = dict(conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (ids,)))
                claims.append(
                    {
                        "claim": c["text"],
                        "cited": [
                            {"quote": x["quote"], "chunk_text": texts.get(x["chunk_id"])}
                            for x in c["citations"]
                        ],  # fmt: skip
                        "verifier_passed": c["verifier_passed"],
                        "judge_supported": c["judge_supported"],
                    }
                )
            items.append(
                {
                    "contract_id": r["contract_id"],
                    "clause_type": r["clause_type"],
                    "answer": r["answer"],
                    "claims": claims,
                    "human_grounded": None,
                    "human_note": "",
                }
            )
    out = Path(report_path).with_name(Path(report_path).stem + f"_labels_n{n}_seed{seed}.json")
    out.write_text(
        json.dumps(
            {
                "report": Path(report_path).name,
                "instructions": "For each item set human_grounded to true if EVERY claim is "
                "supported by the contract text it cites, else false. Judge from chunk_text, "
                "not from the verifier_passed/judge_supported fields (hide them while labeling "
                "if you can).",
                "items": items,
            },
            indent=1,
        )
    )
    return out


def score(labels_path: str) -> dict:
    data = json.loads(Path(labels_path).read_text())
    items = [i for i in data["items"] if i["human_grounded"] is not None]
    if not items:
        raise SystemExit("no labeled items yet")

    def agreement(predict) -> dict:
        pairs = [(i["human_grounded"], predict(i)) for i in items]
        return {
            "agreement": sum(h == p for h, p in pairs) / len(pairs),
            "both_grounded": sum(h and p for h, p in pairs),
            "both_ungrounded": sum(not h and not p for h, p in pairs),
            "human_grounded_auto_ungrounded": sum(h and not p for h, p in pairs),
            "human_ungrounded_auto_grounded": sum(not h and p for h, p in pairs),
        }

    result = {
        "labels": Path(labels_path).name,
        "n_labeled": len(items),
        "human_ungrounded_rate": sum(not i["human_grounded"] for i in items) / len(items),
        "verifier": agreement(lambda i: all(c["verifier_passed"] for c in i["claims"])),
        "judge": agreement(lambda i: all(c["judge_supported"] is not False for c in i["claims"])),
    }
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("report")
    d.add_argument("--n", type=int, default=30)
    d.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("labels")
    args = ap.parse_args()
    if args.cmd == "draw":
        print("wrote", draw(args.report, args.n, args.seed))
    else:
        score(args.labels)
