"""Run an evaluation and write a JSON report to evals/reports/ (docs/SPEC.md §7).

    uv run python -m evals.run_eval --split dev --stage retrieval
        [--subset FILE] [--limit N] [--shard I/N] [--question-style plain|cuad_template]

Retrieval stage: for every answerable case, retrieve within the case's contract and
measure recall@k for BM25-only, vector-only, hybrid (RRF) and hybrid + rerank. The
pipeline runs once per case in "rerank" mode; the other modes' rankings are read from its
per-stage outputs, which equals running them separately because every stage is
deterministic. Also reports an exact random-k baseline and recall restricted to contracts
with more than k chunks (in smaller contracts, top-k is the whole contract).

Each case is stored as a small record (first relevant rank per mode, chunk counts,
timings) and all metrics are computed from records, so `--shard I/N` runs can be merged
with `evals.merge_reports` and every number can be recomputed from the report.

Every report records the git commit (and whether the tree was dirty), date, config and
metrics. Numbers quoted in docs must come from a complete report from a clean tree.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from clausewitz import pipeline
from clausewitz.config import get_settings
from clausewitz.db import connect
from clausewitz.retrieval.rerank import RERANK_MODEL
from evals.build_cases import QUESTION_STYLES, build_cases
from evals.metrics import first_relevant_rank, is_relevant, percentiles, random_hit_probability

REPORTS_DIR = Path("evals/reports")
KS = (1, 3, 8)
POOL_K = pipeline.CANDIDATES_K
FINAL_K = pipeline.FINAL_K
MODES = ("bm25", "vector", "hybrid", "rerank")


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def run_retrieval_cases(conn, cases) -> list[dict]:
    """Retrieve for each case and return one record per case."""
    chunks: dict[int, list[tuple[int, int, int]]] = defaultdict(list)  # contract -> chunks
    for cid, chunk_id, s, e in conn.execute(
        "SELECT contract_id, id, start_char, end_char FROM chunks ORDER BY id"
    ):
        chunks[cid].append((chunk_id, s, e))
    span_of = {chunk_id: (s, e) for rows in chunks.values() for chunk_id, s, e in rows}

    records = []
    for case in tqdm(cases, desc="retrieval eval", mininterval=10):
        r = pipeline.retrieve(case.question, case.contract_id, conn, mode="rerank")
        ranked = {
            "bm25": r.candidates["bm25"],
            "vector": r.candidates["vector"],
            "hybrid": r.candidates["hybrid"],
            "rerank": r.chunk_ids,
        }
        contract_chunks = chunks[case.contract_id]
        records.append(
            {
                "contract_id": case.contract_id,
                "clause_type": case.clause_type,
                "n_chunks": len(contract_chunks),
                "n_relevant": sum(is_relevant((s, e), case.gold) for _, s, e in contract_chunks),
                "first_relevant_rank": {
                    m: first_relevant_rank([span_of[c] for c in ranked[m]], case.gold)
                    for m in MODES
                },
                "timings_ms": {k: round(v, 2) for k, v in r.timings_ms.items()},
            }
        )
    return records


def retrieval_metrics(records: list[dict]) -> dict:
    """All retrieval metrics, computed only from per-case records."""

    def mean(xs) -> float:
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    def recall(recs, mode, k) -> float:
        return mean((r["first_relevant_rank"][mode] or k + 1) <= k for r in recs)

    def random_baseline(recs) -> float:
        return mean(random_hit_probability(r["n_chunks"], r["n_relevant"], FINAL_K) for r in recs)

    big = [r for r in records if r["n_chunks"] > FINAL_K]
    timings = defaultdict(list)
    for r in records:
        for stage, ms in r["timings_ms"].items():
            timings[stage].append(ms)
    return {
        "n_cases": len(records),
        f"n_cases_contract_gt_{FINAL_K}_chunks": len(big),
        f"recall@{FINAL_K}": {m: recall(records, m, FINAL_K) for m in MODES},
        "recall_at_k": {m: {str(k): recall(records, m, k) for k in KS} for m in MODES},
        f"recall@{POOL_K}_candidate_pool": {
            m: recall(records, m, POOL_K) for m in ("bm25", "vector", "hybrid")
        },
        f"random_baseline_recall@{FINAL_K}": random_baseline(records),
        f"contracts_gt_{FINAL_K}_chunks": {
            f"recall@{FINAL_K}": {m: recall(big, m, FINAL_K) for m in MODES},
            f"random_baseline_recall@{FINAL_K}": random_baseline(big),
        },
        "latency_ms": {stage: percentiles(v) for stage, v in timings.items()},
    }


def environment() -> dict:
    from clausewitz.ingest.embed import get_model

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": str(get_model().device),
    }


def print_summary(metrics: dict) -> None:
    k = FINAL_K
    print(
        f"\n{metrics['n_cases']} answerable cases "
        f"({metrics[f'n_cases_contract_gt_{k}_chunks']} in contracts with >{k} chunks)"
    )
    print(f"{'mode':<8} {'R@1':>6} {'R@3':>6} {'R@8':>6}  R@8 (contracts >{k} chunks)")
    for m in MODES:
        r = metrics["recall_at_k"][m]
        big = metrics[f"contracts_gt_{k}_chunks"][f"recall@{k}"][m]
        print(f"{m:<8} {r['1']:>6.3f} {r['3']:>6.3f} {r['8']:>6.3f}  {big:.3f}")
    rb = metrics[f"random_baseline_recall@{k}"]
    rb_big = metrics[f"contracts_gt_{k}_chunks"][f"random_baseline_recall@{k}"]
    print(f"{'random':<8} {'':>6} {'':>6} {rb:>6.3f}  {rb_big:.3f}")


def write_report(report: dict, name: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    return path


def parse_shard(value: str) -> tuple[int, int]:
    i, n = (int(x) for x in value.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError("shard must be I/N with 0 <= I < N")
    return i, n


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--stage", choices=["retrieval", "full"], required=True)
    ap.add_argument("--subset", help="JSON list of {title, clause_type}")
    ap.add_argument("--limit", type=int, help="first N answerable cases (smoke runs)")
    ap.add_argument("--shard", type=parse_shard, help="I/N: run every Nth case starting at I")
    ap.add_argument("--question-style", choices=QUESTION_STYLES, default="plain")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--model")
    args = ap.parse_args(argv)
    if args.stage == "full":
        sys.exit("--stage full arrives in Phase 3")

    started = time.time()
    with connect() as conn:
        cases = build_cases(conn, args.split, args.subset, args.question_style)
        answerable = [c for c in cases if c.answerable]
        if args.limit:
            answerable = answerable[: args.limit]
        if args.shard:
            i, n = args.shard
            answerable = answerable[i::n]
        records = run_retrieval_cases(conn, answerable)

    git = git_state()
    report = {
        "command": "python -m evals.run_eval " + " ".join(sys.argv[1:] if argv is None else argv),
        "git": git,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "environment": environment(),
        "config": {
            "split": args.split,
            "stage": args.stage,
            "subset": args.subset,
            "limit": args.limit,
            "shard": "/".join(map(str, args.shard)) if args.shard else None,
            "scope": "single contract (contract_id given)",
            "question_style": args.question_style,
            "candidates_k": pipeline.CANDIDATES_K,
            "rerank_candidates": pipeline.RERANK_CANDIDATES,
            "final_k": FINAL_K,
            "embed_model": get_settings().embed_model,
            "rerank_model": RERANK_MODEL,
        },
        "cases": {"total": len(cases), "answerable_evaluated": len(records)},
        "metrics": retrieval_metrics(records),
        "records": records,
    }
    name = f"{report['date'][:10]}_{args.split}_{args.stage}_{git['commit'][:7]}"
    if args.question_style != "plain":
        name += f"_{args.question_style}"
    if args.limit:
        name += f"_limit{args.limit}"
    if args.shard:
        name += f"_shard{args.shard[0]}of{args.shard[1]}"
    if git["dirty"]:
        name += "_dirty"
    path = write_report(report, name)
    print_summary(report["metrics"])
    if git["dirty"]:
        print("WARNING: working tree is dirty; don't quote these numbers.")
    print(f"report: {path}")
    return path


if __name__ == "__main__":
    main()
