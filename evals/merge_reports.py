"""Merge sharded eval reports (`run_eval --shard I/N`) into one complete report.

    uv run python -m evals.merge_reports evals/reports/*_shard*of8.json

Checks that the shards share one commit and config and together cover every shard index
exactly once, then recomputes all metrics from the combined per-case records.
"""

import json
import sys
from pathlib import Path

from evals.run_eval import print_summary, retrieval_metrics, write_report


def merge(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("no reports to merge")
    first = reports[0]
    for r in reports:
        if r["git"] != first["git"]:
            raise ValueError("shards come from different commits or a dirty tree")
        if {**r["config"], "shard": None} != {**first["config"], "shard": None}:
            raise ValueError("shards were run with different configs")
    shards = sorted(tuple(map(int, r["config"]["shard"].split("/"))) for r in reports)
    n = shards[0][1]
    if shards != [(i, n) for i in range(n)]:
        raise ValueError(f"expected shards 0..{n - 1} of {n} exactly once, got {shards}")

    records = [rec for r in reports for rec in r["records"]]
    return {
        "command": [r["command"] for r in reports],
        "git": first["git"],
        "date": max(r["date"] for r in reports),
        "duration_s": sum(r["duration_s"] for r in reports),
        "environment": first["environment"],
        "config": {**first["config"], "shard": f"merged {n} shards"},
        "cases": {
            "total": first["cases"]["total"],
            "answerable_evaluated": len(records),
        },
        "metrics": retrieval_metrics(records),
        "records": records,
    }


def main(paths: list[str]) -> Path:
    report = merge([json.loads(Path(p).read_text()) for p in paths])
    c = report["config"]
    name = f"{report['date'][:10]}_{c['split']}_{c['stage']}_{report['git']['commit'][:7]}"
    if c["limit"]:
        name += f"_limit{c['limit']}"
    name += f"_merged{len(report['command'])}"  # never overwrite an unsharded run's report
    if report["git"]["dirty"]:
        name += "_dirty"
    path = write_report(report, name)
    print_summary(report["metrics"])
    print(f"report: {path}")
    return path


if __name__ == "__main__":
    main(sys.argv[1:])
