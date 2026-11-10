"""CI retrieval gate: fail if recall@8 fell too far below the stored main baseline (§7).

    uv run python -m evals.check_regression REPORT [--baseline evals/subsets/ci_50_baseline.json]

The baseline file holds the rerank (production path) recall@8 that `main` scored on the
`ci_50` subset, plus the report it was copied from. The check fails (exit 1) if the new
report's value is more than `MAX_DROP_POINTS` below it, or if the report was run on a
different subset or with too few cases. All four retrieval modes are printed for context;
only the production path is gated.
"""

import argparse
import json
import sys
from pathlib import Path

MAX_DROP_POINTS = 2.0
GATED_MODE = "rerank"
METRIC = "recall@8"


def check(report: dict, baseline: dict) -> tuple[bool, str]:
    """Return (passed, message) for a retrieval report against the baseline."""
    if report["config"]["subset"] != baseline["subset"]:
        return False, f"report subset {report['config']['subset']!r} != {baseline['subset']!r}"
    n = report["metrics"]["n_cases"]
    if n != baseline["n_cases"]:
        return False, f"report evaluated {n} cases, baseline has {baseline['n_cases']}"
    new = report["metrics"][METRIC][GATED_MODE]
    old = baseline[METRIC]
    change = 100 * (new - old)
    drop = -change
    msg = f"{GATED_MODE} {METRIC}: {new:.3f} (main {old:.3f}, change {change:+.1f} points)"
    if drop > MAX_DROP_POINTS + 1e-9:
        return False, msg + f" -- dropped more than {MAX_DROP_POINTS:g} points"
    return True, msg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("--baseline", default="evals/subsets/ci_50_baseline.json")
    args = ap.parse_args(argv)
    report = json.loads(Path(args.report).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    for mode, value in report["metrics"][METRIC].items():
        print(f"{mode:<8} {METRIC} {value:.3f}")
    passed, msg = check(report, baseline)
    print(("PASS " if passed else "FAIL ") + msg)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
