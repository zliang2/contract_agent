from evals.check_regression import check

BASELINE = {"subset": "evals/subsets/ci_50.json", "n_cases": 50, "recall@8": 0.90}


def report(value, subset="evals/subsets/ci_50.json", n=50):
    return {
        "config": {"subset": subset},
        "metrics": {"n_cases": n, "recall@8": {"bm25": 0.5, "rerank": value}},
    }


def test_equal_or_better_passes():
    assert check(report(0.90), BASELINE)[0]
    assert check(report(0.96), BASELINE)[0]


def test_drop_of_exactly_two_points_passes():
    # One case out of 50 is exactly 2 points; the spec fails only drops of *more* than 2.
    assert check(report(0.88), BASELINE)[0]


def test_drop_of_more_than_two_points_fails():
    passed, msg = check(report(0.86), BASELINE)
    assert not passed
    assert "dropped more than 2 points" in msg


def test_wrong_subset_or_case_count_fails():
    assert not check(report(0.90, subset=None), BASELINE)[0]
    assert not check(report(0.90, n=49), BASELINE)[0]
