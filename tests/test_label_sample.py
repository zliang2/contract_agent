import json

import pytest

from evals.label_sample import score


def item(human, verifier, judge):
    return {"human_grounded": human,
            "claims": [{"verifier_passed": verifier, "judge_supported": judge}]}  # fmt: skip


def test_score_agreement(tmp_path):
    items = [item(True, True, True), item(False, False, False), item(False, True, False),
             item(True, False, True), item(None, True, True)]  # fmt: skip
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"items": items}))
    r = score(str(p))
    assert r["n_labeled"] == 4
    assert r["verifier"]["agreement"] == pytest.approx(2 / 4)
    assert r["judge"]["agreement"] == pytest.approx(4 / 4)
    assert r["verifier"]["human_ungrounded_auto_grounded"] == 1
