import pytest

from evals.build_cases import plain_question

TEMPLATE = (
    'Highlight the parts (if any) of this contract related to "Termination For Convenience" '
    "that should be reviewed by a lawyer. Details: Can a party terminate this  contract "
    "without cause (solely by giving a notice and allowing a waiting  period to expire)?"
)


def test_plain_question_keeps_clause_name_and_details():
    assert plain_question(TEMPLATE) == (
        "Termination For Convenience: Can a party terminate this contract without cause "
        "(solely by giving a notice and allowing a waiting period to expire)?"
    )


def test_plain_question_rejects_unexpected_text():
    with pytest.raises(ValueError):
        plain_question("What is the governing law?")
