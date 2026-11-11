import pytest
from latex2sympy2_extended.sets import FiniteSet
from sympy import Rational

from math_verify import LatexExtractionConfig, parse


@pytest.mark.parametrize(
    "answer",
    [
        r"The final answer is $1$ and $\frac{2}{}$.",
        r"The final answer is $1$ and $\frac{2}{}$ and $3$.",
        r"The final answer is $\frac{2}{}$ and $1$.",
    ],
)
def test_first_match_rejects_an_incomplete_multipart_answer(answer):
    assert (
        parse(
            answer,
            extraction_mode="first_match",
            fallback_mode="no_fallback",
            raise_on_error=True,
        )
        == []
    )


def test_malformed_multipart_fallback_keeps_all_components():
    answer = r"The final answer is $1$ and $\frac{2}{}$."
    assert parse(answer, extraction_mode="first_match") == [r"1 and \frac{2}{}"]


def test_valid_multipart_answer_keeps_all_components():
    answer = r"The final answer is $1$ and $\frac{2}{3}$."
    assert parse(
        answer, extraction_mode="first_match", fallback_mode="no_fallback"
    ) == [FiniteSet(1, Rational(2, 3))]


def test_any_match_can_continue_to_an_independent_candidate():
    answer = (
        r"The intermediate result was $4$. The final answer is $1$ and $\frac{2}{}$."
    )
    assert parse(answer, [LatexExtractionConfig()], fallback_mode="no_fallback") == [4]
