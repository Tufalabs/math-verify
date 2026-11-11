import sys

import pytest
from sympy import Integer

from math_verify import parse


@pytest.fixture
def integer_digit_limit():
    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("Python does not limit integer string conversion")
    previous_limit = sys.get_int_max_str_digits()
    limit = sys.int_info.str_digits_check_threshold
    sys.set_int_max_str_digits(limit)
    try:
        yield limit
    finally:
        sys.set_int_max_str_digits(previous_limit)


@pytest.mark.parametrize("fallback_mode", ["no_fallback", "first_match"])
def test_rejects_integer_beyond_digit_limit(integer_digit_limit, fallback_mode):
    literal = "7" * (integer_digit_limit + 1)

    assert (
        parse(
            f"Final answer is {literal}.",
            fallback_mode=fallback_mode,
            extraction_mode="first_match",
            raise_on_error=True,
        )
        == []
    )


def test_integer_at_digit_limit_is_extracted(integer_digit_limit):
    literal = "7" * integer_digit_limit

    assert parse(
        f"Final answer is {literal}.",
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        raise_on_error=True,
    ) == [Integer(int(literal))]
