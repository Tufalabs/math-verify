import pytest

from math_verify import LatexExtractionConfig, parse, verify


@pytest.mark.parametrize(
    ("gold", "prediction", "expected"),
    [
        pytest.param(r"x \ne 2", r"2 \ne x", True, id="swapped-unequality"),
        pytest.param(r"x \ne 2", r"x+1 \ne 3", True, id="shifted-unequality"),
        pytest.param(r"x \ne 2", r"x \ne 3", False, id="different-excluded-value"),
        pytest.param(r"x \ne 2", "x < 2", False, id="different-relation"),
        pytest.param("1 < x < 3", "1 < x < 3 < y", False, id="extra-predicted-clause"),
        pytest.param(
            "1 < x < 3 < y", "1 < x < 3", False, id="missing-predicted-clause"
        ),
        pytest.param("1 < x < 3", "0 < x-1 < 2", True, id="equivalent-chain"),
        pytest.param("1 < x < 3", "1 < x < 4", False, id="different-chain-bound"),
        pytest.param("(2,1)", r"\{2,1,2\}", False, id="extra-ordered-element"),
        pytest.param("(2,1)", r"\{2,1\}", True, id="same-input-order"),
        pytest.param("(2,1)", r"\{1,2\}", False, id="different-input-order"),
    ],
)
def test_parsed_comparison_returns_a_verdict(gold, prediction, expected):
    options = {
        "extraction_config": [LatexExtractionConfig()],
        "extraction_mode": "first_match",
        "fallback_mode": "no_fallback",
        "raise_on_error": True,
    }
    gold_objects = parse(f"$ {gold} $", **options)
    prediction_objects = parse(f"$ {prediction} $", **options)
    assert gold_objects and prediction_objects
    assert verify(gold_objects, prediction_objects, raise_on_error=True) is expected
