import pytest

from math_verify import LatexExtractionConfig, parse, verify


@pytest.mark.parametrize(
    ("gold", "prediction", "expected"),
    [
        pytest.param("x=1", "x=2=1", False, id="contradictory-assignment"),
        pytest.param("x=2=1", "x=1", False, id="contradictory-gold"),
        pytest.param("x=2=1", "x=2=1", False, id="two-contradictory-chains"),
        pytest.param("1", "x=2=1", False, id="contradictory-scalar-prediction"),
        pytest.param("1", "x^2=2=1", False, id="contradictory-equation"),
        pytest.param("x=2=1", "1", False, id="contradictory-gold-to-scalar"),
        pytest.param("1", "x=2=1=1", False, id="contradiction-before-last-link"),
        pytest.param("1", r"x=\frac{1}{2}=1", False, id="contradictory-fraction"),
        pytest.param(
            r"\frac{46}{5}", r"x=\frac{46}{5}=9.2", True, id="equal-fraction-decimal"
        ),
        pytest.param(
            r"\frac{1}{3}", r"x=\frac{1}{3}=0.333333", True, id="rounded-decimal"
        ),
        pytest.param("2", "x=2=2.0", True, id="equal-integer-decimal"),
        pytest.param("x=1", "x=y=1", True, id="symbolic-chain"),
    ],
)
def test_numeric_links_in_parsed_chains(gold, prediction, expected):
    options = {
        "extraction_config": [LatexExtractionConfig()],
        "extraction_mode": "first_match",
        "fallback_mode": "no_fallback",
        "raise_on_error": True,
    }
    gold_objects = parse(f"${gold}$", **options)
    prediction_objects = parse(f"${prediction}$", **options)
    assert gold_objects and prediction_objects
    assert verify(gold_objects, prediction_objects, raise_on_error=True) is expected
