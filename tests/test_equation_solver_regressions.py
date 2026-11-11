import pytest
from sympy import Eq, Symbol, pi, sin

from math_verify import LatexExtractionConfig, parse, verify


@pytest.mark.parametrize("prediction", [r"$2x^2=2$", r"$2=2x^2$"])
def test_scaled_quadratic_has_same_roots(prediction):
    extraction = [LatexExtractionConfig()]
    gold = parse(r"$x^2=1$", extraction, fallback_mode="no_fallback")
    pred = parse(prediction, extraction, fallback_mode="no_fallback")

    assert verify(gold, pred, raise_on_error=True)
    assert verify(pred, gold, raise_on_error=True)


def test_decimal_scaling_preserves_root_equivalence():
    x = Symbol("x", real=True)
    polynomial = (x - 2) * (x**2 - 2)
    gold = Eq(polynomial, 0)
    pred = Eq(0.5 * polynomial, 0)

    assert verify(gold, pred, raise_on_error=True)
    assert verify(pred, gold, raise_on_error=True)


def test_different_roots_or_variables_are_not_equivalent():
    x, y = Symbol("x", real=True), Symbol("y", real=True)
    gold = Eq(x**2, 1)

    for pred in (Eq(x**2, 4), Eq(x, 1), Eq(y**2, 1)):
        assert not verify(gold, pred, raise_on_error=True)
        assert not verify(pred, gold, raise_on_error=True)


def test_empty_solver_results_do_not_establish_equivalence():
    x = Symbol("x", real=True)

    identity = Eq(x, x, evaluate=False)
    contradiction = Eq(x, x + 1, evaluate=False)
    assert not verify(identity, contradiction, raise_on_error=True)


def test_partial_roots_do_not_establish_equivalence():
    x = Symbol("x", real=True)

    assert not verify(Eq(sin(x), 0), Eq(x * (x - pi), 0), raise_on_error=True)
    assert not verify(Eq(x * (x**5 - pi * x - 1), 0), Eq(x, 0), raise_on_error=True)
