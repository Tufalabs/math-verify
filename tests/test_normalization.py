import pytest

from math_verify import LatexExtractionConfig, parse, verify
from math_verify.normalization import (
    normalize_math_presentation,
    prepare_standalone_math,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"\displaystyle \sqrt{5}", r"\sqrt{5}"),
        (r"\textstyle \scriptstyle \scriptscriptstyle 1", "1"),
        (r"\mathbf{+2}", "+2"),
        (r"\mathrm{-2.5}.", "-2.5"),
        (r"\mathit{.5}", ".5"),
        (r"\mathbf{2}3", r"\mathbf{2}3"),
        (r"\mathbf{2}\mathbf{3}", r"\mathbf{2}\mathbf{3}"),
        (r"\mathrm{m}", r"\mathrm{m}"),
    ],
)
def test_normalize_math_presentation(text, expected):
    assert normalize_math_presentation(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"  -\dfrac{1}{2}.  ", r"\[-\dfrac{1}{2}\]"),
        ("sqrt(13)", r"\[\sqrt{13}\]"),
        ("2 sqrt(13) + 5", r"\[2 \sqrt{13} + 5\]"),
        ("sqrt(2+sqrt((9)))", r"\[\sqrt{2+\sqrt{(9)}}\]"),
        ("2 sqrt((13)", ""),
        ("2 sqrt(13))", ""),
        ("42", None),
        (r"The answer is \sqrt{5}.", None),
        (r"An intermediate is \sqrt{5}. The answer is $7$.", None),
        (r"$\sqrt{5}$", None),
        (r"\(\sqrt{5}\)", None),
        (r"\[\sqrt{5}\]", None),
        (r"\boxed{\sqrt{5}}", None),
        (r"\sqrt{5} \approx 100", None),
        (r"sqrt(4)=\sqrt{289}", None),
    ],
)
def test_prepare_standalone_math(text, expected):
    assert prepare_standalone_math(text) == expected


def _parse_latex(text):
    return parse(
        text,
        extraction_config=(LatexExtractionConfig(),),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        raise_on_error=True,
    )


@pytest.mark.parametrize(
    "text",
    [
        r"\sqrt{5}",
        r"\frac{3\sqrt{2}}{4}",
        r"\dfrac{123}{1024}",
        r"\frac{2\sqrt{5}}{5}",
        r"4 + i, 4 - i, 4 + i \sqrt{5}, 4 - i \sqrt{5}",
    ],
)
def test_audit_expressions_match_their_delimited_forms(text):
    prepared = prepare_standalone_math(text)
    assert prepared
    assert verify(
        _parse_latex("$" + text + "$"), _parse_latex(prepared), raise_on_error=True
    )


@pytest.mark.parametrize(
    ("prediction", "target", "expected"),
    [
        (r"\sqrt[3]{8}", "2", True),
        (r"\sqrt{\frac{1}{4}}", "0.5", True),
        (r"\frac{1}{2} + \sqrt{2}", r"\sqrt{2} + \frac{1}{2}", True),
        ("sqrt(2+sqrt((9)))", r"\sqrt{5}", True),
        (r"2\sqrt{5}", "2", False),
        (r"\sqrt{5} + 2", "2", False),
        ("sqrt(13) + 5", "5", False),
        ("2 sqrt((13)", "2", False),
    ],
)
def test_standalone_math_parsing_uses_the_full_expression(prediction, target, expected):
    prepared = prepare_standalone_math(prediction)
    assert prepared is not None
    assert (
        verify(
            _parse_latex("$" + target + "$"),
            _parse_latex(prepared),
            raise_on_error=True,
        )
        is expected
    )
