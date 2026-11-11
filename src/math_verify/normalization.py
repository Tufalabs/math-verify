"""Opt-in presentation and standalone math normalization."""

import re

_MATH_STYLE = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)(?![A-Za-z])\s*"
)
_STYLED_NUMBER = re.compile(
    r"\\(?:mathbf|mathrm|mathit)\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}"
)
_BARE_LATEX = re.compile(r"\\(?:dfrac|tfrac|frac|sqrt)(?![A-Za-z])")
_EXPLICIT_LATEX = re.compile(r"\$|\\(?:[\[(]|boxed(?![A-Za-z]))")
# Math-Verify can truncate at \approx. Limit this path to arithmetic commands.
_LATEX_COMMAND = re.compile(
    r"\\(?:(?:dfrac|tfrac|frac|sqrt|left|right|cdot|times|div|pi|infty|pm|mp)(?![A-Za-z])|[,!:;%{} ])"
)
_ASCII_ROOT = re.compile(r"(?<![A-Za-z\\])sqrt\s*\(")
_ROOT_OR_PAREN = re.compile(_ASCII_ROOT.pattern + r"|[()]")
_MATH_CHARACTERS = re.compile(r"[A-Za-z0-9\s{}\[\]()+.,*/^_|!:-]*")
_PROSE_WORD = re.compile(r"[A-Za-z]{2}")


def normalize_math_presentation(text: str) -> str:
    """Strip display styles and unwrap a whole styled numeric literal."""
    text = _MATH_STYLE.sub("", text)
    if styled := _STYLED_NUMBER.fullmatch(text.strip().removesuffix(".").rstrip()):
        return styled[1]
    return text


def _ascii_roots_to_latex(text: str) -> str | None:
    pieces, closing = [], []
    offset = 0
    for match in _ROOT_OR_PAREN.finditer(text):
        pieces.append(text[offset : match.start()])
        offset = match.end()
        token = match[0]
        if token == ")":
            if not closing:
                return None
            pieces.append(closing.pop())
        elif token == "(":
            pieces.append(token)
            closing.append(")")
        else:
            pieces.append(r"\sqrt{")
            closing.append("}")
    if closing:
        return None
    pieces.append(text[offset:])
    return "".join(pieces)


def prepare_standalone_math(text: str) -> str | None:
    """Prepare supported standalone roots and fractions for LaTeX-only parsing.

    Return display math for a recognized expression and an empty string for a
    recognized ASCII root with unbalanced parentheses. Return None for other
    inputs so the caller can use its ordinary extraction path.
    """
    candidate = text.strip().removesuffix(".").rstrip()
    has_ascii_root = _ASCII_ROOT.search(candidate) is not None
    if _EXPLICIT_LATEX.search(candidate) or not (
        has_ascii_root or _BARE_LATEX.search(candidate)
    ):
        return None
    unescaped = _LATEX_COMMAND.sub(" ", _ASCII_ROOT.sub("(", candidate))
    if not _MATH_CHARACTERS.fullmatch(unescaped) or _PROSE_WORD.search(unescaped):
        return None
    if has_ascii_root:
        candidate = _ascii_roots_to_latex(candidate)
        if candidate is None:
            return ""
    return rf"\[{candidate}\]"
