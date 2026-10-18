"""One exact real number read from a short piece of a reply, for the graders that read many of them.

The `inequality` and `matrix` graders read a reply as a structure (bounds and intervals, rows and entries) and
then need each piece as an exact number. A piece is read as a plain number first (`3`, `-2.5`, `7/3`, `2 1/2`),
and only otherwise as a closed-form constant (`√2`, `2sqrt(3)/3`, `π/4`, `ln 2`, `e^2`) through the same
restricted parser the expression grader uses, never the spoken-word fallback. A piece that names a variable, is
not real, or is infinite is not a number: None.
"""

from __future__ import annotations

import re

import sympy

from .safe_parse import TooCostly, parse_expression_safely
from .typed import _TRANSFORMATIONS, _PI_SYMBOL, _parse_typed_fraction, _sympy_text

#: Longest piece read as one value. A bound or an entry in the library is a dozen characters at most.
MAX_VALUE_LENGTH = 40
#: The characters a closed-form constant is written with; anything else is not a number.
_CONSTANT_CHARS_RE = re.compile(r"[0-9a-zA-Z.+\-*/^()√π× \t]+")
_SPACED_DIGITS_RE = re.compile(r"\d[ \t]+\d")
_EULER_POWER_RE = re.compile(r"(?<![\w.])e(?=\s*\*\*)")


def read_value(text: str):
    """`text` as an exact SymPy real number, or None."""
    text = (text or "").strip()
    if not text or len(text) > MAX_VALUE_LENGTH:
        return None
    plain = _parse_typed_fraction(text)
    if plain is not None:
        return sympy.Rational(plain.numerator, plain.denominator)
    if not _CONSTANT_CHARS_RE.fullmatch(text) or _SPACED_DIGITS_RE.search(text):
        return None  # "2 000" is not 2*000: only a mixed number ("2 1/2", read above) may have digits side by side
    candidate = _sympy_text(text)
    if candidate is None:
        return None
    candidate = _EULER_POWER_RE.sub("E", candidate)  # "e^2" is exp(2); a lone "e" is still not a number here
    try:
        value = parse_expression_safely(candidate, transformations=_TRANSFORMATIONS)
    except TooCostly:
        return None
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return None
    except Exception:  # noqa: BLE001 - tokenize.TokenError and whatever else a malformed piece raises
        return None
    if not isinstance(value, sympy.Basic):
        return None
    if _PI_SYMBOL in value.free_symbols:
        value = value.xreplace({_PI_SYMBOL: sympy.pi})
    if value.free_symbols or not value.is_number or value.is_real is not True or not value.is_finite:
        return None
    return value


def same_number(left, right) -> bool:
    """Whether two exact real numbers are equal. Both come from `read_value` or from the generator."""
    if left == right:
        return True
    if not isinstance(left, sympy.Basic) or not isinstance(right, sympy.Basic):
        return False  # never hand text to `simplify`: it would sympify it
    try:
        return bool(sympy.simplify(left - right) == 0)
    except Exception:  # noqa: BLE001 - a comparison SymPy cannot decide is not equality
        return False
