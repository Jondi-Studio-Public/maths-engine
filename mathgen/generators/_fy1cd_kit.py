"""Small shared helpers for the first-year complex-number, discrete-maths and integration builds.

A formula is written once as a pair, its learner-facing text and its TeX, and ``say`` joins prose and formulas into
the question's ``text`` and ``latex`` together, so the two never drift apart. The rest are the distractor builders the
builds share and the decimal-answer helper.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import sympy

from ..answers import Distractor, exact_value
from ..core import (
    _escape_prose,
    check,
    join_terms,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)

SUP = {1: "", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶"}


class F(tuple):
    """A formula: ``(plain text, TeX)``."""

    def __new__(cls, plain: str, tex: str):
        return super().__new__(cls, (plain, tex))


def say(*items) -> tuple[str, str]:
    """Join prose (plain strings) and formulas (``F``) into ``(text, latex)``."""
    text, tex = [], []
    for item in items:
        if isinstance(item, F):
            text.append(item[0])
            tex.append(item[1])
        else:
            text.append(item)
            tex.append(rf"\text{{{_escape_prose(item)}}}")
    return "".join(text), "".join(tex)


# -- numbers -------------------------------------------------------------------------------------------------


def frac(value) -> Fraction:
    return Fraction(value)


def pi_text(p) -> tuple[str, str]:
    """``p`` times pi as plain text and TeX: 0, π, -π/4, 2π/3."""
    p = Fraction(p)
    if p == 0:
        return "0", "0"
    sign = "-" if p < 0 else ""
    num, den = abs(p.numerator), p.denominator
    top = "π" if num == 1 else f"{num}π"
    ttop = r"\pi" if num == 1 else rf"{num}\pi"
    if den == 1:
        return sign + top, sign + ttop
    return f"{sign}{top}/{den}", rf"{sign}\frac{{{ttop}}}{{{den}}}"


def reduce_arg(p) -> Fraction:
    """``p`` (an argument in units of pi) moved into (-1, 1]."""
    p = Fraction(p)
    q = p - 2 * ((p + 1 - Fraction(1, 10**9)) // 2)
    check(-1 < q <= 1, f"argument {p} not reduced")
    return q


def surd(coef, root: int = 1) -> tuple[str, str]:
    """``coef`` times the square root of ``root`` (1 for a plain number), plain and TeX, in lowest form."""
    coef = Fraction(coef)
    if root == 1:
        return rational_text(coef), rational_text(coef)
    mag = abs(coef)
    sign = "-" if coef < 0 else ""
    if mag.denominator == 1:
        head = "" if mag == 1 else str(mag.numerator)
        return f"{sign}{head}√{root}", rf"{sign}{head}\sqrt{{{root}}}"
    head = "" if mag.numerator == 1 else str(mag.numerator)
    return (
        f"{sign}{head}√{root}/{mag.denominator}",
        rf"{sign}\frac{{{head or 1}\sqrt{{{root}}}}}{{{mag.denominator}}}",
    )


# -- distractors ---------------------------------------------------------------------------------------------


def fraction_dist(value, tag: str) -> Distractor:
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


def int_dist(value: int, tag: str) -> Distractor:
    return Distractor(str(int(value)), tag, int(value))


def decimal_text(value, places: int) -> tuple[float, str]:
    """``value`` (a SymPy expression or number) as ``(float, text)`` rounded half up to exactly ``places`` places."""
    if isinstance(value, (int, Fraction)):
        exact = Fraction(value)
    elif isinstance(value, float):
        exact = Fraction(repr(value))
    else:
        exact = Fraction(str(sympy.N(sympy.sympify(value), 40)))
    rounded = round_half_up(exact, places) if exact >= 0 else -round_half_up(-exact, places)
    text = stated_precision_string(rounded, places)
    return float(rounded), text


def decimal_dist(value, places: int, tag: str) -> Distractor:
    number, text = decimal_text(value, places)
    return Distractor(text, tag, number)


def pick_decimal(answer, candidates, places: int, n: int = 4, gap: float = 0.005) -> list[Distractor]:
    """Distractors for a decimal answer: each candidate is ``(value, tag)``; candidates within ``gap`` of the answer
    (or of an earlier pick) are dropped, so the grader cannot confuse them."""
    answer_value, answer_text = decimal_text(answer, places)
    chosen: list[Distractor] = []
    seen = [answer_value]
    for value, tag in candidates:
        try:
            number, _ = decimal_text(value, places)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if any(abs(number - s) < gap for s in seen):
            continue
        seen.append(number)
        chosen.append(decimal_dist(value, places, tag))
        if len(chosen) == n:
            break
    return pick_distractors(answer_text, chosen, n)


def poly_text(coeffs, var: str = "x") -> str:
    """A polynomial from its coefficients, highest power first: ``[1, -4, 6]`` is x² - 4x + 6."""
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        terms.append((c, "" if power == 0 else var + SUP[power] if power > 1 else var))
    return join_terms(terms)


def poly_tex(coeffs, var: str = "x") -> str:
    deg = len(coeffs) - 1
    out = ""
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = deg - i
        mag = abs(c)
        body = "" if (mag == 1 and power > 0) else str(mag)
        if power > 0:
            body += var + ("" if power == 1 else f"^{{{power}}}")
        out += (" - " if c < 0 else (" + " if out else "")) + body
        if not out.strip():
            out = body
    return out.strip() if out else "0"


def divisor_count(n: int) -> int:
    return sum(1 for d in range(1, n + 1) if n % d == 0)


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def ctext(re: int, im: int) -> str:
    """``re + im i`` the way a learner writes it: 3 - 2i, 4i, -i, 5."""
    if im == 0:
        return str(re)
    mag = "i" if abs(im) == 1 else f"{abs(im)}i"
    imag = mag if im > 0 else f"-{mag}"
    if re == 0:
        return imag
    return f"{re} {'+' if im > 0 else '-'} {mag}"
