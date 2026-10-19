"""Helpers shared by the `r4e_*` senior calculus and functions generators (not a generator itself).

Builds on the part builders in `exd_x_r2b_support`: every question is written once as a "part" (a prompt, an
answer, tagged wrong answers) and `single` turns one part into a single-answer question, so the tiers and the exam
chain share one source of distractor and rounding logic. Adds the two part kinds that module lacks: a list of
numbers and an inequality (a union of open intervals).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import latex_answer, rational_text
from .exd_x_r2b_support import exact_number_part as _exact_number_part
from .exd_x_r2b_support import fraction_part as _fraction_part
from .exd_x_r2b_support import (  # noqa: F401  (re-exported for the r4e generators)
    Redraw,
    _finish,
    _first_wrong,
    clear_of_boundary,
    draw,
    expression_part,
    number_part,
    real,
    typed,
)


def _whole(v) -> bool:
    return Fraction(v).denominator == 1


def _like_key_first(value, wrong):
    """Wrong answers of the key's own shape (whole number or not) first, so the key is not the only one of its shape."""
    same = [w for w in wrong if w[0] is not None and _whole(w[0]) == _whole(value)]
    return same + [w for w in wrong if w not in same]


def fraction_part(prompt, value, marks, working, wrong, **kw):
    return _fraction_part(prompt, value, marks, working, _like_key_first(value, wrong), **kw)


def exact_number_part(prompt, value, marks, working, wrong, **kw):
    return _exact_number_part(prompt, value, marks, working, _like_key_first(value, wrong), **kw)


def single(stem: str, part: dict) -> dict:
    """A single-answer question from `stem` and one part dict (its prompt joins the stem)."""
    text = f"{stem} {part['prompt']}".strip()
    return dict(
        text=text,
        answer_kind=part["answer_kind"],
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_latex=part["answer_latex"],
        working=part.get("working", []),
        distractors=part["distractors"],
    )


def num(value) -> str:
    """A rational as a learner writes it: `3`, `3/2`, `-1/2`."""
    return rational_text(sympy.Rational(value))


def list_text(values) -> str:
    return ", ".join(num(v) for v in sorted(values, key=sympy.Rational))


def list_part(prompt, values, marks, working, wrong) -> dict:
    """A part answered by a list of rational numbers in any order (kind `list`). `wrong` holds `(values, tag)`."""
    values = sorted((sympy.Rational(v) for v in values))
    key_text = list_text(values)

    def as_json(vs):
        return [int(v) if v.q == 1 else float(v) for v in sorted(sympy.Rational(x) for x in vs)]

    def build(vs, tag):
        return Distractor(list_text(vs), tag, as_json(vs))

    def same(vs):
        return sorted(sympy.Rational(x) for x in vs) == values

    cands = [(vs, t) for vs, t in wrong if vs]
    options = _first_wrong(key_text, cands, same, build)
    return _finish(
        dict(
            prompt=prompt,
            answer_kind="list",
            answer=as_json(values),
            answer_text=key_text,
            answer_latex=f"plain:{key_text}",
            marks=marks,
            working=working,
            distractors=options,
        ),
        None,
        None,
    )


def _parts(iv):
    """(low, high, low_closed, high_closed) from a 2- or 4-tuple (open by default)."""
    lo, hi = iv[0], iv[1]
    lc, hc = (iv[2], iv[3]) if len(iv) == 4 else (False, False)
    return lo, hi, bool(lc and lo is not None), bool(hc and hi is not None)


def interval_text(ivs, var: str = "x") -> str:
    """Intervals `(low, high)` or `(low, high, low_closed, high_closed)` (None for unbounded, open by default)
    written as inequalities in `var`, joined by `or`."""
    out = []
    for iv in ivs:
        lo, hi, lc, hc = _parts(iv)
        if lo is None and hi is None:
            return "all real numbers"
        if lo is None:
            out.append(f"{var} {'≤' if hc else '<'} {num(hi)}")
        elif hi is None:
            out.append(f"{var} {'≥' if lc else '>'} {num(lo)}")
        else:
            out.append(f"{num(lo)} {'≤' if lc else '<'} {var} {'≤' if hc else '<'} {num(hi)}")
    return " or ".join(out)


def _canon(ivs):
    def one(iv):
        lo, hi, lc, hc = _parts(iv)
        return (
            None if lo is None else sympy.Rational(lo),
            None if hi is None else sympy.Rational(hi),
            lc,
            hc,
        )

    return tuple(
        sorted(
            (
                (
                    x[0] is not None,
                    x[0] if x[0] is not None else 0,
                    x[1] is not None,
                    x[1] if x[1] is not None else 0,
                    x[2],
                    x[3],
                )
                for x in map(one, ivs)
            )
        )
    )


def _sort_key(iv):
    lo = iv[0]
    return (lo is not None, sympy.Rational(lo) if lo is not None else 0)


def inequality_part(prompt, ivs, marks, working, wrong, *, var: str = "x") -> dict:
    """A part answered by a set of numbers written as inequalities in `var` (kind `inequality`). `ivs` and each wrong
    answer are lists of intervals `(low, high)` or `(low, high, low_closed, high_closed)`, None for no bound."""
    ivs = sorted(ivs, key=_sort_key)
    key_text = interval_text(ivs, var)
    key = _canon(ivs)

    def build(v, tag):
        text = interval_text(sorted(v, key=_sort_key), var)
        return Distractor(text, tag, text)

    def same(v):
        return _canon(v) == key

    cands = [(v, t) for v, t in wrong if v]
    options = _first_wrong(key_text, cands, same, build)
    return _finish(
        dict(
            prompt=prompt,
            answer_kind="inequality",
            answer=key_text,
            answer_text=key_text,
            answer_latex="plain:" + key_text,
            marks=marks,
            working=working,
            distractors=options,
        ),
        None,
        None,
    )


# -- polynomials as coefficient lists (highest power first), in exact arithmetic ----------------------------------------
#
# SymPy costs tens of milliseconds a call; the calculus generators only ever need values, derivatives and integrals of
# low-degree polynomials with rational coefficients, which these three functions do exactly.


def pev(coeffs, at) -> Fraction:
    """Value of the polynomial at `at` (Horner's rule)."""
    total = Fraction(0)
    for c in coeffs:
        total = total * Fraction(at) + Fraction(c)
    return total


def pder(coeffs) -> list:
    """Coefficients of the derivative."""
    n = len(coeffs) - 1
    return [Fraction(c) * (n - i) for i, c in enumerate(coeffs[:-1])] or [Fraction(0)]


def pint(coeffs) -> list:
    """Coefficients of the antiderivative with constant term 0."""
    n = len(coeffs)
    return [Fraction(c) / (n - i) for i, c in enumerate(coeffs)] + [Fraction(0)]


def definite(coeffs, lo, hi) -> Fraction:
    """The integral of the polynomial from lo to hi."""
    anti = pint(coeffs)
    return pev(anti, hi) - pev(anti, lo)


def poly_text(coeffs, var: str = "x") -> str:
    """The polynomial written highest power first, as a learner would; a fractional coefficient is bracketed,
    `(3/2)x^2`."""
    n = len(coeffs) - 1
    out = []
    for i, c in enumerate(coeffs):
        c = Fraction(c)
        if c == 0:
            continue
        power = n - i
        mag = abs(c)
        name = "" if power == 0 else (var if power == 1 else f"{var}^{power}")
        if mag.denominator != 1:
            body = f"({mag.numerator}/{mag.denominator}){name}"
        elif mag == 1 and name:
            body = name
        else:
            body = f"{mag.numerator}{name}"
        out.append(("-" if c < 0 else "+", body))
    if not out:
        return "0"
    first_sign, first = out[0]
    text = ("-" if first_sign == "-" else "") + first
    for sign, body in out[1:]:
        text += f" {sign} {body}"
    return text


def poly_code(coeffs, var: str) -> str:
    """The polynomial as a SymPy-readable string in `var`, for a follow-through expression."""
    n = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        c = Fraction(c)
        if c == 0:
            continue
        coef = f"({c.numerator}/{c.denominator})" if c.denominator != 1 else f"({c.numerator})"
        power = n - i
        parts.append(coef if power == 0 else (f"{coef}*{var}" if power == 1 else f"{coef}*{var}**{power}"))
    return " + ".join(parts) or "0"


__all__ = [
    "pev",
    "pder",
    "pint",
    "definite",
    "poly_text",
    "poly_code",
    "Fraction",
    "Redraw",
    "clear_of_boundary",
    "draw",
    "exact_number_part",
    "expression_part",
    "fraction_part",
    "inequality_part",
    "interval_text",
    "latex_answer",
    "list_part",
    "list_text",
    "num",
    "number_part",
    "real",
    "single",
    "typed",
]
