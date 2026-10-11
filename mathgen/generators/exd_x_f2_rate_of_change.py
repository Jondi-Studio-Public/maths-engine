"""Exam-tier source for the average rate of change of a fully specified polynomial.

The practice tiers give an expanded quadratic or cubic and one interval. Here the function is still fully
specified and the answer is still one average rate (or a combination of two), but the learner has to organise
the calculation: compare two intervals of a quartic, differentiate nothing but compose two functions first, or
work from a function given as an unexpanded product of factors.

Three question shapes, each with whole-number coefficients and endpoints, so every average rate is a whole
number (the divided difference of an integer polynomial at integer points is an integer):

- ``compare_intervals``: a quartic and two intervals; by how much does the average rate over the first interval
  exceed that over the second?
- ``composite``: ``f(x) = x^2 + bx`` (or ``x^3 + bx``) and ``g(x) = mx + d``; the average rate of change of the
  composite ``f(g(x))`` over an interval of ``x``.
- ``factored``: a cubic or quartic written as a product of linear factors; the average rate over an interval.

Every key is cross-checked by an independent route: the printed function is expanded with sympy and the
difference quotient is computed symbolically before the endpoints are substituted. Distractors, each a named
slip:

- ``compare_intervals``: reporting only the first rate; reporting the sum of the two rates; reporting the
  difference in the wrong order.
- ``composite``: composing in the opposite order, ``g(f(x))``; taking the average rate of ``f`` alone over the
  interval; reporting the total change in ``f(g(x))`` without dividing by the width.
- ``factored``: reporting the total change instead of the rate; reporting the negative of the rate; reporting
  the value ``f(x2)`` at the right endpoint.
"""

from __future__ import annotations

import random

from sympy import Poly, Symbol, cancel, expand
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..core import (
    VerificationError,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    join_terms,
    phrase,
    pick_distractors,
)

x = Symbol("x")
_X1, _X2 = Symbol("x1"), Symbol("x2")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_RESAMPLE = 300

_COMPARE = (
    "Let f(x) = {f}. By how much does the average rate of change of f(x) from x = {a1} to x = {a2} exceed "
    "its average rate of change from x = {b1} to x = {b2}?",
    "The function f is given by f(x) = {f}. Find the average rate of change of f over [{a1}, {a2}] minus the "
    "average rate of change of f over [{b1}, {b2}].",
)
_COMPOSITE = (
    "Let f(x) = {f} and g(x) = {g}. Find the average rate of change of the composite function f(g(x)) from "
    "x = {x1} to x = {x2}.",
    "Given f(x) = {f} and g(x) = {g}, find the average rate of change of f(g(x)) over the interval from "
    "x = {x1} to x = {x2}.",
)
_FACTORED = (
    "A function is given by f(x) = {f}. Find the average rate of change of f(x) from x = {x1} to x = {x2}.",
    "Let f(x) = {f}. Find the average rate of change of f over the interval from x = {x1} to x = {x2}.",
)


def _shift(k: int) -> str:
    return f"x - {k}" if k >= 0 else f"x + {-k}"


def _parse(text: str):
    return parse_expr(text, local_dict={"x": x}, transformations=_TRANSFORMS)


def _poly_text(coeffs: dict[int, int]) -> str:
    terms = []
    for power in sorted(coeffs, reverse=True):
        var = "" if power == 0 else ("x" if power == 1 else f"x^{power}")
        terms.append((coeffs[power], var))
    return join_terms(terms)


def _rate(expr, x1: int, x2: int) -> int:
    """Average rate of change by plain integer arithmetic on the sympy expression's values."""
    f = lambda v: int(expr.subs(x, v))  # noqa: E731
    diff = f(x2) - f(x1)
    assert diff % (x2 - x1) == 0
    return diff // (x2 - x1)


def _symbolic_rate(expr, x1: int, x2: int) -> int:
    """Independent route: cancel the difference quotient symbolically, then substitute the endpoints."""
    quotient = cancel((expr.subs(x, _X2) - expr.subs(x, _X1)) / (_X2 - _X1))
    return int(quotient.subs({_X1: x1, _X2: x2}))


def _compare_intervals(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        coeffs = {
            4: rng.choice([1, 2]),
            3: rng.choice([v for v in range(-4, 5) if v != 0]),
            2: rng.choice([v for v in range(-6, 7) if v != 0]),
            1: rng.choice([v for v in range(-9, 10) if v != 0]),
            0: rng.choice([v for v in range(-9, 10) if v != 0]),
        }
        expr = _parse(_poly_text(coeffs))
        a1 = rng.randint(-3, 2)
        a2 = a1 + rng.randint(2, 3)
        b1 = rng.randint(-3, 2)
        b2 = b1 + rng.randint(2, 3)
        if (a1, a2) == (b1, b2):
            continue
        r1, r2 = _rate(expr, a1, a2), _rate(expr, b1, b2)
        if r1 < r2:
            a1, a2, b1, b2, r1, r2 = b1, b2, a1, a2, r2, r1
        answer = r1 - r2
        wrong = [r1, r1 + r2, r2 - r1]
        printed = {a1, a2, b1, b2}
        if answer == 0 or set(wrong) & (printed | {answer}) or answer in printed:
            continue
        try:
            check_pairwise_distinct(answer, *wrong)
        except VerificationError:
            continue
        break
    else:
        raise VerificationError("compare_intervals: no collision-free draw")
    check_equivalent(_symbolic_rate(expr, a1, a2) - _symbolic_rate(expr, b1, b2), answer, "compare_intervals: mismatch")
    f_text = _poly_text(coeffs)
    text = phrase(rng, _COMPARE, f=f_text, a1=a1, a2=a2, b1=b1, b2=b2)
    working = [
        f"Average rate over [{a1}, {a2}] = (f({a2}) - f({a1})) / {a2 - a1} = {r1}.",
        f"Average rate over [{b1}, {b2}] = (f({b2}) - f({b1})) / {b2 - b1} = {r2}.",
        f"Difference = {r1} - ({r2}) = {answer}.",
    ]
    return _build(text, answer, wrong, working)


def _composite(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        cubic = rng.random() < 0.4
        b = rng.choice([v for v in range(-6, 7) if v != 0])
        m = rng.choice([2, 3, -2, -3])
        d = rng.choice([v for v in range(-5, 6) if v != 0])
        f_expr = x**3 + b * x if cubic else x**2 + b * x
        g_expr = m * x + d
        h = expand(f_expr.subs(x, g_expr))
        swapped = expand(g_expr.subs(x, f_expr))
        x1 = rng.randint(-3, 2)
        x2 = x1 + rng.randint(2, 4)
        answer = _rate(h, x1, x2)
        wrong = [_rate(swapped, x1, x2), _rate(f_expr, x1, x2), int(h.subs(x, x2) - h.subs(x, x1))]
        printed = {x1, x2, b, m, d}
        if answer == 0 or set(wrong) & (printed | {answer}) or answer in printed:
            continue
        try:
            check_pairwise_distinct(answer, *wrong)
        except VerificationError:
            continue
        break
    else:
        raise VerificationError("composite: no collision-free draw")
    check_equivalent(_symbolic_rate(h, x1, x2), answer, "composite: mismatch")
    f_text = _poly_text({3: 1, 1: b}) if cubic else _poly_text({2: 1, 1: b})
    g_text = _poly_text({1: m, 0: d})
    text = phrase(rng, _COMPOSITE, f=f_text, g=g_text, x1=x1, x2=x2)
    working = [
        f"f(g(x)) = {_poly_text({int(p): int(c) for (p,), c in Poly(h, x).terms()})}.",
        f"Average rate = (h({x2}) - h({x1})) / {x2 - x1} = {answer}.",
    ]
    return _build(text, answer, wrong, working)


def _factored(rng: random.Random) -> dict:
    for _ in range(_RESAMPLE):
        roots = rng.sample([v for v in range(-4, 5) if v != 0], 3)
        quartic = rng.random() < 0.5
        factors = [f"({_shift(r)})" for r in roots]
        if quartic:
            factors = ["x"] + factors
        f_text = "".join(factors)
        expr = _parse(f_text)
        x1 = rng.randint(-3, 2)
        x2 = x1 + rng.randint(2, 4)
        answer = _rate(expr, x1, x2)
        total = int(expr.subs(x, x2) - expr.subs(x, x1))
        wrong = [total, -answer, int(expr.subs(x, x2))]
        printed = {x1, x2} | set(roots)
        if answer == 0 or set(wrong) & (printed | {answer}) or answer in printed:
            continue
        try:
            check_pairwise_distinct(answer, *wrong)
        except VerificationError:
            continue
        break
    else:
        raise VerificationError("factored: no collision-free draw")
    check_equivalent(_symbolic_rate(expr, x1, x2), answer, "factored: mismatch")
    text = phrase(rng, _FACTORED, f=f_text, x1=x1, x2=x2)
    working = [
        f"f({x2}) = {int(expr.subs(x, x2))} and f({x1}) = {int(expr.subs(x, x1))}.",
        f"Average rate = ({int(expr.subs(x, x2))} - ({int(expr.subs(x, x1))})) / {x2 - x1} = {answer}.",
    ]
    return _build(text, answer, wrong, working)


def _build(text: str, answer: int, wrong: list[int], working: list[str]) -> dict:
    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(str(answer), [str(w) for w in wrong], n=3),
    )


@generator(
    "exd_x_f2_rate_of_change",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f2_rate_of_change(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier average-rate question: two intervals of a quartic, a composite function, or a factored form."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}; this source only has 'exam'")
    shape = rng.choice(["compare_intervals", "composite", "factored"])
    if shape == "compare_intervals":
        return _compare_intervals(rng)
    if shape == "composite":
        return _composite(rng)
    return _factored(rng)
