"""Exam-tier source for composing the reciprocal function ``f(x) = 1/(x - k)`` and the square-root function
``g(x) = sqrt(x - a)``.

The practice tiers substitute one of these into the other once. Every exam question here needs more than one
substitution, or a substitution followed by a change of form that has to be chosen by the learner.

Four question shapes, each built from chosen integers ``a`` and ``k``:

- ``rationalised``: ``f(g(x)) = 1/(sqrt(x - a) - k)``, to be given with a rational denominator. Multiplying by the
  conjugate gives ``(sqrt(x - a) + k)/(x - a - k^2)``. Graded with the ``rationalised`` answer form.
- ``self_composition``: ``f(f(x)) = 1/(1/(x - k) - k)``, to be written as a single fraction,
  ``(x - k)/(k^2 + 1 - kx)``. Graded with the ``single_fraction`` answer form.
- ``three_fold``: ``f(g(f(x))) = 1/(sqrt(1/(x - k) - a) - k)``, three substitutions in the stated order.
- ``root_of_root``: ``g(g(x)) = sqrt(sqrt(x - a) - a)``.

Every key is cross-checked by an independent route: the printed key is parsed back with sympy and compared
with the composition built by substituting sympy expressions directly, and for the two form-constrained shapes
the key is also compared numerically with the unrationalised or unsimplified composition at sample points.

Distractors, each a named slip:

- ``rationalised``: multiplying top and bottom by the same radical expression instead of the conjugate; the sign
  of ``k^2`` wrong in the denominator; ``k`` instead of ``k^2`` in the denominator.
- ``self_composition``: dropping the ``k^2`` that comes from ``k(x - k)``; the sign of the ``kx`` and ``k^2``
  terms reversed; shifting twice (``1/(x - 2k)``) instead of composing.
- ``three_fold``: composing in the opposite order, ``g(f(g(x)))``; stopping after ``f(g(x))``; moving ``a`` out of
  the radical.
- ``root_of_root``: adding the shifts (``sqrt(x - 2a)``); applying only one root; adding ``a`` instead of
  subtracting it inside the outer root.
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, simplify, sqrt
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..core import check, check_distinct, generator, join_terms, latex_answer, phrase, pick_distractors

x = Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

_RATIONALISED = (
    "Let f(x) = {f} and g(x) = {g}. Find f(g(x)), giving your answer with a rational denominator.",
    "Given f(x) = {f} and g(x) = {g}, find f(g(x)) and rationalise the denominator.",
)
_SELF = (
    "Let f(x) = {f}. Find f(f(x)), writing your answer as a single fraction.",
    "Given f(x) = {f}, find f(f(x)) as a single algebraic fraction.",
)
_THREE = (
    "Let f(x) = {f} and g(x) = {g}. Find f(g(f(x))).",
    "Given f(x) = {f} and g(x) = {g}, find the composite function f(g(f(x))).",
)
_ROOT = (
    "Let g(x) = {g}. Find g(g(x)).",
    "Given g(x) = {g}, find the composite function g(g(x)).",
)


def _parse(text: str):
    return parse_expr(text, local_dict={"x": x}, transformations=_TRANSFORMS)


def _shift(k: int) -> str:
    """``x - k`` with the sign chosen so the text never reads ``- -n``."""
    return join_terms([(1, "x"), (-k, "")])


def _inner(k: int, base: str = "x") -> str:
    return f"{base} - {k}" if k >= 0 else f"{base} + {-k}"


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _check_same_at_points(key_text: str, direct, points) -> None:
    """The printed key must agree numerically with the composition built directly from sympy functions."""
    key = _parse(key_text)
    compared = 0
    for pt in points:
        expected = direct.subs(x, pt)
        if not expected.is_finite or not expected.is_real:
            continue  # the composition is undefined here; nothing to compare
        compared += 1
        check(
            abs(float(key.subs(x, pt)) - float(expected)) < 1e-9,
            f"key {key_text!r} disagrees with the direct composition at x = {pt}",
        )
    check(compared >= 2, "fewer than two usable sample points")


def _result(text: str, answer_text: str, direct, points, wrong: list[str], working: list[str], form=None) -> dict:
    key = _parse(answer_text)
    _check_same_at_points(answer_text, direct, points)
    check(simplify(key - direct) == 0, f"key {answer_text!r} is not equal to the direct composition")
    for candidate in wrong:
        check_distinct(key, _parse(candidate), msg=f"distractor {candidate!r} equals the key")
    result = dict(
        text=text,
        answer=str(key),
        answer_text=answer_text,
        answer_latex=latex_answer(key),
        working=working,
        distractors=pick_distractors(answer_text, wrong, n=3),
    )
    if form is not None:
        result["answer_form"] = form
    return result


def _rationalised(rng: random.Random) -> dict:
    a = rng.randint(1, 9)
    k = rng.choice([v for v in range(-6, 7) if abs(v) >= 2])  # |k| = 1 would make k and k^2 coincide
    root = f"sqrt({_inner(a)})"
    f_text, g_text = f"1/({_inner(k)})", root
    denominator = join_terms([(1, "x"), (-(a + k * k), "")])
    answer_text = f"({root} {'+' if k >= 0 else '-'} {abs(k)})/({denominator})"
    wrong = [
        f"({root} {'-' if k >= 0 else '+'} {abs(k)})/({denominator})",
        f"({root} {'+' if k >= 0 else '-'} {abs(k)})/({join_terms([(1, 'x'), (-a + k * k, '')])})",
        f"({root} {'+' if k >= 0 else '-'} {abs(k)})/({join_terms([(1, 'x'), (-(a + k), '')])})",
    ]
    direct = 1 / (sqrt(x - a) - k)
    # sample points above x = a + k^2 + 1 keep the composition away from its excluded point and defined
    points = [a + k * k + 1, a + k * k + 5, a + 2 * k * k + 7]
    text = phrase(rng, _RATIONALISED, f=f_text, g=g_text)
    working = [
        f"f(g(x)) = 1/({root} {'-' if k >= 0 else '+'} {abs(k)}).",
        f"Multiply top and bottom by the conjugate ({root} {'+' if k >= 0 else '-'} {abs(k)}): the denominator becomes ({root})^2 - {abs(k)}^2 = {denominator}.",
        f"f(g(x)) = {answer_text}.",
    ]
    return _result(text, answer_text, direct, points, wrong, working, form="rationalised")


def _self_composition(rng: random.Random) -> dict:
    k = rng.choice([v for v in range(-6, 7) if abs(v) >= 2])
    f_text = f"1/({_inner(k)})"
    numerator = join_terms([(1, "x"), (-k, "")])
    denominator = join_terms([(-k, "x"), (k * k + 1, "")])
    answer_text = f"({numerator})/({denominator})"
    wrong = [
        f"({numerator})/({join_terms([(-k, 'x'), (1, '')])})",
        f"({numerator})/({join_terms([(k, 'x'), (-k * k + 1, '')])})",
        f"1/({join_terms([(1, 'x'), (-2 * k, '')])})",
    ]
    direct = 1 / (1 / (x - k) - k)
    # sample points that avoid x = k and x = k + 1/k
    points = [k + 3, k - 4, k + 7]
    text = phrase(rng, _SELF, f=f_text)
    working = [
        f"f(f(x)) = 1/(1/({_inner(k)}) - {k})." if k > 0 else f"f(f(x)) = 1/(1/({_inner(k)}) + {-k}).",
        f"Put the denominator over {_inner(k)}: 1/({_inner(k)}) - ({k})({_inner(k)})/({_inner(k)}) = ({join_terms([(-k, 'x'), (k * k + 1, '')])})/({_inner(k)}).",
        f"Inverting gives {answer_text}.",
    ]
    return _result(text, answer_text, direct, points, wrong, working, form="single_fraction")


def _three_fold(rng: random.Random) -> dict:
    a = rng.randint(1, 9)
    k = _nonzero(rng, -6, 6)
    f_text, g_text = f"1/({_inner(k)})", f"sqrt({_inner(a)})"
    answer_text = f"1/(sqrt(1/({_inner(k)}) - {a}) - {k})" if k > 0 else f"1/(sqrt(1/({_inner(k)}) - {a}) + {-k})"
    wrong_order = f"sqrt(1/(sqrt({_inner(a)}) {'-' if k > 0 else '+'} {abs(k)}) - {a})"
    stops_early = f"1/(sqrt({_inner(a)}) {'-' if k > 0 else '+'} {abs(k)})"
    radical_slip = f"1/(sqrt(1/({_inner(k)})) - {a} {'-' if k > 0 else '+'} {abs(k)})"
    direct = 1 / (sqrt(1 / (x - k) - a) - k)
    # choose points where 1/(x - k) > a and the outer denominator is nonzero: x just above k, small steps
    points = [k + Rational(1, 2 * a + 3), k + Rational(1, 3 * a + 5), k + Rational(1, 5 * a + 7)]
    text = phrase(rng, _THREE, f=f_text, g=g_text)
    working = [
        f"f(x) = {f_text}, so g(f(x)) = sqrt(1/({_inner(k)}) - {a}).",
        f"Then f(g(f(x))) = 1/(g(f(x)) {'-' if k > 0 else '+'} {abs(k)}).",
        f"f(g(f(x))) = {answer_text}.",
    ]
    return _result(text, answer_text, direct, points, [wrong_order, stops_early, radical_slip], working)


def _root_of_root(rng: random.Random) -> dict:
    a = rng.randint(1, 9)
    g_text = f"sqrt({_inner(a)})"
    answer_text = f"sqrt(sqrt({_inner(a)}) - {a})"
    wrong = [
        f"sqrt({_inner(2 * a)})",
        f"sqrt({_inner(a)}) - {a}",
        f"sqrt(sqrt({_inner(a)}) + {a})",
    ]
    direct = sqrt(sqrt(x - a) - a)
    points = [a + a * a + 1, a + a * a + 6, a + (a + 2) ** 2]
    text = phrase(rng, _ROOT, g=g_text)
    working = [
        f"g(g(x)) = sqrt(g(x) - {a}) = sqrt(sqrt({_inner(a)}) - {a}).",
    ]
    return _result(text, answer_text, direct, points, wrong, working)


@generator(
    "exd_x_f2_compose",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f2_compose(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier composition: repeated substitution, or substitution followed by a required change of form."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}; this source only has 'exam'")
    shape = rng.choice(["rationalised", "self_composition", "three_fold", "root_of_root"])
    if shape == "rationalised":
        return _rationalised(rng)
    if shape == "self_composition":
        return _self_composition(rng)
    if shape == "three_fold":
        return _three_fold(rng)
    return _root_of_root(rng)
