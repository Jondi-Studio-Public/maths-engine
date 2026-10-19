"""Generates `exd_x_r2a_parallel_tangent_normal` questions: a multi-part exam chain on a tangent parallel to a line.

Skill: senior-find-the-slope-of-a-tangent-and-the (exam tier). The point of contact is not given: the tangent to a
cubic is parallel to a stated line, and there is exactly one such point with a stated sign of x. The question runs:

(a) find the x-coordinate of that point P                         fraction,   2 marks
(b) hence the equation of the tangent at P                        expression, 2 marks, slope of the line through P
(c) hence the equation of the normal at P                         expression, 2 marks, -1/(slope of (b)) through P
(d) hence where the normal meets the y-axis                       fraction,   1 mark,  from the learner's (c)

## Construction: choose the contact point first

``f(x) = a x^3 + b x^2 + c x + d`` has ``f'(x) = 3a (x - r1)(x - r2)`` for chosen integers r1 < 0 < r2, which fixes
``b = -3a(r1 + r2)/2`` (an integer: a = 2, or a = 1 with r1 + r2 even) and ``c = m + 3a r1 r2`` where m is the slope of
the line. Exactly one root is positive (or negative), so "the point where x > 0" names r2. The constant d and the
line's intercept are drawn freely. f(r2) and m are exact, and the tangent's gradient is non-zero.
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, diff, expand, latex

from ..core import check, generator, join_terms
from .exd_x_r2a_common import (
    X,
    EXACT_NOTE,
    Redraw,
    build_with_redraw,
    expr_part,
    fraction_part,
    paren,
    part_prompt,
    pretty,
    prose,
    same_function,
    line_intercept_ft,
    line_slope_ft,
)

_PATTERNS = (
    "The curve with equation {yf} has {fx}. The tangent to the curve at the point P, where x is {sign}, is parallel to "
    "the line {line}.",
    "Let {fx}. At the point P on the graph of {yf} where x is {sign}, the tangent is parallel to the line {line}.",
    "Consider the curve {yf}, where {fx}. P is the point on the curve with {signed} at which the tangent is parallel to "
    "{line}.",
)
_YF = ("y = f(x)", "y = f(x)")


def _build(rng: random.Random):
    a = rng.choice([1, 2, 2])
    r1 = rng.choice([-3, -2, -1])  # the negative root of f'(x) = m
    r2 = rng.choice([1, 2, 3])  # the positive root
    if a == 1 and (r1 + r2) % 2:
        raise Redraw
    positive = rng.choice([True, True, False])
    target, other = (r2, r1) if positive else (r1, r2)
    b = Rational(-3 * a * (r1 + r2), 2)
    if b.q != 1:
        raise Redraw
    b = int(b)
    m = rng.choice([m for m in range(-12, 13) if m != 0])
    c = m + 3 * a * r1 * r2
    d = rng.choice([m2 for m2 in range(-6, 7) if m2 != 0])
    k = rng.choice([m2 for m2 in range(-6, 7) if m2 != 0])
    f = a * X**3 + b * X**2 + c * X + d
    fprime = diff(f, X)
    check(same_function(fprime - m, 3 * a * (X - r1) * (X - r2)), "f' - m does not factor as constructed")
    if abs(c) > 40:
        raise Redraw
    f0 = f.subs(X, target)
    if f0 == 0 or abs(f0) > 120:
        raise Redraw
    tangent = m * (X - target) + f0
    nslope = Rational(-1, m)
    normal = nslope * (X - target) + f0
    c_n = normal.subs(X, 0)
    if c_n == 0:
        raise Redraw
    tangent_x = tangent.subs(X, 0)
    if tangent_x == k:
        raise Redraw  # the tangent would be the given line itself

    sign_word = "positive" if positive else "negative"
    f_text = join_terms([(a, "x^3"), (b, "x^2"), (c, "x"), (d, "")])
    line_text = "y = " + join_terms([(m, "x"), (k, "")])

    part_a = fraction_part(
        part_prompt("Find the x-coordinate of P. " + EXACT_NOTE),
        target,
        2,
        [
            f"f'(x) = {pretty(fprime)}. The tangent is parallel to the line, so f'(x) = {m}.",
            f"{pretty(expand(fprime - m))} = 0 gives x = {r1} or x = {r2}; x is {sign_word}, so x = {target}.",
        ],
        [
            ("wrong-root-chosen", other),
            ("sign-error-moving-term", -target),
            ("slope-used-as-x", m),
        ],
    )
    f_at_a = str(f.subs(X, Symbol("part_a")))
    part_b = expr_part(
        part_prompt(
            "Hence find the equation of the tangent to the curve at P, in the form {form}.",
            form=("y = mx + c", "y = mx + c"),
        ),
        tangent,
        2,
        [
            f"The gradient is {m} and P is ({target}, {f0}).",
            f"y - {paren(f0)} = {m}(x - {paren(target)}), so y = {pretty(expand(tangent))}.",
        ],
        [
            ("given-line-used-as-tangent", m * X + k),
            ("used-perpendicular-gradient", normal),
            ("y-value-as-intercept", m * X + f0),
            ("forgot-point-y-value", m * (X - target)),
        ],
        follow_through=f"{m}*(x - part_a) + {f_at_a}",
        prefix="y = ",
    )
    slope_b = line_slope_ft("b")
    part_c = expr_part(
        part_prompt(
            "Hence find the equation of the normal to the curve at P, in the form {form}.",
            form=("y = mx + c", "y = mx + c"),
        ),
        normal,
        2,
        [
            f"The normal is perpendicular to the tangent, so its gradient is -1/{m} = {Rational(-1, m)}.",
            f"y - {paren(f0)} = {Rational(-1, m)}(x - {paren(target)}), so y = {pretty(expand(normal))}.",
        ],
        [
            ("tangent-instead-of-normal", tangent),
            ("normal-slope-reciprocal-only", Rational(1, m) * (X - target) + f0),
            ("normal-slope-negated-only", -m * (X - target) + f0),
            ("y-value-as-intercept", nslope * X + f0),
        ],
        follow_through=f"-1/{slope_b}*(x - part_a) + {f_at_a}",
        prefix="y = ",
    )
    part_d = fraction_part(
        part_prompt("Find the y-coordinate of the point where this normal meets the y-axis. " + EXACT_NOTE),
        c_n,
        1,
        [f"Put x = 0 in the normal: y = {c_n}."],
        [
            ("tangent-instead-of-normal", tangent_x),
            ("y-value-as-intercept", f0),
            ("intercepts-confused", -c_n / nslope),
            ("sign-reversed-result", -c_n),
        ],
        follow_through=line_intercept_ft("c"),
    )
    stem, stem_latex = prose(
        rng,
        _PATTERNS,
        fx=(f"f(x) = {f_text}", f"f(x) = {latex(f)}"),
        yf=_YF,
        sign=(sign_word, rf"\text{{{sign_word}}}"),
        signed=(
            f"{'positive' if positive else 'negative'} x-coordinate",
            rf"\text{{{sign_word} }}x\text{{-coordinate}}",
        ),
        line=(line_text, line_text),
    )
    return dict(text=stem, latex=stem_latex, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2a_parallel_tangent_normal", level=16, presets={"exam": {}})
def exd_x_r2a_parallel_tangent_normal(rng: random.Random):
    """A cubic and a parallel line: the point of contact, the tangent, the normal, and its y-intercept."""
    return build_with_redraw(rng, _build)
