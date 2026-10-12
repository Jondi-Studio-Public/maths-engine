"""Generates `exd_x_r2a_trig_turning` questions: a multi-part exam chain on a trigonometric function with a linear term.

Skill: differentiate-trig (exam tier). The function is ``f(x) = c x + n sin(bx) + h`` or ``f(x) = m x + n cos(bx) + h``
on one period of the trigonometric part, 0 <= x <= 2π/b, built so that f'(x) = 0 has exact solutions (the cosine or
sine of bx is plus or minus a half). The question uses the derivative to find a maximum:

(a) find f'(x)                                                    expression, 2 marks
(b) hence the x-coordinate of the local maximum                   expression, 2 marks, a solution of the learner's f'
(c) hence the value of f at that local maximum, in exact form    expression, 2 marks, f at the learner's x

## Construction

``f'(x) = P + Q cos(bx)`` (sine family) or ``P + Q sin(bx)`` (cosine family) with ``|Q| = 2|P|``. In the sine family
Q = nb > 0 and the maximum is at x = arccos(-P/Q)/b; in the cosine family the slope is positive (m > 0), Q = -nb < 0
and the maximum is at x = arcsin(-P/Q)/b. Both are in the domain and are the maximum because f'' < 0 there. The
other turning point (the minimum) feeds the distractors.

## Follow-through

(b) reads P and Q from the learner's (a): it is evaluated where the cosine (or sine) vanishes and at x = 0, so any
equivalent form of (a) works, and the sign of Q picks the maximum rather than the minimum of the learner's own f. A
derivative that has lost the right trigonometric function (sin differentiated to sin) is not followed through.
(c) is f at the learner's (b).
"""

from __future__ import annotations

import random

from sympy import Integer, Symbol, acos, asin, cos, diff, latex, pi, sin

from ..core import check, generator, join_terms
from .exd_x_r2a_common import (
    X,
    Redraw,
    build_with_redraw,
    expr_part,
    part_prompt,
    pretty,
    prose,
    derivative_agrees,
    same_function,
)

_PATTERNS = (
    "The function f is defined by {fx} for {dom}.",
    "Let {fx}, where {dom}.",
    "Consider the function {fx} on the domain {dom}.",
)


def _domain(b: int):
    end = 2 * pi / b
    end_text = {1: "2π", 2: "π", 3: "2π/3", 4: "π/2"}[b]
    return f"0 ≤ x ≤ {end_text}", rf"0 \le x \le {latex(end)}"


def _build(rng: random.Random):
    family = rng.choice(["sin", "cos"])
    b = rng.choice([1, 2, 3, 4])
    m = rng.randint(1, 4)
    if (2 * m) % b:
        raise Redraw
    n = 2 * m // b
    s = rng.choice([1, -1]) if family == "sin" else 1
    h = rng.choice([0, 0, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    slope = s * m
    if family == "sin":
        f = slope * X + n * sin(b * X) + h
        fprime = slope + n * b * cos(b * X)
        theta = acos(-slope / Integer(n * b))
        p_ft = f"Subs(part_a, (x,), (pi/{2 * b},))"
        q_ft = f"(Subs(part_a, (x,), (0,)) - {p_ft})"
        sign = f"sign({q_ft})"  # +1 or -1
        x_max_ft = f"(pi - {sign}*(pi - acos(-{p_ft}/{q_ft})))/{b}"
        trig_text, trig = f"sin({b}x)" if b > 1 else "sin(x)", sin
        other_theta = 2 * pi - theta
    else:
        f = slope * X + n * cos(b * X) + h
        fprime = slope - n * b * sin(b * X)
        theta = asin(slope / Integer(n * b))
        p_ft = "Subs(part_a, (x,), (0,))"
        q_ft = f"(Subs(part_a, (x,), (pi/{2 * b},)) - {p_ft})"
        sign = f"sign({q_ft})"  # +1 or -1
        x_max_ft = f"((1 - {sign})/2*asin(-{p_ft}/{q_ft}) + (1 + {sign})/2*(pi - asin(-{p_ft}/{q_ft})))/{b}"
        trig_text, trig = f"cos({b}x)" if b > 1 else "cos(x)", cos
        other_theta = pi - theta
    check(derivative_agrees(f, fprime), "derivative disagrees with the numerical derivative")
    x_max, x_min = theta / b, other_theta / b
    f_max, f_min = f.subs(X, x_max), f.subs(X, x_min)
    check(0 < x_max < 2 * pi / b and f_max > f_min, "the maximum is not the larger turning value")
    check(diff(f, X, 2).subs(X, x_max) < 0 < diff(f, X, 2).subs(X, x_min), "the second derivative does not classify")
    n_b = n * b
    if family == "sin":
        wrong_a = [
            ("forgot-inner-derivative", slope + n * cos(b * X)),
            ("sign-error-trig-derivative", slope - n * b * cos(b * X)),
            ("trig-function-not-changed-by-derivative", slope + n * b * sin(b * X)),
            ("sign-error-trig-derivative", slope - n * cos(b * X)),
        ]
    else:
        wrong_a = [
            ("forgot-inner-derivative", slope - n * sin(b * X)),
            ("sign-error-trig-derivative", slope + n_b * sin(b * X)),
            ("trig-function-not-changed-by-derivative", slope - n_b * cos(b * X)),
            ("sign-error-trig-derivative", slope + n * sin(b * X)),
        ]
    rng.shuffle(wrong_a)
    part_a = expr_part(
        "Find f'(x).",
        fprime,
        2,
        [
            f"The chain rule puts a factor of {b} in front of the derivative of {trig_text}."
            if b > 1
            else f"The derivative of {trig_text} is {'cos(x)' if family == 'sin' else '-sin(x)'}.",
            f"f'(x) = {pretty(fprime)}.",
        ],
        wrong_a,
    )

    flip = acos(slope / Integer(n * b)) if family == "sin" else asin(-slope / Integer(n * b))
    wrong_b = [
        ("second-derivative-sign-reversed", x_min),
        ("sign-error-moving-term", flip / b),
        ("forgot-to-divide-by-coefficient", theta if b > 1 else None),
    ]
    part_b = expr_part(
        part_prompt("Hence find the x-coordinate of the local maximum of f on this domain, in exact form."),
        x_max,
        2,
        [
            f"f'(x) = 0 gives {'cos' if family == 'sin' else 'sin'}({b}x) = {pretty(-slope / Integer(n * b) if family == 'sin' else slope / Integer(n * b))}.",
            f"In the domain the solutions are x = {pretty(x_max)} and x = {pretty(x_min)}; f'' < 0 at x = {pretty(x_max)}, so this is the maximum.",
        ],
        wrong_b,
        follow_through=x_max_ft,
    )
    f_in_b = str(f.subs(X, Symbol("part_b")))
    wrong_c = [
        ("second-derivative-sign-reversed", f_min),
        ("x-given-for-y", x_max),
        ("dropped-constant-term", f_max - h if h else None),
        ("forgot-to-divide-by-coefficient", f.subs(X, theta) if b > 1 else None),
    ]
    part_c = expr_part(
        part_prompt("Hence find the value of f at this local maximum, in exact form."),
        f_max,
        2,
        [f"f({pretty(x_max)}) = {pretty(f_max)}."],
        wrong_c,
        follow_through=f_in_b,
    )
    f_text = join_terms([(slope, "x"), (n, trig_text), (h, "")])
    dom_plain, dom_latex = _domain(b)
    stem, stem_latex = prose(rng, _PATTERNS, fx=(f"f(x) = {f_text}", f"f(x) = {latex(f)}"), dom=(dom_plain, dom_latex))
    return dict(text=stem, latex=stem_latex, parts=[part_a, part_b, part_c])


@generator("exd_x_r2a_trig_turning", level=16, presets={"exam": {}})
def exd_x_r2a_trig_turning(rng: random.Random):
    """A linear plus trigonometric function: f'(x), the local maximum, its exact value."""
    return build_with_redraw(rng, _build)
