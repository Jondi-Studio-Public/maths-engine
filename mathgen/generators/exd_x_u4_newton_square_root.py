"""Exam-tier source for `newton-method-iteration`: Newton's method for a length that is only described.

The practice tiers hand the learner `f(x)` and a starting value for one step. Here the function is never given.
A rectangle with a stated ratio of length to width has a stated area, and the learner must

1. turn "m w^2 = A" into an equation `f(w) = m w^2 - A = 0` and take one Newton step from `w0`,
2. measure how far the area for that estimate lies from the true area, `f(w1)`, and
3. use that residual to take the second step, `w2 = w1 - f(w1) / f'(w1)`.

Parts (b) and (c) carry follow-through, so a learner whose (a) is wrong still earns the later parts for using their
own value correctly.

Built from an integer start `w0`, a ratio `m` in {1, 2, 3} and an area `A` for which `sqrt(A / m)` is irrational, so
the iteration never terminates and every answer is a genuine fraction. The independent route evaluates the step with
SymPy from `f` and `f'` rather than from the closed form `(w + A/(m w)) / 2` used to build the key.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import isqrt

from sympy import Rational, Symbol, diff

from mathgen.core import check, check_equivalent, generator

from ._u4_kit import fraction_distractors, fraction_payload, survives_nsimplify

_W = Symbol("w")

_SETTINGS = {
    1: [
        "A square courtyard has an area of {A} m². Let w be the side length of the courtyard in metres.",
        "A square vegetable garden has an area of {A} m². Let w be its side length in metres.",
        "A square pool cover has an area of {A} m². Let w be its side length in metres.",
    ],
    2: [
        "A rectangular paddock is twice as long as it is wide and has an area of {A} m². Let w be its width in metres.",
        "A rectangular deck is twice as long as it is wide and has an area of {A} m². Let w be its width in metres.",
    ],
    3: [
        "A rectangular vegetable bed is three times as long as it is wide and has an area of {A} m². "
        "Let w be its width in metres.",
        "A rectangular cricket practice mat is three times as long as it is wide and has an area of {A} m². "
        "Let w be its width in metres.",
    ],
}

_STEM_ENDS = [
    "Newton's method is used to approximate w, starting from w₀ = {w0}.",
    "A student approximates w by Newton's method for the equation that w satisfies, starting from w₀ = {w0}.",
]


def _step(w: Fraction, m: int, area: int) -> Fraction:
    return (w + Fraction(area, m) / w) / 2


@generator("exd_x_u4_newton_square_root", level=16, presets={"exam": dict(tier="exam")})
def exd_x_u4_newton_square_root(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: two Newton steps for a width defined by an area, then the area residual."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        m = rng.choice([1, 1, 2, 2, 3, 3])
        area = rng.randint(3, 40)
        if isqrt(area * m) ** 2 == area * m:
            continue  # sqrt(A/m) must be irrational
        root = isqrt(area // m)
        w0 = rng.randint(max(1, root - 1), root + 2)
        w1 = _step(Fraction(w0), m, area)
        residual = m * w1 * w1 - area
        w2 = w1 - residual / (2 * m * w1)
        if {area, w0} & {w1, w2, residual} or w1 == w2 or residual == 0:
            continue
        if 1 in (w1.denominator, w2.denominator, residual.denominator):
            continue
        if not survives_nsimplify(w1, w2, residual):
            continue  # no answer may repeat a number in the prompt, and the iteration must still be moving
        break
    else:
        check(False, "no usable start, ratio and area in 1000 tries")

    # Independent route: the Newton step from f and f' with SymPy.
    f = m * _W**2 - area
    df = diff(f, _W)

    def rat(v: Fraction):
        return Rational(v.numerator, v.denominator)

    for value, previous in ((w1, Fraction(w0)), (w2, w1)):
        at = rat(previous)
        check_equivalent(at - f.subs(_W, at) / df.subs(_W, at), rat(value), "Newton step disagrees with f and f'")
    check_equivalent(f.subs(_W, rat(w1)), rat(residual), "residual disagrees with f(w1)")
    check(w2 > 0 and residual > 0, "Newton iterates from above a convex root must stay above it")

    stem = " ".join(
        [rng.choice(_SETTINGS[m]).format(A=area), rng.choice(_STEM_ENDS).format(w0=w0)]
        + ["Give exact values as fractions."]
    )

    def w_text(v):
        return fraction_payload(v)[1]

    a_candidates = [
        (Fraction(w0) + Fraction(area, m * w0), "newton-halving-forgotten"),
        (Fraction(w0) + (m * w0 * w0 - area) / Fraction(2 * m * w0), "newton-sign-of-correction"),
        (Fraction(w0) - (m * w0 * w0 - area), "newton-correction-no-derivative"),
        (Fraction(w0) - (m * w0 * w0 - area) / Fraction(m * w0), "newton-derivative-missing-factor-two"),
    ]
    b_candidates = [
        (w1, "residual-of-estimate-not-value"),
        (m * w0 * w0 - area, "newton-residual-at-previous-estimate"),
        (area - m * w1 * w1, "gap-sign-reversed"),
        (m * w1, None),
    ]
    c_candidates = [
        (w1, "newton-reapplied-from-start"),
        (w1 + Fraction(area, m) / w1, "newton-halving-forgotten"),
        (w1 + residual / (2 * m * w1), "newton-sign-of-correction"),
        (w1 - residual / (m * w1), "newton-derivative-missing-factor-two"),
        (w1 - residual, "newton-correction-no-derivative"),
    ]
    a_text, b_text, c_text = (fraction_payload(v)[1] for v in (w1, residual, w2))
    coef = str(m) if m != 1 else ""
    shape = f"{m}w² = {area}" if m != 1 else f"w² = {area}"

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the first improved estimate w₁ (an exact fraction).",
                answer_kind="fraction",
                answer=fraction_payload(w1)[0],
                answer_text=a_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"w satisfies {shape}, so take f(w) = {coef}w² − {area} and f′(w) = {2 * m}w.",
                    f"w₁ = {w0} − f({w0})/f′({w0}) = {a_text}.",
                ],
                distractors=fraction_distractors(w1, a_candidates),
            ),
            dict(
                prompt=f"Using your w₁, find f(w₁) = {coef}w₁² − {area}: how far the area {coef}w₁² is from the true "
                "area (an exact fraction).",
                answer_kind="fraction",
                answer=fraction_payload(residual)[0],
                answer_text=b_text,
                marks=1,
                follow_through=f"{m}*part_a**2 - {area}",
                working=[f"f(w₁) = {coef}({a_text})² − {area} = {b_text}."],
                distractors=fraction_distractors(residual, b_candidates),
            ),
            dict(
                prompt="Using your w₁ and f(w₁), find the second estimate w₂ (an exact fraction).",
                answer_kind="fraction",
                answer=fraction_payload(w2)[0],
                answer_text=c_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a - part_b/({2 * m}*part_a)",
                working=[f"w₂ = w₁ − f(w₁)/f′(w₁) = {a_text} − ({b_text})/({2 * m}·{a_text}) = {c_text}."],
                distractors=fraction_distractors(w2, c_candidates),
            ),
        ],
    )
