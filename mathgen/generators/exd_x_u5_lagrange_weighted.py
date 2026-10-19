"""Exam-tier source for `lagrange-constrained-point`: a weighted objective under a linear constraint.

The practice tiers ask for the point of a line nearest a target, which is the plain squared distance. Here the
objective is a weighted sum of squares ``f(x, y) = x^2 + k y^2`` and the learner uses the Lagrange condition
``grad f = lambda grad g`` for the constraint ``g(x, y) = a x + b y = c`` to carry a four-step chain:

(a) the multiplier lambda                          fraction, 2 marks
(b) the x-coordinate of the constrained minimum    fraction, 1 mark,  ``part_a * a / 2``
(c) the y-coordinate                               fraction, 1 mark,  ``part_a * b / (2 k)``
(d) the minimum value of f                         fraction, 2 marks, ``part_b^2 + k part_c^2``

From ``2 x = lambda a`` and ``2 k y = lambda b`` the constraint gives ``lambda = 2 k c / D`` with
``D = k a^2 + b^2``, hence ``x = k a c / D``, ``y = b c / D`` and ``f = k c^2 / D``. Each is re-derived below by
solving the stationarity equations with exact fractions rather than from these closed forms.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

from ._exd_distinct_parts import answers_distinct, junk_safe


_EXACT = "Give an exact answer: a fraction in simplest form, or an integer."


def _lam(c: int) -> str:
    """``c`` times λ as written on paper: ``λ``, ``-λ``, ``3λ``."""
    return {1: "λ", -1: "-λ"}.get(c, f"{c}λ")


def _fraction_part(prompt, value, marks, working, candidates, follow_through=None):
    """A `fraction` part whose answer is `value`; `candidates` are (misconception id or None, value) pairs."""
    text = rational_text(value)
    options = [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in candidates if v != value]
    part = dict(
        prompt=f"{prompt} {_EXACT}",
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, options, 3),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _draw(rng: random.Random) -> dict | None:
    a = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
    b = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
    k = rng.choice((2, 3, 4))
    c = rng.choice([n for n in range(-15, 16) if n])
    # x = a lambda / 2 and y = b lambda / (2k) must not be lambda itself (or the follow-through is a restatement).
    if a == 2 or b == 2 * k:
        return None

    # Solve the stationarity equations 2x = lambda a, 2 k y = lambda b and the constraint exactly.
    d = k * a * a + b * b
    lam = Fraction(2 * k * c, d)
    x = lam * a / 2
    y = lam * b / (2 * k)
    f_min = x * x + k * y * y
    check(a * x + b * y == c, "the minimum is not on the constraint")
    check(2 * x == lam * a and 2 * k * y == lam * b, "stationarity fails")
    check(f_min == lam * c / 2, "minimum value disagrees with lambda c / 2")

    # The unweighted problem (f = x^2 + y^2), the usual slip when the weight is dropped from the gradient.
    d_flat = a * a + b * b
    lam_flat = Fraction(2 * c, d_flat)

    g_text = join_terms([(a, "x"), (b, "y")])
    f_text = f"x² + {k}y²"
    stem = phrase(
        rng,
        (
            "Minimise f(x, y) = {f} subject to the constraint {g} = {c}. Use the Lagrange condition ∇f = λ∇g, "
            "where g(x, y) = {g}.",
            "A function f(x, y) = {f} is to be minimised on the line {g} = {c}. Take g(x, y) = {g} and use the "
            "Lagrange condition ∇f = λ∇g.",
            "Use the method of Lagrange multipliers, with ∇f = λ∇g and g(x, y) = {g}, to minimise f(x, y) = {f} "
            "subject to {g} = {c}.",
        ),
        f=f_text,
        g=g_text,
        c=c,
    )

    part_a = _fraction_part(
        "Find the value of λ.",
        lam,
        2,
        [
            f"∇f = (2x, {2 * k}y) and ∇g = ({a}, {b}), so 2x = {_lam(a)} and {2 * k}y = {_lam(b)}.",
            f"Substituting x = {_lam(a)}/2 and y = {_lam(b)}/{2 * k} into {g_text} = {c} gives λ = {rational_text(lam)}.",
        ],
        [
            ("dropped-weight-in-gradient", lam_flat),
            ("dropped-factor-two-in-derivative", lam / 2),
            (None, -lam),
            (None, Fraction(2 * c, d)),
        ],
    )
    part_b = _fraction_part(
        "Hence find the x-coordinate of the minimum.",
        x,
        1,
        [f"x = {_lam(a)}/2 = {rational_text(x)}."],
        [
            ("dropped-factor-two-in-derivative", lam * a),
            ("dropped-weight-in-gradient", lam_flat * a / 2),
            (None, Fraction(c, a)),
            (None, lam * b / 2),
        ],
        follow_through=f"part_a*({a})/2",
    )
    part_c = _fraction_part(
        "Hence find the y-coordinate of the minimum.",
        y,
        1,
        [f"y = {_lam(b)}/{2 * k} = {rational_text(y)}."],
        [
            ("dropped-weight-in-gradient", lam * b / 2),
            ("dropped-weight-in-gradient", lam_flat * b / 2),
            ("dropped-factor-two-in-derivative", lam * b / k),
            (None, Fraction(c, b)),
        ],
        follow_through=f"part_a*({b})/(2*{k})",
    )
    part_d = _fraction_part(
        "Hence find the minimum value of f.",
        f_min,
        2,
        [f"f = x² + {k}y² = {rational_text(x * x)} + {k} × {rational_text(y * y)} = {rational_text(f_min)}."],
        [
            ("dropped-weight-in-gradient", x * x + y * y),
            (None, x * x + k * y),
            (None, lam * c),
            (None, k * x * x + y * y),
        ],
        follow_through=f"part_b**2 + {k}*part_c**2",
    )
    if any(len(p["distractors"]) < 3 for p in (part_a, part_b, part_c, part_d)):
        return None
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_u5_lagrange_weighted", level=17, presets={"exam": dict(tier="exam")})
def exd_x_u5_lagrange_weighted(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the multiplier, the constrained minimum point and the minimum value, for a weighted objective."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a draw with three distinct distractors per part"):
        question = _draw(rng)
        if question is not None and answers_distinct(question["parts"]) and junk_safe(question["parts"]):
            return question
