"""Exam source of `autonomous-equilibrium-stability`: a harvested population with two equilibria.

The practice tiers classify one equilibrium of x' = f(x) by the sign of f'. Here a population with a fixed annual
harvest is modelled by

    dP/dt = f(P) = pP - qP^2 - h,        f(P) = -q (P - P1)(P - P2),   0 < P1 < P2,

with the two equilibria drawn first so the coefficients are exact two-place decimals (p = q(P1 + P2), h = q P1 P2):

    (a) the larger equilibrium P2                                  (2 marks)
    (b) f'(P2) = p - 2 q P2, negative, so P2 is stable, to 2 d.p.  (2 marks, follow-through from (a))
    (c) the smaller equilibrium P1, below which the stock dies out (2 marks, follow-through from (a))

Part (c) uses the sum of the roots, P1 = p/q - P2, so it needs the larger equilibrium found in (a). The derivative
test is applied by the learner in (b); the sign of the answer is the classification.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import sympy
from sympy import Symbol

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    exact_decimal_string,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    stated_precision_string,
    stated_precision_tolerance,
)

P = Symbol("P")
_RATES = [Fraction(1, 100), Fraction(2, 100), Fraction(4, 100), Fraction(5, 100)]
_STORIES = [
    ("The number of fish P, in thousands, in a stocked dam", "caught"),
    ("The number of rock lobsters P, in thousands, in a fishing zone off the south coast of South Australia", "taken"),
    ("The population P, in thousands, of a kangaroo herd on a grazing property", "culled"),
    ("The number of prawns P, in thousands, in a Queensland trawl region", "trawled"),
]


def _dec(value: Fraction) -> Decimal:
    return Decimal(exact_decimal_string(value, 2))


def _dist(value, tag, places=None):
    value = Fraction(value)
    if places is None:  # whole thousands, or a half when two whole numbers are averaged
        places = 0 if value.denominator == 1 else 1
    return Distractor(stated_precision_string(value, places), tag, value)


def _fraction_text(value: Fraction) -> str:
    return f"({value.numerator}/{value.denominator})"


@generator("exd_x_sc1_harvested_equilibria", level=15, presets={"exam": dict(tier="exam")})
def exd_x_sc1_harvested_equilibria(rng, tier="exam"):
    for _ in attempts(1000, "two positive equilibria"):
        q = rng.choice(_RATES)
        p1 = rng.randint(1, 18)
        p2 = rng.randint(p1 + 2, p1 + 40)
        if p1 + p2 > 60:
            continue
        p = q * (p1 + p2)
        h = q * p1 * p2
        break

    qd, pd, hd = _dec(q), _dec(p), _dec(h)
    f = sympy.Rational(p.numerator, p.denominator) * P - sympy.Rational(q.numerator, q.denominator) * P**2
    f = f - sympy.Rational(h.numerator, h.denominator)
    # Independent route: solve f(P) = 0, then differentiate.
    roots = sorted(sympy.solve(f, P))
    check(roots == [p1, p2], "the equilibria are not the drawn roots")
    slope = Fraction(str(sympy.diff(f, P).subs(P, p2)))
    check(slope == p - 2 * q * p2 == -q * (p2 - p1), "f'(P2) disagrees")
    check(slope < 0 and sympy.diff(f, P).subs(P, p1) > 0, "stability pattern is not stable above, unstable below")

    tol = stated_precision_tolerance(2)
    # (a) the larger equilibrium
    a_cands = [
        _dist(p1, "chose-the-other-equilibrium"),
        _dist(Fraction(p1 + p2, 2), "took-the-vertex-as-the-equilibrium"),
        _dist(p1 + p2, "sum-of-roots-as-root"),
        _dist(p1 * p2, "product-of-roots-as-root"),
    ]
    a_dist = pick_distractors(str(p2), a_cands, 3)
    # (b) f' at P2
    b_cands = [
        _dist(p - q * p2, "derivative-lost-factor-of-power", 2),
        _dist(-slope, "derivative-at-other-equilibrium", 2),
        _dist(0, "evaluated-function-not-derivative", 2),
        _dist(2 * q * p2, "derivative-dropped-linear-term", 2),
    ]
    b_text = stated_precision_string(slope, 2)
    b_dist = pick_distractors(b_text, b_cands, 3, tolerance=tol)
    # (c) the smaller equilibrium
    c_cands = [
        _dist(p1 * p2, "product-of-roots-as-root"),
        _dist(p1 + p2, "sum-of-roots-as-root"),
        _dist(Fraction(p1 + p2, 2), "took-the-vertex-as-the-equilibrium"),
        _dist(p2 - p1, "gap-between-equilibria-as-root"),
    ]
    c_dist = pick_distractors(str(p1), c_cands, 3)

    who, verb = rng.choice(_STORIES)
    model = join_terms([(pd, "P"), (-qd, "P^2"), (-hd, "")])
    stem = (
        f"{who} is modelled by dP/dt = f(P), where f(P) = {model} and t is in years. "
        f"The constant term is a fixed number of thousands {verb} each year."
    )
    sum_roots = p1 + p2
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the larger of the two equilibrium populations.",
                answer_kind="number",
                answer=p2,
                answer_text=str(p2),
                answer_latex=latex_answer(str(p2)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"f(P) = 0 gives {exact_decimal_string(q, 2)}(P - {p1})(P - {p2}) = 0, so P = {p1} or P = {p2}.",
                    f"The larger is {p2}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Find f'(P) at that equilibrium, correct to 2 decimal places. A negative value means it is stable.",
                answer_kind="number",
                answer=float(slope),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                working=[f"f'(P) = {pd} - {_dec(2 * q)}P, so f'({p2}) = {b_text}.", "f' is negative, so it is stable."],
                distractors=b_dist,
                follow_through=f"{_fraction_text(p)} - 2*{_fraction_text(q)}*part_a",
            ),
            dict(
                prompt="Hence find the smaller equilibrium, below which the population falls to zero.",
                answer_kind="number",
                answer=p1,
                answer_text=str(p1),
                answer_latex=latex_answer(str(p1)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The two equilibria add to {sum_roots}, because the roots of f(P) = 0 add to p/q.",
                    f"The smaller equilibrium is {sum_roots} - {p2} = {p1}; f'({p1}) > 0, so it is unstable.",
                ],
                distractors=c_dist,
                follow_through=f"{sum_roots} - part_a",
            ),
        ],
    )
