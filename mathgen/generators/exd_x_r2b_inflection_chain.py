"""Generates `exd_x_r2b_inflection_chain` questions: a multi-part exam chain on concavity and a point of inflection.

Skill: senior-understand-the-concepts-of-concavity-and-points-of (exam tier).

A cubic `f(x) = a x^3 + k x^2 + B x + C` has an unknown `k` and a point of inflection at a stated `x = p`:

(a) find k from f''(p) = 0                                        fraction, 2 marks
(b) f'' at a nearby x0 and the concavity there                    ordered_list, 2 marks
(c) the y-coordinate of the point of inflection                   fraction, 1 mark,  f(p) from k
(d) the tangent at the point of inflection, as y = mx + c         expression, 2 marks, from k and f(p)

## Construction: choose the inflection first

`a`, `p`, `B` and `C` are drawn and `k = -3 a p`, so `f''(x) = 6a(x - p)` changes sign at `p` and `f''(p) = 0`
exactly. `f'(p) = B - 3 a p^2`, so the tangent at the inflection depends on `k` only through the gradient, and a
learner's wrong `k` carries into (c) and (d). Every key is re-derived with SymPy from the finished polynomial.
"""

from __future__ import annotations

import sympy
from sympy import Rational, diff, symbols

from ..core import attempts, check, generator, join_terms, paren_negative, phrase
from .exd_x_r2b_support import expression_part, fraction_part, typed, words_part, draw

x, k = symbols("x k")

_A = (-3, -2, -1, 1, 2, 3)
_STEMS = (
    "The function f(x) = {f}, where k is a constant, has a point of inflection at x = {p}.",
    "Let f(x) = {f}, where k is a constant. The graph of f has a point of inflection at x = {p}.",
    "The graph of y = f(x), where f(x) = {f} and k is a constant, has a point of inflection where x = {p}.",
)


def _shape(second) -> str:
    return "concave up" if second > 0 else "concave down"


def _line(slope, intercept) -> str:
    return "y = " + typed(slope * x + intercept)


def _build(rng):
    """A cubic with an unknown coefficient and an inflection at x = p: k, concavity, the point, the tangent."""
    for _ in attempts(1000, "a cubic with a point of inflection that gives a non-trivial tangent"):
        a = rng.choice(_A)
        p = rng.choice([v for v in range(-3, 4) if v])
        b = rng.choice([v for v in range(-9, 10) if v])
        c = rng.choice([v for v in range(-12, 13) if v])
        d = rng.choice([-2, -1, 1, 2])
        k_val = -3 * a * p
        slope = b - 3 * a * p * p
        y0 = a * p**3 + k_val * p**2 + b * p + c
        if slope != 0 and y0 != 0 and abs(slope) <= 40 and abs(y0) <= 80:
            break

    # ---- independent derivation from the finished polynomial ---------------------------------------------------
    poly = a * x**3 + k * x**2 + b * x + c
    check(sympy.solve(diff(poly, x, 2).subs(x, p), k) == [k_val], "k does not give an inflection at p")
    f = poly.subs(k, k_val)
    check(sympy.solve(diff(f, x, 2), x) == [p], "f'' does not vanish only at p")
    x0 = p + d
    second = int(diff(f, x, 2).subs(x, x0))
    check(second == 6 * a * d and second != 0, "f''(x0) disagrees with the construction")
    check(int(f.subs(x, p)) == y0 and int(diff(f, x).subs(x, p)) == slope, "point or gradient disagree")
    tangent = slope * (x - p) + y0

    f_text = join_terms([(a, "x^3"), (1, "kx^2"), (b, "x"), (c, "")])
    stem = phrase(rng, _STEMS, f=f_text, p=p)

    # ---- (a) k --------------------------------------------------------------------------------------------------
    a_wrong = [
        (Rational(3 * a * p), "sign-error-moving-term"),  # 6ap + 2k = 0 rearranged with the wrong sign
        (Rational(-6 * a * p), "forgot-power-multiplier"),  # kx^2 differentiated twice without the 2
        (Rational(-(3 * a * p * p + b), 2 * p), "first-derivative-set-to-zero"),  # f'(p) = 0 solved instead
        (Rational(-a * p, 1), "forgets-the-multiplier"),  # the 3 in 6a = 3 * 2a lost
    ]
    part_a = fraction_part(
        "Find the value of k.",
        k_val,
        2,
        [
            f"f''(x) = {join_terms([(6 * a, 'x'), (2, 'k')])}, and f''({p}) = 0.",
            f"{6 * a * p} + 2k = 0, so k = {k_val}.",
        ],
        a_wrong,
    )

    # ---- (b) f'' at x0 and the concavity -------------------------------------------------------------------------
    at_other = int(diff(f, x, 2).subs(x, p - d))
    dropped = 6 * a * x0  # k left out of f''
    b_wrong = [
        ([second, _shape(-second)], "flipped-concavity"),
        ([at_other, _shape(at_other)], "second-derivative-at-wrong-point"),
        ([dropped, _shape(dropped)] if dropped != second and dropped else None, "forgets-the-multiplier"),
        ([-second, _shape(second)], "sign-reversed-result"),
    ]
    part_b = words_part(
        f"Find f''({x0}) and state whether the graph of f is concave up or concave down at x = {x0}. "
        "Give the value, then concave up or concave down, separated by a comma.",
        [second, _shape(second)],
        2,
        [
            f"f''(x) = {6 * a}x {'+' if 2 * k_val >= 0 else '-'} {abs(2 * k_val)}, so f''({x0}) = {second}.",
            f"f''({x0}) is {'positive' if second > 0 else 'negative'}, so the graph is {_shape(second)} there.",
        ],
        [w for w in b_wrong if w[0] is not None],
    )

    # ---- (c) the point of inflection ---------------------------------------------------------------------------
    c_wrong = [
        (int(f.subs(x, -p)), "substitution-sign-error"),
        (int(f.subs(x, x0)), "evaluated-at-the-wrong-x"),
        (int((a * x**3 + b * x + c).subs(x, p)), "forgets-the-multiplier"),  # the kx^2 term left out
        (slope, "evaluated-wrong-function-order"),
    ]
    part_c = fraction_part(
        "Find the y-coordinate of the point of inflection.",
        y0,
        1,
        [f"f({p}) = {y0}."],
        c_wrong,
        follow_through=f"{paren_negative(a * p**3)} + part_a*{p * p} + {paren_negative(b * p)} + {paren_negative(c)}",
    )

    # ---- (d) the tangent at the inflection ---------------------------------------------------------------------
    d_wrong = [
        (sympy.Integer(y0), "second-derivative-as-gradient"),  # f''(p) = 0 used as the gradient
        (slope * (x + p) + y0, "sign-slip-point-slope"),
        (slope * (x - p), "forgot-point-y-value"),
        (-sympy.Rational(1, slope) * (x - p) + y0, "normal-instead-of-tangent"),
    ]
    part_d = expression_part(
        "Find the equation of the tangent to the graph at the point of inflection, in the form y = mx + c.",
        tangent,
        2,
        [
            f"f'(x) = {join_terms([(3 * a, 'x^2'), (2 * k_val, 'x'), (b, '')])}, so the gradient at x = {p} is {slope}.",
            f"y - {paren_negative(y0)} = {paren_negative(slope)}(x - {paren_negative(p)}), so {_line(slope, y0 - slope * p)}.",
        ],
        [(sympy.expand(e), t) for e, t in d_wrong],
        prefix="y = ",
        follow_through=(
            f"(3*{a}*{p * p} + 2*part_a*{paren_negative(p)} + {paren_negative(b)})*(x - {paren_negative(p)}) + part_c"
        ),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_inflection_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_inflection_chain(rng, tier: str = "exam"):
    """A cubic with an unknown coefficient and an inflection at x = p: k, concavity, the point, the tangent."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
