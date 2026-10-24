"""Exam-tier source for `solve-ode-separable-poly`: a gradient given as a product, an initial condition, a use.

The stem gives a curve's gradient as a product of two linear factors, dy/dx = (x - p)(x - q) with p < q, and a
point (x0, y0) the curve passes through, with x0 not 0 so the constant has to be found rather than read off.

* (a) the equation of the curve: expand, integrate term by term, and fix the constant from the point;
* (b) the value of y at a further x, followed through from the learner's own equation in (a);
* (c) the y-coordinate of the local maximum, which is at the smaller root p of the gradient; also followed
  through from (a).

Built from the roots and the point, then re-solved with `sympy.dsolve` using the initial condition (an
independent route to the equation), and the stationary point is checked to be a maximum from the sign of the
second derivative.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy
from sympy import Rational, symbols

from ..answers import Distractor, exact_value
from ._fy1ms_common import plain_caret
from ..core import check, check_equivalent, generator, pick_distractors, rational_text, value_safe_str

x = symbols("x")
_STEMS = [
    "A curve has gradient dy/dx = ({fp})({fq}) at every point and passes through the point ({x0}, {y0}).",
    "The gradient of a curve is given by dy/dx = ({fp})({fq}), and the curve passes through ({x0}, {y0}).",
    "Along a curve, dy/dx = ({fp})({fq}). The curve passes through the point ({x0}, {y0}).",
]
_B_PROMPTS = ["Find the value of y when x = {x1}.", "Find the y-coordinate of the point on the curve where x = {x1}."]
_C_PROMPTS = [
    "The curve has a local maximum. Find its y-coordinate.",
    "Find the y-coordinate of the local maximum of the curve.",
]


def _signed(n: int) -> str:
    return f"+ {n}" if n >= 0 else f"- {-n}"


def _factor(root: int) -> str:
    return f"x {_signed(-root)}" if root != 0 else "x"


def _product(p: int, q: int) -> str:
    """The gradient's two factors, ``(x + 3)(x - 2)``; a bare ``x`` factor is not bracketed."""
    fp, fq = _factor(p), _factor(q)
    return (fp if fp == "x" else f"({fp})") + (fq if fq == "x" else f"({fq})")


def _draw(rng: random.Random):
    for _ in range(1000):
        p, q = sorted(rng.sample(range(-5, 7), 2))
        if q - p < 2:
            continue
        x0 = rng.choice([v for v in range(-3, 5) if v != 0])
        y0 = rng.randint(-8, 9)
        x1 = rng.choice([v for v in range(-3, 6) if v not in (0, x0, p, q)])
        antiderivative = x**3 / 3 - Rational(p + q, 2) * x**2 + p * q * x
        constant = y0 - antiderivative.subs(x, x0)
        curve = antiderivative + constant
        key_b, key_c = curve.subs(x, x1), curve.subs(x, p)
        if constant.is_integer or len({constant, key_b, key_c, Rational(y0)}) < 4:
            continue
        return p, q, x0, y0, x1, antiderivative, constant, curve, key_b, key_c
    raise RuntimeError("no suitable roots and point in 1000 draws")


def _frac_dist(value, tag):
    value = Fraction(int(value.p), int(value.q))
    return Distractor(rational_text(value), tag, exact_value(value))


@generator("exd_x_u1_ode_initial_value", level=17, presets={"exam": dict(tier="exam")})
def exd_x_u1_ode_initial_value(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: solve dy/dx = (x - p)(x - q) through a point, then use the curve."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p, q, x0, y0, x1, antiderivative, constant, curve, key_b, key_c = _draw(rng)

    # Independent route: let SymPy solve the initial value problem from the expanded gradient.
    y = sympy.Function("y")
    solved = sympy.dsolve(sympy.Eq(y(x).diff(x), sympy.expand((x - p) * (x - q))), y(x), ics={y(x0): y0}).rhs
    check_equivalent(solved, curve, "curve disagrees with dsolve")
    check(sympy.diff(curve, x, 2).subs(x, p) < 0, "the stationary point at the smaller root is not a maximum")
    check(sympy.expand(curve.diff(x) - (x - p) * (x - q)) == 0, "curve has the wrong gradient")
    check(curve.subs(x, x0) == y0, "curve misses the given point")

    answer_text = "y = " + value_safe_str(sympy.expand(curve))
    expanded_gradient = sympy.expand((x - p) * (x - q))

    def curve_text(expr):
        return "y = " + value_safe_str(sympy.expand(expr))

    reversed_constant = antiderivative.subs(x, x0) - y0  # F(x0) - y0, the wrong way round
    a_cands = [
        Distractor(curve_text(antiderivative), "forgot-constant", curve_text(antiderivative)),
        Distractor(
            curve_text(x * expanded_gradient + constant),
            "raised-power-ode",
            curve_text(x * expanded_gradient + constant),
        ),
        Distractor(curve_text(antiderivative + y0), "initial-value-taken-as-constant", curve_text(antiderivative + y0)),
        Distractor(
            curve_text(antiderivative + reversed_constant),
            "constant-sign-reversed",
            curve_text(antiderivative + reversed_constant),
        ),
    ]
    gradient_x1 = Rational((x1 - p) * (x1 - q))
    b_cands = [
        _frac_dist(antiderivative.subs(x, x1), "forgot-constant"),
        _frac_dist(gradient_x1, "gradient-read-as-value"),
        _frac_dist((antiderivative + y0).subs(x, x1), "initial-value-taken-as-constant"),
        _frac_dist((antiderivative + reversed_constant).subs(x, x1), "constant-sign-reversed"),
    ]
    c_cands = [
        _frac_dist(curve.subs(x, q), "wrong-stationary-point"),
        _frac_dist(antiderivative.subs(x, p), "forgot-constant"),
        _frac_dist(Rational(y0), None),
        _frac_dist(Rational(p), "x-coordinate-given-for-y"),
    ]
    key_b_text, key_c_text = (
        rational_text(Fraction(int(key_b.p), int(key_b.q))),
        rational_text(Fraction(int(key_c.p), int(key_c.q))),
    )
    a_dist = pick_distractors(answer_text, a_cands, 3)
    b_dist = pick_distractors(key_b_text, b_cands, 3)
    c_dist = pick_distractors(key_c_text, c_cands, 3)
    check(len(a_dist) >= 2 and len(b_dist) >= 2 and len(c_dist) >= 2, "too few distractors")

    def as_answer(value):
        return exact_value(Fraction(int(value.p), int(value.q)))

    return dict(
        text=rng.choice(_STEMS).format(fp=_factor(p), fq=_factor(q), x0=x0, y0=y0),
        parts=[
            dict(
                prompt="Find the equation of the curve, giving y in terms of x.",
                answer_kind="expression",
                answer=value_safe_str(sympy.expand(curve)),
                answer_text=answer_text,
                marks=3,
                working=[
                    f"dy/dx = {_product(p, q)} = {plain_caret(expanded_gradient)}.",
                    f"Integrating, y = {plain_caret(sympy.expand(antiderivative))} + C.",
                    f"At ({x0}, {y0}): C = {y0} - ({antiderivative.subs(x, x0)}) = {constant}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=rng.choice(_B_PROMPTS).format(x1=x1),
                answer_kind="fraction",
                answer=as_answer(key_b),
                answer_text=key_b_text,
                marks=1,
                follow_through=f"Subs(part_a, (x,), ({x1},))",
                working=[f"y({x1}) = {key_b_text}."],
                distractors=b_dist,
            ),
            dict(
                prompt=rng.choice(_C_PROMPTS),
                answer_kind="fraction",
                answer=as_answer(key_c),
                answer_text=key_c_text,
                marks=2,
                follow_through=f"Subs(part_a, (x,), ({p},))",
                working=[
                    f"dy/dx = 0 at x = {p} and x = {q}. The gradient changes from positive to negative at x = {p}, "
                    "so the maximum is there.",
                    f"y({p}) = {key_c_text}.",
                ],
                distractors=c_dist,
            ),
        ],
    )
