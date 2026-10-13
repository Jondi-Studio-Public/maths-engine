"""Generates `exd_x_r3f_initial_value_chain` questions: an exam chain from a supplied general solution to a tangent.

The general solution y = F(x) + C of a first-order equation is supplied already integrated, with one point on the
solution. The chain:

(a) the constant C from the point                                        fraction,   2 marks   C = y0 - F(x0)
(b) the value of y at a second stated x                                  fraction,   1 mark    F(x1) + part_a
(c) the equation of the tangent to the solution curve at that x          expression, 2 marks   m (x - x1) + part_b

The gradient m in (c) is the derivative of the supplied F, so the learner differentiates but never integrates.
Part (b) follows through from (a) and part (c) from (b), so a learner whose C is wrong still earns the later parts
for using it correctly. Two settings share the construction: a curve y(x) and a particle's displacement s(t)
(where t is a positive time). Exact rational arithmetic throughout.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, pick_distractors, rational_text
from ._exd_r3f_parts import fraction_part
from ._exd_distinct_parts import distinct_part_answers

_SETTINGS = (
    # (independent, dependent, positive_only, curve phrase, stem)
    (
        "x",
        "y",
        False,
        "the solution curve",
        "The general solution of a differential equation is y = {f} + C. The solution curve passes through the "
        "point ({x0}, {y0}).",
    ),
    (
        "x",
        "y",
        False,
        "the solution curve",
        "A differential equation has general solution y = {f} + C, where C is a constant. The point ({x0}, {y0}) "
        "lies on the solution curve.",
    ),
    (
        "t",
        "s",
        True,
        "the graph of s against t",
        "The displacement s metres of a particle from the origin at time t seconds satisfies s = {f} + C for some "
        "constant C. When t = {x0}, s = {y0}.",
    ),
)

Poly = dict  # degree -> Fraction coefficient


def _eval(poly: Poly, at) -> Fraction:
    return sum((c * Fraction(at) ** d for d, c in poly.items()), Fraction(0))


def _text(poly: Poly, var: str) -> str:
    """The polynomial as a learner types it, highest power first: `3*x^2/2 - x/3 + 5/4`."""
    out = []
    for degree in sorted(poly, reverse=True):
        c = poly[degree]
        if c == 0:
            continue
        mag = abs(c)
        if degree == 0:
            body = rational_text(mag)
        else:
            power = var if degree == 1 else f"{var}^{degree}"
            numer = power if mag.numerator == 1 else f"{mag.numerator}*{power}"
            body = numer if mag.denominator == 1 else f"{numer}/{mag.denominator}"
        out.append(("-" if c < 0 else "+", body))
    if not out:
        return "0"
    first_sign, first = out[0]
    text = ("-" if first_sign == "-" else "") + first
    for sign, body in out[1:]:
        text += f" {sign} {body}"
    return text


def _gradient_text(poly: Poly, var: str) -> str:
    """An integer-coefficient gradient written the way a paper prints it: `4x^3 - x^2 + 3x`."""
    return _text(poly, var).replace("*", "")


@generator("exd_x_r3f_initial_value_chain", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_initial_value_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Constant of integration, the value at a second point, then the tangent there."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    var, dep, positive, curve, template = rng.choice(_SETTINGS)
    for _ in attempts(1000, "a solution with two or three terms, distinct points and a non-zero gradient"):
        degrees = sorted(rng.sample([0, 1, 2], rng.choice([2, 3])), reverse=True)
        grad: Poly = {d: Fraction(rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])) for d in degrees}
        x0 = rng.randint(1, 4) if positive else rng.choice([-3, -2, -1, 1, 2, 3])
        x1 = rng.randint(1, 4) if positive else rng.choice([-3, -2, -1, 1, 2, 3])
        y0 = rng.randint(-9, 9)
        anti: Poly = {d + 1: c / (d + 1) for d, c in grad.items()}
        c_value = y0 - _eval(anti, x0)
        slope = _eval(grad, x1)
        if x1 != x0 and y0 != 0 and max(degrees) >= 2 and c_value != 0 and slope != 0:
            break

    # Independent checks: differentiating F returns the gradient, Simpson's rule (exact for a cubic) gives the
    # same change in y between 0 and x0, and the tangent touches the solution at x1 with the stated gradient.
    check({d - 1: c * d for d, c in anti.items()} == grad, "F does not differentiate back to the gradient")
    simpson = Fraction(x0, 6) * (_eval(grad, 0) + 4 * _eval(grad, Fraction(x0, 2)) + _eval(grad, x0))
    check(simpson == _eval(anti, x0), "F disagrees with Simpson's rule")
    solution: Poly = {**anti, 0: c_value}
    check(_eval(solution, x0) == y0, "the solution misses the given point")
    value = _eval(solution, x1)
    tangent: Poly = {1: slope, 0: value - slope * x1}
    check(_eval(tangent, x1) == value, "the tangent misses the curve at x1")

    anti_x0, anti_x1 = _eval(anti, x0), _eval(anti, x1)
    c_candidates = [
        (Fraction(y0), "forgets-to-subtract-antiderivative"),
        (y0 + anti_x0, "sign-error-on-antiderivative"),
        (anti_x0, "reports-antiderivative-as-constant"),
        (y0 - _eval(grad, x0), "evaluated-derivative-instead-of-function"),
        (y0 - _eval(anti, y0) if y0 != x0 else None, "antiderivative-evaluated-at-wrong-value"),
    ]
    value_candidates = [
        (anti_x1, "particular-solution-omits-constant"),
        (anti_x1 - c_value, "particular-solution-constant-sign-error"),
        (anti_x1 + y0, "particular-solution-used-y0-as-constant"),
        (c_value, "reports-constant-not-value"),
    ]

    def line(m, point_y, shift=0) -> Poly:
        """y = m (x - (x1 + shift)) + point_y."""
        return {1: Fraction(m), 0: point_y - Fraction(m) * (x1 + shift)}

    tangent_candidates = [
        (line(-1 / slope, value), "tangent-and-normal-confused"),
        (line(slope, Fraction(0)), "tangent-point-slope-dropped-y-value"),
        (line(slope, value, shift=2 * -x1), "tangent-point-slope-sign-error"),
        (line(value, value), "y-value-used-as-gradient"),
    ]

    tangent_text = _text(tangent, var)
    distractors = pick_distractors(
        tangent_text,
        [
            Distractor(_text(p, var), tag, _text(p, var))
            for p, tag in tangent_candidates
            if p != tangent and any(p.values())
        ],
        3,
    )
    check(distractors, "no tangent distractor survives")
    part_c = dict(
        prompt=f"Find the equation of the tangent to {curve} at {var} = {x1}, in the form {dep} = m{var} + c.",
        answer_kind="expression",
        answer=tangent_text,
        answer_text=tangent_text,
        answer_latex=latex_answer(tangent_text),
        marks=2,
        follow_through=f"({rational_text(slope)})*({var} - ({x1})) + part_b",
        working=[
            f"{dep}' = {_text(grad, var)}, so the gradient at {var} = {x1} is {rational_text(slope)}.",
            f"{dep} - ({rational_text(value)}) = {rational_text(slope)}({var} - ({x1})), so {dep} = {tangent_text}.",
        ],
        distractors=distractors,
    )
    part_a = fraction_part(
        "Find the value of the constant C.",
        c_value,
        c_candidates,
        marks=2,
        working=[f"{y0} = {rational_text(anti_x0)} + C, so C = {rational_text(c_value)}."],
    )
    part_b = fraction_part(
        f"Hence find the value of {dep} when {var} = {x1}.",
        value,
        value_candidates,
        marks=1,
        follow_through=_follow_value(anti, var, x1),
        working=[f"{dep}({x1}) = {rational_text(anti_x1)} + ({rational_text(c_value)}) = {rational_text(value)}."],
    )
    stem = template.format(f=_text(anti, var).replace("*", ""), x0=x0, y0=y0)
    return dict(text=stem, parts=[part_a, part_b, part_c])


def _follow_value(anti: Poly, var: str, x1: int) -> str:
    """`F(x1) + part_a` as a follow-through expression."""
    return f"({_text(anti, var).replace(var, f'({x1})')}) + part_a"
