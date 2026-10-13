"""Exam-tier source for `senior-find-and-use-the-derivative-of-the-inverse`: a tangent to an arctan curve.

A curve y = f(x) is built from an inverse tangent of an inner function plus a linear term:

* family A: f(x) = a arctan(kx + m) + bx, examined at the integer x0 where kx0 + m = 1;
* family B: f(x) = a arctan(x/k) + bx, examined at x0 = k, where x/k = 1.

At x0 the inner function is 1, so arctan(1) = pi/4 and the gradient is exact. The chain:

(a) f'(x)                                   expression, 2 marks
(b) the gradient at x = x0                  fraction,   1 mark,  Subs(part_a, x, x0)
(c) where the tangent meets the x-axis      number,     2 marks, x0 - f(x0)/part_b   (2 decimal places)

The linear term bx does not change the tangent's y-intercept (it cancels), but it does change the x-intercept, so
the question asks for the x-intercept and every part of f matters.

The learner needs f(x0) = a pi/4 + b x0, which is the "use the exact value of arctan 1" step. All arithmetic is exact
(Fractions, with pi to 50 digits for the one rounded answer). The key f'(x) is checked against a central difference
of f as written, and the gradient and intercept are checked in the tests against SymPy and a numerical tangent.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    check,
    generator,
    join_terms,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ._exd_distinct_parts import distinct_part_answers

PI = Fraction("3.14159265358979323846264338327950288419716939937510")
_PLACES = 2


def _coef(a: int) -> str:
    """The coefficient written before "arctan": nothing for 1, a bare sign for -1, else the number and a space."""
    return "" if a == 1 else "-" if a == -1 else f"{a} "


def _quotient(num: Fraction, den: str) -> str:
    """ "num/den" with num an int or Fraction, written as a student would type it."""
    num = Fraction(num)
    if num.denominator == 1:
        return f"{int(num)}/{den}"
    return f"{num.numerator}/({num.denominator}*{den})"


def _with_linear(term: str, b: int) -> str:
    return f"{term} {'+' if b > 0 else '-'} {abs(b)}" if b else term


def _draw(rng):
    for _ in range(1000):
        family = rng.choice("AAB")
        a = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        b = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        if family == "A":
            k = rng.choice([2, 3, 4, 5])
            x0 = rng.choice([1, 2, 3, 4])
            m = 1 - k * x0
            slope = Fraction(a * k, 2) + b
        else:
            k = rng.choice([2, 3, 4, 5, 6])
            x0, m = k, 0
            slope = Fraction(a, 2 * k) + b
        if abs(slope) < Fraction(1, 2):
            continue
        f_x0 = Fraction(a, 4) * PI + b * x0
        target = x0 - f_x0 / slope
        if abs(target) < Fraction(1, 5) or abs(target) > 40:
            continue
        if abs(target - round_half_up(target, _PLACES)) < Fraction(1, 2000):  # clear of a rounding boundary
            continue
        return family, a, b, k, m, x0, slope, f_x0, target
    check(False, "no usable curve found")


def _candidate(tag: str, text: str, fn, b: int):
    """A wrong f'(x) as (tag, typed text, its function of x) with the linear term b added."""
    return tag, _with_linear(text, b), (lambda v: fn(v) + b)


@generator("exd_x_r3e_inverse_trig_tangent", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_inverse_trig_tangent(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    family, a, b, k, m, x0, slope, f_x0, target = _draw(rng)

    if family == "A":
        inner = join_terms([(k, "x"), (m, "")])
        inner_typed = join_terms([(k, "*x"), (m, "")])
        func_core = f"{_coef(a)}arctan({inner})"
        func_latex = rf"{_coef(a)}\arctan\left({inner}\right)"

        def u(v):
            return k * v + m

        def f(v):
            return a * math.atan(float(u(v)))

        den_true = f"(1 + ({inner_typed})^2)"
        key_core = _quotient(a * k, den_true)
        key_fn = lambda v: Fraction(a * k) / (1 + u(v) ** 2)  # noqa: E731
        slips = [
            _candidate(
                "inverse-trig-sign-flip", _quotient(-a * k, den_true), lambda v: Fraction(-a * k) / (1 + u(v) ** 2), b
            ),
            _candidate(
                "chain-rule-inner-derivative-dropped",
                _quotient(a, den_true),
                lambda v: Fraction(a) / (1 + u(v) ** 2),
                b,
            ),
            _candidate(
                "dropped-inner-constant-in-formula",
                _quotient(a * k, f"(1 + ({k}*x)^2)"),
                lambda v: Fraction(a * k) / (1 + (k * v) ** 2),
                b,
            ),
            _candidate(
                "arctan-derivative-missing-square",
                _quotient(a * k, f"(1 + {inner_typed})"),
                lambda v: Fraction(a * k) / (1 + u(v)),
                b,
            ),
        ]
        slip_slopes = [
            ("inverse-trig-sign-flip", Fraction(-a * k, 2) + b),
            ("chain-rule-inner-derivative-dropped", Fraction(a, 2) + b),
            ("dropped-inner-constant-in-formula", Fraction(a * k, 1 + (k * x0) ** 2) + b),
            ("dropped-linear-term-in-derivative", Fraction(a * k, 2)),
        ]
    else:
        func_core = f"{_coef(a)}arctan(x/{k})"
        func_latex = rf"{_coef(a)}\arctan\left(\frac{{x}}{{{k}}}\right)"

        def u(v):
            return Fraction(v) / k

        def f(v):
            return a * math.atan(float(v) / k)

        den_true = f"({k * k} + x^2)"
        den_u = f"(1 + (x/{k})^2)"
        key_core = _quotient(a * k, den_true)
        key_fn = lambda v: Fraction(a * k) / (k * k + Fraction(v) ** 2)  # noqa: E731
        slips = [
            _candidate(
                "inverse-trig-sign-flip",
                _quotient(-a * k, den_true),
                lambda v: Fraction(-a * k) / (k * k + Fraction(v) ** 2),
                b,
            ),
            _candidate(
                "chain-rule-inner-derivative-dropped", _quotient(a, den_u), lambda v: Fraction(a) / (1 + u(v) ** 2), b
            ),
            _candidate("inverted-ratio", _quotient(a * k, den_u), lambda v: Fraction(a * k) / (1 + u(v) ** 2), b),
            _candidate(
                "arctan-derivative-missing-square",
                _quotient(a * k, f"({k * k} + x)"),
                lambda v: Fraction(a * k) / (k * k + Fraction(v)),
                b,
            ),
        ]
        slip_slopes = [
            ("inverse-trig-sign-flip", Fraction(-a, 2 * k) + b),
            ("chain-rule-inner-derivative-dropped", Fraction(a, 2) + b),
            ("inverted-ratio", Fraction(a * k, 2) + b),
            ("dropped-linear-term-in-derivative", Fraction(a, 2 * k)),
        ]
    func_text = func_core
    if b:
        tail = f" {'+' if b > 0 else '-'} {abs(b) if abs(b) != 1 else ''}x"
        func_text += tail
        func_latex += tail
    key_text = _with_linear(key_core, b)

    # ---- independent check of (a): a central difference of f as written, and the exact gradient at x0 -----------
    def full(v):
        return f(v) + b * float(v)

    for probe in (Fraction(1, 3), Fraction(5, 4), Fraction(-7, 5)):
        h = 1e-6
        numeric = (full(float(probe) + h) - full(float(probe) - h)) / (2 * h)
        check(abs(numeric - float(key_fn(probe) + b)) < 1e-6, "f'(x) disagrees with a numerical derivative")
    check(key_fn(x0) + b == slope, "gradient wrong")
    check(abs(float(f_x0) - full(x0)) < 1e-9, "f(x0) wrong")

    # ---- (a) distractors ------------------------------------------------------------------------------------
    a_cands = slips + [
        ("dropped-linear-term-in-derivative", key_core, key_fn),
    ]
    rng.shuffle(a_cands)
    probes = (Fraction(2, 7), Fraction(13, 9), Fraction(-5, 11), Fraction(17, 13), Fraction(3, 19))
    key_values = tuple(key_fn(p) + b for p in probes)
    a_dist, seen = [], [key_values]
    for tag, text, fn in a_cands:
        values = (
            tuple(fn(p) for p in probes)
            if tag != "dropped-linear-term-in-derivative"
            else tuple(key_fn(p) for p in probes)
        )
        if values in seen:
            continue
        seen.append(values)
        a_dist.append(Distractor(text, tag, value=text))
    a_dist = a_dist[:3]

    # ---- (b) gradient ---------------------------------------------------------------------------------------
    slope_text = rational_text(slope)
    rng.shuffle(slip_slopes)
    b_cands = [Distractor(rational_text(v), tag, value=v) for tag, v in slip_slopes if v != slope]
    b_dist = pick_distractors(slope_text, b_cands, 3)

    # ---- (c) x-intercept ------------------------------------------------------------------------------------
    shown = round_half_up(target, _PLACES)
    c_text = stated_precision_string(shown, _PLACES)
    deg_f = Fraction(a * 45 + b * x0)
    half_pi_f = Fraction(a, 2) * PI + b * x0
    c_slips = [
        ("inverse-trig-value-in-degrees", x0 - deg_f / slope),
        ("inverse-trig-exact-value-wrong", x0 - half_pi_f / slope),
        ("x-intercept-sign-slip-on-quotient", x0 + f_x0 / slope),
        ("x-intercept-shift-forgotten", -f_x0 / slope),
        ("multiplied-by-gradient-not-divided", x0 - f_x0 * slope),
    ]
    rng.shuffle(c_slips)
    c_cands = []
    for tag, v in c_slips:
        v2 = round_half_up(v, _PLACES)
        if abs(v2 - shown) >= Fraction(1, 20) and abs(v2) < 10**6:
            c_cands.append(Distractor(stated_precision_string(v2, _PLACES), tag, value=v2))
    c_dist = pick_distractors(c_text, c_cands, 3, tolerance=Fraction(1, 20))
    check(len(c_dist) >= 3 and len(b_dist) >= 3 and len(a_dist) >= 3, "too few distractors")

    stem = f"A curve has equation y = f(x), where f(x) = {func_text}."
    stem_latex = rf"\text{{A curve has equation }}y = f(x)\text{{, where }}f(x) = {func_latex}."
    pi_over_4 = Fraction(a, 4)
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                prompt="Find f'(x).",
                answer_kind="expression",
                answer=key_text,
                answer_text=key_text,
                answer_latex=latex_answer(key_text),
                marks=2,
                working=[f"f'(x) = {key_text}."],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Find the gradient of the curve at the point where x = {x0}.",
                answer_kind="fraction",
                answer={"p": slope.numerator, "q": slope.denominator},
                answer_text=slope_text,
                answer_latex=latex_answer(slope_text),
                marks=1,
                working=[f"At x = {x0} the inside function equals 1, so the gradient is {slope_text}."],
                distractors=b_dist,
                follow_through=f"Subs(part_a, (x,), ({x0},))",
            ),
            dict(
                prompt=(
                    f"Hence find the x-coordinate of the point where the x-axis meets the tangent to the curve at "
                    f"x = {x0}, correct to {_PLACES} decimal places."
                ),
                answer_kind="number",
                answer=numeric_answer(shown),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                working=[
                    f"f({x0}) = {a}·π/4 + {b}·{x0}, since arctan 1 = π/4.",
                    f"The tangent meets the x-axis where x = {x0} - f({x0})/{slope_text} = {c_text}.",
                ],
                distractors=c_dist,
                follow_through=f"{x0} - ({pi_over_4.numerator}*pi/{pi_over_4.denominator} + {b * x0})/part_b",
            ),
        ],
    )
