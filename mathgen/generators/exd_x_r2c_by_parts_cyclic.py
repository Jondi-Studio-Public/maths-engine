"""Exam chain for integration by parts: the cyclic integrals of ``e^(ax) sin(bx)`` and ``e^(ax) cos(bx)``.

Two definite integrals ``I`` and ``J`` over ``[0, L]`` (``L`` a multiple of ``pi / (2b)``), one with sin and one
with cos, and the standard exam route through them:

(a) one by-parts step on I:  ``I = P + kJ``                              fraction, 2 marks
(b) the same step on J, substituted: ``c I = Q``; find c                 fraction, 2 marks, follow-through (a)
(c) the exact value of I                                                 expression, 3 marks, follow-through (a), (b)
(d) hence the exact value of J                                           expression, 2 marks, follow-through (c)

## The algebra (checked on every question)

With f the function in I and g the other one, taking ``u`` as the trig factor each time gives

    I = (1/a) e^(ax) f(bx) |0..L  +  k1 J        k1 = -b/a if f = sin, +b/a if f = cos
    J = (1/a) e^(ax) g(bx) |0..L  +  k2 I        k2 = -k1

so ``(1 - k1 k2) I = (1/a) [e^(ax) (f(bx) + k1 g(bx))] |0..L`` and ``c = 1 - k1 k2 = 1 + b^2/a^2``. The key for
I and J is checked against numerical quadrature of the integrands themselves, to 1e-10.

## Distractors

* (a): the sign of k wrong, the ratio inverted, k left as b (the division by a forgotten).
* (b): the cyclic integral subtracted instead of added (c = 1 - b^2/a^2), the leading 1 forgotten, the ratio
  not squared, the ratio inverted.
* (c): the boundary term not divided by c, the other sign on k in the boundary term, the lower limit forgotten.
* (d): the I term left out, its sign flipped, the lower limit forgotten.
"""

from __future__ import annotations

from fractions import Fraction

from functools import cache


@cache
def _np():
    import numpy

    return numpy


from sympy import Integer, Rational, cos, exp, pi, sin

from mathgen.answers import Distractor, exact_value
from mathgen.core import check, generator, latex_answer, phrase, pick_distractors, rational_text

_STEMS = (
    "Let I = ∫ from x = 0 to x = {lim} of {fi} dx and J = ∫ from x = 0 to x = {lim} of {gj} dx.",
    "Two integrals are defined by I = ∫ from x = 0 to x = {lim} of {fi} dx and "
    "J = ∫ from x = 0 to x = {lim} of {gj} dx.",
    "Consider the integrals I = ∫ from x = 0 to x = {lim} of {fi} dx and J = ∫ from x = 0 to x = {lim} of {gj} dx.",
)


@cache
def _gl():
    return _np().polynomial.legendre.leggauss(24)


def _quadrature(fun, upper: float, panels: int) -> float:
    """Integral of `fun` over [0, upper] by 24-point Gauss-Legendre on equal panels."""
    _NODES, _WEIGHTS = _gl()
    edges = _np().linspace(0.0, upper, panels + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        total += float(_np().sum(_WEIGHTS * fun((hi - lo) / 2 * _NODES + (hi + lo) / 2)) * (hi - lo) / 2)
    return total


def _typed(expr) -> str:
    return str(expr).replace("**", "^")


def _with_term(edge, k, name) -> str:
    """ "edge + (k)name", or just "(k)name" when the boundary value is zero."""
    tail = f"({rational_text(k)}){name}"
    return tail if edge == 0 else f"{_typed(edge)} + {tail}"


def _fraction_part(prompt, value, marks, working, wrong, follow=None):
    text = rational_text(value)
    cands = [Distractor(rational_text(v), tag, exact_value(Fraction(int(v.p), int(v.q)))) for v, tag in wrong]
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(Fraction(int(value.p), int(value.q))),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, cands),
    )
    if follow:
        part["follow_through"] = follow
    return part


def _expression_part(prompt, value, marks, working, wrong, follow):
    """An expression part. `working` lines may contain `{value}`, filled with the answer's text."""
    text = _typed(value)
    target = float(value)
    cands = []
    for v, tag in wrong:
        if abs(float(v) - target) > 1e-9:
            shown = _typed(v)
            cands.append(Distractor(shown, tag, shown))
    return dict(
        prompt=prompt,
        answer_kind="expression",
        answer=str(value),
        answer_text=text,
        answer_latex=latex_answer(value),
        marks=marks,
        working=[line.replace("{value}", text) for line in working],
        distractors=pick_distractors(text, cands),
        follow_through=follow,
    )


@generator("exd_x_r2c_by_parts_cyclic", level=17, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_by_parts_cyclic(rng, tier: str = "exam"):
    """I and J over [0, L]: one by-parts step, the cyclic equation, then both exact values."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    a = rng.randint(1, 5)
    b = rng.choice([v for v in range(1, 6) if v != a])  # a = b would make every ratio +-1 and the distractors collapse
    f_is_sin = rng.random() < 0.5
    m = rng.randint(1, 4)
    f, g = (sin, cos) if f_is_sin else (cos, sin)
    fn, gn = ("sin", "cos") if f_is_sin else ("cos", "sin")
    limit = Rational(m, 2 * b) * pi
    k1 = Rational(-b, a) if f_is_sin else Rational(b, a)
    k2 = -k1
    c = 1 - k1 * k2

    # Exact values at the ends: f(bL) and g(bL) are 0, +-1.
    f_end, g_end = f(m * pi / 2), g(m * pi / 2)
    f_start, g_start = Integer(f(0)), Integer(g(0))
    exp_end = exp(a * limit)

    def boundary(kk, *, lower=True, g_term=True):
        lo_part = (f_start + kk * g_start) if lower else 0
        return (exp_end * (f_end + kk * g_end) - lo_part) / a if g_term else (exp_end * f_end - lo_part) / a

    i_value = boundary(k1) / c
    # The by-parts boundary values, as numbers: I = I_edge + k1 J and J = J_edge + k2 I, so cI = I_edge + k1 J_edge.
    i_edge = (exp_end * f_end - f_start) / a
    j_edge = (exp_end * g_end - g_start) / a
    q_text = _typed(boundary(k1))
    j_value = (exp_end * g_end - g_start) / a + k2 * i_value

    # Independent check: Gauss-Legendre quadrature of the integrands themselves, one panel per quarter period.
    for value, fun in (
        (i_value, _np().sin if f_is_sin else _np().cos),
        (j_value, _np().cos if f_is_sin else _np().sin),
    ):
        target = _quadrature(lambda t, fun=fun: _np().exp(a * t) * fun(b * t), float(limit), m)
        check(abs(float(value.evalf(20)) - target) < 1e-9 * (1 + abs(target)), "a value disagrees with quadrature")

    def exp_text(coefficient):
        return "e^x" if coefficient == 1 else f"e^({coefficient}x)"

    def trig_text(name):
        return f"{name}(x)" if b == 1 else f"{name}({b}x)"

    lim_text = _typed(limit).replace("*pi", "π").replace("pi", "π")
    fi, gj = f"{exp_text(a)}·{trig_text(fn)}", f"{exp_text(a)}·{trig_text(gn)}"
    stem = phrase(rng, _STEMS, lim=lim_text, fi=fi, gj=gj)

    boundary_name = f"[(1/{a})·{exp_text(a)}·{trig_text(fn)}]" if a != 1 else f"[{exp_text(a)}·{trig_text(fn)}]"
    part_a = _fraction_part(
        f"Integrating I by parts once, with u = {trig_text(fn)}, gives I = {boundary_name} evaluated from x = 0 to "
        f"x = {lim_text}, plus kJ. Find the value of k.",
        k1,
        2,
        [f"With u = {trig_text(fn)} and dv = {exp_text(a)} dx, the remaining integral is ({rational_text(k1)})J."],
        [
            (-k1, "sign-reversed-result"),
            (1 / abs(k1), "inverted-ratio") if k1 > 0 else (-1 / abs(k1), "inverted-ratio"),
            (Rational(b) if k1 > 0 else Rational(-b), "forgot-to-divide-by-coefficient"),
            (k1 / a, "extra-factor-introduced"),
        ],
    )
    part_b = _fraction_part(
        "Applying the same method to J and substituting into the equation for I gives cI = Q, where Q does not "
        "involve I or J. Using your value of k, find c.",
        c,
        2,
        [
            f"I = {_with_term(i_edge, k1, 'J')} and J = {_with_term(j_edge, k2, 'I')}, "
            f"so cI = {q_text} with c = 1 - ({rational_text(k1)})({rational_text(k2)}) = {rational_text(c)}."
        ],
        [
            (1 - Rational(b * b, a * a), "sign-reversed-result"),
            (Rational(b * b, a * a), "cyclic-forgot-original-integral"),
            (1 + Rational(b, a), "ratio-not-squared"),
            (1 + Rational(a * a, b * b), "inverted-ratio"),
        ],
        follow=f"1 - part_a*({rational_text(k2)})",
    )

    c_follow = (
        f"((exp({_typed(a * limit)})*({f_end} + part_a*({g_end})) - ({f_start} + part_a*({g_start})))/{a})/part_b"
    )
    part_c = _expression_part(
        "Hence find the exact value of I.",
        i_value,
        3,
        [f"I = ({q_text}) ÷ {rational_text(c)} = {{value}}."],
        [
            (boundary(k1), "forgot-to-divide-by-coefficient"),
            (boundary(-k1) / c, "sign-reversed-result"),
            (boundary(k1, lower=False) / c, "upper-limit-only"),
        ],
        c_follow,
    )
    part_d = _expression_part(
        "Hence find the exact value of J.",
        j_value,
        2,
        [f"J = {_with_term(j_edge, k2, 'I')} = {{value}}."],
        [
            ((exp_end * g_end - g_start) / a, "cyclic-forgot-original-integral"),
            ((exp_end * g_end - g_start) / a - k2 * i_value, "sign-reversed-result"),
            ((exp_end * g_end) / a + k2 * i_value, "upper-limit-only"),
        ],
        f"(exp({_typed(a * limit)})*({g_end}) - ({g_start}))/{a} + ({rational_text(k2)})*part_c",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
