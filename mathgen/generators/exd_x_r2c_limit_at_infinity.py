"""Exam chain for limits at infinity: a difference of square roots, both ends of the graph.

``f(x) = sqrt(a^2 x^2 + b x + c) - sqrt(a^2 x^2 + e x + g)`` has the form infinity minus infinity at both ends, so
it has to be rewritten, and the two ends are not the same limit (``sqrt(x^2) = |x|``). Set the way a senior
exam sets it:

(a) the numerator N(x) after multiplying by the conjugate          expression, 2 marks
(b) the limit as x tends to infinity                                fraction,   2 marks, follow-through (a)
(c) the limit as x tends to negative infinity                       fraction,   2 marks, follow-through (b)
(d) the vertical distance between the two horizontal asymptotes    fraction,   1 mark,  follow-through (b), (c)

## The algebra (checked on every question)

Multiplying by the conjugate gives ``f(x) = N(x) / (sqrt1 + sqrt2)`` with ``N(x) = (b - e) x + (c - g)``. Dividing
through by ``x`` (``|x|`` when x is negative) gives

    x -> +inf:  (b - e) / (2a)          x -> -inf:  -(b - e) / (2a)          distance  |b - e| / a

The limits are also found by evaluating f at x = +-10^8 with 50-digit arithmetic, which must land within 10^-6 of
the key. The follow-through for (b) reads the coefficient of x in the learner's own N(x).

## Distractors

* (a): the signs reversed, the radicands added instead of subtracted, the constant term dropped.
* (b): zero (infinity minus infinity "is" zero), the factor 2 forgotten, the ratio inverted.
* (c): the same value as (b), zero, the factor 2 forgotten.
* (d): a single limit offered as the gap, zero (the two limits taken as equal), the gap taken negative.
"""

from __future__ import annotations

from fractions import Fraction

from mpmath import mp, mpf, sqrt as mp_sqrt
from sympy import Rational, Symbol

from mathgen.answers import Distractor, exact_value
from mathgen.core import check, generator, join_terms, latex_answer, phrase, pick_distractors, rational_text

_X = Symbol("x")

_STEMS = (
    "Consider the function f(x) = {expr}.",
    "The function f is defined by f(x) = {expr}.",
    "Let f(x) = {expr}. The graph of y = f(x) has a horizontal asymptote at each end.",
)


def _root(a: int, b: int, c: int) -> str:
    return "√(" + join_terms([(a * a, "x^2"), (b, "x"), (c, "")]) + ")"


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


@generator("exd_x_r2c_limit_at_infinity", level=16, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_limit_at_infinity(rng, tier: str = "exam"):
    """A difference of square roots: the conjugate numerator, the two limits at infinity, the gap between them."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    a = rng.randint(1, 4)
    b, e = rng.sample(range(-9, 10), 2)
    c, g = rng.sample(range(-6, 7), 2)
    top, bottom = Rational(b - e, 2 * a), Rational(e - b, 2 * a)

    def f(x):
        return mp_sqrt(a * a * x * x + b * x + c) - mp_sqrt(a * a * x * x + e * x + g)

    with mp.workdps(50):  # local, so make() leaves mpmath's global precision as it found it
        check(abs(f(mpf(10) ** 8) - mpf(top.p) / top.q) < mpf("1e-6"), "f(10^8) is not near the limit at +inf")
        check(abs(f(-(mpf(10) ** 8)) - mpf(bottom.p) / bottom.q) < mpf("1e-6"), "f(-10^8) is not near -inf's limit")

    n_slope, n_const = b - e, c - g
    numerator = n_slope * _X + n_const
    numerator_text = join_terms([(n_slope, "x"), (n_const, "")])
    gap = abs(top - bottom)
    expr = f"{_root(a, b, c)} - {_root(a, e, g)}"
    stem = phrase(rng, _STEMS, expr=expr)

    part_a = dict(
        prompt=f"Write f(x) in the form N(x) / ({_root(a, b, c)} + {_root(a, e, g)}), where N(x) is a polynomial. "
        "Find N(x), simplified.",
        answer_kind="expression",
        answer=str(numerator),
        answer_text=numerator_text,
        answer_latex=latex_answer(numerator_text),
        marks=2,
        working=[
            "The conjugate is the sum of the two roots; the product of the roots' difference and sum is the "
            "difference of the radicands.",
            f"N(x) = ({join_terms([(a * a, 'x^2'), (b, 'x'), (c, '')])}) - ({join_terms([(a * a, 'x^2'), (e, 'x'), (g, '')])})"
            f" = {numerator_text}.",
        ],
        distractors=pick_distractors(
            numerator_text,
            [
                Distractor(join_terms([(-n_slope, "x"), (-n_const, "")]), "sign-reversed-result", None),
                Distractor(join_terms([(b + e, "x"), (c + g, "")]), "conjugate-sum-of-radicands", None),
                Distractor(join_terms([(n_slope, "x")]), "dropped-constant-term", None),
            ],
        ),
    )
    part_b = _fraction_part(
        "Hence find the limit of f(x) as x tends to infinity.",
        top,
        2,
        [
            f"Divide the numerator and both roots by x: the denominator tends to {2 * a}, the numerator to {n_slope}.",
            f"The limit is {rational_text(top)}.",
        ],
        [
            (Rational(0), "infinity-minus-infinity-zero"),
            (Rational(b - e, a), "forgot-factor-two-from-conjugate"),
            (Rational(2 * a, b - e), "inverted-ratio"),
        ],
        follow=f"(Subs(part_a, (x,), (1,)) - Subs(part_a, (x,), (0,)))/{2 * a}",
    )
    part_c = _fraction_part(
        "Find the limit of f(x) as x tends to negative infinity.",
        bottom,
        2,
        [
            "For negative x, √(x²) = -x, so each root is -ax(1 + …): the denominator tends to "
            f"{-2 * a} when the numerator is divided by x.",
            f"The limit is {rational_text(bottom)}.",
        ],
        [
            (top, "root-sign-at-negative-infinity"),
            (Rational(0), "infinity-minus-infinity-zero"),
            (Rational(e - b, a), "forgot-factor-two-from-conjugate"),
        ],
        follow="-part_b",
    )
    part_d = _fraction_part(
        "Hence find the vertical distance between the two horizontal asymptotes of the graph of y = f(x).",
        gap,
        1,
        [
            f"The asymptotes are y = {rational_text(top)} and y = {rational_text(bottom)}, a distance {rational_text(gap)} apart."
        ],
        [
            (abs(top), "gap-given-as-one-limit"),
            (Rational(0), "root-sign-at-negative-infinity"),
            (-gap, "sign-reversed-result"),
        ],
        follow="abs(part_b - part_c)",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
