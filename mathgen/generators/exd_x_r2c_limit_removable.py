"""Exam chain for removable-discontinuity limits: a hidden constant fixed by the existence of the limit.

``f(x) = (m x^2 + p x + k) / (x - a)`` for ``x != a``, where ``k`` is an unknown constant and the limit of ``f`` as
x tends to ``a`` exists. Set the way a senior exam sets it:

(a) the value of k                                                 number,     2 marks
(b) f(x) in simplified form for x != a                              expression, 2 marks
(c) the limit of f(x) as x tends to a                               number,     1 mark,  follow-through (b)
(d) the x-intercept of the graph of y = f(x)                        fraction,   2 marks, follow-through (b)

## Construction: choose m, p and a first

The numerator must vanish at ``x = a`` for the limit to be finite, so ``k = -(m a^2 + p a)`` and then
``m x^2 + p x + k = (x - a)(m x + m a + p)``. The simplified function is ``m x + (m a + p)``, the limit is
``2 m a + p`` and the x-intercept is ``-(m a + p) / m``. The limit is kept non-zero so that the intercept is not
the removed point, and ``a`` is non-zero so ``k`` is not trivially zero. As a second route the quotient is
evaluated at ``a +- 10^-12`` with 60-digit arithmetic and must land within 10^-8 of the key.

## Distractors

* (a): the sign of k wrong, the leading coefficient forgotten, the linear term dropped.
* (b): the sign slip when factoring, the shift dropped, the leading coefficient forgotten.
* (c): "undefined" (stopped at 0/0), zero, the constant term of the simplified function.
* (d): the sign slip on the intercept, the shift forgotten.
"""

from __future__ import annotations

from fractions import Fraction

from mpmath import mp, mpf
from sympy import Rational, Symbol

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative as pn,
    phrase,
    pick_distractors,
    rational_text,
)

_X = Symbol("x")

_STEMS = (
    "Let f(x) = ({num})/({den}) for x ≠ {a}, where k is a constant. As x tends to {a}, the limit of f(x) exists.",
    "The function f is defined for x ≠ {a} by f(x) = ({num})/({den}), where k is a constant, "
    "and f(x) has a limit as x tends to {a}.",
    "Consider f(x) = ({num})/({den}) with x ≠ {a} and k a constant. It is known that f(x) has a limit as x tends "
    "to {a}.",
)


def _at(coefficient: int, power: str) -> str:
    """``coefficient`` times ``power``, leaving out a coefficient of 1: ``5²``, ``-5²``, ``3·5²``."""
    return power if coefficient == 1 else f"-{power}" if coefficient == -1 else f"{coefficient}·{power}"


def _number(prompt, value, marks, working, wrong, follow=None):
    """A whole-number part; `wrong` lists (value or text, id)."""
    text = str(value)
    cands = [Distractor(str(v), tag, int(v) if not isinstance(v, str) else None) for v, tag in wrong if v is not None]
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, cands),
    )
    if follow:
        part["follow_through"] = follow
    return part


@generator("exd_x_r2c_limit_removable", level=16, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_limit_removable(rng, tier: str = "exam"):
    """A quotient with a hidden constant: find it, simplify, take the limit, find the intercept."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(1000):  # bounded rejection loop
        m = rng.choice([-3, -2, -1, 1, 2, 3])
        p = rng.choice([v for v in range(-9, 10) if v != 0])
        a = rng.choice([v for v in range(-5, 6) if v != 0])
        if 2 * m * a + p != 0 and m * a + p != 0 and m * a * a + p * a != 0:  # no zero limit, shift or k
            break
    else:  # pragma: no cover
        check(False, "no usable draw")
    k = -(m * a * a + p * a)
    shift = m * a + p
    limit = 2 * m * a + p
    intercept = Rational(-shift, m)
    check(m * a * a + p * a + k == 0, "the numerator does not vanish at a")
    check((m * _X**2 + p * _X + k).subs(_X, a) == 0, "the numerator does not vanish at a")

    def quotient(x):
        return (m * x * x + p * x + k) / (x - a)

    with mp.workdps(60):  # local, so make() leaves mpmath's global precision as it found it
        for side in (mpf(10) ** -12, -(mpf(10) ** -12)):
            check(abs(quotient(mpf(a) + side) - limit) < mpf("1e-8"), "the quotient disagrees with the limit")
    check(intercept != a and Rational(m) * intercept + shift == 0, "the intercept is the removed point")

    num_text = join_terms([(m, "x^2"), (p, "x"), (1, "k")])
    den_text = f"x - {a}" if a > 0 else f"x + {-a}"
    stem = phrase(rng, _STEMS, num=num_text, den=den_text, a=a)

    simplified = join_terms([(m, "x"), (shift, "")])
    part_a = _number(
        "Find the value of k.",
        k,
        2,
        [
            f"The numerator must be zero at x = {a}: "
            f"{_at(m, f'{pn(a)}²')} {'+' if p > 0 else '-'} {abs(p)}·{pn(a)} + k = 0, so k = {k}."
        ],
        [
            (-k, "sign-reversed-result"),
            (-(a * a + p * a), "forgot-leading-coefficient"),
            (-m * a * a, "left-out-a-term-when-solving-for-constant"),
        ],
    )
    wrong_forms = [
        (join_terms([(m, "x"), (p - m * a, "")]), "factorising-sign-slip"),
        (join_terms([(m, "x"), (p, "")]), "dropped-constant-term"),
        (join_terms([(1, "x"), (shift, "")]), "forgot-leading-coefficient"),
    ]
    part_b = dict(
        prompt=f"Write f(x) in simplified form, with no common factor, for x ≠ {a}.",
        answer_kind="expression",
        answer=str(m * _X + shift),
        answer_text=simplified,
        answer_latex=latex_answer(simplified),
        marks=2,
        working=[
            f"With k = {k}: {join_terms([(m, 'x^2'), (p, 'x'), (k, '')])} = ({den_text})({simplified}).",
            f"So f(x) = {simplified} for x ≠ {a}.",
        ],
        distractors=pick_distractors(simplified, [Distractor(t, tag, t) for t, tag in wrong_forms]),
    )
    part_c = _number(
        f"Find the limit of f(x) as x tends to {a}.",
        limit,
        1,
        [f"The limit is the value of {simplified} at x = {a}: {limit}."],
        [
            ("undefined", "stopped-at-undefined"),
            (0, "answered-zero"),
            (shift, "dropped-the-x-term-at-the-limit"),
        ],
        follow=f"Subs(part_b, (x,), ({a},))",
    )
    inter_text = rational_text(intercept)
    part_d = dict(
        prompt="Find the x-coordinate of the point where the graph of y = f(x) crosses the x-axis. "
        "Give an exact answer: a fraction in simplest form, or an integer.",
        answer_kind="fraction",
        answer=exact_value(Fraction(int(intercept.p), int(intercept.q))),
        answer_text=inter_text,
        answer_latex=latex_answer(inter_text),
        marks=2,
        working=[f"{simplified} = 0 gives x = {inter_text}, which is not x = {a}."],
        distractors=pick_distractors(
            inter_text,
            [
                Distractor(
                    rational_text(-intercept),
                    "sign-error-moving-term",
                    exact_value(Fraction(int(-intercept.p), int(intercept.q))),
                ),
                Distractor(rational_text(Rational(-shift)), "forgot-to-divide-by-coefficient", -shift),
                Distractor(
                    rational_text(Rational(m * a - p, m)),
                    "factorising-sign-slip",
                    exact_value(Fraction(m * a - p, m)),
                ),
                Distractor(
                    rational_text(Rational(-p, m)),
                    "dropped-constant-term",
                    exact_value(Fraction(-p, m)),
                ),
                Distractor(
                    rational_text(Rational(shift, m)), "sign-error-moving-term", exact_value(Fraction(shift, m))
                ),
            ],
        ),
        follow_through="-Subs(part_b, (x,), (0,))/(Subs(part_b, (x,), (1,)) - Subs(part_b, (x,), (0,)))",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
