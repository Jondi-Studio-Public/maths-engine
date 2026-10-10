"""Exam-tier source for ``integrate-polynomial``: recover a function from its second derivative.

The learner is given ``f''(x)`` (powers of ``x``, sometimes with one reciprocal power such as
``4/x^3``) together with the values of ``f`` and ``f'`` at one point, and must integrate twice,
fixing a separate constant of integration at each stage from the right condition.  Two
antiderivatives and two constants, with the conditions applying to different stages, is a change
in kind from integrating a single expression once.

Built backwards: ``f''`` is drawn first, then the two integration constants ``k1`` and ``k2``
(integers), and the stated values ``f'(x0)`` and ``f(x0)`` are read off the resulting function, so the
unique answer is known exactly.  Only reciprocal powers ``x^-3`` and ``x^-4`` are used, so that
neither antiderivative ever needs a logarithm.

Distractors:

- stopping after a single integration (the answer is offered as ``f'``);
- dropping the first constant of integration (``f'(x)`` taken to have no constant);
- applying the two conditions to the wrong stages (``f(x0)`` used for ``f'`` and vice versa).
"""

from __future__ import annotations

import random

from sympy import Add, Rational, Symbol, diff, expand

from ..core import check, check_distinct, generator, join_terms, phrase, pick_distractors, rational_text

_X = Symbol("x")

_PATTERNS = (
    "Given that f''(x) = {second}, f'({x0}) = {d1} and f({x0}) = {d0}, find f(x).",
    "A function f satisfies f''(x) = {second}. Also f({x0}) = {d0} and f'({x0}) = {d1}. Find f(x).",
    "Find f(x) if f''(x) = {second}, f({x0}) = {d0} and f'({x0}) = {d1}.",
)


def _second_derivative_text(positive: list[tuple[int, int]], recip: tuple[int, int] | None) -> str:
    """``6x^2 - x + 4 + 5/x^3`` style text: powers via ``join_terms``, the reciprocal term appended."""
    text = join_terms([(c, "x" + (f"^{p}" if p > 1 else "") if p else "") for c, p in positive])
    if recip is not None:
        coef, power = recip
        sign = "+" if coef > 0 else "-"
        mag = abs(coef)
        tail = f"{mag}/x^{power}" if mag != 1 else f"1/x^{power}"
        text = f"{text} {sign} {tail}"
    return text


def _display(expr) -> str:
    """Learner-facing text: ``^`` for powers, never Python's ``**``."""
    return str(expr).replace("**", "^")


def _integrate(expr):
    """Reverse power rule term by term (never an exponent of -1, so no logarithm); much cheaper than sympy.integrate."""
    total = Rational(0)
    for term in Add.make_args(expand(expr)):
        coef, power = term.as_coeff_exponent(_X)
        total += coef * _X ** (power + 1) / (power + 1)
    return total


def _solve_for(second, x0, d1, d0):
    """Integrate ``second`` twice, fixing each constant from its own condition."""
    first = _integrate(second)
    first += d1 - first.subs(_X, x0)
    func = _integrate(first)
    return func + (d0 - func.subs(_X, x0))


def _exam(rng: random.Random) -> dict:
    powers = rng.sample([0, 1, 2, 3], k=rng.choice([2, 3]))
    positive = [(rng.choice([1, 2, 3, 4, 5, 6, -1, -2, -3, -4]), p) for p in sorted(powers, reverse=True)]
    recip = None
    if rng.random() < 0.5:
        recip = (rng.choice([1, 2, 3, 4, 5, -1, -2, -3]), rng.choice([3, 4]))
    second = sum(c * _X**p for c, p in positive)
    if recip is not None:
        second += recip[0] * _X ** (-recip[1])

    x0 = rng.choice([1, 2])
    while True:
        k1, k2 = rng.choice([v for v in range(-5, 6) if v]), rng.randint(-5, 5)  # k1 != 0 so dropping it matters
        # the swapped-conditions distractor must differ from the key, so f(x0) and f'(x0) must differ
        if _integrate(_integrate(second)).subs(_X, x0) + k1 * x0 + k2 != _integrate(second).subs(_X, x0) + k1:
            break
    func = _integrate(_integrate(second)) + k1 * _X + k2
    d1 = diff(func, _X).subs(_X, x0)
    d0 = func.subs(_X, x0)

    check(expand(diff(func, _X, 2) - second) == 0, "f'' does not match the stated second derivative")
    answer = _solve_for(second, x0, d1, d0)
    check(expand(answer - func) == 0, "two-constant solve disagrees with the constructed function")
    check(answer.subs(_X, x0) == d0 and diff(answer, _X).subs(_X, x0) == d1, "conditions not met")

    one_step = _integrate(second) + (d1 - _integrate(second).subs(_X, x0))
    no_first_constant = _integrate(_integrate(second))
    no_first_constant += d0 - no_first_constant.subs(_X, x0)
    swapped = _solve_for(second, x0, d0, d1)
    candidates = [one_step, no_first_constant, swapped]
    check_distinct(answer, *candidates)

    text = phrase(
        rng,
        _PATTERNS,
        second=_second_derivative_text(positive, recip),
        x0=x0,
        d1=rational_text(d1),
        d0=rational_text(d0),
    )
    answer_text = _display(answer)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=pick_distractors(answer_text, [_display(c) for c in candidates]),
    )


@generator("exd_x_c2_integrate_polynomial", level=15, presets={"exam": dict(tier="exam")})
def exd_x_c2_integrate_polynomial(rng: random.Random, tier: str = "exam") -> dict:
    """Recover f from f'', f(x0) and f'(x0)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
