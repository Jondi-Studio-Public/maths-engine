"""Exam source for `senior-solve-linear-equations`: fractional equations whose quadratic terms cancel.

The practice tiers have constant denominators. Here the unknown is in the denominators on both sides, so clearing them
produces ``x^2`` terms on both sides which cancel and leave a linear equation. The learner has to choose the method,
see the cancellation coming and keep the excluded values in mind. The answer is an exact fraction. Three forms:

  * ``pair``   -- ``(x + a)/(x + b) = (x + c)/(x + d)``
  * ``minus``  -- ``(x + a)/(x + b) - 1 = c/(x + d)``
  * ``sum``    -- ``1/(x + a) + 1/(x + b) = 2/(x + c)``

Parameters are drawn and then screened with SymPy: the cleared numerator must have exact degree 1, its root must be
a non-integer rational (this is a fraction skill) and must not be an excluded value, so the key is unique and exact.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Poly, Rational, Symbol, cancel, denom, numer, solve, together

from ..core import check, generator, phrase, pick_distractors, rational_text

_X = Symbol("x")


def _br(b: int) -> str:
    return f"(x + {b})" if b > 0 else f"(x - {-b})"


def _nz(rng: random.Random, lo: int = -9, hi: int = 9) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _solve_unique(lhs, rhs, poles: list[int]):
    """The unique rational root of the cleared linear equation, or None if the draw is unsuitable."""
    expr = together(lhs - rhs)
    num = Poly(numer(expr), _X)
    if num.degree() != 1:
        return None
    root = solve(num.as_expr(), _X)[0]
    if root.q == 1 or any(root == -b for b in poles):
        return None
    check(denom(expr).subs(_X, root) != 0)
    return Rational(root)


def _result(text: str, root: Rational, cands: list, working: list[str]) -> dict:
    ans = rational_text(root)
    d = pick_distractors(ans, [rational_text(c) for c in cands if c != root and c.q != 0], 3)
    if len(d) < 3:
        return None
    return dict(
        text=text,
        latex="plain:" + text,
        answer={"p": int(root.p), "q": int(root.q)},
        answer_text=ans,
        distractors=d,
        working=working,
    )


def _pair(rng: random.Random) -> dict | None:
    a, b, c, d = (_nz(rng) for _ in range(4))
    if b == d or a == c or a + d == b + c:
        return None
    lhs, rhs = (_X + a) / (_X + b), (_X + c) / (_X + d)
    root = _solve_unique(lhs, rhs, [b, d])
    if root is None:
        return None
    check(root == Rational(c * b - a * d, a + d - b - c))
    R = Rational
    cands = [
        R(c * d - a * b, a + b - c - d) if a + b != c + d else R(0),  # cross-multiplied numerator with numerator
        -root,  # sign of the result lost
        R(c * b, a + d - b - c),  # the constant product a*d is dropped
        R(c * b - a * d, a + d + b + c) if a + b + c + d else R(0),  # x terms gathered with the wrong sign
    ]
    text = phrase(rng, ["Solve for x: {l} = {r}", "Solve: {l} = {r}"], l=f"{_br(a)}/{_br(b)}", r=f"{_br(c)}/{_br(d)}")
    return _result(
        text,
        root,
        cands,
        [
            f"Cross-multiply: {_br(a)}{_br(d)} = {_br(c)}{_br(b)}; the x^2 terms cancel",
            f"x = {rational_text(root)}",
        ],
    )


def _minus(rng: random.Random) -> dict | None:
    a, b, c, d = (_nz(rng) for _ in range(4))
    if a == b or b == d or a - b == c:
        return None
    lhs, rhs = (_X + a) / (_X + b) - 1, Rational(c) / (_X + d)
    root = _solve_unique(lhs, rhs, [b, d])
    if root is None:
        return None
    R = Rational
    k = a - b
    cands = [
        R(c * b - (a + b) * d, a + b - c) if a + b != c else R(0),  # 1 subtracted as (x + b) with the wrong sign
        R(c * d - k * b, k - c),  # cross-multiplied the wrong way round
        -root,
        R(c * b - k * d, k + c) if k + c else R(0),  # sign slip gathering the x terms
    ]
    text = phrase(
        rng,
        ["Solve for x: {l} - 1 = {r}", "Solve: {l} - 1 = {r}"],
        l=f"{_br(a)}/{_br(b)}",
        r=f"{c}/{_br(d)}",
    )
    return _result(text, root, cands, [f"{_br(a)}/{_br(b)} - 1 = {k}/{_br(b)}", f"x = {rational_text(root)}"])


def _sum(rng: random.Random) -> dict | None:
    a, b, c = (_nz(rng, -8, 8) for _ in range(3))
    if len({a, b, c}) < 3:
        return None
    lhs, rhs = 1 / (_X + a) + 1 / (_X + b), 2 / (_X + c)
    root = _solve_unique(lhs, rhs, [a, b, c])
    if root is None:
        return None
    R = Rational
    cands = [
        R(c - a - b),  # adds the denominators: 2/(2x + a + b)
        R(2 - a - b, 2),  # drops the denominators on the right
        -root,
        R(2 * a * b + (a + b) * c, 2 * c - a - b) if 2 * c != a + b else R(0),  # sign slip on the constant terms
    ]
    text = phrase(
        rng,
        ["Solve for x: 1/{a} + 1/{b} = 2/{c}", "Solve: 1/{a} + 1/{b} = 2/{c}"],
        a=_br(a),
        b=_br(b),
        c=_br(c),
    )
    return _result(text, root, cands, ["Clear denominators; the x^2 terms cancel", f"x = {rational_text(root)}"])


_BUILDERS = {"pair": _pair, "minus": _minus, "sum": _sum}


@generator(
    "exd_x_a3_linear_fractions_cancelling_square",
    level=11,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_a3_linear_fractions_cancelling_square(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        out = _BUILDERS[rng.choice(sorted(_BUILDERS))](rng)
        if out is not None:
            return out
