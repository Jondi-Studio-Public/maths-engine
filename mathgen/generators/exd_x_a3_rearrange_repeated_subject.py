"""Exam source for `rearrange-simple-formulas`: the new subject appears in more than one place.

The practice tiers isolate a variable that occurs once. Here the target occurs in a denominator, or on both sides
after clearing a fraction, so the learner must clear denominators, gather the target's terms and factorise before
dividing. Three forms (letters are drawn per seed):

  * ``mobius``     -- ``y = (a*x + b)/(c*x + d)``, make ``x`` the subject
  * ``reciprocal`` -- ``1/f = 1/u + 1/v``, make ``v`` (or ``u``) the subject
  * ``offset``     -- ``y = a + b/(x - c)``, make ``x`` the subject

The key is written in closed form and verified by substituting it back into the original formula at several exact
rational points (no symbolic solve, so a draw costs a few milliseconds). Distractors are named slips (a sign lost
while gathering, a factor not taken out, a reciprocal not inverted).
"""

from __future__ import annotations

import random

from sympy import Eq, Rational, Symbol, latex

from ..core import check, generator, pick_distractors

_POOL = "abcdfghkmnpqrstuvwz"


def _letters(rng: random.Random, n: int) -> list[str]:
    return rng.sample(_POOL, n)


def _tex(eq: Eq) -> str:
    return latex(eq)


def _mobius(rng: random.Random) -> dict:
    x, y, a, b, c, d = _letters(rng, 6)
    X, Y, A, B, C, D = (Symbol(s) for s in (x, y, a, b, c, d))
    sign = rng.choice([1, -1])  # denominator c*x + d or c*x - d
    num = A * X + B
    den = C * X + sign * D
    eq = Eq(Y, num / den)
    key = (B - sign * D * Y) / (C * Y - A)
    text = f"Make {x} the subject of the formula {y} = ({a}{x} + {b})/({c}{x} {'+' if sign > 0 else '-'} {d})"
    # slips
    cands = [
        (B - sign * D * Y) / (A - C * Y),  # sign flipped in the denominator after gathering
        (B + sign * D * Y) / (C * Y - A),  # sign lost on the term moved across
        (B - sign * D * Y) / (C - A),  # y dropped from the factorised bracket
        (B - sign * D * Y) / (C * Y),  # the a x term never moved across
    ]
    return _build(text, eq, X, key, cands)


def _reciprocal(rng: random.Random) -> dict:
    f, u, v = _letters(rng, 3)
    F, U, V = Symbol(f), Symbol(u), Symbol(v)
    target = rng.choice([(v, V, U), (u, U, V)])
    tname, T, O = target
    eq = Eq(1 / F, 1 / U + 1 / V)
    key = F * O / (O - F)
    text = f"Make {tname} the subject of the formula 1/{f} = 1/{u} + 1/{v}"
    cands = [
        F - O,  # takes reciprocals term by term
        F * O / (F - O),  # the difference taken the wrong way round
        F * O / (O + F),  # adds where it should subtract
        (O - F) / (F * O),  # inverted
    ]
    return _build(text, eq, T, key, cands)


def _offset(rng: random.Random) -> dict:
    x, y, a, b, c = _letters(rng, 5)
    X, Y, A, B, C = (Symbol(s) for s in (x, y, a, b, c))
    sign = rng.choice([1, -1])
    eq = Eq(Y, A + B / (X - sign * C))
    key = sign * C + B / (Y - A)
    text = f"Make {x} the subject of the formula {y} = {a} + {b}/({x} {'-' if sign > 0 else '+'} {c})"
    cands = [
        B / (Y - A),  # the constant c is never added back
        sign * C + B / (Y + A),  # a added instead of subtracted
        -sign * C + B / (Y - A),  # sign slip on c
        sign * C + (B - A) / Y,  # divides b - a by y
    ]
    return _build(text, eq, X, key, cands)


_POINTS = (
    (Rational(2), Rational(3), Rational(5), Rational(7), Rational(11), Rational(13)),
    (Rational(3, 2), Rational(-5, 3), Rational(7, 4), Rational(2, 5), Rational(-9, 7), Rational(4, 3)),
    (Rational(-4), Rational(9, 2), Rational(1, 3), Rational(-6, 5), Rational(8), Rational(5, 7)),
)


def _values(expr, symbols: list[Symbol]) -> tuple:
    """`expr` evaluated exactly at three fixed rational points (None where a denominator vanishes)."""
    out = []
    for point in _POINTS:
        val = expr.xreplace(dict(zip(symbols, point)))
        out.append(None if val.has(Symbol) or not val.is_finite else val)
    return tuple(out)


def _build(text: str, eq: Eq, target: Symbol, key, cands: list) -> dict:
    symbols = sorted(eq.free_symbols, key=str)
    # The key must satisfy the original formula exactly at every test point.
    others = [v for v in symbols if v != target]
    verified = 0
    for point in _POINTS:
        env = dict(zip(others, point))
        k = key.xreplace(env)
        if not k.is_finite:  # a test point that hits a pole says nothing
            continue
        residual = (eq.lhs - eq.rhs).xreplace(env).xreplace({target: k})
        check(residual == 0, "key does not satisfy the formula")
        verified += 1
    check(verified >= 2, "too few usable test points")
    key_values = _values(key, symbols)
    key_text = str(key)
    # Every distractor must be a different formula, not just different text.
    kept, seen = [], {key_values}
    for c in cands:
        vals = _values(c, symbols)
        if vals in seen:
            continue
        seen.add(vals)
        kept.append(str(c))
    d = pick_distractors(key_text, kept, 3)
    check(len(d) == 3, "fewer than three distinct slips")
    return dict(
        text=text,
        latex=r"\text{Make } " + latex(target) + r" \text{ the subject of } " + latex(eq),
        answer=key_text,
        answer_text=key_text,
        answer_latex=latex(key),
        working=[f"Start from {text.split('formula ')[1]}", f"{target} = {key_text}"],
        distractors=d,
    )


_BUILDERS = {"mobius": _mobius, "reciprocal": _reciprocal, "offset": _offset}


@generator(
    "exd_x_a3_rearrange_repeated_subject",
    level=7,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_a3_rearrange_repeated_subject(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        try:
            return _BUILDERS[rng.choice(sorted(_BUILDERS))](rng)
        except Exception as exc:  # a degenerate draw (fewer than 3 slips); redraw
            from ..core import VerificationError

            if not isinstance(exc, VerificationError) or "slips" not in str(exc):
                raise
