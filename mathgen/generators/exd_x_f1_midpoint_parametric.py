"""Exam-tier source for ``senior-determine-the-coordinates-of-the-midpoint-of-two``.

The practice tiers find one coordinate of a midpoint, or an endpoint from a midpoint. Here the two
endpoints have coordinates that depend on an unknown ``t`` (``A(3t + 1, 2 - t)``), and the learner is
told the midpoint lies on a stated line. They must form the midpoint coordinates as expressions in
``t``, substitute them into the line and solve the resulting linear equation. Two kinds of line are
used: slope-intercept ``y = m x + c`` and general ``a x + b y = c``.

Built backwards from the answer ``t0``: the midpoint's coordinates and endpoint A are drawn as linear
expressions in ``t``, B is ``2M - A``, and the line is chosen to pass through ``M(t0)``. The
distractors come from solving the same equation under a named misreading of "midpoint".
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Rational, Symbol, solve

from ..core import check, generator, join_terms, phrase, pick_distractors

t = Symbol("t")

_WORDINGS = (
    "The points A({ax}, {ay}) and B({bx}, {by}) are such that their midpoint lies on the line {line}. "
    "Find the value of t.",
    "Let M be the midpoint of A({ax}, {ay}) and B({bx}, {by}). M lies on the line {line}. Find t.",
    "Find t, given that the midpoint of the segment joining A({ax}, {ay}) and B({bx}, {by}) lies on the line {line}.",
)


def _lin(const: int, coef: int) -> str:
    return join_terms([(coef, "t"), (const, "")])


def _solve(line, x0, ux, y0, uy):
    """t for which (x0 + ux t, y0 + uy t) is on line (lx, ly, lc): lx X + ly Y = lc. None if degenerate."""
    lx, ly, lc = line
    denom = lx * ux + ly * uy
    if denom == 0:
        return None
    return Fraction(lc - lx * x0 - ly * y0, 1) / denom


def _draw(rng: random.Random) -> dict | None:
    t0 = rng.choice([n for n in range(-6, 7) if n != 0])
    # (constant, t-coefficient) for the midpoint and for A; B = 2M - A.
    mx = (rng.randint(-5, 5), rng.randint(-2, 2))
    my = (rng.randint(-5, 5), rng.randint(-2, 2))
    ax = (rng.randint(-6, 6), rng.randint(-3, 3))
    ay = (rng.randint(-6, 6), rng.randint(-3, 3))
    bx = (2 * mx[0] - ax[0], 2 * mx[1] - ax[1])
    by = (2 * my[0] - ay[0], 2 * my[1] - ay[1])
    if sum(1 for c in (ax, ay, bx, by) if c[1] != 0) < 3:
        return None
    if rng.random() < 0.5:
        m = rng.choice([-3, -2, -1, 1, 2, 3])
        line = (m, -1, 0)
    else:
        line = (rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-3, -2, 2, 3]), 0)
    lc = line[0] * (mx[0] + mx[1] * t0) + line[1] * (my[0] + my[1] * t0)
    line = (line[0], line[1], lc)
    if line[0] * mx[1] + line[1] * my[1] == 0:
        return None
    if line[1] == -1:  # y = m x + c  <=>  m x - y = -c
        line_text = "y = " + join_terms([(line[0], "x"), (-lc, "")])
    else:
        line_text = f"{join_terms([(line[0], 'x'), (line[1], 'y')])} = {lc}"
    return dict(t0=t0, mx=mx, my=my, ax=ax, ay=ay, bx=bx, by=by, line=line, line_text=line_text)


def _build(rng: random.Random) -> dict:
    for _ in range(300):
        d = _draw(rng)
        if d is None:
            continue
        t0, line = d["t0"], d["line"]
        mx, my, ax, ay, bx, by = (d[k] for k in ("mx", "my", "ax", "ay", "bx", "by"))
        # Independent route: build the midpoint from the endpoints as sympy expressions and solve.
        ex = Rational(1, 2) * ((ax[0] + ax[1] * t) + (bx[0] + bx[1] * t))
        ey = Rational(1, 2) * ((ay[0] + ay[1] * t) + (by[0] + by[1] * t))
        sol = solve(line[0] * ex + line[1] * ey - line[2], t)
        check(sol == [t0], "midpoint equation does not give the drawn t")
        pool = [
            # forgot to halve (used the sum of the endpoints)
            _solve(line, ax[0] + bx[0], ax[1] + bx[1], ay[0] + by[0], ay[1] + by[1]),
            # used only A, ignoring B
            _solve(line, ax[0], ax[1], ay[0], ay[1]),
            # used only B, ignoring A
            _solve(line, bx[0], bx[1], by[0], by[1]),
            # subtracted instead of averaged: (A - B) / 2
            _solve(
                line,
                Fraction(ax[0] - bx[0], 2),
                Fraction(ax[1] - bx[1], 2),
                Fraction(ay[0] - by[0], 2),
                Fraction(ay[1] - by[1], 2),
            ),
            # slip in the sign when isolating t
            Fraction(-t0),
        ]
        wrong: list[int] = []
        for v in pool:
            # keep whole-number misreadings only, so a fraction never gives the key away
            if v is not None and v.denominator == 1 and int(v) != t0 and int(v) not in wrong:
                wrong.append(int(v))
        if len(wrong) < 3:
            continue
        wrong = wrong[:3]
        answer_text = str(t0)
        return dict(
            text=phrase(
                rng,
                _WORDINGS,
                ax=_lin(*ax),
                ay=_lin(*ay),
                bx=_lin(*bx),
                by=_lin(*by),
                line=d["line_text"],
            ),
            latex=rf"A({_lin(*ax)}, {_lin(*ay)}),\ B({_lin(*bx)}, {_lin(*by)}),\ M \in {d['line_text']}",
            answer=t0,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"M = ({_lin(*mx)}, {_lin(*my)}). Substituting into {d['line_text']} gives a linear equation in t "
                f"with solution t = {t0}."
            ],
            distractors=pick_distractors(answer_text, [str(w) for w in wrong]),
        )
    check(False, "could not build a midpoint question")


@generator(
    "exd_x_f1_midpoint_parametric",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f1_midpoint_parametric(rng: random.Random, tier: str = "exam") -> dict:
    """Midpoint of two parametric points constrained to a line; solve for the parameter."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _build(rng)
