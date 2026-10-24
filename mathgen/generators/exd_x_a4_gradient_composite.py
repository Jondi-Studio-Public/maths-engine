"""Exam tier for "rate of change and gradient".

The practice tiers read a gradient straight off two points, a table or a story. Here the gradient is not
sitting in the data; the learner has to produce it by a method of their own choosing. Three question types
share this source:

- ``standard``: the line is given as ``ax + by = c``, so the learner must rearrange to ``y = mx + c'`` (or
  otherwise work out the rate of change of y with respect to x);
- ``perpendicular``: the gradient of a line perpendicular to the line through two given points (find the
  gradient, then take the negative reciprocal);
- ``midpoint``: the gradient of the line from one point to the midpoint of a segment (find the midpoint
  first, then the gradient).

Built backwards from the key. The key is re-derived from the printed prompt alone with sympy / exact
fractions, independently of how each prompt was built.

Distractors are named misconceptions: the sign dropped when moving ``ax`` across the equals sign, the
reciprocal taken the wrong way, the intercept read as the gradient, the perpendicular step skipped, the
endpoint used in place of the midpoint, and the coordinates summed rather than averaged.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

from sympy import Eq, Rational, Symbol, latex, solve
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from ..core import check, fmt_signed, generator, join_terms, paren_negative, phrase, pick_distractors, rational_text

_X, _Y = Symbol("x"), Symbol("y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

_STANDARD = (
    "A line has the equation {lhs} = {rhs}. Find its gradient.",
    "Find the rate of change of y with respect to x for the relationship {lhs} = {rhs}.",
    "What is the gradient of the line {lhs} = {rhs}?",
)
_PERPENDICULAR = (
    "Line m passes through A({ax}, {ay}) and B({bx}, {by}). Find the gradient of a line perpendicular to m.",
    "Find the gradient of any line that is perpendicular to the line through ({ax}, {ay}) and ({bx}, {by}).",
)
_MIDPOINT = (
    "M is the midpoint of the segment joining B({bx}, {by}) and C({cx}, {cy}). Find the gradient of the line "
    "through A({ax}, {ay}) and M.",
    "Points are A({ax}, {ay}), B({bx}, {by}) and C({cx}, {cy}). Find the gradient of the line from A to the "
    "midpoint of BC.",
)


def _br(value: Fraction | int) -> str:
    """A number as written after an operator: negatives and fractions in brackets."""
    t = rational_text(value)
    return f"({t})" if t.startswith("-") or "/" in t else t


def _frac(num: int, den: int) -> Fraction:
    return Fraction(num, den)


def _pair(f: Fraction) -> dict:
    return {"p": f.numerator, "q": f.denominator}


def _texts(values: list[Fraction | None]) -> list[str]:
    return [rational_text(v) for v in values if v is not None]


def _standard(rng: random.Random) -> tuple[str, Fraction, list[str], list[str]]:
    while True:
        a = rng.choice([-1, 1]) * rng.randint(1, 9)
        b = rng.choice([-1, 1]) * rng.randint(2, 9)
        c = rng.randint(-20, 20)
        if c and abs(a) != abs(b):
            break
    lhs = join_terms([(a, "x"), (b, "y")])
    text = phrase(rng, _STANDARD, lhs=lhs, rhs=c)
    # Independent route: parse the printed equation and solve it for y.
    m = re.search(r"([-+0-9xy ]*x[-+0-9xy ]*) = (-?\d+)", text)
    check(m is not None, "printed equation not found")
    eq = Eq(parse_expr(m.group(1), transformations=_TRANSFORMS), int(m.group(2)))
    sol = solve(eq, _Y)[0]
    slope = sol.diff(_X)
    key = Fraction(int(slope.p), int(slope.q))
    check(key == _frac(-a, b), "re-derived gradient disagrees with the construction")
    wrong = [_frac(a, b), _frac(-b, a), _frac(c, b), _frac(-a, 1)]
    answer_text = rational_text(key)
    working = [
        f"Make y the subject: {fmt_signed(b, 'y')} = {join_terms([(-a, 'x'), (c, '')])}.",
        f"Divide by {b}: the coefficient of x is {paren_negative(-a)} ÷ {paren_negative(b)} = {answer_text}, "
        f"so the gradient is {answer_text}.",
    ]
    return text, key, _texts(wrong), working


def _perpendicular(rng: random.Random) -> tuple[str, Fraction, list[str], list[str]]:
    while True:
        dx = rng.choice([-1, 1]) * rng.randint(1, 8)
        dy = rng.choice([-1, 1]) * rng.randint(1, 8)
        if abs(dx) != abs(dy):
            break
    ax, ay = rng.randint(-6, 6), rng.randint(-6, 6)
    bx, by = ax + dx, ay + dy
    text = phrase(rng, _PERPENDICULAR, ax=ax, ay=ay, bx=bx, by=by)
    nums = [int(v) for v in re.findall(r"-?\d+", text)]
    check(len(nums) == 4, "exactly the four printed coordinates")
    m_ref = _frac(nums[3] - nums[1], nums[2] - nums[0])
    key = -1 / m_ref
    check(key == _frac(-dx, dy), "re-derived gradient disagrees with the construction")
    wrong = [m_ref, -m_ref, 1 / m_ref]
    working = [
        f"Gradient of the line through the two points = ({nums[3]} - {paren_negative(nums[1])}) ÷ "
        f"({nums[2]} - {paren_negative(nums[0])}) = {rational_text(m_ref)}.",
        f"A perpendicular line has gradient -1 ÷ {_br(m_ref)} = {rational_text(key)}.",
    ]
    return text, key, _texts(wrong), working


def _midpoint(rng: random.Random) -> tuple[str, Fraction, list[str], list[str]]:
    while True:
        mx, my = rng.randint(-6, 6), rng.randint(-6, 6)
        px, py = rng.randint(1, 5), rng.randint(-5, 5)
        dx, dy = rng.choice([-1, 1]) * rng.randint(1, 8), rng.choice([-1, 1]) * rng.randint(1, 8)
        ax, ay = mx - dx, my - dy
        if abs(dx) != abs(dy):
            break
    bx, by, cx, cy = mx - px, my - py, mx + px, my + py
    text = phrase(rng, _MIDPOINT, ax=ax, ay=ay, bx=bx, by=by, cx=cx, cy=cy)
    printed = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"([ABC])\((-?\d+), (-?\d+)\)", text)}
    check(sorted(printed) == ["A", "B", "C"], "exactly the three printed points")
    (qax, qay), (qbx, qby), (qcx, qcy) = printed["A"], printed["B"], printed["C"]
    mid = _frac(qbx + qcx, 2), _frac(qby + qcy, 2)
    key = (mid[1] - qay) / (mid[0] - qax)
    check(key == _frac(dy, dx), "re-derived gradient disagrees with the construction")
    sx, sy = qbx + qcx, qby + qcy  # summed, never halved
    wrong = [
        _frac(qby - qay, qbx - qax) if qbx != qax else None,  # endpoint B used in place of the midpoint
        _frac(sy - qay, sx - qax) if sx != qax else None,  # midpoint coordinates summed, not averaged
        _frac(dx, dy),  # rise and run reversed
        -key,  # points taken in the opposite order for one coordinate
    ]
    working = [
        f"M = (({qbx} + {paren_negative(qcx)}) ÷ 2, ({qby} + {paren_negative(qcy)}) ÷ 2) = "
        f"({rational_text(mid[0])}, {rational_text(mid[1])}).",
        f"Gradient of AM = ({rational_text(mid[1])} - {paren_negative(qay)}) ÷ "
        f"({rational_text(mid[0])} - {paren_negative(qax)}) = {rational_text(key)}.",
    ]
    return text, key, _texts(wrong), working


@generator(
    "exd_x_a4_gradient_composite",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a4_gradient_composite(rng: random.Random, tier: str = "exam") -> dict:
    """Gradient that must be derived in two steps. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    kind = rng.choice(["standard", "perpendicular", "midpoint"])
    build = {"standard": _standard, "perpendicular": _perpendicular, "midpoint": _midpoint}[kind]
    text, key, wrong, working = build(rng)
    answer_text = rational_text(key)
    return dict(
        text=text,
        latex=text,
        answer=_pair(key),
        answer_text=answer_text,
        answer_latex=latex(Rational(key.numerator, key.denominator)),
        working=working,
        distractors=pick_distractors(answer_text, wrong),
    )
