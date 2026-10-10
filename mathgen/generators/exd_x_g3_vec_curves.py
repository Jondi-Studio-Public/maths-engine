"""Exam-tier source for the skill "Use vector equations of curves ... involving a parameter".

The practice tiers eliminate the parameter, or evaluate the Cartesian relation,
for a curve whose constants are all stated.  Here a constant of the curve
``r(t) = (a t + p, b / t + c)`` is unknown and must be recovered from points the
curve passes through:

* ``one_point`` -- ``b`` is unknown; one point gives ``t`` from the x-coordinate
  and then ``b`` from the y-coordinate.
* ``two_points`` -- ``b`` and ``c`` are both unknown; two points give a pair of
  simultaneous equations in ``1/t`` terms.  The question asks for ``b`` or for ``c``.

Built backwards from the constants and the chosen parameter values, so all
coordinates are integers.
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, solve

from ..core import check, generator, pick_distractors

_T = [-3, -2, -1, 1, 2, 3, 4, 6]


def _signed_add(base: str, k: int) -> str:
    return f"{base} {'+' if k > 0 else '-'} {abs(k)}" if k else base


def _build(rng: random.Random) -> dict:
    form = rng.choice(["one_point", "two_points"])
    for _ in range(400):
        a = rng.choice([-3, -2, 2, 3, 4])
        if form == "one_point":
            p = rng.randint(-5, 5)
            c = rng.choice([v for v in range(-8, 9) if v != 0])
            t = rng.choice(_T)
            w = rng.choice([v for v in range(-6, 7) if v != 0])
            b = t * w
            if abs(b) < 2:
                continue
            x, y = a * t + p, w + c
            # Independent route: solve for b symbolically.
            bs = Symbol("b")
            sol = solve(bs / Rational(x - p, a) + c - y, bs)
            check(sol == [b], "b is determined by the point")
            ytxt = f"b/t {'+' if c > 0 else '-'} {abs(c)}"
            text = (
                f"A curve has vector equation r(t) = ({_signed_add(f'{a}t', p)}, {ytxt}) for t not equal to 0, "
                f"where b is a constant. The curve passes through the point ({x}, {y}). Find b."
            )
            answer = b
            cands = [
                y - c,  # takes t = 1
                y * t,  # forgets the offset c
                (y + c) * t,  # sign slip on the offset
                (y - c) * x,  # uses the x-coordinate in place of t
            ]
        else:
            t1, t2 = rng.sample(_T, 2)
            p = 0
            b = 12 * rng.choice([-2, -1, 1, 2])
            c = rng.choice([v for v in range(-8, 9) if v != 0])
            if b % t1 or b % t2:
                continue
            x1, y1 = a * t1, b // t1 + c
            x2, y2 = a * t2, b // t2 + c
            unknown = rng.choice(["b", "c"])
            bs, cs = Symbol("b"), Symbol("c")
            sol = solve(
                [bs / Rational(x1, a) + cs - y1, bs / Rational(x2, a) + cs - y2],
                [bs, cs],
                dict=True,
            )
            check(len(sol) == 1 and sol[0][bs] == b and sol[0][cs] == c, "b and c are determined by the points")
            ytxt = "b/t + c"
            text = (
                f"A curve has vector equation r(t) = ({a}t, {ytxt}) for t not equal to 0, "
                f"where b and c are constants. The curve passes through the points ({x1}, {y1}) and ({x2}, {y2}). "
                f"Find {unknown}."
            )
            answer = b if unknown == "b" else c
            if unknown == "c":
                cands = [y1, y2, b]
                # takes t = x (forgets the factor a): solve the same pair with t1 = x1, t2 = x2
                bb = Rational(y1 - y2) / (Rational(1, x1) - Rational(1, x2))
                cc = y1 - bb / x1
                if cc.q == 1:
                    cands.append(int(cc))
            else:
                cands = [c, y1 - y2, -b]
                bb = Rational(y1 - y2) / (Rational(1, x1) - Rational(1, x2))
                if bb.q == 1:
                    cands.append(int(bb))
        distractors = pick_distractors(str(answer), [str(v) for v in cands])
        if len(distractors) >= 3:
            break
    else:  # pragma: no cover
        raise ValueError("could not draw a question")
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=["Use the x-coordinate to find t, then the y-coordinate to find the constant."],
        distractors=distractors[:3],
    )


@generator(
    "exd_x_g3_vec_curves",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g3_vec_curves(rng: random.Random, tier: str = "exam") -> dict:
    """Recover an unknown constant of r(t) = (a t + p, b/t + c) from points on the curve."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
