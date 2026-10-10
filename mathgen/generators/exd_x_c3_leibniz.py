"""Exam tier for Leibniz notation: derivatives composed through a middle variable.

The hard tier reads `dy/dx = f'(x)` off a quadratic. Here the learner chains rates written in Leibniz notation:
either `dy/dx = dy/du * du/dx` for `y` a function of `u` and `u` a function of `x`, or `dy/dt = dy/dx * dx/dt`
for a quantity changing with time. Answers are whole numbers.
"""

from __future__ import annotations

import random

from sympy import diff, symbols

from ..core import GRADER_TOLERANCE, check, generator, pick_distractors
from .exd_x_c3_support import Redraw, draw

x, u, t = symbols("x u t")


def _chain_through_u(rng: random.Random) -> dict:
    n = rng.choice([2, 3])
    k = rng.choice([-3, -2, -1, 1, 2, 3])
    a = rng.choice([1, 2, 3])
    c = rng.randint(-4, 4)
    x0 = rng.choice([-3, -2, -1, 1, 2, 3])
    y_of_u = u**n + k * u
    u_of_x = a * x**2 + c
    u0 = int(u_of_x.subs(x, x0))
    dy_du = int(diff(y_of_u, u).subs(u, u0))
    du_dx = int(diff(u_of_x, x).subs(x, x0))
    answer = dy_du * du_dx
    # Independent route: substitute first, then differentiate with respect to x.
    check(int(diff(y_of_u.subs(u, u_of_x), x).subs(x, x0)) == answer, "composite derivative disagrees with the chain")
    wrong_u = int(diff(y_of_u, u).subs(u, x0))
    cands = [
        dy_du,  # stops at dy/du
        du_dx,  # stops at du/dx
        dy_du + du_dx,  # adds the two rates
        wrong_u * du_dx,  # evaluates dy/du at x0 instead of at u0
    ]
    y_text = f"u^{n} {'+' if k > 0 else '-'} {abs(k) if abs(k) != 1 else ''}u"
    u_text = f"{a if a != 1 else ''}x^2 {'+' if c > 0 else '-'} {abs(c)}" if c else f"{a if a != 1 else ''}x^2"
    text = (
        f"y = {y_text}, where u = {u_text}. Using the Leibniz chain dy/dx = dy/du × du/dx, "
        f"find the value of dy/dx when x = {x0}."
    )
    return _pack(text, answer, cands)


def _rate_in_time(rng: random.Random) -> dict:
    a = rng.choice([1, 2, 3])
    b = rng.randint(-6, 6)
    p = rng.choice([-3, -2, -1, 1, 2, 3])
    rate = rng.choice([-4, -3, -2, 2, 3, 4, 5])
    f = a * x**3 + b * x
    slope = int(diff(f, x).subs(x, p))
    answer = slope * rate
    # Independent route: let x move linearly through p at the given rate, then differentiate y with respect to t.
    check(int(diff(f.subs(x, p + rate * t), t).subs(t, 0)) == answer, "rate disagrees with the chain rule")
    cands = [
        slope,  # forgets dx/dt
        rate * int(f.subs(x, p)),  # uses y instead of dy/dx
        slope + rate,  # adds the rates
        int(diff(a * x**2 + b * x, x).subs(x, p)) * rate,  # wrong power rule for the cubic
    ]
    b_text = f" {'+' if b > 0 else '-'} {abs(b) if abs(b) != 1 else ''}x" if b else ""
    text = (
        f"y = {a if a != 1 else ''}x^3{b_text}, and x changes with time t so that dx/dt = {rate} when x = {p}. "
        f"Using the Leibniz chain dy/dt = dy/dx × dx/dt, find the value of dy/dt at that moment."
    )
    return _pack(text, answer, cands)


def _pack(text: str, answer: int, cands) -> dict:
    if answer == 0:
        raise Redraw
    distractors = pick_distractors(str(answer), [str(c) for c in cands], tolerance=GRADER_TOLERANCE)
    if len(distractors) < 3:
        raise Redraw
    return dict(text=text, latex="", answer=answer, answer_text=str(answer), distractors=distractors)


_KINDS = (_chain_through_u, _rate_in_time)


@generator("exd_x_c3_leibniz", level=13, presets={"exam": dict(tier="exam")})
def exd_x_c3_leibniz(rng: random.Random, tier: str = "exam") -> dict:
    """Chain two Leibniz-notation rates to find a derivative at a point."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(rng.choice(_KINDS), rng)
