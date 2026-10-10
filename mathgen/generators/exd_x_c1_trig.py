"""Exam-tier source for `differentiate-trig`.

The hard tier differentiates ``a sin(bx) + c cos(dx)``: a sum of trig functions of a *linear* argument. Here the
function is a single term in one of two forms the learner must tell apart:

- a *power of a trig function*, ``k sin^n(bx)`` or ``k cos^n(bx)`` with n = 2 or 3, which needs the chain rule
  twice (the power, then the trig function, then the linear argument); or
- a trig function of a *quadratic* argument, ``k sin(ax^2 + bx)`` or ``k cos(ax^2 + bx)``, whose inner derivative
  ``2ax + b`` is not a constant.

Built forwards from the parameters. The independent route is sympy's own ``diff``.
"""

from __future__ import annotations

import random

from sympy import cos, diff, sin

from ..core import check_distinct, check_equivalent, generator, join_terms, phrase, pick_distractors, value_safe_str
from .exd_x_c1_common import X, tex

_PATTERNS = (
    "Find f'(x) given that f(x) = {f}.",
    "If f(x) = {f}, find f'(x).",
    "Find dy/dx when y = {f}.",
)


def _power_form(rng: random.Random) -> dict:
    fn_name = rng.choice(["sin", "cos"])
    fn, dfn_sign = (sin, 1) if fn_name == "sin" else (cos, -1)
    other = cos if fn_name == "sin" else sin
    n = rng.choice([2, 3])
    b = rng.randint(2, 5)
    k = rng.choice([m for m in range(-5, 6) if m not in (0, 1, -1)])
    f = k * fn(b * X) ** n
    answer = dfn_sign * k * n * b * fn(b * X) ** (n - 1) * other(b * X)
    check_equivalent(diff(f, X), answer, "trig exam: power-form derivative disagrees with sympy")

    candidates = [
        dfn_sign * k * n * fn(b * X) ** (n - 1) * other(b * X),  # dropped the inner factor b
        dfn_sign * k * n * b * fn(b * X) ** (n - 1),  # forgot the derivative of the trig function itself
        dfn_sign * k * n * b * fn(b * X) ** n * other(b * X),  # forgot to lower the power
    ]
    if fn_name == "cos":
        candidates.append(k * n * b * fn(b * X) ** (n - 1) * other(b * X))  # sign error on the derivative of cos
    return dict(
        f_text=f"{k}{fn_name}^{n}({b}x)",
        f_tex=rf"{tex(k)}\{fn_name}^{{{n}}}({b}x)",
        answer=answer,
        candidates=candidates,
        working=[
            f"Outer power {n}, then {fn_name}, then the argument {b}x: three layers.",
            f"f'(x) = {k}*{n}*{fn_name}^{n - 1}({b}x) * ({'-' if fn_name == 'cos' else ''}{other.__name__}({b}x)) * {b}",
        ],
    )


def _quadratic_argument_form(rng: random.Random) -> dict:
    fn_name = rng.choice(["sin", "cos"])
    fn, dfn_sign = (sin, 1) if fn_name == "sin" else (cos, -1)
    other = cos if fn_name == "sin" else sin
    a = rng.randint(1, 3)
    b = rng.choice([m for m in range(-4, 5) if m != 0])
    k = rng.choice([m for m in range(-5, 6) if m not in (0, 1, -1)])
    arg = a * X**2 + b * X
    f = k * fn(arg)
    answer = dfn_sign * k * (2 * a * X + b) * other(arg)
    check_equivalent(diff(f, X), answer, "trig exam: quadratic-argument derivative disagrees with sympy")

    candidates = [
        dfn_sign * k * other(arg),  # dropped the inner derivative
        dfn_sign * k * (2 * a * X) * other(arg),  # inner derivative of the quadratic term only
        dfn_sign * k * (2 * a * X + b) * fn(arg),  # left the trig function unchanged
    ]
    if fn_name == "cos":
        candidates.append(k * (2 * a * X + b) * other(arg))  # sign error on the derivative of cos
    else:
        candidates.append(-k * (2 * a * X + b) * other(arg))  # attached cos's minus sign to sin
    arg_text = join_terms([(a, "x^2"), (b, "x")])
    return dict(
        f_text=f"{k}{fn_name}({arg_text})",
        f_tex=rf"{tex(k)}\{fn_name}\left({tex(arg)}\right)",
        answer=answer,
        candidates=candidates,
        working=[
            f"Inner function {arg_text} has derivative {value_safe_str(2 * a * X + b)}.",
            f"f'(x) = {value_safe_str(answer)}",
        ],
    )


def _build(rng: random.Random) -> dict:
    d = _power_form(rng) if rng.random() < 0.5 else _quadratic_argument_form(rng)
    answer = d["answer"]
    check_distinct(answer, *d["candidates"], msg="trig exam: a distractor equals the answer")
    answer_text = value_safe_str(answer)
    return dict(
        text=phrase(rng, _PATTERNS, f=d["f_text"]),
        latex=rf"\text{{Find }} f'(x) \text{{ given that }} f(x) = {d['f_tex']}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=tex(answer),
        working=d["working"],
        distractors=pick_distractors(answer_text, [value_safe_str(c) for c in d["candidates"]]),
    )


@generator("exd_x_c1_trig", level=14, presets={"exam": dict(tier="exam")})
def exd_x_c1_trig(rng: random.Random, tier: str = "exam") -> dict:
    """Differentiate a power of sin/cos, or sin/cos of a quadratic argument."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
