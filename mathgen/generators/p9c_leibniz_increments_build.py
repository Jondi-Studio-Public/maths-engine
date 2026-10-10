"""Generates questions on the bare Leibniz increments δx and δy (ACMMM078), answered as a labelled pair.

**What this asks that the ACMMM082 neighbour does not.** This skill asks for the
two BARE increments δx and δy themselves -- never their ratio, never a
limit, never an estimate. There is no quotient anywhere in this generator; see
``mathgen/generators/p5e_first_principles_build.py`` (the difference-quotient /
derivative-definition skill) for the quotient/limit skill this one is not.

``y = f(x)``. A point x1 and a step δx are drawn as bounded integers, x2 =
x1 + δx, and the learner reports the pair (δx, δy) where δy =
f(x2) - f(x1):

- easy: f linear, δx drawn positive -- δy is a fixed multiple of δx
  (the slope), independent of where x1 sits.
- medium: f quadratic, δx still drawn positive -- δy now genuinely
  depends on WHERE x1 is, so the learner must evaluate f at two points rather
  than read off a constant.
- hard: f quadratic, δx drawn NEGATIVE (x is decreasing) -- δx and
  δy may now carry independent signs, a genuinely different reasoning step
  from applying a fixed rule in one direction.
"""

from __future__ import annotations

from sympy import expand, symbols

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    paren_negative,
    pick_distractors,
)

_x, _dx = symbols("x dx")

_DX_POSITIVE = (1, 2, 3, 4)
_DX_NEGATIVE = (-1, -2, -3, -4)
_MAX_ATTEMPTS = 500


def _fmt_pair(dx: int, dy: int) -> str:
    return f"δx = {dx}, δy = {dy}"


def _general_delta_y(f_expr):
    """Symbolic δy = f(x+dx) - f(x), expanded to a closed form in the
    GENERIC symbols _x, _dx -- the independent route:
    substitute x1/δx into this general result, rather than the
    direct-substitution-then-subtract route construction itself uses."""
    return expand(f_expr.subs(_x, _x + _dx) - f_expr)


def _sample_linear(rng):
    m = rng.choice([v for v in range(-9, 10) if v != 0])
    k = rng.randint(-9, 9)
    f_expr = m * _x + k
    f_text = join_terms([(m, "x"), (k, "")])
    return f_expr, f_text


def _sample_quadratic(rng):
    a = rng.randint(1, 5)
    b = rng.choice([v for v in range(-9, 10) if v != 0])
    c = rng.randint(-9, 9)
    f_expr = a * _x**2 + b * _x + c
    f_text = join_terms([(a, "x^2"), (b, "x"), (c, "")])
    return f_expr, f_text


def _build(rng, *, tier, sample_f, dx_pool, x1_lo=-6, x1_hi=6):
    """Shared build for every tier. ``sample_f(rng)`` draws a fresh (f_expr,
    f_text) pair; ``dx_pool`` is the pool δx is drawn from."""
    for _attempt in range(_MAX_ATTEMPTS):
        f_expr, f_text = sample_f(rng)
        dx = rng.choice(dx_pool)
        x1 = rng.randint(x1_lo, x1_hi)
        if x1 == dx:
            continue  # guard: the used-x1-instead-of-dx misconception needs x1 != dx
        x2 = x1 + dx
        check(x2 == x1 + dx, "x2 construction failed")

        y1 = int(f_expr.subs(_x, x1))
        y2 = int(f_expr.subs(_x, x2))
        dy = y2 - y1

        if dx == dy:  # the swapped-order distractor would collide with the answer
            continue
        if dy == 0:  # the sign-flipped-dy distractor would collide with the answer
            continue
        break
    else:
        raise RuntimeError(
            f"{tier}: could not draw a non-colliding Leibniz-increment question in {_MAX_ATTEMPTS} attempts"
        )

    # Independent check: a general closed form in generic x, dx, THEN
    # substitute the specific x1/dx -- a genuinely different route from
    # construction's direct substitution of f at x1 and x2 followed by a
    # subtraction.
    general = _general_delta_y(f_expr)
    dy_general = general.subs({_x: x1, _dx: dx})
    check_equivalent(dy_general, dy, f"{tier}: independent check failed for x1={x1}, dx={dx}")

    answer_text = _fmt_pair(dx, dy)
    swapped = (dy, dx)
    sign_flipped = (dx, -dy)
    check_distinct((dx, dy), swapped, msg=f"{tier}: swapped-order collides with the answer")
    check_distinct((dx, dy), sign_flipped, msg=f"{tier}: sign-flipped-dy collides with the answer")
    candidates = [_fmt_pair(*swapped), _fmt_pair(*sign_flipped)]

    if tier == "hard":
        dropped_sign = (abs(dx), dy)
        check_distinct((dx, dy), dropped_sign, msg=f"{tier}: dropped-sign-on-dx collides with the answer")
        candidates.append(_fmt_pair(*dropped_sign))
    else:
        used_x1 = (x1, dy)
        check_distinct((dx, dy), used_x1, msg=f"{tier}: used-x1-instead-of-dx collides with the answer")
        candidates.append(_fmt_pair(*used_x1))

    distractors = pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE)

    text = f"y = f(x) = {f_text}. x changes from x1 = {x1} to x2 = {x2}. What are δx and δy?"
    working = [
        f"δx = x2 - x1 = {x2} - {paren_negative(x1)} = {dx}.",
        f"f(x1) = {y1}, f(x2) = {y2}.",
        f"δy = f(x2) - f(x1) = {y2} - {paren_negative(y1)} = {dy}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer={"dx": dx, "dy": dy},
        answer_text=answer_text,
        answer_latex=rf"\delta x = {dx},\; \delta y = {dy}",
        working=working,
        distractors=distractors,
    )


@generator(
    "p9c_leibniz_delta_x_delta_y_increments",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9c_leibniz_delta_x_delta_y_increments(rng, tier="easy"):
    """Report the labelled pair (δx, δy) for a bounded change from x1
    to x2 = x1 + δx under y = f(x). See module docstring for the
    per-tier design."""
    if tier == "easy":
        return _build(rng, tier=tier, sample_f=_sample_linear, dx_pool=_DX_POSITIVE)
    if tier == "medium":
        return _build(rng, tier=tier, sample_f=_sample_quadratic, dx_pool=_DX_POSITIVE)
    if tier == "hard":
        return _build(rng, tier=tier, sample_f=_sample_quadratic, dx_pool=_DX_NEGATIVE)
    raise ValueError(f"unknown tier: {tier!r}")
