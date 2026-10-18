"""Generates `w6a_aux_angle` questions (easy, medium, hard) and the exam chain `exd_x_w6a_aux_angle`.

``auxiliary-angle-max-min-solve`` (``trigonometry``, answer ``expression``): use the auxiliary-angle form
``a cos x + b sin x = R cos(x - α)`` to find extreme values and where they occur, and to solve ``a cos x + b sin x = c``
on ``[0, 2π)``. The conversion itself is practised in ``senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm``; here it is
a tool, and nobody is told R or α.

* easy: the maximum or minimum of ``s(a cos x + b sin x) + d`` for whole-number a, b (a Pythagorean pair, or a pair
  that makes R a surd), including a multiplier and a vertical shift.
* medium: the smallest positive x at which ``a cos x + b sin x`` is largest or smallest, for coefficients that are
  whole numbers or multiples of √3, so α is a multiple of π/6 or π/4 in any of the four quadrants.
* hard: the larger or smaller solution of ``a cos x + b sin x = c`` on ``[0, 2π)``, or the least whole number c for
  which the equation has no solution.
* exam (``exd_x_w6a_aux_angle``): (a) R, (b) the angle α in ``[0, 2π)``, (c) the larger solution of the equation
  (follows through from (b)), (d) the greatest value of the function plus a constant (follows through from (a)).

Every key is derived exactly with ``atan2`` and then checked numerically: the function is sampled across [0, 2π) to
find its largest and smallest values and the places they occur, and each solution is substituted back.
"""

from __future__ import annotations

import math

import sympy
from sympy import Rational, pi, sqrt

from ..core import attempts, check, generator, phrase
from ._ucv1_calc_markup import value_text as vt
from ._w6a_kit import enough, expr_dist, expr_fields, part, retry_draw

TOPIC = "w6a_aux_angle"
EXAM_TOPIC = "exd_x_w6a_aux_angle"

AREA = "auxiliary-angle"
MISCONCEPTIONS = [
    ("aux-sums-coefficients", "Takes R to be a + b instead of the square root of a² + b².", AREA),
    ("aux-larger-coefficient", "Takes the larger of the two coefficients as the amplitude R.", AREA),
    ("aux-forgets-root", "Gives R² = a² + b² as the amplitude, forgetting the square root.", AREA),
    ("aux-shift-ignored", "Leaves out the vertical shift, giving the extreme value of the wave alone.", AREA),
    (
        "aux-multiplier-ignored",
        "Leaves out the multiplier in front of the bracket when finding the extreme value.",
        AREA,
    ),
    (
        "aux-wrong-end",
        "Applies the vertical shift to the wrong end, giving the maximum where the minimum was asked or the reverse.",
        AREA,
    ),
    (
        "aux-alpha-swapped",
        "Finds α from arctan(a/b) instead of arctan(b/a), swapping the roles of the two coefficients.",
        AREA,
    ),
    (
        "aux-no-quadrant",
        "Reads α straight from the arctangent without placing it in the quadrant given by the signs of a and b.",
        AREA,
    ),
    (
        "aux-min-for-max",
        "Gives the position of the minimum where the maximum was asked, or the reverse (an error of π).",
        AREA,
    ),
    ("aux-angle-left-negative", "Leaves α negative instead of adding 2π to bring it into [0, 2π).", AREA),
    (
        "aux-other-solution",
        "Gives the other solution of cos(x - α) = c/R, the smaller where the larger was asked or the reverse.",
        AREA,
    ),
    (
        "aux-offset-not-added",
        "Gives the offset δ from cos δ = c/R without adding α, solving the equation for x - α only.",
        AREA,
    ),
    ("aux-c-not-scaled", "Solves cos(x - α) = c instead of cos(x - α) = c/R.", AREA),
    (
        "aux-threshold-rounded-down",
        "Gives the whole part of R as the least c with no solution, instead of the next whole number.",
        AREA,
    ),
    ("aux-threshold-rounded", "Rounds R to the nearest whole number to find the least c with no solution.", AREA),
]


# -- the pairs (a, b), their R and α ---------------------------------------------------------------------------------


def _coef(k: int, surd: bool):
    return k * sqrt(3) if surd else sympy.Integer(k)


def _term_text(coef, trig: str) -> str:
    """One term such as ``3 cos x``, ``√3 sin x`` or ``-2√3 cos x`` (the sign is added by the caller)."""
    mag = abs(sympy.sympify(coef))
    if mag.has(sqrt(3)):
        k = mag / sqrt(3)
        body = ("" if k == 1 else str(k)) + "√3"
        return f"{body} {trig}"
    return f"{trig}" if mag == 1 else f"{mag} {trig}"


def _function_text(a, b) -> str:
    first = ("-" if a < 0 else "") + _term_text(a, "cos x")
    second = ("- " if b < 0 else "+ ") + _term_text(b, "sin x")
    return f"{first} {second}"


def _alpha(a, b):
    """α in [0, 2π), exactly."""
    value = sympy.atan2(b, a)
    if value < 0:
        value += 2 * pi
    return sympy.nsimplify(value) if not value.is_number else value


def _check_extreme(a, b, d, s, maximum: bool, expected):
    """Numerically: the largest (or smallest) value of s(a cos x + b sin x) + d over a fine grid of [0, 2π)."""
    fa, fb = float(a), float(b)
    values = [s * (fa * math.cos(t) + fb * math.sin(t)) + d for t in (2 * math.pi * i / 6000 for i in range(6000))]
    got = max(values) if maximum else min(values)
    check(abs(got - float(expected)) < 2e-3 * max(1.0, abs(float(expected))), "numeric extreme differs")


def _check_location(a, b, maximum: bool, expected):
    fa, fb = float(a), float(b)
    xs = [2 * math.pi * i / 12000 for i in range(12000)]
    vals = [fa * math.cos(t) + fb * math.sin(t) for t in xs]
    best = max(range(12000), key=lambda i: vals[i] if maximum else -vals[i])
    check(
        abs(xs[best] - float(expected)) < 1e-3 or abs(abs(xs[best] - float(expected)) - 2 * math.pi) < 1e-3,
        "numeric location differs",
    )


def _wave_text(s, a, b, d) -> str:
    inner = _function_text(a, b)
    body = inner if s == 1 else f"-({inner})" if s == -1 else f"{s}({inner})"
    if d == 0:
        return body
    if d < 0:
        return f"{body} - {abs(d)}"
    return f"{d} - {body[1:]}" if body.startswith("-") else f"{d} + {body}"


# -- easy ---------------------------------------------------------------------------------------------------------------

_PYTH = ((3, 4), (4, 3), (5, 12), (12, 5), (8, 15), (15, 8), (6, 8), (8, 6), (7, 24), (9, 12), (20, 21))
_SURDY = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 3),
    (3, 2),
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (3, 3),
    (5, 1),
    (1, 5),
    (4, 3),
    (2, 5),
)


def _signed_pair(rng, pair):
    a, b = pair
    return a * rng.choice((1, -1)), b * rng.choice((1, -1))


def _easy(rng):
    if rng.random() < 0.6:
        a, b = _signed_pair(rng, rng.choice(_PYTH))
    else:
        a, b = _signed_pair(rng, rng.choice(_SURDY))
    s = rng.choice((1, 1, 2, 3, -1))
    d = rng.choice((0, 0, 1, 2, 3, 4, -1, -2, 5))
    if s == 1 and d == 0:
        d = rng.choice((1, 2, 3, -2))
    maximum = rng.random() < 0.55
    R = sqrt(a * a + b * b)
    top, bottom = d + abs(s) * R, d - abs(s) * R
    value = top if maximum else bottom
    _check_extreme(a, b, d, s, maximum, value)
    wave = _wave_text(s, a, b, d)
    swap = bottom if maximum else top
    cands = [
        (swap, "aux-wrong-end"),
        (d + abs(s) * (abs(a) + abs(b)) * (1 if maximum else -1), "aux-sums-coefficients"),
        (d + abs(s) * (a * a + b * b) * (1 if maximum else -1), "aux-forgets-root"),
        (abs(s) * R * (1 if maximum else -1), "aux-shift-ignored") if d != 0 else None,
        (d + R * (1 if maximum else -1), "aux-multiplier-ignored") if abs(s) != 1 else None,
    ]
    dist = expr_dist(value, [c for c in cands if c is not None])
    enough(dist)
    word = "maximum" if maximum else "minimum"
    other = "greatest" if maximum else "least"
    stem = phrase(
        rng,
        (
            "Find the {word} value of f(x) = {wave}.",
            "The function f is defined by f(x) = {wave} for all real x. What is the {other} value f takes?",
            "Without using calculus, find the {word} value of y = {wave}.",
        ),
        word=word,
        other=other,
        wave=wave,
    )
    working = [
        f"Write {_function_text(a, b)} in the form R cos(x - α) with R = √({vt(a * a)} + {vt(b * b)}) = {vt(R)}.",
        f"The inner wave runs from -{vt(R)} to {vt(R)}, so f has {word} value {vt(value)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="expression", **expr_fields(value))


# -- medium ---------------------------------------------------------------------------------------------------------------


def _special_pair(rng):
    """(a, b): coefficients with α a multiple of π/6 or π/4 and R a whole number or a surd."""
    kind = rng.choice(("sixth", "sixth", "quarter"))
    k = rng.randint(1, 3)
    if kind == "sixth":
        ca, cb = (k, True), (k, False)
        if rng.random() < 0.5:
            ca, cb = cb, ca
        a, b = _coef(*ca), _coef(*cb)
    else:
        a, b = sympy.Integer(k), sympy.Integer(k)
    return a * rng.choice((1, -1)), b * rng.choice((1, -1))


def _medium(rng):
    a, b = _special_pair(rng)
    alpha = _alpha(a, b)
    maximum = rng.random() < 0.6
    far = alpha + pi if alpha + pi < 2 * pi else alpha - pi
    where = sympy.simplify(alpha if maximum else far)
    _check_location(a, b, maximum, where)
    raw = sympy.atan(sympy.simplify(b / a))
    swapped = sympy.atan2(a, b)
    swapped = swapped + 2 * pi if swapped < 0 else swapped
    cands = [
        (far if maximum else alpha, "aux-min-for-max"),
        (raw, "aux-no-quadrant"),
        (swapped, "aux-alpha-swapped"),
        (sympy.atan2(b, a) if sympy.atan2(b, a) < 0 else None, "aux-angle-left-negative"),
    ]
    dist = expr_dist(where, [(sympy.simplify(c), t) for c, t in cands if c is not None and sympy.simplify(c).is_real])
    # a minimum whose answer is the same as another tagged value is dropped by expr_dist; top up with a standard slip
    if len(dist) < 2:
        dist = expr_dist(
            where,
            [
                (sympy.simplify(alpha + pi / 2), "aux-alpha-swapped"),
                (sympy.simplify(-alpha + 2 * pi), "aux-no-quadrant"),
                (sympy.simplify(raw), "aux-no-quadrant"),
                (sympy.simplify(swapped), "aux-alpha-swapped"),
            ],
        )
    enough(dist)
    fn = _function_text(a, b)
    word = "greatest" if maximum else "least"
    stem = phrase(
        rng,
        (
            "For 0 <= x < 2π, the function f(x) = {fn} takes its {word} value at exactly one x. Find that value of x.",
            "Find the x in the interval [0, 2π) at which f(x) = {fn} reaches its {word} value.",
            "At what value of x, with 0 <= x < 2π, is {fn} {word_adj}?",
        ),
        fn=fn,
        word=word,
        word_adj="largest" if maximum else "smallest",
    )
    working = [
        f"Write {fn} as R cos(x - α): cos α = a/R and sin α = b/R put α = {vt(alpha)} in [0, 2π).",
        f"R cos(x - α) is largest at x = α and smallest at x = α ± π, so the answer is x = {vt(where)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="expression", **expr_fields(where))


# -- hard -------------------------------------------------------------------------------------------------------------------

_DELTAS = (pi / 6, pi / 4, pi / 3, pi / 2, 2 * pi / 3, 3 * pi / 4)


def _solutions(alpha, delta):
    """The two solutions of cos(x - alpha) = cos(delta) in [0, 2*pi), sorted, as exact values."""
    out = []
    for v in (alpha + delta, alpha - delta):
        v = sympy.simplify(v)
        while v < 0:
            v += 2 * pi
        while v >= 2 * pi:
            v -= 2 * pi
        out.append(sympy.simplify(v))
    out = sorted(set(out), key=float)
    return out


def _check_solutions(a, b, c, sols):
    fa, fb, fc = float(a), float(b), float(c)
    for x in sols:
        check(
            abs(fa * math.cos(float(x)) + fb * math.sin(float(x)) - fc) < 1e-9,
            "a solution does not satisfy the equation",
        )
    # no other solution: count sign changes of the function minus c over a fine grid
    grid = [2 * math.pi * (i + 0.37) / 20000 for i in range(20000)]
    vals = [fa * math.cos(t) + fb * math.sin(t) - fc for t in grid]
    changes = sum(1 for u, v in zip(vals, vals[1:]) if u * v < 0)
    check(changes == len(sols), "the number of solutions differs")


def _hard_solve(rng):
    for _ in attempts(500, "an equation with two solutions in range"):
        a, b = _special_pair(rng)
        alpha = _alpha(a, b)
        delta = rng.choice(_DELTAS)
        R = sympy.simplify(sqrt(a * a + b * b))
        c = sympy.simplify(R * sympy.cos(delta))
        sols = _solutions(alpha, delta)
        if len(sols) == 2 and c != 0 and sols[0] != 0:
            break
    _check_solutions(a, b, c, sols)
    larger = rng.random() < 0.55
    value = sols[1] if larger else sols[0]
    other = sols[0] if larger else sols[1]
    cands = [
        (other, "aux-other-solution"),
        (sympy.simplify(delta), "aux-offset-not-added"),
        (sympy.simplify(alpha), "aux-offset-not-added"),
        (sympy.simplify(sympy.acos(c)) if abs(float(c)) <= 1 else None, "aux-c-not-scaled"),
        (sympy.simplify(alpha + delta + pi) if float(alpha + delta + pi) < 2 * float(pi) else None, "aux-min-for-max"),
    ]
    dist = expr_dist(value, [(v, t) for v, t in cands if v is not None])
    enough(dist)
    fn = _function_text(a, b)
    c_text = expr_fields(c)["answer_text"]
    word = "larger" if larger else "smaller"
    stem = phrase(
        rng,
        (
            "Solve {fn} = {c} for 0 <= x < 2π. Give the {word} of the two solutions.",
            "The equation {fn} = {c} has two solutions with 0 <= x < 2π. What is the {word} one?",
            "On the interval [0, 2π), find the {word} root of the equation {fn} = {c}.",
        ),
        fn=fn,
        c=c_text,
        word=word,
    )
    working = [
        f"Write the left side as R cos(x - α) with R = {vt(R)} and α = {vt(alpha)}.",
        f"Then cos(x - α) = {vt(sympy.simplify(c / R))}, so x - α = ±{vt(delta)} and x = {vt(sols[0])} or {vt(sols[1])}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="expression", **expr_fields(value))


def _hard_threshold(rng):
    """The least whole number c for which a cos x + b sin x = c has no solution: floor(R) + 1."""
    for _ in attempts(500, "a coefficient pair with a non-integer R"):
        a, b = rng.randint(1, 8) * rng.choice((1, -1)), rng.randint(1, 8) * rng.choice((1, -1))
        r2 = a * a + b * b
        if math.isqrt(r2) ** 2 != r2 and 3 <= math.sqrt(r2) <= 11:
            break
    r = math.sqrt(r2)
    value = math.isqrt(r2) + 1
    # Second route: scan the integers: the first c above max f has no solution, the one below has.
    fa, fb = float(a), float(b)
    top = max(fa * math.cos(2 * math.pi * i / 8000) + fb * math.sin(2 * math.pi * i / 8000) for i in range(8000))
    check(value - 1 <= top + 1e-3 < value, "numeric threshold differs")
    cands = [
        (value - 1, "aux-threshold-rounded-down"),
        (round(r) if round(r) != value else value + 1, "aux-threshold-rounded"),
        (abs(a) + abs(b), "aux-sums-coefficients"),
        (r2, "aux-forgets-root"),
    ]
    dist = expr_dist(value, cands)
    enough(dist)
    fn = _function_text(a, b)
    stem = phrase(
        rng,
        (
            "What is the least positive whole number c for which {fn} = c has no solution?",
            "The equation {fn} = c has real solutions for some whole numbers c and none for others. Find the smallest positive whole number c with no solution.",
            "Find the smallest positive integer c such that {fn} = c cannot be solved for x.",
        ),
        fn=fn,
    )
    working = [
        f"The left side is R cos(x - α) with R = √({a * a} + {b * b}) = √{r2} ≈ {r:.2f}.",
        f"It never exceeds R, so the equation fails once c > √{r2}: the least such whole number is {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, answer_kind="expression", **expr_fields(value))


_TIERS = {"easy": (_easy,), "medium": (_medium,), "hard": (_hard_solve, _hard_threshold)}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6a_aux_angle(rng, tier="easy"):
    """Extreme values, where they occur, and equations a cos x + b sin x = c; see the module docstring."""
    pool = _TIERS.get(tier)
    check(pool is not None, f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w6a_aux_angle(rng, tier="exam"):
    """Exam chain: R, α, the larger solution of f(x) = c, and the greatest value of f(x) plus a constant."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return retry_draw(rng, _exam)


def _exam(rng):
    for _ in attempts(500, "an exam function whose larger solution stays below 2π"):
        a, b = _special_pair(rng)
        alpha = _alpha(a, b)
        delta = rng.choice((pi / 6, pi / 4, pi / 3))
        R = sympy.simplify(sqrt(a * a + b * b))
        c = sympy.simplify(R * sympy.cos(delta))
        sols = _solutions(alpha, delta)
        if len(sols) == 2 and alpha + delta < 2 * pi and c != 0 and alpha - delta > 0 and sols[0] != 0:
            break
    larger = sols[1]
    check(sympy.simplify(larger - (alpha + delta)) == 0, "the larger solution is not alpha + delta")
    _check_solutions(a, b, c, sols)
    shift = rng.choice((1, 2, 3, 4, 5))
    greatest = sympy.simplify(shift + R)
    _check_extreme(a, b, shift, 1, True, greatest)
    fn = _function_text(a, b)
    raw = sympy.atan(sympy.simplify(b / a))
    swapped = sympy.atan2(a, b)
    swapped = swapped + 2 * pi if swapped < 0 else swapped
    a_dist = expr_dist(
        R,
        [
            (abs(a) + abs(b), "aux-sums-coefficients"),
            (a * a + b * b, "aux-forgets-root"),
            (max(abs(a), abs(b)), "aux-larger-coefficient"),
        ],
        3,
    )
    b_dist = expr_dist(
        alpha,
        [
            (sympy.simplify(raw), "aux-no-quadrant"),
            (sympy.simplify(swapped), "aux-alpha-swapped"),
            (sympy.simplify(alpha + pi if alpha + pi < 2 * pi else alpha - pi), "aux-min-for-max"),
            (sympy.simplify(alpha - 2 * pi), "aux-angle-left-negative"),
        ],
        3,
    )
    c_dist = expr_dist(
        larger,
        [
            (sols[0], "aux-other-solution"),
            (delta, "aux-offset-not-added"),
            (alpha, "aux-offset-not-added"),
            (sympy.simplify(sympy.acos(c)) if abs(float(c)) <= 1 else None, "aux-c-not-scaled"),
        ],
        3,
    )
    d_dist = expr_dist(
        greatest,
        [
            (shift + abs(a) + abs(b), "aux-sums-coefficients"),
            (R, "aux-shift-ignored"),
            (shift - R, "aux-wrong-end"),
            (shift + a * a + b * b, "aux-forgets-root"),
        ],
        3,
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        enough(dist)
    c_text = expr_fields(c)["answer_text"]
    delta_text = expr_fields(delta)["answer"]
    return dict(
        text=f"Let f(x) = {fn}, for 0 <= x < 2π.",
        parts=[
            part(
                "Write f(x) in the form R cos(x - α) with R > 0. Find R.",
                "expression",
                expr_fields(R),
                1,
                [f"R = √({vt(a * a)} + {vt(b * b)}) = {vt(R)}."],
                a_dist,
            ),
            part(
                "Find α, where 0 <= α < 2π.",
                "expression",
                expr_fields(alpha),
                2,
                [f"cos α = a/R and sin α = b/R put α in the right quadrant: α = {vt(alpha)}."],
                b_dist,
            ),
            part(
                f"Solve f(x) = {c_text}, and give the larger solution in the interval.",
                "expression",
                expr_fields(larger),
                2,
                [
                    f"cos(x - α) = {vt(sympy.simplify(c / R))}, so x - α = ±{vt(delta)}, and the larger solution is x = α + {vt(delta)} = {vt(larger)}."
                ],
                c_dist,
                follow_through=f"part_b + {delta_text}",
            ),
            part(
                f"Find the greatest value of g(x) = {shift} + f(x).",
                "expression",
                expr_fields(greatest),
                1,
                [f"f has greatest value R = {vt(R)}, so g has greatest value {shift} + {vt(R)} = {vt(greatest)}."],
                d_dist,
                follow_through=f"{shift} + part_a",
            ),
        ],
    )
