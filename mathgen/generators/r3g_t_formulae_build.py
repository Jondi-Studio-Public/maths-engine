"""Generates `r3g_t_formulae` questions (easy, medium, hard) and the exam chain `r3g_exam_t_formulae`.

Skill ``t-formulae-half-angle`` (``trigonometry``, answer ``expression``): the substitution
``t = tan(x/2)``, which gives ``sin x = 2t/(1 + t²)``, ``cos x = (1 - t²)/(1 + t²)`` and ``tan x = 2t/(1 - t²)``.

* easy: sin x, cos x or tan x as a fraction, given ``t = tan(x/2)`` as a fraction.
* medium: tan(x/2) from cos x or from tan x. Either cos x and the quadrant of x (the sign of tan(x/2) comes from the
  quadrant of x/2), or the exact half-angle value of a standard angle (tan 15°, tan 22.5°, ...) from tan x = 1,
  tan x = 1/√3 and so on, which gives a quadratic in t with two roots, one of them rejected.
* hard: ``a sin x + b cos x = c`` for 0 ≤ x < 2π becomes a quadratic in t; the larger or smaller solution exactly, as 2 arctan t (plus 2π when t is negative).
* exam (``r3g_exam_t_formulae``): the same equation set up for one pair of positive roots: (a) the smaller root t, (b) the
  solution x it gives (follows through from (a)), (c) the larger root and (d) its solution x (follows through from (c)).

Each result is confirmed by building x = 2 arctan t with ``math`` and evaluating the equation or the ratio directly.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..core import attempts, check, generator, round_half_up
from ._r3g_kit import (
    vary_symbol,
    decimal_dist,
    decimal_set_dist,
    decimal_set_fields,
    ex,
    latex,
    number_fields,
    pick,
    retry_short,
    typed,
)

R = sp.Rational
F = Fraction


def _rational(p, q):
    return R(p, q)


def _easy(rng):
    for _ in attempts(300, "a half-angle tangent with 1 - t² nonzero"):
        q = rng.randint(2, 7)
        p = rng.randint(1, 9) * rng.choice((1, -1))
        if math.gcd(abs(p), q) != 1 or abs(p) == q or abs(p) > 2 * q:
            continue
        t = R(p, q)
        break
    fn = rng.choice(("sin", "cos", "tan"))
    sin_x, cos_x = 2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)
    value = {"sin": sin_x, "cos": cos_x, "tan": 2 * t / (1 - t**2)}[fn]
    x = 2 * math.atan(float(t))
    expected = {"sin": math.sin(x), "cos": math.cos(x), "tan": math.tan(x)}[fn]
    check(abs(float(value) - expected) < 1e-9, "value differs from the float")
    wrong = {
        "sin": [2 * t / (1 - t**2), (1 - t**2) / (1 + t**2), 2 * t / (1 + t), t / (1 + t**2)],
        "cos": [(1 + t**2) / (1 - t**2), 2 * t / (1 + t**2), (1 - t) / (1 + t), (1 - t**2) / (1 + t)],
        "tan": [2 * t / (1 + t**2), (1 - t**2) / (2 * t), 2 * t / (1 - t), 2 * t],
    }[fn]
    tags = {
        "sin": [
            "r3g-t-formula-tan-formula-used",
            "r3g-t-formula-functions-swapped",
            "r3g-t-formula-denominator-not-squared",
            "r3g-t-formula-numerator-wrong",
        ],
        "cos": [
            "r3g-t-formula-reciprocal",
            "r3g-t-formula-functions-swapped",
            "r3g-t-formula-denominator-not-squared",
            "r3g-t-formula-denominator-not-squared",
        ],
        "tan": [
            "r3g-t-formula-sin-formula-used",
            "r3g-t-formula-reciprocal",
            "r3g-t-formula-denominator-not-squared",
            "r3g-t-formula-doubling-only",
        ],
    }[fn]
    text = typed(value)
    formula = {"sin": "2t/(1 + t²)", "cos": "(1 - t²)/(1 + t²)", "tan": "2t/(1 - t²)"}[fn]
    return dict(
        text=f"Let t = tan(x/2). Given that t = {typed(t)}, find the value of {fn} x.",
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=[f"{fn} x = {formula} with t = {typed(t)}.", f"{fn} x = {text}."],
        distractors=pick(text, [ex(w, g) for w, g in zip(wrong, tags)], 3),
    )


_HALF_ANGLES = {
    # tan x (exact) -> x in degrees, tan(x/2) positive root
    30: (sp.sqrt(3) / 3, 2 - sp.sqrt(3)),
    45: (sp.Integer(1), sp.sqrt(2) - 1),
    60: (sp.sqrt(3), sp.sqrt(3) / 3),
    120: (-sp.sqrt(3), sp.sqrt(3)),
    135: (sp.Integer(-1), sp.sqrt(2) + 1),
}


def _medium_cos(rng):
    for _ in attempts(300, "a half-angle tangent with a rational cosine"):
        q = rng.randint(2, 6)
        p = rng.randint(1, 6)
        if math.gcd(p, q) != 1 or p >= 2 * q:
            continue
        t = R(p, q)
        if t == 1:
            continue
        break
    first_half = rng.random() < 0.5
    signed = t if first_half else -t
    cos_x = (1 - t**2) / (1 + t**2)
    x = 2 * math.atan(float(signed)) % (2 * math.pi)
    check(abs(math.cos(x) - float(cos_x)) < 1e-9, "cos x differs from the float")
    check(abs(math.tan(x / 2) - float(signed)) < 1e-9, "tan(x/2) differs from the float")
    rng_text = "0 < x < π" if first_half else "π < x < 2π"
    wrong = [
        (-signed, "r3g-t-formula-half-angle-sign"),
        (t**2, "r3g-t-formula-root-not-taken"),
        (1 / signed, "r3g-t-formula-reciprocal"),
        (sp.sqrt((1 - cos_x) / 2) * (1 if first_half else -1), "r3g-t-formula-sine-half-angle-used"),
    ]
    text = typed(signed)
    return dict(
        text=f"Given that cos x = {typed(cos_x)} and {rng_text}, find the exact value of tan(x/2).",
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(signed),
        working=[
            "With t = tan(x/2), cos x = (1 - t²)/(1 + t²), so t² = (1 - cos x)/(1 + cos x).",
            f"t² = {typed(t**2)}. x/2 is in the {'first' if first_half else 'second'} quadrant, so t is {'positive' if first_half else 'negative'}.",
            f"tan(x/2) = {text}.",
        ],
        distractors=pick(text, [ex(w, g) for w, g in wrong], 3),
    )


def _medium_exact(rng):
    deg = rng.choice(tuple(_HALF_ANGLES))
    tan_x, t_pos = _HALF_ANGLES[deg]
    t = sp.symbols("t")
    # 2t/(1 - t²) = tan x  ->  tan x t² + 2t - tan x = 0
    roots = sp.solve(sp.Eq(2 * t, tan_x * (1 - t**2)), t)
    positive = [sp.simplify(r) for r in roots if r.is_positive]
    check(
        len(roots) == 2 and len(positive) == 1 and sp.simplify(positive[0] - t_pos) == 0, "roots differ from the table"
    )
    other = [sp.simplify(r) for r in roots if r.is_negative][0]
    check(abs(math.tan(math.radians(deg / 2)) - float(t_pos)) < 1e-9, "exact value differs from the float")
    wrong = [
        (other, "r3g-t-formula-wrong-root"),
        (tan_x / 2, "r3g-t-formula-halved-tangent"),
        (1 / t_pos, "r3g-t-formula-reciprocal"),
        (t_pos**2, "r3g-t-formula-root-not-taken"),
    ]
    text = typed(sp.radsimp(t_pos))
    return dict(
        text=(
            f"Use the substitution t = tan(x/2) with tan {deg}° = {typed(tan_x)} to find the exact value of "
            f"tan {deg / 2:g}°. Type surds as sqrt(3), for example."
        ),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(sp.radsimp(t_pos)),
        working=[
            f"tan {deg}° = 2t/(1 - t²) with t = tan {deg / 2:g}°, so {typed(tan_x)}(1 - t²) = 2t.",
            f"The quadratic has roots {typed(sp.radsimp(t_pos))} and {typed(sp.radsimp(other))}; {deg / 2:g}° is acute, so t > 0.",
        ],
        distractors=pick(text, [ex(sp.radsimp(w), g) for w, g in wrong], 3),
    )


def _medium(rng):
    return rng.choice([_medium_cos, _medium_cos, _medium_exact, _medium_exact])(rng)


def _equation_text(a, b, c) -> str:
    def term(coef, name, first):
        if coef == 0:
            return ""
        mag = abs(coef)
        shown = name if mag == 1 else f"{mag} {name}"
        return (f"-{shown}" if coef < 0 else shown) if first else f" {'-' if coef < 0 else '+'} {shown}"

    left = term(a, "sin x", True) + term(b, "cos x", a == 0)
    return f"{left} = {c}"


def _roots_pair(rng, positive_only: bool):
    for _ in attempts(300, "two distinct integer roots with a rounded solution well away from a tie"):
        lo, hi = (1, 6) if positive_only else (-7, 9)
        t1, t2 = rng.randint(lo, hi), rng.randint(lo, hi)
        if t1 >= t2 or 0 in (t1, t2) or t1 * t2 == 1 or t1 + t2 == 0:
            continue
        xs = [(2 * math.atan(t)) % (2 * math.pi) for t in (t1, t2)]
        if any(abs((x * 100) % 1 - 0.5) < 0.02 for x in xs):
            continue
        return t1, t2
    raise AssertionError  # pragma: no cover


def _setup(t1: int, t2: int):
    """a sin x + b cos x = c whose t = tan(x/2) quadratic has roots t1, t2."""
    return t1 + t2, 1 - t1 * t2, 1 + t1 * t2


def _check_equation(a, b, c, t1, t2):
    sols = sorted((2 * math.atan(t)) % (2 * math.pi) for t in (t1, t2))
    for x in sols:
        check(abs(a * math.sin(x) + b * math.cos(x) - c) < 1e-9, "a solution does not satisfy the equation")
    # scan for roots on a fine grid: exactly two sign changes
    changes, prev = 0, None
    for k in range(20000):
        x = 2 * math.pi * (k + 0.5) / 20000
        v = a * math.sin(x) + b * math.cos(x) - c
        if prev is not None and prev * v < 0:
            changes += 1
        prev = v
    check(changes == 2, f"{changes} sign changes in the scan, expected 2")
    return sols


def _hard(rng):
    t1, t2 = _roots_pair(rng, positive_only=False)
    a, b, c = _setup(t1, t2)
    sols = _check_equation(a, b, c, t1, t2)
    want_larger = rng.random() < 0.5
    exact = {tt: 2 * sp.atan(tt) + (2 * sp.pi if tt < 0 else 0) for tt in (t1, t2)}
    ordered = sorted((t1, t2), key=lambda tt: float(exact[tt]))
    t = ordered[1] if want_larger else ordered[0]
    other = ordered[0] if want_larger else ordered[1]
    shift = 2 * sp.pi if t < 0 else 0
    value = exact[t]
    check(abs(float(value) - sols[1 if want_larger else 0]) < 1e-9, "exact solution differs from the numeric one")
    text = typed(value)
    cands = [
        ex(sp.atan(t) + shift / 2, "r3g-t-formula-half-angle-not-doubled"),
        ex(2 * sp.atan(other) + (2 * sp.pi if other < 0 else 0), "r3g-t-formula-one-root-only"),
        ex(t, "r3g-t-formula-roots-given-as-solutions"),
        ex(2 * sp.atan(other), "r3g-t-formula-one-root-only"),
    ]
    if t < 0:
        cands.append(ex(2 * sp.atan(t), "r3g-t-formula-negative-solution-not-shifted"))
    cands = [d for d in cands if d.text != text]
    eq = _equation_text(a, b, c)
    which = "larger" if want_larger else "smaller"
    return dict(
        text=(
            f"Use t = tan(x/2) to solve {eq} for 0 ≤ x < 2π. Give the {which} of the two solutions exactly, "
            "typed like 2atan(3) + 2pi."
        ),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=[
            f"With sin x = 2t/(1 + t²) and cos x = (1 - t²)/(1 + t²), the equation becomes {c + b}t² - {2 * a}t + {c - b} = 0, "
            f"which has roots t = {t1} and t = {t2}.",
            f"x = 2 arctan t, taken into 0 ≤ x < 2π, gives x = {sols[0]:.2f} and x = {sols[1]:.2f}; the {which} is {text}.",
        ],
        distractors=pick(text, cands, 3),
    )


@generator(
    "r3g_t_formulae",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
@vary_symbol("x", ("x", "θ", "φ"))
def r3g_t_formulae(rng, tier="easy"):
    """The t = tan(x/2) substitution; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3g_exam_t_formulae", level=12, presets={"exam": dict(tier="exam")})
@retry_short
@vary_symbol("x", ("x", "θ", "φ"))
def r3g_exam_t_formulae(rng, tier="exam"):
    """Exam chain: the two roots of the quadratic in t and the solution of the equation each gives."""
    check(tier == "exam", f"unknown tier {tier!r}")
    t1, t2 = _roots_pair(rng, positive_only=True)
    a, b, c = _setup(t1, t2)
    sols = _check_equation(a, b, c, t1, t2)
    x1, x2 = sols
    check(
        abs(x1 - 2 * math.atan(t1)) < 1e-12 and abs(x2 - 2 * math.atan(t2)) < 1e-12, "solutions differ from 2 arctan t"
    )
    eq = _equation_text(a, b, c)

    def num_part(prompt, value, places, marks, working, wrongs, follow=None):
        fields = number_fields(value, places)
        d = dict(prompt=prompt, marks=marks, working=working, sensible_range=(0, None), **fields)
        d["distractors"] = pick(fields["answer_text"], [decimal_dist(w, places, g) for w, g in wrongs], 3)
        if follow:
            d["follow_through"] = follow
        return d

    return dict(
        text=[
            f"Consider the equation {eq}, where 0 ≤ x < 2π. Let t = tan(x/2), so that sin x = 2t/(1 + t²) and "
            "cos x = (1 - t²)/(1 + t²).",
            f"The equation {eq} is to be solved for 0 ≤ x < 2π using the substitution t = tan(x/2), where "
            "sin x = 2t/(1 + t²) and cos x = (1 - t²)/(1 + t²).",
            f"Solve {eq} for 0 ≤ x < 2π in stages. Write t = tan(x/2); then sin x = 2t/(1 + t²) and "
            "cos x = (1 - t²)/(1 + t²).",
        ][rng.randrange(3)],
        parts=[
            num_part(
                "The substitution gives a quadratic equation in t with two integer roots. Find the smaller root.",
                F(t1),
                0,
                1 + 1,
                [
                    f"The equation becomes {c + b}t² - {2 * a}t + {c - b} = 0, which is t² - {t1 + t2}t + {t1 * t2} = 0, so t = {t1} or t = {t2}."
                ],
                [
                    (t2, "r3g-t-formula-wrong-root"),
                    (-t1, "r3g-t-formula-root-sign-slip"),
                    (t1 + t2, "r3g-t-formula-roots-added"),
                    (t1 * t2, "r3g-t-formula-roots-added"),
                ],
            ),
            num_part(
                "Find the solution x of the equation that this root gives, correct to 2 decimal places.",
                F(str(round(x1, 8))),
                2,
                1,
                [f"x = 2 arctan t = 2 arctan {t1} = {x1:.2f}."],
                [
                    (math.atan(t1), "r3g-t-formula-half-angle-not-doubled"),
                    (2 * math.atan(t2), "r3g-t-formula-wrong-root"),
                    (math.degrees(2 * math.atan(t1)), "r3g-t-formula-degrees-for-radians"),
                ],
                "2*atan(part_a)",
            ),
            num_part(
                "Find the larger root of the quadratic in t.",
                F(t2),
                0,
                1,
                [f"The larger root is t = {t2}."],
                [
                    (t1, "r3g-t-formula-wrong-root"),
                    (-t2, "r3g-t-formula-root-sign-slip"),
                    (t1 + t2, "r3g-t-formula-roots-added"),
                    (t1 * t2, "r3g-t-formula-roots-added"),
                ],
            ),
            num_part(
                "Find the other solution of the equation, correct to 2 decimal places.",
                F(str(round(x2, 8))),
                2,
                1,
                [f"x = 2 arctan {t2} = {x2:.2f}."],
                [
                    (math.atan(t2), "r3g-t-formula-half-angle-not-doubled"),
                    (2 * math.atan(t1), "r3g-t-formula-wrong-root"),
                    (math.degrees(2 * math.atan(t2)), "r3g-t-formula-degrees-for-radians"),
                ],
                "2*atan(part_c)",
            ),
        ],
    )
