"""Generates `salg_reciprocal_trig_values` questions (easy, medium, hard) and the exam chain
`salg_exam_reciprocal_trig_values`.

Skill ``senior-define-the-reciprocal-trigonometric-functions`` (``trigonometry``, answer ``expression``): the exact
values of sec, cosec and cot at standard angles, from sec = 1/cos, cosec = 1/sin and cot = cos/sin.

* easy: sec, cosec or cot of 30°, 45° or 60°, in degrees.
* medium: the same functions at the standard angles in the other three quadrants, in degrees (120° to 330°), so the sign has to be decided as well as the value.
* hard: radians. The angle is a multiple of π/6, π/4 or π/3 that may be negative or more than a full turn, so it has to
  be reduced to a standard angle and the quadrant found before the value is.
* exam (``salg_exam_reciprocal_trig_values``): for a stated angle θ in radians outside the first quadrant, (a) sin θ;
  (b) cos θ; (c) cosec θ, using (a); (d) cot θ, using (a) and (b).

Every value is computed twice: exactly with SymPy's trigonometric functions of the rational multiple of π, and as a
float from the ``math`` module. A distractor is the value a learner gets from one named slip: not taking the
reciprocal, swapping the reciprocal functions, reading cot as tan, taking the sign from the wrong quadrant, or using
the complementary reference angle.
"""

from __future__ import annotations

import math
from functools import lru_cache

import sympy as sp

from ..answers import Distractor
from ..core import check, draw_until, generator, latex_answer, phrase, pick_distractors
from ._salg_kit import typed

_PI = sp.pi
_FUNCS = {
    "sin": lambda t: sp.sin(t),
    "cos": lambda t: sp.cos(t),
    "tan": lambda t: sp.tan(t),
    "sec": lambda t: 1 / sp.cos(t),
    "cosec": lambda t: 1 / sp.sin(t),
    "cot": lambda t: sp.cos(t) / sp.sin(t),
}
_FLOAT = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sec": lambda r: 1 / math.cos(r),
    "cosec": lambda r: 1 / math.sin(r),
    "cot": lambda r: math.cos(r) / math.sin(r),
}


@lru_cache(maxsize=None)
def value(fn: str, theta: sp.Expr) -> sp.Expr:
    """The exact value of `fn` at `theta` (radians), checked against the float route."""
    exact = sp.radsimp(sp.nsimplify(_FUNCS[fn](theta)))
    check(exact.is_real, f"{fn}({theta}) is not real")
    check(abs(float(exact) - _FLOAT[fn](float(theta))) < 1e-9, f"{fn}({theta}) differs from the float route")
    return exact


@lru_cache(maxsize=None)
def _clean(val: sp.Expr) -> sp.Expr:
    """`val` in simplest radical form; cached, because the same few exact values recur across questions."""
    return sp.radsimp(sp.nsimplify(val))


def defined(fn: str, theta: sp.Expr) -> bool:
    """Whether `fn` has a value at `theta`: sec needs cos != 0, cosec and cot need sin != 0."""
    denominator = {"sec": "cos", "cosec": "sin", "cot": "sin"}.get(fn)
    return denominator is None or abs(math.cos(float(theta)) if denominator == "cos" else math.sin(float(theta))) > 1e-9


def _reference(theta: sp.Expr):
    """The acute reference angle of a non-axis `theta`, or None on an axis."""
    turn = float(theta) % (2 * math.pi)
    if min(abs(turn - k * math.pi / 2) for k in range(5)) < 1e-9:
        return None
    phi = sp.Mod(theta, 2 * _PI)
    quarter = int(turn // (math.pi / 2))
    return [phi, _PI - phi, phi - _PI, 2 * _PI - phi][quarter]


def wrong_values(fn: str, theta: sp.Expr, true: sp.Expr) -> list[tuple[sp.Expr, str]]:
    """(value, error id) for the slips a learner makes finding `fn` at `theta`."""
    out: list[tuple[sp.Expr, str]] = []
    plain = {"sec": "cos", "cosec": "sin"}
    if fn in plain:
        out.append((value(plain[fn], theta), "recip-not-taken"))
        other = "cosec" if fn == "sec" else "sec"
        if defined(other, theta):
            out.append((value(other, theta), "recip-sec-cosec-swapped"))
    elif fn == "cot":
        if defined("cot", theta) and math.cos(float(theta)) != 0 and abs(math.cos(float(theta))) > 1e-9:
            out.append((value("tan", theta), "recip-cot-as-tan"))
        if defined("cosec", theta):
            out.append((value("cosec", theta), "recip-cot-as-cosec"))
    elif fn in ("sin", "cos"):
        out.append((value("cos" if fn == "sin" else "sin", theta), "trig-sin-cos-swapped"))
        out.append((value("cosec" if fn == "sin" else "sec", theta), "recip-reciprocal-given"))
    elif fn == "tan":
        out.append((value("cot", theta), "recip-cot-as-tan"))
    if true != 0:
        out.append((-true, "trig-sign-wrong-quadrant" if true > 0 else "trig-quadrant-sign-ignored"))
    ref = _reference(theta)
    if ref is not None:
        sign = 1 if true > 0 else -1
        comp = sp.Rational(1, 2) * _PI - ref
        for name in [fn] if fn in _FUNCS else []:
            if defined(name, comp):
                out.append((sign * abs(value(name, comp)), "trig-wrong-reference-angle"))
    return out


def distractors(fn, theta, true, count=3, extra=()):
    cands = []
    seen = [true]
    for val, tag in [*wrong_values(fn, theta, true), *extra]:
        val = _clean(val)
        if any(abs(float(val) - float(s)) < 1e-9 for s in seen):
            continue
        seen.append(val)
        cands.append(Distractor(typed(val), tag, typed(val)))
    picked = pick_distractors(typed(true), cands, count)
    check(len(picked) >= 2, f"too few distractors for {fn}({theta})")
    return picked


# -- angle text -------------------------------------------------------------------------------------------------


def deg_text(d: int) -> str:
    return f"{d}°"


def rad_text(frac: sp.Rational) -> str:
    """A rational multiple of π as 7π/6, -π/3, 2π, or 0."""
    if frac == 0:
        return "0"
    num, den = frac.p, frac.q
    mag = "π" if abs(num) == 1 else f"{abs(num)}π"
    sign = "-" if num < 0 else ""
    return f"{sign}{mag}" if den == 1 else f"{sign}{mag}/{den}"


def _fn_text(fn: str) -> str:
    return fn


def _say(expr) -> str:
    """A typed value as the working writes it: 2*sqrt(3)/3 -> 2√(3)/3."""
    return typed(expr).replace("*sqrt(", "√(").replace("sqrt(", "√(")


def _working(fn, angle_text, theta, true):
    sin_v, cos_v = value("sin", theta), value("cos", theta)
    lines = [f"sin {angle_text} = {_say(sin_v)} and cos {angle_text} = {_say(cos_v)}."]
    form = {"sec": "1/cos", "cosec": "1/sin", "cot": "cos/sin"}[fn]
    lines.append(f"{fn} {angle_text} = {form} {angle_text} = {_say(true)}.")
    return lines


_STEMS = (
    "Find the exact value of {f} {a}.",
    "Evaluate {f} {a}, giving the exact value.",
    "Write down the exact value of {f} {a}.",
)


def _question(rng, fn, theta, angle_text):
    true = value(fn, theta)
    check(true != 0 or fn == "cot", "unexpected zero")
    angle = angle_text if "/" not in angle_text else f"({angle_text})"
    return dict(
        text=phrase(rng, _STEMS, f=fn, a=angle),
        answer_kind="expression",
        answer=typed(true),
        answer_text=typed(true),
        answer_latex=latex_answer(true),
        working=_working(fn, angle, theta, true),
        distractors=distractors(fn, theta, true),
    )


_FNS = ("sec", "cosec", "cot")


def _degree_question(rng, angles):
    def draw(r):
        return r.choice(_FNS), r.choice(angles)

    fn, d = draw_until(
        rng, draw, lambda fd: defined(fd[0], sp.Rational(fd[1], 180) * _PI), what="a defined reciprocal value"
    )
    theta = sp.Rational(d, 180) * _PI
    return _question(rng, fn, theta, deg_text(d))


def _easy(rng):
    return _degree_question(rng, [30, 45, 60])


def _medium(rng):
    return _degree_question(rng, [120, 135, 150, 210, 225, 240, 300, 315, 330])


def _hard(rng):
    def draw(r):
        den = r.choice([3, 4, 6])
        num = r.choice([n for n in range(-2 * den * 2, 4 * den + 1) if n != 0])
        return r.choice(_FNS), sp.Rational(num, den)

    fn, frac = draw_until(
        rng,
        draw,
        lambda fd: defined(fd[0], fd[1] * _PI) and not (-1 <= fd[1] <= 1) and (fd[1] * 2).q != 1,
        what="a reduced radian angle",
    )
    return _question(rng, fn, frac * _PI, rad_text(frac))


@generator(
    "salg_reciprocal_trig_values",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def salg_reciprocal_trig_values(rng, tier="easy"):
    """The exact value of sec, cosec or cot at a standard angle; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _exam_angles() -> list[sp.Rational]:
    """Multiples of π/3, π/4 and π/6 between -2π and 2π that are not on an axis and whose reduced angle is not in
    the first quadrant (the first quadrant is what the practice tiers cover)."""
    out = set()
    for den in (3, 4, 6):
        for num in range(-3 * den, 3 * den + 1):
            frac = sp.Rational(num, den)
            if (frac * 2).q == 1 or frac in out:
                continue
            if 0 < float(frac % 2) < 0.5:
                continue
            out.add(frac)
    return sorted(out)


_EXAM_ANGLES = _exam_angles()


_FIRST_PROMPTS = (
    "Find the exact value of sin θ.",
    "Write down the exact value of sin θ.",
    "Use the unit circle to find the exact value of sin θ.",
)


@generator("salg_exam_reciprocal_trig_values", level=11, presets={"exam": dict(tier="exam")})
def salg_exam_reciprocal_trig_values(rng, tier="exam"):
    """Exam chain: sin θ, cos θ, hence cosec θ, hence cot θ at a standard angle outside the first quadrant."""
    check(tier == "exam", f"unknown tier {tier!r}")
    frac = rng.choice(_EXAM_ANGLES)
    theta = frac * _PI
    at = rad_text(frac)
    s, c = value("sin", theta), value("cos", theta)
    cosec, cot = value("cosec", theta), value("cot", theta)
    check(sp.simplify(1 / s - cosec) == 0 and sp.simplify(c / s - cot) == 0, "reciprocal values differ")

    def part(prompt, fn, true, marks, ft=None, working=()):
        p = dict(
            prompt=prompt,
            answer_kind="expression",
            answer=typed(true),
            answer_text=typed(true),
            answer_latex=latex_answer(true),
            marks=marks,
            working=list(working),
            distractors=distractors(fn, theta, true),
        )
        if ft:
            p["follow_through"] = ft
        return p

    return dict(
        text=f"Let θ = {at}.",
        parts=[
            part(phrase(rng, _FIRST_PROMPTS), "sin", s, 1, working=[f"sin θ = {_say(s)}."]),
            part("Find the exact value of cos θ.", "cos", c, 1, working=[f"cos θ = {_say(c)}."]),
            part(
                "Hence find the exact value of cosec θ.",
                "cosec",
                cosec,
                2,
                ft="1/part_a",
                working=[f"cosec θ = 1/sin θ = 1/({_say(s)}) = {_say(cosec)}."],
            ),
            part(
                "Hence find the exact value of cot θ.",
                "cot",
                cot,
                2,
                ft="part_b/part_a",
                working=[f"cot θ = cos θ / sin θ = ({_say(c)}) / ({_say(s)}) = {_say(cot)}."],
            ),
        ],
    )
