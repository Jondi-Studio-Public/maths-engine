"""Generates `r3g_compound_angle` questions (easy, medium, hard) and the exam chain `r3g_exam_compound_angle`.

Skill ``trig-compound-angle-exact`` (``trigonometry``, answer ``expression``): exact values from the sum, difference and
double angle formulas for sine, cosine and tangent, typed as a surd or a fraction. The existing proof skills ask for the
identities themselves; this one uses them.

* easy: the exact value of sin or cos at 15°, 75°, 105° or 165° (or the same angle in radians), found by splitting the
  angle into two standard angles.
* medium: sin(A + B), cos(A + B), sin(A - B) or cos(A - B) when sin A and cos B are given as fractions and both angles
  are acute, so every other ratio is found from a Pythagorean triple.
* hard: tan(A + B) or cos(A + B) when A and B lie in stated quadrants, so signs have to be chosen; or sin 2x, cos 2x or
  tan 2x from one ratio and the quadrant of x.
* exam (``r3g_exam_compound_angle``): A in the second quadrant with sin A given and B in the fourth quadrant with cos B
  given; (a) cos A, (b) sin B, (c) sin(A + B) and (d) cos(A + B), both following through from (a) and (b).

Each result is confirmed by building the angles with the ``math`` module and evaluating the compound expression directly.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..core import attempts, check, generator, phrase
from ._r3g_kit import ex as _ex
from ._r3g_kit import latex, pick, retry_short, typed

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]
_S2, _S3 = sp.sqrt(2), sp.sqrt(3)
# sin, cos of 30, 45, 60 degrees
_SC = {
    30: (sp.Rational(1, 2), _S3 / 2),
    45: (_S2 / 2, _S2 / 2),
    60: (_S3 / 2, sp.Rational(1, 2)),
}
R = sp.Rational


def clean(expr):
    """`expr` with surds expanded and any surd in a denominator rationalised: sqrt(3)/(3 - sqrt(3)) + ... -> sqrt(3) + 2."""
    return sp.expand(sp.radsimp(sp.expand(sp.sympify(expr))))


def ex(expr, tag):
    return _ex(clean(expr), tag)


def _deg_text(deg: int, radians: bool) -> str:
    if not radians:
        return f"{deg}°"
    from ._r3g_kit import pi_shown

    return f"({pi_shown(Fraction(deg, 180))})"


def _split(deg: int):
    """(A, sign, B) with A, B in {30, 45, 60} and deg = A + sign*B, A != B."""
    table = {15: (60, -1, 45), 75: (45, 1, 30), 105: (60, 1, 45), 165: (60, 1, 45)}
    if deg == 165:
        return None
    return table[deg]


def _compound(fn: str, a: int, sign: int, b: int):
    sa, ca = _SC[a]
    sb, cb = _SC[b]
    if fn == "sin":
        return sa * cb + sign * ca * sb
    if fn == "cos":
        return ca * cb - sign * sa * sb
    ta, tb = sa / ca, sb / cb
    return (ta + sign * tb) / (1 - sign * ta * tb)


def _exact(fn: str, deg: int):
    return {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan}[fn](sp.pi * R(deg, 180))


def _angle_value_float(fn: str, deg: int) -> float:
    return getattr(math, fn)(math.radians(deg))


_STEM_EASY = (
    "Find the exact value of {e}. Type surds as sqrt(3), for example.",
    "Evaluate {e} exactly. Type surds as sqrt(3), for example.",
)


def _easy(rng):
    fn = rng.choice(("sin", "cos", "tan"))
    deg = rng.choice((15, 75, 105))
    mult = rng.choice((1, 1, 2, 4))
    a, sign, b = _split(deg)
    base = _compound(fn, a, sign, b)
    check(sp.simplify(base - _exact(fn, deg)) == 0, "compound value differs from SymPy")
    check(abs(float(base) - _angle_value_float(fn, deg)) < 1e-9, "compound value differs from the float")
    value = mult * base
    radians = rng.random() < 0.4
    label = f"{fn} {_deg_text(deg, radians)}" if not radians else f"{fn}{_deg_text(deg, True)}"
    if mult != 1:
        label = f"{mult} {label}"
    text = typed(clean(value))
    sa, ca = _SC[a]
    sb, cb = _SC[b]
    other = {"sin": "cos", "cos": "sin", "tan": "tan"}[fn]
    cands = [
        ex(sp.expand(mult * _compound(fn, a, -sign, b)), "r3g-compound-sign-slip"),
        ex(sp.expand(-value), "r3g-compound-wrong-quadrant-sign"),
    ]
    if fn == "sin":
        cands += [
            ex(sp.expand(mult * (sa + sb)), "r3g-compound-values-added"),
            ex(sp.expand(mult * sa * sb), "r3g-compound-single-product"),
        ]
    elif fn == "cos":
        cands += [
            ex(sp.expand(mult * (ca + cb)), "r3g-compound-values-added"),
            ex(sp.expand(mult * ca * cb), "r3g-compound-single-product"),
        ]
    else:
        ta, tb = sa / ca, sb / cb
        cands += [
            ex(sp.expand(mult * (ta + sign * tb)), "r3g-compound-tan-denominator-dropped"),
            ex(sp.expand(mult * (ta + sign * tb) / (1 + sign * ta * tb)), "r3g-compound-sign-slip"),
        ]
    if fn != "tan":
        cands.append(ex(sp.expand(mult * _compound(other, a, sign, b)), "r3g-compound-sin-cos-swapped"))
    if mult != 1:
        cands.append(ex(sp.expand(base), "r3g-compound-multiplier-dropped"))
    op = "+" if sign > 0 else "-"
    formula = {
        "sin": f"sin {a}° cos {b}° {op} cos {a}° sin {b}°",
        "cos": f"cos {a}° cos {b}° {'-' if sign > 0 else '+'} sin {a}° sin {b}°",
        "tan": f"(tan {a}° {op} tan {b}°) ÷ (1 {'-' if sign > 0 else '+'} tan {a}° tan {b}°)",
    }[fn]
    working = [f"Write {deg}° = {a}° {op} {b}°.", f"{fn} {deg}° = {formula} = {typed(clean(base))}."]
    if mult != 1:
        working.append(f"Multiplying by {mult} gives {text}.")
    return dict(
        text=phrase(rng, _STEM_EASY, e=label),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(clean(value)),
        working=working,
        distractors=pick(text, cands, 3),
    )


def _triple_pair(rng):
    for _ in attempts(100, "two different triples"):
        t1, t2 = rng.choice(_TRIPLES), rng.choice(_TRIPLES)
        if t1 != t2:
            return t1, t2
    raise AssertionError  # pragma: no cover


def _angle_vals(sin_a, cos_a, sin_b, cos_b, fn: str, sign: int):
    if fn == "sin":
        return sin_a * cos_b + sign * cos_a * sin_b
    return cos_a * cos_b - sign * sin_a * sin_b


def _float_compound(fn: str, a_rad: float, sign: int, b_rad: float) -> float:
    angle = a_rad + sign * b_rad
    return getattr(math, fn)(angle)


def _medium(rng):
    (p1, q1, h1), (p2, q2, h2) = _triple_pair(rng)
    if rng.random() < 0.5:
        p1, q1 = q1, p1
    sin_a, cos_a = R(p1, h1), R(q1, h1)
    cos_b, sin_b = R(p2, h2), R(q2, h2)
    fn = rng.choice(("sin", "cos"))
    sign = rng.choice((1, -1))
    value = _angle_vals(sin_a, cos_a, sin_b, cos_b, fn, sign)
    a_rad, b_rad = math.asin(float(sin_a)), math.acos(float(cos_b))
    check(abs(float(value) - _float_compound(fn, a_rad, sign, b_rad)) < 1e-12, "value differs from the float")
    op = "+" if sign > 0 else "-"
    other = _angle_vals(sin_a, cos_a, sin_b, cos_b, fn, -sign)
    other_fn = "cos" if fn == "sin" else "sin"
    text = typed(value)
    cands = [
        ex(other, "r3g-compound-sign-slip"),
        ex(sin_a * cos_b if fn == "sin" else cos_a * cos_b, "r3g-compound-single-product"),
        ex(sin_a + sin_b if fn == "sin" else cos_a + cos_b, "r3g-compound-values-added"),
        ex(_angle_vals(sin_a, cos_a, sin_b, cos_b, other_fn, sign), "r3g-compound-formula-of-other-function"),
    ]
    stem = (
        f"Angle A is acute with sin A = {typed(sin_a)} and angle B is acute with cos B = {typed(cos_b)}. "
        f"Find the exact value of {fn}(A {op} B)."
    )
    working = [
        f"cos A = {typed(cos_a)} and sin B = {typed(sin_b)}, both positive because the angles are acute.",
        (
            f"sin(A {op} B) = sin A cos B {op} cos A sin B"
            if fn == "sin"
            else f"cos(A {op} B) = cos A cos B {'-' if sign > 0 else '+'} sin A sin B"
        ),
        f"{fn}(A {op} B) = {text}.",
    ]
    return dict(
        text=stem,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=working,
        distractors=pick(text, cands, 3),
    )


_QUADS = {
    2: ("second", 1, -1),
    3: ("third", -1, -1),
    4: ("fourth", -1, 1),
}


def _hard_sum(rng):
    for _ in attempts(300, "tan of a sum with a finite value"):
        (p1, q1, h1), (p2, q2, h2) = _triple_pair(rng)
        qa, qb = rng.choice((2, 3, 4)), rng.choice((2, 3, 4))
        a_name, sa, ca = _QUADS[qa]
        b_name, sb, cb = _QUADS[qb]
        if rng.random() < 0.5:
            p1, q1 = q1, p1
        sin_a, cos_a = sa * R(p1, h1), ca * R(q1, h1)
        sin_b, cos_b = sb * R(p2, h2), cb * R(q2, h2)
        fn = rng.choice(("tan", "cos"))
        sign = rng.choice((1, -1))
        if fn == "tan":
            ta, tb = sin_a / cos_a, sin_b / cos_b
            if 1 + sign * ta * tb == 0:
                continue
            value = (ta + sign * tb) / (1 - sign * ta * tb)
            # given: sin A and cos B (so each of the other ratios needs the quadrant)
        else:
            value = cos_a * cos_b - sign * sin_a * sin_b
        if value == 0:
            continue
        break
    a_rad = (math.pi - math.asin(float(sin_a))) if cos_a < 0 else math.asin(float(sin_a))
    b_rad = math.acos(float(cos_b)) * (1 if sin_b >= 0 else -1)
    angle = a_rad + sign * b_rad
    float_value = math.tan(angle) if fn == "tan" else math.cos(angle)
    check(abs(float(value) - float_value) < 1e-9, "value differs from the float")
    op = "+" if sign > 0 else "-"
    text = typed(value)
    # wrong quadrant signs
    pos_cos_a = abs(cos_a)
    pos_sin_b = abs(sin_b)
    if fn == "tan":
        ta_pos, tb_pos = sin_a / pos_cos_a, pos_sin_b / cos_b
        cands = [
            ex(
                (ta_pos + sign * tb_pos) / (1 - sign * ta_pos * tb_pos) if 1 - sign * ta_pos * tb_pos != 0 else None,
                "r3g-compound-quadrant-sign-ignored",
            ),
            ex(
                (sin_a / cos_a - sign * sin_b / cos_b) / (1 + sign * sin_a / cos_a * sin_b / cos_b)
                if 1 + sign * ta * tb != 0
                else None,
                "r3g-compound-sign-slip",
            ),
            ex(ta + sign * tb, "r3g-compound-tan-denominator-dropped"),
            ex(-value, "r3g-compound-wrong-quadrant-sign"),
        ]
        cands = [c for c in cands if c.text != "zoo"]
        what = f"tan(A {op} B)"
        working = [
            f"cos A = {typed(cos_a)} (A is in the {a_name} quadrant) and sin B = {typed(sin_b)} (B is in the {b_name} quadrant).",
            f"tan A = {typed(ta)} and tan B = {typed(tb)}.",
            f"tan(A {op} B) = (tan A {op} tan B) ÷ (1 {'-' if sign > 0 else '+'} tan A tan B) = {text}.",
        ]
    else:
        cands = [
            ex(abs(cos_a) * cos_b - sign * sin_a * sin_b, "r3g-compound-quadrant-sign-ignored"),
            ex(cos_a * cos_b + sign * sin_a * sin_b, "r3g-compound-sign-slip"),
            ex(cos_a * cos_b, "r3g-compound-single-product"),
            ex(-value, "r3g-compound-wrong-quadrant-sign"),
        ]
        what = f"cos(A {op} B)"
        working = [
            f"cos A = {typed(cos_a)} (A is in the {a_name} quadrant) and sin B = {typed(sin_b)} (B is in the {b_name} quadrant).",
            f"cos(A {op} B) = cos A cos B {'-' if sign > 0 else '+'} sin A sin B = {text}.",
        ]
    stem = (
        f"Angle A is in the {a_name} quadrant with sin A = {typed(sin_a)}, and angle B is in the {b_name} quadrant "
        f"with cos B = {typed(cos_b)}. Find the exact value of {what}."
    )
    return dict(
        text=stem,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=working,
        distractors=pick(text, cands, 3),
    )


def _hard_double(rng):
    for _ in attempts(300, "a double angle with a nonzero value"):
        p, q, h = rng.choice(_TRIPLES)
        quad = rng.choice((2, 3, 4))
        _name, ss, cs = _QUADS[quad]
        if rng.random() < 0.5:
            p, q = q, p
        sin_x, cos_x = ss * R(p, h), cs * R(q, h)
        given = rng.choice(("sin", "cos"))
        ask = rng.choice(("sin 2x", "cos 2x", "tan 2x"))
        t = sin_x / cos_x
        value = {
            "sin 2x": 2 * sin_x * cos_x,
            "cos 2x": cos_x**2 - sin_x**2,
            "tan 2x": 2 * t / (1 - t**2) if t**2 != 1 else None,
        }[ask]
        if value is not None and value != 0:
            break
    x_rad = math.atan2(float(sin_x), float(cos_x))
    fv = {"sin 2x": math.sin(2 * x_rad), "cos 2x": math.cos(2 * x_rad), "tan 2x": math.tan(2 * x_rad)}[ask]
    check(abs(float(value) - fv) < 1e-9, "double angle value differs from the float")
    ranges = {2: "π/2 < x < π", 3: "π < x < 3π/2", 4: "3π/2 < x < 2π"}
    given_text = f"{given} x = {typed(sin_x if given == 'sin' else cos_x)}"
    text = typed(value)
    # wrong: other ratio with the wrong sign
    flipped_sin = -sin_x if given == "cos" else sin_x
    flipped_cos = -cos_x if given == "sin" else cos_x
    tw = flipped_sin / flipped_cos
    alt = {
        "sin 2x": 2 * flipped_sin * flipped_cos,
        "cos 2x": flipped_cos**2 - flipped_sin**2,
        "tan 2x": 2 * tw / (1 - tw**2) if tw**2 != 1 else None,
    }[ask]
    cands = [
        ex(alt, "r3g-compound-quadrant-sign-ignored"),
        ex(2 * (sin_x if given == "sin" else cos_x), "r3g-compound-double-the-ratio"),
        ex(-value, "r3g-compound-wrong-quadrant-sign"),
    ]
    if ask == "sin 2x":
        cands.append(ex(sin_x * cos_x, "r3g-compound-double-angle-formula-slip"))
    elif ask == "tan 2x":
        cands.append(ex(2 * t, "r3g-compound-double-angle-formula-slip"))
    else:
        cands.append(ex(2 * cos_x**2, "r3g-compound-double-angle-formula-slip"))
    stem = f"Given that {given_text} and {ranges[quad]}, find the exact value of {ask}."
    other = "cos" if given == "sin" else "sin"
    working = [
        f"x is in the {_QUADS[quad][0]} quadrant, so {other} x = {typed(cos_x if given == 'sin' else sin_x)}.",
        {
            "sin 2x": f"sin 2x = 2 sin x cos x = {text}.",
            "cos 2x": f"cos 2x = cos² x - sin² x = {text}.",
            "tan 2x": f"tan x = {typed(t)}, so tan 2x = 2 tan x ÷ (1 - tan² x) = {text}.",
        }[ask],
    ]
    return dict(
        text=stem,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=working,
        distractors=pick(text, cands, 3),
    )


@generator(
    "r3g_compound_angle",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3g_compound_angle(rng, tier="easy"):
    """Exact values from compound and double angle formulas; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_sum, _hard_sum, _hard_double])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3g_exam_compound_angle", level=12, presets={"exam": dict(tier="exam")})
@retry_short
def r3g_exam_compound_angle(rng, tier="exam"):
    """Exam chain: cos A, sin B, then sin(A + B) and cos(A + B) following through from them."""
    check(tier == "exam", f"unknown tier {tier!r}")
    (p1, q1, h1), (p2, q2, h2) = _triple_pair(rng)
    if rng.random() < 0.5:
        p1, q1 = q1, p1
    sin_a = R(p1, h1)
    cos_a = -R(q1, h1)
    cos_b = R(p2, h2)
    sin_b = -R(q2, h2)
    sin_ab = sin_a * cos_b + cos_a * sin_b
    cos_ab = cos_a * cos_b - sin_a * sin_b
    a_rad = math.pi - math.asin(float(sin_a))
    b_rad = -math.acos(float(cos_b))
    check(abs(float(sin_ab) - math.sin(a_rad + b_rad)) < 1e-9, "sin(A + B) differs from the float")
    check(abs(float(cos_ab) - math.cos(a_rad + b_rad)) < 1e-9, "cos(A + B) differs from the float")
    ft_sin = f"({typed(sin_a)})*({typed(cos_b)}) + part_a*part_b"
    ft_cos = f"part_a*({typed(cos_b)}) - ({typed(sin_a)})*part_b"

    def part(prompt, value, marks, working, cands, follow=None):
        text = typed(value)
        d = dict(
            prompt=prompt,
            answer_kind="expression",
            answer=text,
            answer_text=text,
            answer_latex=latex(value),
            marks=marks,
            working=working,
            distractors=pick(text, cands, 3),
        )
        if follow:
            d["follow_through"] = follow
        return d

    variant = rng.randrange(3)
    return dict(
        text=(
            f"Angle A is in the second quadrant with sin A = {typed(sin_a)}. Angle B is in the fourth quadrant "
            f"with cos B = {typed(cos_b)}."
            if variant == 0
            else f"Given sin A = {typed(sin_a)} with A obtuse, and cos B = {typed(cos_b)} with B in the fourth quadrant."
            if variant == 1
            else f"Angles A and B satisfy sin A = {typed(sin_a)} and cos B = {typed(cos_b)}, where A lies in the second quadrant and B in the fourth."
        ),
        parts=[
            part(
                "Find the exact value of cos A.",
                cos_a,
                1,
                [
                    f"cos² A = 1 - {typed(sin_a**2)} = {typed(cos_a**2)}; cosine is negative in the second quadrant, so cos A = {typed(cos_a)}."
                ],
                [
                    ex(-cos_a, "r3g-compound-quadrant-sign-ignored"),
                    ex(sin_a, "r3g-compound-sin-cos-swapped"),
                    ex(1 - sin_a, "r3g-compound-pythagorean-slip"),
                    ex(cos_a**2, "r3g-compound-root-not-taken"),
                ],
            ),
            part(
                "Find the exact value of sin B.",
                sin_b,
                1,
                [
                    f"sin² B = 1 - {typed(cos_b**2)} = {typed(sin_b**2)}; sine is negative in the fourth quadrant, so sin B = {typed(sin_b)}."
                ],
                [
                    ex(-sin_b, "r3g-compound-quadrant-sign-ignored"),
                    ex(cos_b, "r3g-compound-sin-cos-swapped"),
                    ex(1 - cos_b, "r3g-compound-pythagorean-slip"),
                    ex(sin_b**2, "r3g-compound-root-not-taken"),
                ],
            ),
            part(
                "Hence find the exact value of sin(A + B).",
                sin_ab,
                2,
                [f"sin(A + B) = sin A cos B + cos A sin B = {typed(sin_ab)}."],
                [
                    ex(sin_a * cos_b - cos_a * sin_b, "r3g-compound-sign-slip"),
                    ex(sin_a * cos_b + abs(cos_a) * abs(sin_b), "r3g-compound-quadrant-sign-ignored"),
                    ex(sin_a + sin_b, "r3g-compound-values-added"),
                    ex(sin_a * cos_b, "r3g-compound-single-product"),
                ],
                ft_sin,
            ),
            part(
                "Hence find the exact value of cos(A + B).",
                cos_ab,
                2,
                [f"cos(A + B) = cos A cos B - sin A sin B = {typed(cos_ab)}."],
                [
                    ex(cos_a * cos_b + sin_a * sin_b, "r3g-compound-sign-slip"),
                    ex(abs(cos_a) * cos_b - sin_a * abs(sin_b), "r3g-compound-quadrant-sign-ignored"),
                    ex(cos_a + cos_b, "r3g-compound-values-added"),
                    ex(cos_a * cos_b, "r3g-compound-single-product"),
                ],
                ft_cos,
            ),
        ],
    )
