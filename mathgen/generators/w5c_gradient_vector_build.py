"""Generates `w5c_gradient_vector` questions and the exam chain `exd_x_w5c_gradient_vector`.

``gradient-vector`` (``partial-derivatives``): the gradient of a function of two variables, its value at a point, and
what the gradient says about the surface: the direction of steepest ascent, the rate of change in a stated
direction, the point where the gradient takes a given value, and the direction along a level curve.

The skill used to be deferred because its answer was a ``list`` of expressions, which no grader reads. Here every
single-answer question is a ``pair``: the value of the gradient at a point, written ``(a, b)``, a unit vector written
to 2 decimal places, a point, or a labelled pair ``D = ..., M = ...``. The exam chain is four typed numbers with
follow-through.

Tiers (topic ``w5c_gradient_vector``):

* easy   -- the gradient at a point of a polynomial in x and y whose terms mix the two variables;
* medium -- the gradient at a point of a function that needs the chain, product or quotient rule: ``ln(x^2 + y^2)``,
  ``u e^(g)``, ``sqrt(x^2 + y^2)`` or a ratio of linear expressions. The point is chosen so that the exponential,
  logarithm or root is a clean value;
* hard   -- one of four questions that use the gradient: the unit vector of steepest ascent (or descent); the point
  where the gradient equals a stated vector; the unit tangent to the level curve; the rate of change in a stated
  direction together with the greatest rate of change, as a labelled pair.

The exam tier (topic ``exd_x_w5c_gradient_vector``): (a) f_x at P, (b) f_y at P, (c) the rate of change from P towards
a second point Q (follow-through from (a) and (b)), (d) the greatest rate of increase at P (follow-through).

Everything is built backwards: the hard tier fixes the gradient first (a whole-number multiple of a Pythagorean
vector, so its length and its unit vector are clean) and then sets the linear terms of the function to produce it.
Everything is exact (integers and Fractions); `tests/test_w5c_gradient_vector.py` re-derives every key with SymPy.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    VerificationError,
    attempts,
    check,
    exact_decimal_string,
    generator,
    join_terms,
    paren_negative,
    pick_distractors,
    prompt_safe_str,
    round_half_up,
    stated_precision_string,
)
from .exd_x_r2k_support import number_part, retry_build

_TOL = Fraction(51, 1000)

# ---------------------------------------------------------------------------
# Exact arithmetic helpers


def _mono(px: int, py: int) -> str:
    return ("x" if px == 1 else f"x^{px}" if px else "") + ("y" if py == 1 else f"y^{py}" if py else "")


def _poly_grad(terms, x0: int, y0: int) -> tuple[Fraction, Fraction]:
    """Exact (f_x, f_y) at (x0, y0) of a sum of c x^p y^q terms given as (c, p, q)."""
    gx = sum(c * p * x0 ** (p - 1) * y0**q for c, p, q in terms if p)
    gy = sum(c * q * x0**p * y0 ** (q - 1) for c, p, q in terms if q)
    return Fraction(gx), Fraction(gy)


def _poly_partial_text(terms, wrt: str) -> str:
    out = []
    for c, p, q in terms:
        if wrt == "x" and p:
            out.append((c * p, _mono(p - 1, q)))
        if wrt == "y" and q:
            out.append((c * q, _mono(p, q - 1)))
    return join_terms(out)


def _decimal(value: Fraction) -> str:
    return exact_decimal_string(value, 3)


def _pair_text(a: Fraction, b: Fraction) -> str:
    return f"({_decimal(a)}, {_decimal(b)})"


def _pair_text_dp(a: Fraction, b: Fraction, dp: int) -> str:
    return f"({stated_precision_string(round_half_up(a, dp), dp)}, {stated_precision_string(round_half_up(b, dp), dp)})"


def _p2(value: Fraction) -> str:
    text = stated_precision_string(value, 2)
    return f"({text})" if value < 0 else text


def _root_exact(n: int) -> Fraction | None:
    r = math.isqrt(n)
    return Fraction(r) if r * r == n else None


def _clear(value: Fraction, dp: int = 2, margin: Fraction = Fraction(1, 10)) -> bool:
    scaled = abs(value) * 10**dp
    frac = scaled - math.floor(scaled)
    return abs(frac - Fraction(1, 2)) > margin


def _pair_distractors(
    answer: tuple[Fraction, Fraction], cands: list[tuple[tuple[Fraction, Fraction], str]], text_of, keep: int = 3
) -> list[Distractor]:
    ax, ay = answer
    out: list[Distractor] = []
    seen = {text_of(ax, ay)}
    for (cx, cy), tag in cands:
        if abs(cx - ax) <= _TOL and abs(cy - ay) <= _TOL:
            continue
        text = text_of(cx, cy)
        if text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, {"x": float(cx), "y": float(cy)}))
    return out[:keep] if len(out) >= 2 else out


def _pair_question(text, answer, dists, working, text_of=_pair_text) -> dict:
    check(len(dists) >= 2, "too few gradient distractors")
    return dict(
        text=text,
        answer={"x": _num(answer[0]), "y": _num(answer[1])},
        answer_text=text_of(*answer),
        answer_kind="pair",
        working=working,
        distractors=dists,
    )


def _num(value: Fraction):
    return int(value) if value.denominator == 1 else float(value)


# ---------------------------------------------------------------------------
# Easy: a polynomial with mixed terms

_POLY_POOL = [
    (3, 0, False),
    (2, 1, True),
    (1, 2, True),
    (0, 3, False),
    (2, 0, False),
    (1, 1, True),
    (0, 2, False),
    (1, 0, False),
    (0, 1, False),
]

_SETTINGS = [
    (
        "The temperature, in degrees Celsius, of a heated metal plate at a point (x, y) centimetres from one corner is",
        "T",
    ),
    ("The height, in metres, of a hillside above a point (x, y) kilometres east and north of a campsite is", "h"),
    ("The concentration of a pollutant, in milligrams per litre, at a point (x, y) metres across a pond is", "c"),
]


def _easy_question(rng: random.Random) -> dict:
    for _ in attempts(2000, "an easy gradient question"):
        picks = sorted(rng.sample(range(len(_POLY_POOL)), 4))
        if not any(_POLY_POOL[i][2] for i in picks):
            continue
        terms = [(rng.choice((-4, -3, -2, -1, 1, 2, 3, 4, 5)), _POLY_POOL[i][0], _POLY_POOL[i][1]) for i in picks]
        mixed = [t for t, i in zip(terms, picks) if _POLY_POOL[i][2]]
        pure = [t for t, i in zip(terms, picks) if not _POLY_POOL[i][2]]
        ftext = join_terms([(c, _mono(p, q)) for c, p, q in terms])
        x0, y0 = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
        gx, gy = _poly_grad(terms, x0, y0)
        if gx == 0 or gy == 0 or gx == gy or abs(gx) > 70 or abs(gy) > 70:
            continue
        # the wrong readings, built from the same terms
        gx_drop, gy_drop = _poly_grad(pure, x0, y0)
        # product rule applied to BOTH variables in each partial of a mixed term
        both_x, both_y = _poly_grad(mixed, x0, y0)
        mx = sum(c * q * x0**p * y0 ** (q - 1) for c, p, q in mixed if q)
        my = sum(c * p * x0 ** (p - 1) * y0**q for c, p, q in mixed if p)
        tx = gx_drop + both_x + mx
        ty = gy_drop + both_y + my
        cands = [
            ((gy, gx), "gradient-vector-components-swapped"),
            ((gx, gx), "gradient-partial-repeated"),
            ((gy, gy), "gradient-partial-repeated"),
            ((gx_drop, gy_drop), "gradient-term-omitted"),
            ((tx, ty), "partial-other-variable-also-differentiated"),
        ]
        dists = _pair_distractors((gx, gy), cands, _pair_text, 4)
        if len(dists) >= 3:
            break
    setting, name = rng.choice(_SETTINGS)
    shape = rng.randrange(3)
    pt = f"({x0}, {y0})"
    if shape == 0:
        text = f"Let f(x, y) = {ftext}. Find the gradient vector ∇f at the point {pt}, written as (f_x, f_y)."
    elif shape == 1:
        text = f"{setting} {name}(x, y) = {ftext}. Find ∇{name} at {pt}, written as an ordered pair."
    else:
        text = f"For f(x, y) = {ftext}, calculate ∇f({x0}, {y0}). Give the two components as (f_x, f_y)."
    return _pair_question(
        text,
        (gx, gy),
        dists,
        [
            f"f_x = {_poly_partial_text(terms, 'x')} and f_y = {_poly_partial_text(terms, 'y')}.",
            f"At {pt}: ∇f = {_pair_text(gx, gy)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Medium: chain, product and quotient rules at a point where the special function is clean

_LOG_POINTS = [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
    (2, 6),
    (6, 2),
    (5, 5),
    (6, 8),
    (8, 6),
]
_ROOT_POINTS = [(3, 4), (4, 3), (6, 8), (8, 6)]


def _signed(rng, pt):
    return rng.choice((-1, 1)) * pt[0], rng.choice((-1, 1)) * pt[1]


def _log_template(rng):
    c = rng.randint(1, 5)
    m = rng.choice((-3, -2, -1, 1, 2, 3))
    x0, y0 = _signed(rng, rng.choice(_LOG_POINTS))
    ftext = join_terms([(c, "ln(x^2 + y^2)"), (m, "xy")])
    s = x0 * x0 + y0 * y0
    grad = (Fraction(2 * c * x0, s) + m * y0, Fraction(2 * c * y0, s) + m * x0)
    wrong_inner = (Fraction(c * x0, s) + m * y0, Fraction(c * y0, s) + m * x0)  # inner derivative 2x dropped
    wrong_m = (Fraction(2 * c * x0, s), Fraction(2 * c * y0, s))  # product term dropped
    return (
        grad,
        ftext,
        (x0, y0),
        [(wrong_inner, "gradient-chain-inner-derivative-dropped"), (wrong_m, "gradient-term-omitted")],
    )


def _exp_template(rng):
    kind = rng.choice(("xy", "diff", "sq"))
    if kind == "xy":
        a = rng.choice((-4, -3, -2, 2, 3, 4, 5))
        pt = (a, 0) if rng.random() < 0.5 else (0, a)
        gtext, dg = "xy", (pt[1], pt[0])
    elif kind == "diff":
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        pt = (a, a)
        gtext, dg = "x - y", (1, -1)
    else:
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        pt = (a, rng.choice((-1, 1)) * a)
        gtext, dg = "x^2 - y^2", (2 * pt[0], -2 * pt[1])
    p, q = rng.choice((-3, -2, -1, 1, 2, 3, 4)), rng.choice((-3, -2, -1, 1, 2, 3, 4))
    lin = p * pt[0] + q * pt[1]  # the exponent is 0 at the point, so e^g = 1 there
    grad = (Fraction(p + lin * dg[0]), Fraction(q + lin * dg[1]))
    ftext = f"({join_terms([(p, 'x'), (q, 'y')])})e^({gtext})"
    # product rule used without the chain-rule factor on the exponential
    return grad, ftext, pt, [((Fraction(p), Fraction(q)), "gradient-chain-inner-derivative-dropped")]


def _root_template(rng):
    c = rng.randint(1, 4)
    m = rng.choice((-3, -2, -1, 1, 2, 3))
    x0, y0 = _signed(rng, rng.choice(_ROOT_POINTS))
    ftext = join_terms([(c, "√(x^2 + y^2)"), (m, "x")])
    r = math.isqrt(x0 * x0 + y0 * y0)
    grad = (Fraction(c * x0, r) + m, Fraction(c * y0, r))
    wrong_no_root = (Fraction(c * x0, 1) + m, Fraction(c * y0, 1))  # x/r written as x
    wrong_half = (Fraction(c * x0, 2 * r) + m, Fraction(c * y0, 2 * r))  # a stray 1/2 kept from the power rule
    return (
        grad,
        ftext,
        (x0, y0),
        [
            (wrong_no_root, "gradient-chain-inner-derivative-dropped"),
            (wrong_half, "gradient-chain-inner-derivative-dropped"),
        ],
    )


def _quotient_template(rng):
    for _ in attempts(500, "a quotient with a clean denominator"):
        al, be, ga, de = (rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(4))
        if al * de - be * ga == 0:
            continue
        x0, y0 = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
        d = ga * x0 + de * y0
        if d not in (1, -1, 2, -2, 4, -4, 5, -5, 8, -8, 10, -10):
            continue
        break
    ftext = f"({join_terms([(al, 'x'), (be, 'y')])})/({join_terms([(ga, 'x'), (de, 'y')])})"
    grad = (Fraction((al * de - be * ga) * y0, d * d), Fraction((be * ga - al * de) * x0, d * d))
    # quotient rule with the numerator subtracted in the wrong order
    wrong = (Fraction(-(al * de - be * ga) * y0, d * d), Fraction((al * de - be * ga) * x0, d * d))
    wrong2 = (Fraction(al, d), Fraction(be, d))  # denominator treated as a constant
    return (
        grad,
        ftext,
        (x0, y0),
        [(wrong, "gradient-quotient-rule-reversed"), (wrong2, "gradient-chain-inner-derivative-dropped")],
    )


_MEDIUM_TEMPLATES = [_log_template, _exp_template, _root_template, _quotient_template]


def _medium_question(rng: random.Random) -> dict:
    for _ in attempts(500, "a medium gradient question"):
        template = rng.choice(_MEDIUM_TEMPLATES)
        (gx, gy), ftext, (x0, y0), wrongs = template(rng)
        if (
            gx == 0
            or gy == 0
            or gx == gy
            or abs(gx) > 70
            or abs(gy) > 70
            or (gx * 1000).denominator != 1
            or (gy * 1000).denominator != 1
        ):
            continue
        cands = (
            [((gy, gx), "gradient-vector-components-swapped")]
            + [(w, tag) for w, tag in wrongs]
            + [((gx, gx), "gradient-partial-repeated")]
        )
        dists = _pair_distractors((gx, gy), [((Fraction(a), Fraction(b)), tag) for (a, b), tag in cands], _pair_text, 4)
        if len(dists) >= 3:
            break
    setting, name = rng.choice(_SETTINGS)
    shape = rng.randrange(3)
    pt = f"({x0}, {y0})"
    if shape == 0:
        text = f"Let f(x, y) = {ftext}. Find the gradient vector ∇f at the point {pt}, written as (f_x, f_y)."
    elif shape == 1:
        text = f"{setting} {name}(x, y) = {ftext}. Find ∇{name} at {pt}, written as an ordered pair."
    else:
        text = f"The surface z = {ftext} has gradient ∇z at each point. Find ∇z at {pt}, as (z_x, z_y)."
    return _pair_question(
        text,
        (gx, gy),
        dists,
        [f"Differentiate with respect to x, then y, and substitute {pt}.", f"∇f = {_pair_text(gx, gy)}."],
    )


# ---------------------------------------------------------------------------
# Hard

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]
_HARD_SETTINGS = [
    (
        "The height, in metres, of a hillside above a point (x, y) kilometres east and north of a campsite is h(x, y) = {f}.",
        "h",
        "height",
    ),
    (
        "The temperature, in degrees Celsius, of a metal plate at a point (x, y) centimetres from one corner is T(x, y) = {f}.",
        "T",
        "temperature",
    ),
    (
        "The pressure, in kilopascals, in a wind tunnel at a point (x, y) metres from the fan is p(x, y) = {f}.",
        "p",
        "pressure",
    ),
]


def _quadratic_surface_with_gradient(rng, x0, y0, g1, g2):
    """f(x, y) = a x^2 y + b x y^2 + c x^2 + d y^2 + e x y + D x + E y with grad f(x0, y0) = (g1, g2)."""
    for _ in attempts(2000, "a surface with the stated gradient"):
        a = rng.choice((0, 0, -2, -1, 1, 2))
        b = rng.choice((0, 0, -2, -1, 1, 2))
        c = rng.choice((-3, -2, -1, 1, 2, 3))
        d = rng.choice((-3, -2, -1, 1, 2, 3))
        e = rng.choice((-3, -2, -1, 1, 2, 3))
        base = [(a, 2, 1), (b, 1, 2), (c, 2, 0), (e, 1, 1), (d, 0, 2)]
        hx, hy = _poly_grad(base, x0, y0)
        D, E = g1 - int(hx), g2 - int(hy)
        if abs(D) > 40 or abs(E) > 40:
            continue
        terms = [(a, 2, 1), (b, 1, 2), (c, 2, 0), (e, 1, 1), (d, 0, 2), (D, 1, 0), (E, 0, 1)]
        check(
            _poly_grad(terms, x0, y0) == (Fraction(g1), Fraction(g2)), "built surface does not have the stated gradient"
        )
        return terms, join_terms([(co, _mono(pp, qq)) for co, pp, qq in terms])
    raise VerificationError("unreachable")  # pragma: no cover


def _pick_gradient(rng: random.Random) -> tuple[int, int]:
    """A gradient (g1, g2) whose length is a whole number and whose components stay below 31."""
    for _ in attempts(500, "a Pythagorean gradient"):
        p, q, _r = rng.choice(_TRIPLES)
        if rng.random() < 0.5:
            p, q = q, p
        m = rng.choice((1, 1, 2))
        if m * max(p, q) > 30:
            continue
        return rng.choice((-1, 1)) * m * p, rng.choice((-1, 1)) * m * q
    raise VerificationError("unreachable")  # pragma: no cover


def _unit_components(g1: int, g2: int) -> tuple[Fraction, Fraction, int]:
    length = math.isqrt(g1 * g1 + g2 * g2)
    check(length * length == g1 * g1 + g2 * g2, "gradient length is not a whole number")
    return Fraction(g1, length), Fraction(g2, length), length


def _steepest_question(rng: random.Random) -> dict:
    g1, g2 = _pick_gradient(rng)
    x0, y0 = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
    _terms, ftext = _quadratic_surface_with_gradient(rng, x0, y0, g1, g2)
    u1, u2, length = _unit_components(g1, g2)
    descent = rng.random() < 0.4
    sgn = -1 if descent else 1
    answer = (sgn * u1, sgn * u2)
    two = lambda a, b: _pair_text_dp(a, b, 2)  # noqa: E731
    cands = [
        ((-answer[0], -answer[1]), "steepest-descent-for-ascent" if not descent else "steepest-ascent-for-descent"),
        ((answer[1], answer[0]), "gradient-vector-components-swapped"),
        (
            (Fraction(g1, abs(g1) + abs(g2)) * sgn, Fraction(g2, abs(g1) + abs(g2)) * sgn),
            "gradient-normalised-by-component-sum",
        ),
        (
            (Fraction(g1, length * length) * sgn, Fraction(g2, length * length) * sgn),
            "gradient-divided-by-squared-length",
        ),
    ]
    dists = _pair_distractors(answer, cands, two, 3)
    setting, name, word = rng.choice(_HARD_SETTINGS)
    shape = rng.randrange(3)
    pt = f"({x0}, {y0})"
    base = setting.format(f=ftext)
    if descent:
        ask = {
            0: f"A raindrop at {pt} runs in the direction in which the {word} falls most rapidly. Give the unit vector of that direction as (u, v), correct to 2 decimal places.",
            1: f"Find the unit vector, correct to 2 decimal places, in the direction of steepest descent of {name} at {pt}.",
            2: f"In which direction does {name} decrease most rapidly at {pt}? Give the unit vector as (u, v), correct to 2 decimal places.",
        }[shape]
    else:
        ask = {
            0: f"A walker at {pt} wants the {word} to increase as quickly as possible. Give the unit vector of the direction to take as (u, v), correct to 2 decimal places.",
            1: f"Find the unit vector, correct to 2 decimal places, in the direction of steepest ascent of {name} at {pt}.",
            2: f"In which direction does {name} increase most rapidly at {pt}? Give the unit vector as (u, v), correct to 2 decimal places.",
        }[shape]
    return _pair_question(
        f"{base} {ask}",
        answer,
        dists,
        [
            f"∇{name}{pt} = {_pair_text(Fraction(g1), Fraction(g2))}, with length {length}.",
            f"The unit vector is {two(*answer)}.",
        ],
        two,
    )


def _solve_for_point_question(rng: random.Random) -> dict:
    for _ in attempts(5000, "a gradient-equals-vector question"):
        A = rng.choice((-3, -2, -1, 1, 2, 3))
        C = rng.choice((-3, -2, -1, 1, 2, 3))
        B = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        det = 4 * A * C - B * B
        if det == 0:
            continue
        D, E = rng.randint(-12, 12), rng.randint(-12, 12)
        x0, y0 = rng.randint(-4, 4), rng.randint(-4, 4)
        if x0 == y0 or x0 == 0 or y0 == 0:
            continue
        g1 = 2 * A * x0 + B * y0 + D
        g2 = B * x0 + 2 * C * y0 + E
        if g1 == 0 or g2 == 0 or abs(g1) > 40 or abs(g2) > 40:
            continue
        # distractor candidates, all whole-number points
        cands = [
            ((y0, x0), "point-coordinates-swapped"),
            ((x0, -y0), "point-sign-slip"),
            ((-x0, y0), "point-sign-slip"),
        ]
        # cross term ignored: x = (g1 - D)/(2A), y = (g2 - E)/(2C)
        ix, iy = Fraction(g1 - D, 2 * A), Fraction(g2 - E, 2 * C)
        if ix.denominator == 1 and iy.denominator == 1:
            cands.append(((int(ix), int(iy)), "solve-gradient-ignores-cross-term"))
        # gradient set to zero instead of to (g1, g2)
        cx, cy = Fraction(-(2 * C * D - B * E), det), Fraction(-(2 * A * E - B * D), det)
        if cx.denominator == 1 and cy.denominator == 1:
            cands.append(((int(cx), int(cy)), "solve-gradient-set-to-zero"))
        dists = _pair_distractors(
            (Fraction(x0), Fraction(y0)), [((Fraction(a), Fraction(b)), t) for (a, b), t in cands], _pair_text, 3
        )
        if len(dists) >= 3:
            break
    ftext = join_terms([(A, "x^2"), (B, "xy"), (C, "y^2"), (D, "x"), (E, "y")])
    # second route: substitute the point back into both equations
    check((2 * A * x0 + B * y0 + D, B * x0 + 2 * C * y0 + E) == (g1, g2), "solve for the point: substitution disagrees")
    shape = rng.randrange(3)
    vec = f"({g1}, {g2})"
    if shape == 0:
        text = f"Let f(x, y) = {ftext}. Find the point (x, y) at which ∇f = {vec}."
    elif shape == 1:
        text = f"The gradient of f(x, y) = {ftext} equals {vec} at exactly one point. Find the coordinates of that point, written as (x, y)."
    else:
        text = f"At which point does the surface z = {ftext} have gradient vector {vec}? Give your answer as (x, y)."
    return _pair_question(
        text,
        (Fraction(x0), Fraction(y0)),
        dists,
        [
            f"∇f = ({join_terms([(2 * A, 'x'), (B, 'y'), (D, '')])}, {join_terms([(B, 'x'), (2 * C, 'y'), (E, '')])}); set this equal to {vec}.",
            f"Solving the two equations gives x = {x0}, y = {y0}.",
        ],
    )


def _tangent_question(rng: random.Random) -> dict:
    g1, g2 = _pick_gradient(rng)
    x0, y0 = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
    _terms, ftext = _quadratic_surface_with_gradient(rng, x0, y0, g1, g2)
    u1, u2, length = _unit_components(g1, g2)
    # perpendicular to the gradient, written with a positive x-component
    t1, t2 = (u2, -u1) if u2 > 0 else (-u2, u1)
    check(
        t1 > 0 and t1 * u1 + t2 * u2 == 0 and t1 * t1 + t2 * t2 == 1,
        "tangent is not a unit vector perpendicular to the gradient",
    )
    # second route: the level curve f = c has slope -f_x/f_y; the tangent direction must have that slope
    slope = Fraction(-g1, g2)
    check(t2 / t1 == slope, "tangent: slope of the level curve disagrees")
    two = lambda a, b: _pair_text_dp(a, b, 2)  # noqa: E731
    par = (u1, u2) if u1 > 0 else (-u1, -u2)
    cands = [
        (par, "tangent-parallel-to-gradient"),
        ((u2, u1) if u2 > 0 else (-u2, -u1), "tangent-perpendicular-not-sign-changed"),
        ((-t1, -t2), "tangent-opposite-orientation"),
        ((t1, -t2), "tangent-component-sign-lost"),
    ]
    dists = _pair_distractors((t1, t2), cands, two, 3)
    setting, name, word = rng.choice(_HARD_SETTINGS)
    base = setting.format(f=ftext)
    shape = rng.randrange(3)
    pt = f"({x0}, {y0})"
    convention = "with a positive first component"
    ask = {
        0: f"The contour {name}(x, y) = {name}{pt} passes through {pt}. Find the unit vector tangent to this contour at {pt}, {convention}, as (u, v) correct to 2 decimal places.",
        1: f"Find the unit vector, {convention}, that points along the level curve of {name} through {pt}. Give it as (u, v), correct to 2 decimal places.",
        2: f"Walking along a path on which the {word} stays constant through {pt}, which unit vector (positive first component) gives the direction of travel? Write it as (u, v), correct to 2 decimal places.",
    }[shape]
    return _pair_question(
        f"{base} {ask}",
        (t1, t2),
        dists,
        [
            f"∇{name}{pt} = {_pair_text(Fraction(g1), Fraction(g2))}; the contour is perpendicular to it.",
            f"Unit tangent: {two(t1, t2)}.",
        ],
        two,
    )


def _labelled(d: str, m: str) -> str:
    return f"D = {d}, M = {m}"


def _directional_pair_question(rng: random.Random) -> dict:
    g1, g2 = _pick_gradient(rng)
    x0, y0 = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
    _terms, ftext = _quadratic_surface_with_gradient(rng, x0, y0, g1, g2)
    length = math.isqrt(g1 * g1 + g2 * g2)
    for _ in attempts(500, "a direction vector of whole length"):
        k = rng.choice((1, 1, 2))
        pp, qq, rr = rng.choice([(3, 4, 5), (4, 3, 5), (5, 0, 5), (0, 5, 5)])
        v1, v2 = rng.choice((-1, 1)) * pp * k, rng.choice((-1, 1)) * qq * k
        vlen = rr * k
        d = Fraction(g1 * v1 + g2 * v2, vlen)
        if d != 0 and d.denominator in (1, 2, 5, 10) and abs(d) != length:
            break
    check(
        d * vlen == g1 * v1 + g2 * v2 and length * length == g1 * g1 + g2 * g2,
        "directional derivative: arithmetic disagrees",
    )
    answer_text = _labelled(_decimal(d), str(length))
    cands = [
        (_labelled(_decimal(Fraction(g1 * v1 + g2 * v2)), str(length)), "direction-not-normalised"),
        (_labelled(_decimal(-d), str(length)), "direction-reversed"),
        (_labelled(str(length), str(length)), "gradient-length-for-directional"),
        (_labelled(_decimal(d), str(length * length)), "gradient-length-no-root"),
        (_labelled(_decimal(Fraction(g2 * v1 + g1 * v2, vlen)), str(length)), "gradient-components-swapped"),
    ]
    seen = {answer_text}
    dists = []
    for text, tag in cands:
        if text in seen:
            continue
        seen.add(text)
        dists.append(Distractor(text, tag))
    check(len(dists) >= 2, "too few directional distractors")
    setting, name, word = rng.choice(_HARD_SETTINGS)
    base = setting.format(f=ftext)
    shape = rng.randrange(3)
    pt = f"({x0}, {y0})"
    vec = f"({v1}, {v2})"
    text = {
        0: f"{base} At {pt}, let D be the rate of change of {name} in the direction of the vector {vec}, and M the greatest rate of change of {name} in any direction. Give your answer in the form D = …, M = ….",
        1: f"{base} Find the rate of change of {name} at {pt} in the direction of {vec}, and the greatest possible rate of change at that point. Write them as D = …, M = ….",
        2: f"{base} A sensor moves from {pt} in the direction {vec}. Let D be the rate at which the {word} changes along that direction and M the largest rate at which it changes in any direction. Write your answer as D = …, M = ….",
    }[shape]
    return dict(
        text=text,
        answer={"D": _num(d), "M": length},
        answer_text=answer_text,
        answer_kind="pair",
        working=[
            f"∇{name}{pt} = {_pair_text(Fraction(g1), Fraction(g2))}.",
            f"D = ∇{name} · v/|v| = ({g1}×{paren_negative(v1)} + {paren_negative(g2)}×{paren_negative(v2)})/{vlen} = {_decimal(d)}; M = |∇{name}| = {length}.",
        ],
        distractors=dists,
    )


@generator(
    "w5c_gradient_vector",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5c_gradient_vector(rng: random.Random, tier: str = "easy") -> dict:
    """Gradient vector: at a point of a polynomial (easy), of a chain/product/quotient function (medium), then its uses (hard)."""
    if tier == "easy":
        return _easy_question(rng)
    if tier == "medium":
        return _medium_question(rng)
    if tier == "hard":
        pick = rng.random()
        if pick < 0.3:
            return _steepest_question(rng)
        if pick < 0.55:
            return _solve_for_point_question(rng)
        if pick < 0.8:
            return _tangent_question(rng)
        return _directional_pair_question(rng)
    raise VerificationError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain


def _exam_function(rng):
    kind = rng.choice(("log", "root", "quotient"))
    if kind == "log":
        c = rng.randint(1, 5)
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        d = rng.choice((-4, -3, -2, 2, 3, 4))
        x0, y0 = _signed(rng, rng.choice(_LOG_POINTS))
        s_ = x0 * x0 + y0 * y0
        grad = (Fraction(2 * c * x0, s_) + a * y0 * y0 + d, Fraction(2 * c * y0, s_) + 2 * a * x0 * y0)
        text = join_terms([(c, "ln(x^2 + y^2)"), (a, "xy^2"), (d, "x")])
    elif kind == "root":
        c = rng.randint(1, 4)
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        d = rng.choice((-4, -3, -2, 2, 3, 4))
        x0, y0 = _signed(rng, rng.choice(_ROOT_POINTS))
        r_ = math.isqrt(x0 * x0 + y0 * y0)
        grad = (Fraction(c * x0, r_) + 2 * a * x0 * y0, Fraction(c * y0, r_) + a * x0 * x0 + d)
        text = join_terms([(c, "√(x^2 + y^2)"), (a, "x^2y"), (d, "y")])
    else:
        for _ in attempts(500, "a quotient with a clean denominator"):
            al, be, ga, de = (rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(4))
            x0, y0 = rng.choice((-3, -2, -1, 1, 2, 3)), rng.choice((-3, -2, -1, 1, 2, 3))
            if al * de - be * ga != 0 and ga * x0 + de * y0 in (1, -1, 2, -2, 4, -4, 5, -5, 8, -8, 10, -10):
                break
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        d = rng.choice((-3, -2, 2, 3))
        dd = ga * x0 + de * y0
        grad = (
            Fraction((al * de - be * ga) * y0, dd * dd) + a * y0,
            Fraction((be * ga - al * de) * x0, dd * dd) + a * x0 + 2 * d * y0,
        )
        text = f"({join_terms([(al, 'x'), (be, 'y')])})/({join_terms([(ga, 'x'), (de, 'y')])}) " + _tail_terms(
            [(a, "xy"), (d, "y^2")]
        )
    return grad, text, (x0, y0)


def _tail_terms(terms) -> str:
    """The terms after a leading expression, each with its sign: '+ 2xy - 3y^2'."""
    body = join_terms(terms)
    return f"- {body[1:]}" if body.startswith("-") else f"+ {body}"


_EXAM_SETTINGS = [
    (
        "The temperature, in degrees Celsius, of a heated metal plate at a point (x, y) centimetres from one corner is",
        "T",
        "centimetre",
    ),
    (
        "The height, in metres, of a hillside above a point (x, y) hundreds of metres east and north of a lookout is",
        "h",
        "hundred metres",
    ),
    (
        "The concentration of a salt solution, in grams per litre, at a point (x, y) centimetres into a tank is",
        "c",
        "centimetre",
    ),
]


def _exam_triples():
    return [(3, 4, 5), (4, 3, 5), (5, 12, 13), (12, 5, 13), (8, 15, 17), (15, 8, 17), (6, 8, 10), (8, 6, 10)]


def _build_exam(rng: random.Random) -> dict | None:
    (gx, gy), ftext, (x0, y0) = _exam_function(rng)
    if gx == 0 or gy == 0 or abs(gx) == abs(gy) or abs(gx) > 60 or abs(gy) > 60:
        return None
    if (gx * 100).denominator != 1 or (gy * 100).denominator != 1:
        return None
    pp, qq, rr = rng.choice(_exam_triples())
    dx, dy = rng.choice((-1, 1)) * pp, rng.choice((-1, 1)) * qq
    directional = (gx * dx + gy * dy) / rr
    steepest_sq = gx * gx + gy * gy
    steepest = Fraction(math.isqrt(steepest_sq.numerator * 10**60 // steepest_sq.denominator), 10**30)
    if not (_clear(directional, 2, Fraction(15, 100)) and _clear(steepest, 2, Fraction(15, 100))):
        return None
    if directional == 0:
        return None
    # the rate from the rounded partials must still land on the same 2 places
    if round_half_up((round_half_up(gx, 2) * dx + round_half_up(gy, 2) * dy) / rr, 2) != round_half_up(directional, 2):
        return None
    qx, qy = x0 + dx, y0 + dy
    setting, name, _unit = rng.choice(_EXAM_SETTINGS)
    stem = f"{setting} {name}(x, y) = {ftext}. Consider the point P({x0}, {y0}) and the point Q({qx}, {qy})."

    a_part = number_part(
        f"Find ∂{name}/∂x at P, correct to 2 decimal places.",
        gx,
        2,
        2,
        [f"{name}_x(P) = {stated_precision_string(gx, 2)}."],
        [
            (gy, "gradient-vector-components-swapped"),
            (-gx, "substitution-sign-error"),
            (gx / 2, "gradient-chain-inner-derivative-dropped"),
        ],
    )
    b_part = number_part(
        f"Find ∂{name}/∂y at P, correct to 2 decimal places.",
        gy,
        2,
        2,
        [f"{name}_y(P) = {stated_precision_string(gy, 2)}."],
        [
            (gx, "gradient-vector-components-swapped"),
            (-gy, "substitution-sign-error"),
            (gy / 2, "gradient-chain-inner-derivative-dropped"),
        ],
    )
    c_part = number_part(
        f"Find the rate of change of {name} at P in the direction from P towards Q, correct to 2 decimal places.",
        directional,
        2,
        3,
        [
            f"PQ = ({dx}, {dy}), |PQ| = {rr}.",
            f"D = ({_p2(gx)}×{paren_negative(dx)} + {_p2(gy)}×{paren_negative(dy)})/{rr}.",
        ],
        [
            ((gx * dx + gy * dy), "direction-not-normalised"),
            (-directional, "direction-reversed"),
            ((gx * dy + gy * dx) / rr, "gradient-components-swapped"),
            (steepest, "gradient-length-for-directional"),
        ],
        follow_through=f"({float(dx)}*part_a + {float(dy)}*part_b)/{float(rr)}",
    )
    d_part = number_part(
        f"Find the greatest rate of increase of {name} at P, correct to 2 decimal places.",
        steepest,
        2,
        2,
        [f"|∇{name}| = √({_p2(gx)}² + {_p2(gy)}²) = {stated_precision_string(round_half_up(steepest, 2), 2)}."],
        [
            (Fraction(steepest_sq), "gradient-length-no-root"),
            (abs(gx) + abs(gy), "magnitude-as-sum-of-components"),
            (max(abs(gx), abs(gy)), "magnitude-as-one-leg"),
            (directional, "gradient-length-for-directional"),
        ],
        follow_through="sqrt(part_a**2 + part_b**2)",
        sensible_range=(0, None),
    )
    if None in (a_part, b_part, c_part, d_part):
        return None
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])


@generator("exd_x_w5c_gradient_vector", level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5c_gradient_vector(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the two partial derivatives at P, the rate towards Q, the greatest rate (9 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return retry_build(rng, _build_exam, "a gradient exam chain", tries=600)
