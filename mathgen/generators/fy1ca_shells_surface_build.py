"""Generates `fy1ca_shells_surface` and `fy1ca_shells_surface_exam` questions.

One first-year skill, ``ucv1-shells-surface-area``: the volume of a solid of revolution by cylindrical shells,
V = 2π ∫ (radius)(height) dx, and the area of a surface of revolution, S = 2π ∫ y √(1 + y'²) dx. (Volumes by discs and
washers are covered by the senior solids-of-revolution skill; this skill is the step up.) Every key is an exact multiple
of π, graded as an expression.

Tiers (the reasoning escalates):

- ``easy``: the region under one curve, rotated about the y-axis: the shell has radius x and height y.
- ``medium``: the shell height is the gap between two curves, or the axis of rotation is a vertical line other than the
  y-axis so the radius is a distance from that line.
- ``hard``: the area of a surface of revolution: the lateral surface of a cone, a paraboloid y = √x, the curve
  y = x³/6 + 1/(2x) (whose arc-length integrand is a perfect square), or a zone of a sphere.
- ``exam`` (topic ``fy1ca_shells_surface_exam``): a four-part chain on the region between y = x² and y = kx rotated about
  the y-axis: the intersection point, the shell height at a given radius, the volume, and the mass of the solid.

Every key is re-derived by a second route: Simpson's rule on the shell integrand, or a sum of frustum areas along the
curve for a surface, agrees with the exact value.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy
from sympy import Rational, pi, sqrt

from ..core import check, generator, latex_answer, phrase
from ._fy1ca_kit import (
    Redraw,
    build_with_redraws,
    clear_of_rounding,
    close,
    expression_distractors,
    fraction_distractors,
    fraction_part,
    need_distractors,
    number_distractors,
    number_part,
    simpson,
    typed,
)

x = sympy.Symbol("x", positive=True)


def _show(value) -> str:
    text = typed(sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value)
    return text.replace("*pi", "π").replace("pi", "π").replace("*", " × ")


def _nice(candidates):
    """Drop wrong keys a learner would never produce by hand (huge denominators, long radicals)."""
    kept = []
    for value, tag in candidates:
        if value is None:
            continue
        value = sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value
        value = sympy.expand(value)
        if value.is_Rational and value.q > 400:
            continue
        if len(typed(value)) > 22:
            continue
        kept.append((value, tag))
    return kept


def _question(text, answer, wrong, working) -> dict:
    expr = sympy.nsimplify(answer) if not isinstance(answer, sympy.Basic) else answer
    distractors = expression_distractors(expr, _nice(wrong))
    need_distractors(distractors)
    return dict(
        text=text,
        answer=str(expr),
        answer_text=typed(expr),
        answer_kind="expression",
        answer_latex=latex_answer(expr),
        working=working,
        distractors=distractors,
    )


# -- shells -----------------------------------------------------------------------------------------------------------


def _radius(axis):
    """The distance from a point x to the axis of rotation (the y-axis when `axis` is None)."""
    if axis is None:
        return x
    return (x - axis) if axis < 0 else (axis - x)


def _shell_volume(top, bottom, lo, hi, axis):
    return 2 * pi * sympy.integrate(_radius(axis) * (top - bottom), (x, lo, hi))


def _shell_numeric(top, bottom, lo, hi, axis) -> float:
    height = sympy.lambdify(x, top - bottom, "math")
    radius = sympy.lambdify(x, _radius(axis), "math")
    return 2 * math.pi * simpson(lambda v: radius(v) * height(v), float(lo), float(hi), 2400)


def _shell_wrong(top, bottom, lo, hi, axis):
    gap = top - bottom
    wrong = [
        (2 * pi * sympy.integrate(gap, (x, lo, hi)), "fy1ca-shell-radius-forgotten"),
        (pi * sympy.integrate(_radius(axis) * gap, (x, lo, hi)), "fy1ca-shell-two-forgotten"),
        (pi * sympy.integrate(top**2 - bottom**2, (x, lo, hi)), "fy1ca-shell-disc-formula"),
        (2 * sympy.integrate(_radius(axis) * gap, (x, lo, hi)), "fy1ca-shell-pi-forgotten"),
    ]
    if bottom != 0:
        wrong.append((2 * pi * sympy.integrate(_radius(axis) * top, (x, lo, hi)), "fy1ca-shell-height-not-difference"))
    if axis is not None:
        wrong.append((2 * pi * sympy.integrate(x * gap, (x, lo, hi)), "fy1ca-shell-axis-ignored"))
    return wrong


_CURVE_TEXT = {
    x: "y = x",
    x**2: "y = x²",
    x**3: "y = x³",
    sqrt(x): "y = √x",
    4 - x**2: "y = 4 - x²",
    9 - x**2: "y = 9 - x²",
}


def _easy(rng: random.Random) -> dict:
    top = rng.choice([x**2, x**3, sqrt(x), 4 - x**2, 9 - x**2, x])
    if top == sqrt(x):
        hi = rng.choice([1, 4, 9])
    elif top == 4 - x**2:
        hi = rng.choice([1, 2])
    elif top == 9 - x**2:
        hi = rng.choice([2, 3])
    else:
        hi = rng.choice([1, 2, 3, 4])
    answer = _shell_volume(top, 0, 0, hi, None)
    check(close(_shell_numeric(top, 0, 0, hi, None), float(answer), 1e-6), "numerical volume disagrees")
    wrong = _shell_wrong(top, 0, 0, hi, None)
    thing = rng.choice(["a ceramic bowl", "a vase", "a fountain basin", "a lampshade mould"])
    text = phrase(
        rng,
        (
            "The region in the first quadrant under the curve {curve}, between x = 0 and x = {hi}, is rotated through 360° about the y-axis to model {thing} (units in centimetres). Use cylindrical shells to find the exact volume of the solid, in cm³.",
            "Let R be the region bounded by {curve}, the x-axis, the y-axis and the line x = {hi}. Find, by the method of cylindrical shells, the exact volume of the solid formed when R is revolved about the y-axis.",
        ),
        curve=_CURVE_TEXT[top],
        hi=hi,
        thing=thing,
    )
    working = [
        f"A shell at x has radius x and height {_show(top)}, so V = 2π times the integral of x({_show(top)}) from 0 to {hi}.",
        f"The integral of {_show(sympy.expand(x * top))} is {_show(sympy.integrate(x * top, x))}, which is {_show(sympy.integrate(x * top, (x, 0, hi)))} between the limits.",
        f"V = 2π × {_show(sympy.integrate(x * top, (x, 0, hi)))} = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


# (top, bottom, lo, hi, text) regions between two curves, rotated about the y-axis
_BETWEEN = (
    (x, x**3, 0, 1, "y = x and y = x³"),
    (sqrt(x), x**2, 0, 1, "y = √x and y = x²"),
    (x**2, x**3, 0, 1, "y = x² and y = x³"),
    (2 * x, x**2, 0, 2, "y = 2x and y = x²"),
    (3 * x, x**2, 0, 3, "y = 3x and y = x²"),
    (4 * x, x**2, 0, 4, "y = 4x and y = x²"),
    (5 * x, x**2, 0, 5, "y = 5x and y = x²"),
    (6 * x, x**2, 0, 6, "y = 6x and y = x²"),
    (x, x**2, 0, 1, "y = x and y = x²"),
    (4 * x, x**3, 0, 2, "y = 4x and y = x³"),
    (9 * x, x**3, 0, 3, "y = 9x and y = x³"),
    (16 * x, x**3, 0, 4, "y = 16x and y = x³"),
)


def _between(rng: random.Random) -> dict:
    top, bottom, lo, hi, curves = rng.choice(_BETWEEN)
    answer = _shell_volume(top, bottom, lo, hi, None)
    check(close(_shell_numeric(top, bottom, lo, hi, None), float(answer), 1e-6), "numerical volume disagrees")
    wrong = _shell_wrong(top, bottom, lo, hi, None)
    text = phrase(
        rng,
        (
            "The region in the first quadrant between the curves {curves} is rotated about the y-axis. Use cylindrical shells to find the exact volume of the solid formed.",
            "Find the exact volume of the solid obtained by rotating about the y-axis the region enclosed between {curves} for x ≥ 0. Use the method of cylindrical shells.",
        ),
        curves=curves,
    )
    gap = sympy.expand(top - bottom)
    working = [
        f"A shell at x has radius x and height {_show(top)} - {_show(bottom)} = {_show(gap)}, and the curves meet at x = {hi}.",
        f"V = 2π times the integral of x({_show(gap)}) from {lo} to {hi} = 2π[{_show(sympy.integrate(x * gap, x))}] from {lo} to {hi}.",
        f"V = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _shifted_axis(rng: random.Random) -> dict:
    hi = rng.choice([1, 2, 3])
    axis = rng.choice([-1, -2, -3, hi + 1, hi + 2, hi + 3])
    top = x**2
    answer = _shell_volume(top, 0, 0, hi, axis)
    check(close(_shell_numeric(top, 0, 0, hi, axis), float(answer), 1e-6), "numerical volume disagrees")
    wrong = _shell_wrong(top, 0, 0, hi, axis)
    radius = f"x + {-axis}" if axis < 0 else f"{axis} - x"
    text = phrase(
        rng,
        (
            "The region under the curve y = x² between x = 0 and x = {hi} is rotated about the vertical line x = {axis}. Use cylindrical shells to find the exact volume of the solid formed.",
            "Let R be the region bounded by y = x², the x-axis and the line x = {hi}. Find the exact volume of the solid formed when R is rotated about the line x = {axis}, using cylindrical shells.",
        ),
        hi=hi,
        axis=axis,
    )
    poly = sympy.expand(_radius(axis) * x**2)
    working = [
        f"A shell at x has radius {radius} (its distance from the line x = {axis}) and height x².",
        f"V = 2π times the integral of ({radius})x² from 0 to {hi} = 2π[{_show(sympy.integrate(poly, x))}] from 0 to {hi}.",
        f"V = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _medium(rng: random.Random) -> dict:
    return rng.choice([_between, _shifted_axis])(rng)


# -- surfaces ------------------------------------------------------------------------------------------------------------


def _frustum_sum(y, lo: float, hi: float, n: int = 4000) -> float:
    """The area of the surface of revolution about the x-axis, as a sum of frustum areas along the curve."""
    f = sympy.lambdify(x, y, "math")
    total = 0.0
    px, py = lo, f(lo)
    for i in range(1, n + 1):
        qx = lo + (hi - lo) * i / n
        qy = f(qx)
        total += math.pi * (py + qy) * math.hypot(qx - px, qy - py)
        px, py = qx, qy
    return total


def _surface_wrong(y, speed, lo, hi):
    """Candidates shared by the families: leave out the root, the radius, the 2, or use the volume formula."""
    return [
        (2 * pi * sympy.integrate(y, (x, lo, hi)), "fy1ca-surface-root-dropped"),
        (2 * pi * sympy.integrate(speed, (x, lo, hi)), "fy1ca-surface-radius-dropped"),
        (pi * sympy.integrate(y * speed, (x, lo, hi)), "fy1ca-surface-two-forgotten"),
        (pi * sympy.integrate(y**2, (x, lo, hi)), "fy1ca-surface-volume-formula"),
        (sympy.integrate(speed, (x, lo, hi)), "fy1ca-surface-arc-length-only"),
    ]


def _cone(rng: random.Random) -> dict:
    m = rng.choice([Rational(3, 4), Rational(4, 3), Rational(5, 12), Rational(12, 5)])
    hi = rng.choice([2, 3, 4, 5, 6])
    y = m * x
    speed = sqrt(1 + m**2)
    answer = sympy.simplify(2 * pi * sympy.integrate(y * speed, (x, 0, hi)))
    check(close(_frustum_sum(y, 0.0, float(hi)), float(answer), 1e-6), "numerical area disagrees")
    wrong = _surface_wrong(y, speed, 0, hi)
    mt = f"{m.p}/{m.q}"
    text = phrase(
        rng,
        (
            "The line segment y = ({mt})x from x = 0 to x = {hi} is rotated about the x-axis to form a cone (units in centimetres). Find the exact curved surface area of the cone, in cm², using the surface-area integral.",
            "A funnel is modelled by rotating the line y = ({mt})x, 0 ≤ x ≤ {hi}, about the x-axis (units in centimetres). Use an integral to find the exact area of its curved surface, in cm².",
        ),
        mt=mt,
        hi=hi,
    )
    working = [
        f"dy/dx = {mt}, so √(1 + (dy/dx)²) = √(1 + {sympy.Rational(m**2)}) = {_show(speed)}.",
        f"S = 2π times the integral of ({mt})x × {_show(speed)} from 0 to {hi} = 2π × {_show(m * speed)} × {Rational(hi**2, 2)}.",
        f"S = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _paraboloid(rng: random.Random) -> dict:
    lo, hi = rng.choice([(0, 2), (0, 6), (0, 12), (2, 6), (2, 12), (6, 12)])
    y = sqrt(x)
    speed = sqrt(1 + 1 / (4 * x))
    integrand = sqrt(x + Rational(1, 4))
    check(sympy.simplify(y * speed - integrand) == 0, "integrand mismatch")
    G = lambda v: Rational(2, 3) * (Rational(v) + Rational(1, 4)) ** Rational(3, 2)  # noqa: E731
    answer = sympy.simplify(2 * pi * (G(hi) - G(lo)))
    check(close(_frustum_sum(y, float(lo), float(hi)), float(answer), 1e-4), "numerical area disagrees")
    wrong = [
        (2 * pi * sympy.integrate(y, (x, lo, hi)), "fy1ca-surface-root-dropped"),
        (pi * (G(hi) - G(lo)), "fy1ca-surface-two-forgotten"),
        (2 * pi * G(hi), "fy1ca-surface-lower-limit-ignored"),
        (pi * sympy.integrate(y**2, (x, lo, hi)), "fy1ca-surface-volume-formula"),
        (2 * pi * (G(hi) - G(lo)) * Rational(3, 2), "fy1ca-surface-wrong-antiderivative"),
        (2 * pi * sympy.integrate(sqrt(x + 1), (x, lo, hi)), "fy1ca-surface-quarter-dropped"),
    ]
    text = phrase(
        rng,
        (
            "The curve y = √x from x = {lo} to x = {hi} is rotated about the x-axis (units in metres). Find the exact area of the surface formed, in m².",
            "A parabolic reflector is made by rotating the arc y = √x, {lo} ≤ x ≤ {hi}, about the x-axis (units in metres). Find the exact surface area of the reflector, in m².",
        ),
        lo=lo,
        hi=hi,
    )
    working = [
        "dy/dx = 1/(2√x), so y√(1 + (dy/dx)²) = √x × √(1 + 1/(4x)) = √(x + 1/4).",
        f"S = 2π times the integral of √(x + 1/4) from {lo} to {hi} = 2π[(2/3)(x + 1/4)^(3/2)] from {lo} to {hi}.",
        f"S = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _perfect_square_curve(rng: random.Random) -> dict:
    lo = 1
    hi = rng.choice([2, 3, 4])
    y = x**3 / 6 + 1 / (2 * x)
    speed = x**2 / 2 + 1 / (2 * x**2)
    check(sympy.simplify(1 + sympy.diff(y, x) ** 2 - speed**2) == 0, "speed mismatch")
    product = sympy.expand(y * speed)
    answer = 2 * pi * sympy.integrate(product, (x, lo, hi))
    check(close(_frustum_sum(y, float(lo), float(hi)), float(answer), 1e-5), "numerical area disagrees")
    wrong = _surface_wrong(y, speed, lo, hi)
    text = phrase(
        rng,
        (
            "The curve y = x³/6 + 1/(2x) from x = {lo} to x = {hi} is rotated about the x-axis. Find the exact area of the surface formed.",
            "Find the exact area of the surface generated when the arc of y = x³/6 + 1/(2x) between x = {lo} and x = {hi} is revolved about the x-axis.",
        ),
        lo=lo,
        hi=hi,
    )
    working = [
        "dy/dx = x²/2 - 1/(2x²), so 1 + (dy/dx)² = (x²/2 + 1/(2x²))² and √(1 + (dy/dx)²) = x²/2 + 1/(2x²).",
        f"y × √(1 + (dy/dx)²) = {_show(product)}; S = 2π times its integral from {lo} to {hi}.",
        f"S = 2π[{_show(sympy.integrate(product, x))}] from {lo} to {hi} = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _sphere_zone(rng: random.Random) -> dict:
    r = rng.choice([3, 5, 10, 13])
    a = rng.choice(list(range(-r + 1, r - 1)))
    b = rng.choice([v for v in range(a + 1, r + 1)])
    y = sqrt(r**2 - x**2)
    # y * sqrt(1 + y'^2) = r exactly, so the area is 2*pi*r*(b - a)
    answer = 2 * pi * r * (b - a)
    expr = lambda v: math.sqrt(r * r - v * v)  # noqa: E731
    check(close(_frustum_sum_func(expr, a, b), float(answer), 2e-4), "numerical area disagrees")
    wrong = [
        (pi * r * (b - a), "fy1ca-surface-two-forgotten"),
        (4 * pi * r**2, "fy1ca-surface-whole-sphere"),
        (2 * pi * r * b, "fy1ca-surface-lower-limit-ignored"),
        (2 * pi * r * (a + b) if a + b > 0 else 2 * pi * r * (b + a + 2), "fy1ca-surface-endpoint-sum"),
        (2 * pi * r**2, "fy1ca-surface-whole-sphere"),
        (2 * pi * (b - a), "fy1ca-surface-radius-dropped"),
    ]
    text = phrase(
        rng,
        (
            "The arc of the circle x² + y² = {r2} with y ≥ 0 between x = {a} and x = {b} is rotated about the x-axis to form a zone of a sphere (units in centimetres). Find its exact surface area, in cm², using the surface-area integral.",
            "A glass dome is the zone of a sphere formed by rotating y = √({r2} - x²) about the x-axis for {a} ≤ x ≤ {b} (units in centimetres). Find the exact area of its curved surface, in cm².",
        ),
        r2=r * r,
        a=a,
        b=b,
    )
    working = [
        f"y = √({r * r} - x²), so dy/dx = -x/y and 1 + (dy/dx)² = ({r * r})/y², giving y√(1 + (dy/dx)²) = {r}.",
        f"S = 2π times the integral of {r} from {a} to {b} = 2π × {r} × {b - a}.",
        f"S = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _frustum_sum_func(f, lo: float, hi: float, n: int = 4000) -> float:
    total = 0.0
    px, py = float(lo), f(lo)
    for i in range(1, n + 1):
        qx = lo + (hi - lo) * i / n
        qy = f(qx)
        total += math.pi * (py + qy) * math.hypot(qx - px, qy - py)
        px, py = qx, qy
    return total


def _hard(rng: random.Random) -> dict:
    return rng.choice([_cone, _paraboloid, _perfect_square_curve, _sphere_zone])(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fy1ca_shells_surface", level=19, presets={t: dict(tier=t) for t in _TIERS})
def fy1ca_shells_surface(rng: random.Random, tier: str = "easy") -> dict:
    """Shells about the y-axis (easy), between curves or about another line (medium), surface area (hard)."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build_with_redraws(rng, _TIERS[tier], f"a {tier} shells or surface-area question")


# -- exam ----------------------------------------------------------------------------------------------------------------


def _exam_build(rng: random.Random) -> dict:
    k = rng.choice([3, 4, 5, 6])
    x0 = rng.choice([1, 2])
    rho = rng.choice(["2.4", "2.7", "7.8"])
    height = k * x0 - x0**2
    volume = pi * Rational(k**4, 6)
    v_float = float(volume)
    top, bottom = k * x, x**2
    check(close(_shell_numeric(top, bottom, 0, k, None), v_float, 1e-6), "numerical volume disagrees")
    mass = v_float * float(rho) / 1000
    if not (clear_of_rounding(v_float, 1) and clear_of_rounding(mass, 2)):
        raise Redraw
    if round(mass, 2) < 0.05:
        raise Redraw

    a_dist = fraction_distractors(
        Fraction(k),
        [
            (k * k, "fy1ca-shell-y-coordinate-given"),
            (Fraction(k, 2), "fy1ca-shell-intersection-slip"),
            (k + 1, "fy1ca-shell-intersection-slip"),
            (k - 1, "fy1ca-shell-intersection-slip"),
        ],
    )
    b_dist = fraction_distractors(
        Fraction(height),
        [
            (x0**2 - k * x0, "fy1ca-shell-height-sign"),
            (k * x0, "fy1ca-shell-height-not-difference"),
            (x0**2, "fy1ca-shell-height-not-difference"),
            (k - x0, "fy1ca-shell-height-slip"),
        ],
    )
    c_dist = number_distractors(
        v_float,
        [
            (float(2 * pi * Rational(k**5, 15)), "fy1ca-shell-disc-formula"),
            (float(pi * Rational(k**3, 3)), "fy1ca-shell-radius-forgotten"),
            (float(pi * Rational(k**4, 12)), "fy1ca-shell-two-forgotten"),
            (float(2 * pi * Rational(k**4, 3)), "fy1ca-shell-height-not-difference"),
            (float(Rational(k**4, 6)), "fy1ca-shell-pi-forgotten"),
        ],
        1,
    )
    d_dist = number_distractors(
        mass,
        [
            (v_float * float(rho), "fy1ca-shell-units-grams"),
            (v_float / float(rho) / 1000, "fy1ca-shell-density-divided"),
            (v_float / 1000, "fy1ca-shell-density-forgotten"),
            (v_float * float(rho) / 100, "fy1ca-shell-units-slip"),
        ],
        2,
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        need_distractors(dist)
    stem = phrase(
        rng,
        (
            "A concrete planter is made by rotating the region R about the y-axis, where R is the region in the first quadrant between the curves y = x² and y = {k}x. Lengths are in centimetres and the concrete has density {rho} g/cm³.",
            "The region R lies in the first quadrant between y = x² and y = {k}x (lengths in centimetres). A solid ornament is formed by rotating R about the y-axis and is cast in a material of density {rho} g/cm³.",
            "A garden bollard is a solid of revolution made by rotating about the y-axis the region R that lies between y = x² and y = {k}x for x ≥ 0 (lengths in centimetres). It is cast from a material of density {rho} g/cm³.",
        ),
        k=k,
        rho=rho,
    )
    parts = [
        fraction_part(
            "Find the x-coordinate of the point, other than the origin, where the two curves meet.",
            Fraction(k),
            1,
            [f"x² = {k}x gives x(x - {k}) = 0, so x = {k}."],
            a_dist,
        ),
        fraction_part(
            f"A thin cylindrical shell of radius x = {x0} is formed. Find its height, in cm.",
            Fraction(height),
            2,
            [f"The height is the distance between the curves: {k}({x0}) - {x0}² = {height}."],
            b_dist,
            follow_through=f"part_a*{x0} - {x0}**2",
        ),
        number_part(
            "Use cylindrical shells to find the volume of the solid, in cm³, correct to 1 decimal place.",
            v_float,
            1,
            3,
            [
                f"V = 2π times the integral of x({k}x - x²) from 0 to {k} = 2π[{k}x³/3 - x⁴/4] from 0 to {k}.",
                f"V = {_show(volume)} = {v_float:.1f} cm³.",
            ],
            c_dist,
            follow_through="pi*part_a**4/6",
            sensible_range=(0, None),
        ),
        number_part(
            "Find the mass of the solid, in kilograms, correct to 2 decimal places.",
            mass,
            2,
            2,
            [f"Mass = {v_float:.1f} × {rho} g = {v_float * float(rho):.1f} g = {mass:.2f} kg."],
            d_dist,
            follow_through=f"part_c*{rho}/1000",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("fy1ca_shells_surface_exam", level=19, presets={"exam": {}})
def fy1ca_shells_surface_exam(rng: random.Random) -> dict:
    """A region between y = x² and y = kx rotated about the y-axis: meeting point, shell height, volume, mass."""
    return build_with_redraws(rng, _exam_build, "a shells exam chain")
