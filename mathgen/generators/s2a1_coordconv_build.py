"""Generates `s2a1_coordconv` questions (easy, medium, hard) and the exam chain `exd_x_s2a1_coordconv`.

``uc2-coordinate-conversion-3d`` (``multivariable-and-vector-calculus``, answer ``expression``): converting points
and surfaces between Cartesian, cylindrical and spherical coordinates. Every question asks for ONE quantity and
states the range it is to lie in, so the answer is unique: θ in [0, 2π), φ in [0, π], r and ρ non-negative.
(Spherical coordinates here follow the usual convention: φ is the angle down from the positive z-axis.)

* easy: a point between Cartesian and cylindrical coordinates: the angle θ (the learner must choose the quadrant), the
  distance r from the axis, or a Cartesian coordinate from (r, θ, z).
* medium: a point between Cartesian, cylindrical and spherical coordinates: ρ, φ or θ from the Cartesian
  coordinates; x, y or z from (ρ, φ, θ); r or z from (ρ, φ, θ); φ from (r, θ, z).
* hard: a surface written in another system. From Cartesian: a sphere through the origin, a plane, a cylinder
  or a cone as ρ in terms of φ, or as the constant φ; a cylinder through the origin as r in terms of θ; a paraboloid,
  a sphere or a hyperboloid as r in terms of z (or z in terms of r). The other way: the radius or centre of
  the shape a spherical or cylindrical equation describes.
* exam (``exd_x_s2a1_coordconv``): a drone at a point P given in Cartesian coordinates: (a) its distance r from the
  vertical axis, (b) the angle θ, (c) its distance ρ from the origin (follows through from (a)), (d) the angle φ down
  from the vertical (follows through from (c)).

Every key is worked out twice: from the defining formulas with exact SymPy values, and by building the point back
from the answer (or, for a surface, by substituting the conversion into the equation) and comparing numerically.
"""

from __future__ import annotations

import functools
import math
import re

import sympy

from ..core import VerificationError, attempts, check, generator
from ._s2a1_common import exam_part, expression_answer, latex_of, stem, wrong_answers

S = sympy.Integer
pi = sympy.pi
sqrt = sympy.sqrt
x_, y_, z_ = sympy.symbols("x y z", real=True)
rho_, phi_, theta_ = sympy.symbols("rho phi theta", positive=True)
r_ = sympy.Symbol("r", positive=True)


def disp(v) -> str:
    """A value as the learner reads it in a stem: ``-2√3``, ``5π/6``."""
    s = str(sympy.expand(sympy.sympify(v))).replace("**", "^")
    s = re.sub(r"sqrt\((\d+)\)", r"√\1", s)
    s = re.sub(r"(\d)\*√", r"\1√", s)
    s = s.replace("*pi", "π").replace("pi", "π").replace("*", "")
    return s


def triple(a, b, c) -> str:
    return f"({disp(a)}, {disp(b)}, {disp(c)})"


def _trig(angle):
    return sympy.cos(angle), sympy.sin(angle)


# ---------------------------------------------------------------------------------------------------------------
# points


_SIXTHS = [sympy.Rational(k, 6) * pi for k in range(12)]
_FOURTHS = [sympy.Rational(2 * k + 1, 4) * pi for k in range(4)]
_PHIS = [pi / 6, pi / 4, pi / 3, pi / 2, 2 * pi / 3, 3 * pi / 4, 5 * pi / 6]


@functools.lru_cache(maxsize=None)
def _xyz(rho, phi, theta):
    """Exact Cartesian coordinates of a spherical point from the tables (the same few hundred points recur)."""
    x = sympy.expand(rho * sympy.sin(phi) * sympy.cos(theta))
    y = sympy.expand(rho * sympy.sin(phi) * sympy.sin(theta))
    z = sympy.expand(rho * sympy.cos(phi))
    return x, y, z, max(len(disp(v)) for v in (x, y, z))


@functools.lru_cache(maxsize=None)
def _xy(r, theta):
    return sympy.expand(r * sympy.cos(theta)), sympy.expand(r * sympy.sin(theta))


def _planar(rng):
    """(r, θ, x, y): a point in the xy-plane whose coordinates are exact and short."""
    if rng.random() < 0.55:
        r, theta = S(rng.randint(1, 6)), rng.choice(_SIXTHS)
    else:
        r, theta = rng.randint(1, 4) * sqrt(2), rng.choice(_FOURTHS)
    x, y = _xy(r, theta)
    return r, theta, x, y  # theta is in [0, 2π) by construction


def _spatial(rng):
    """(ρ, φ, θ, x, y, z) with short exact coordinates."""
    for _ in attempts(200, "a spatial point with short coordinates"):
        rho = S(rng.choice([2, 3, 4, 5, 6, 8]))
        phi = rng.choice(_PHIS)
        theta = rng.choice(_SIXTHS + _FOURTHS)
        x, y, z, width = _xyz(rho, phi, theta)
        if width <= 9 and not (x == 0 and y == 0):
            return rho, phi, theta, x, y, z
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# easy


def _easy(rng):
    kind = rng.choice(["theta", "theta", "r", "cart"])
    z = S(rng.choice([-3, -2, -1, 1, 2, 3, 4, 5]))
    r, theta, x, y = _planar(rng)
    if kind == "theta":
        value = theta
        check(
            sympy.simplify(r * sympy.cos(value) - x) == 0 and sympy.simplify(r * sympy.sin(value) - y) == 0,
            "theta does not rebuild the point",
        )
        text = f"The point P has Cartesian coordinates «{triple(x, y, z)}». Find its cylindrical coordinate «θ», taking «0 ≤ θ < 2π»."
        raw = sympy.atan(y / x) if x != 0 else None
        cands = [
            ("coord-theta-quadrant-ignored", raw),
            ("coord-theta-arctan-inverted", sympy.atan(x / y) if y != 0 else None),
            ("coord-theta-half-turn", (value + pi) % (2 * pi)),
            ("coord-theta-reference-angle", abs(raw) if raw is not None else None),
            ("coord-theta-mirrored", (2 * pi - value) % (2 * pi)),
        ]
        working = [
            f"tan θ = y/x, and the point lies in the quadrant where x = {disp(x)} and y = {disp(y)}, so θ = {disp(value)}."
        ]
    elif kind == "r":
        value = sqrt(x**2 + y**2)
        value = sympy.simplify(value)
        text = f"The point P has Cartesian coordinates «{triple(x, y, z)}». Find its cylindrical coordinate «r», the distance of P from the z-axis."
        cands = [
            ("coord-r-sum-not-root", abs(x) + abs(y)),
            ("coord-r-includes-z", sqrt(x**2 + y**2 + z**2)),
            ("coord-r-root-missing", x**2 + y**2),
            ("coord-r-difference-of-squares", sqrt(abs(x**2 - y**2))),
        ]
        working = [f"r = √(x² + y²) = {disp(value)}."]
    else:
        which = rng.choice(["x", "y"])
        value = x if which == "x" else y
        text = f"A point has cylindrical coordinates «(r, θ, z) = {triple(r, theta, z)}». Find its Cartesian coordinate «{which}»."
        cos, sin = _trig(theta)
        cands = [
            ("coord-sin-cos-swapped", r * (sin if which == "x" else cos)),
            ("coord-trig-factor-dropped", r),
            ("coord-reference-angle-sign", r * abs(cos if which == "x" else sin)),
            ("coord-sign-flipped", -value),
        ]
        working = [f"{which} = r {'cos' if which == 'x' else 'sin'} θ = {disp(value)}."]
    return text, value, cands, working


# ---------------------------------------------------------------------------------------------------------------
# medium


def _medium(rng):
    kind = rng.choice(["rho", "phi", "phi", "theta", "x", "y", "z", "r", "z_cyl", "phi_cyl"])
    rho, phi, theta, x, y, z = _spatial(rng)
    r = sympy.simplify(rho * sympy.sin(phi))
    cartesian = triple(x, y, z)
    sph = f"(ρ, φ, θ) = {triple(rho, phi, theta)}"
    if kind == "rho":
        value = rho
        text = f"The point P has Cartesian coordinates «{cartesian}». Find its spherical coordinate «ρ»."
        cands = [
            ("coord-rho-planar", r),
            ("coord-rho-root-missing", rho**2),
            ("coord-rho-sum-not-root", abs(x) + abs(y) + abs(z)),
            ("coord-rho-height-only", abs(z)),
            ("coord-rho-difference-of-squares", sqrt(abs(r**2 - z**2))),
        ]
        working = [f"ρ = √(x² + y² + z²) = {disp(rho)}."]
    elif kind == "phi":
        value = phi
        text = f"The point P has Cartesian coordinates «{cartesian}». Find its spherical coordinate «φ», the angle down from the positive z-axis, with «0 ≤ φ ≤ π»."
        cands = [
            ("coord-phi-complement", pi / 2 - phi),
            ("coord-phi-reflected", pi - phi),
            ("coord-phi-arctan-inverted", sympy.atan(sympy.nsimplify(z / r)) if r != 0 else None),
            ("coord-phi-sine-for-cosine", sympy.asin(sympy.nsimplify(z / rho))),
            ("coord-phi-quadrant-ignored", sympy.atan(sympy.nsimplify(r / z)) if z != 0 else None),
            ("coord-phi-cosine-for-angle", sympy.nsimplify(z / rho)),
        ]
        working = [f"cos φ = z/ρ = {disp(z)}/{disp(rho)}, so φ = {disp(phi)}."]
    elif kind == "theta":
        value = theta
        text = f"The point P has Cartesian coordinates «{cartesian}». Find its spherical coordinate «θ», taking «0 ≤ θ < 2π»."
        raw = sympy.atan(y / x) if x != 0 else None
        cands = [
            ("coord-theta-quadrant-ignored", raw),
            ("coord-theta-arctan-inverted", sympy.atan(x / y) if y != 0 else None),
            ("coord-theta-half-turn", (value + pi) % (2 * pi)),
            ("coord-theta-reference-angle", abs(raw) if raw is not None else None),
        ]
        working = [f"tan θ = y/x with x = {disp(x)}, y = {disp(y)}, so θ = {disp(value)}."]
    elif kind in ("x", "y", "z"):
        value = {"x": x, "y": y, "z": z}[kind]
        text = f"A point has spherical coordinates «{sph}», with «φ» measured down from the positive z-axis. Find its Cartesian coordinate «{kind}»."
        cands = {
            "x": [
                ("coord-spherical-phi-complement", rho * sympy.cos(phi) * sympy.cos(theta)),
                ("coord-spherical-sin-phi-dropped", rho * sympy.cos(theta)),
                ("coord-sin-cos-swapped", rho * sympy.sin(phi) * sympy.sin(theta)),
                ("coord-spherical-theta-dropped", rho * sympy.sin(phi)),
            ],
            "y": [
                ("coord-spherical-phi-complement", rho * sympy.cos(phi) * sympy.sin(theta)),
                ("coord-spherical-sin-phi-dropped", rho * sympy.sin(theta)),
                ("coord-sin-cos-swapped", rho * sympy.sin(phi) * sympy.cos(theta)),
                ("coord-spherical-theta-dropped", rho * sympy.sin(phi)),
            ],
            "z": [
                ("coord-spherical-phi-complement", rho * sympy.sin(phi)),
                ("coord-spherical-rho-dropped", sympy.cos(phi)),
                ("coord-spherical-sin-for-cos", rho * sympy.sin(phi) * sympy.cos(theta)),
                ("coord-sign-flipped", -z),
            ],
        }[kind]
        working = [f"x = ρ sin φ cos θ, y = ρ sin φ sin θ, z = ρ cos φ, so {kind} = {disp(value)}."]
    elif kind == "r":
        value = r
        text = f"A point has spherical coordinates «{sph}», with «φ» measured down from the positive z-axis. Find its cylindrical coordinate «r»."
        cands = [
            ("coord-spherical-phi-complement", rho * sympy.cos(phi)),
            ("coord-spherical-sin-phi-dropped", rho),
            ("coord-spherical-sin-phi-squared", rho * sympy.sin(phi) ** 2),
            ("coord-spherical-theta-factor", rho * sympy.sin(phi) * sympy.cos(theta)),
        ]
        working = [f"r = ρ sin φ = {disp(value)}."]
    elif kind == "z_cyl":
        value = z
        text = f"A point has spherical coordinates «{sph}», with «φ» measured down from the positive z-axis. Find its cylindrical coordinate «z»."
        cands = [
            ("coord-spherical-phi-complement", rho * sympy.sin(phi)),
            ("coord-spherical-rho-dropped", sympy.cos(phi)),
            ("coord-spherical-cos-squared", rho * sympy.cos(phi) ** 2),
            ("coord-sign-flipped", -z),
        ]
        working = [f"z = ρ cos φ = {disp(value)}."]
    else:
        value = phi
        text = f"A point has cylindrical coordinates «(r, θ, z) = {triple(r, theta, z)}». Find its spherical coordinate «φ», with «0 ≤ φ ≤ π»."
        cands = [
            ("coord-phi-complement", pi / 2 - phi),
            ("coord-phi-reflected", pi - phi),
            ("coord-phi-arctan-inverted", sympy.atan(sympy.nsimplify(z / r)) if r != 0 else None),
            ("coord-phi-sine-for-cosine", sympy.asin(sympy.nsimplify(z / rho))),
            ("coord-phi-quadrant-ignored", sympy.atan(sympy.nsimplify(r / z)) if z != 0 else None),
            ("coord-phi-cosine-for-angle", sympy.nsimplify(z / rho)),
        ]
        working = [f"tan φ = r/z with the angle taken from the positive z-axis, so φ = {disp(phi)}."]
    return text, sympy.nsimplify(value), cands, working


# ---------------------------------------------------------------------------------------------------------------
# hard: surfaces


def _hard_to_spherical(rng):
    kind = rng.choice(["sphere", "plane", "cylinder", "cone", "cone"])
    if kind == "sphere":
        a = rng.randint(1, 5)
        text = f"The sphere «x² + y² + z² = {2 * a}z» is written in spherical coordinates as «ρ = f(φ)», for «0 ≤ φ ≤ π/2». Find «f(φ)», as an expression in phi."
        value = 2 * a * sympy.cos(phi_)
        cands = [
            ("coord-surface-sine-for-cosine", 2 * a * sympy.sin(phi_)),
            ("coord-surface-coefficient-dropped", a * sympy.cos(phi_)),
            ("coord-surface-power-slip", sqrt(2 * a * sympy.cos(phi_))),
            ("coord-surface-constant-radius", S(2 * a)),
        ]
        # check: points of rho = 2a cos(phi) satisfy the equation
        for phi in (0.3, 0.7, 1.2):
            for th in (0.4, 2.0):
                rho = float(value.subs(phi_, phi))
                px, py, pz = rho * math.sin(phi) * math.cos(th), rho * math.sin(phi) * math.sin(th), rho * math.cos(phi)
                check(abs(px * px + py * py + pz * pz - 2 * a * pz) < 1e-9, "sphere does not satisfy its equation")
        working = ["Substitute x² + y² + z² = ρ² and z = ρ cos φ, then divide by ρ."]
    elif kind == "plane":
        c = rng.randint(2, 7)
        text = f"The plane «z = {c}» is written in spherical coordinates as «ρ = f(φ)», for «0 ≤ φ < π/2». Find «f(φ)», as an expression in phi."
        value = c / sympy.cos(phi_)
        cands = [
            ("coord-surface-reciprocal-slip", c * sympy.cos(phi_)),
            ("coord-surface-sine-for-cosine", c / sympy.sin(phi_)),
            ("coord-surface-sine-for-cosine", c * sympy.sin(phi_)),
            ("coord-surface-constant-radius", S(c)),
        ]
        for phi in (0.3, 0.9, 1.3):
            check(abs(float(value.subs(phi_, phi)) * math.cos(phi) - c) < 1e-9, "plane does not satisfy its equation")
        working = ["z = ρ cos φ = c, so ρ = c/cos φ."]
    elif kind == "cylinder":
        c = rng.randint(2, 7)
        text = f"The cylinder «x² + y² = {c * c}» is written in spherical coordinates as «ρ = f(φ)», for «0 < φ < π». Find «f(φ)», as an expression in phi."
        value = c / sympy.sin(phi_)
        cands = [
            ("coord-surface-reciprocal-slip", c * sympy.sin(phi_)),
            ("coord-surface-sine-for-cosine", c / sympy.cos(phi_)),
            ("coord-surface-sine-for-cosine", c * sympy.cos(phi_)),
            ("coord-surface-constant-radius", S(c)),
        ]
        for phi in (0.3, 0.9, 1.3):
            check(
                abs((float(value.subs(phi_, phi)) * math.sin(phi)) ** 2 - c * c) < 1e-9,
                "cylinder does not satisfy its equation",
            )
        working = ["x² + y² = ρ² sin² φ = c², so ρ = c/sin φ."]
    else:
        name = rng.choice(["pi/6", "pi/4", "pi/3"])
        k2 = {"pi/6": "z² = 3(x² + y²)", "pi/4": "z² = x² + y²", "pi/3": "3z² = x² + y²"}[name]
        phi = {"pi/6": pi / 6, "pi/4": pi / 4, "pi/3": pi / 3}[name]
        text = f"The upper half of the cone «{k2}», with «z ≥ 0», is written in spherical coordinates as «φ = α». Find «α», in radians."
        value = phi
        cands = [
            ("coord-phi-complement", pi / 2 - phi),
            ("coord-phi-reflected", pi - phi),
            ("coord-surface-opening-angle", 2 * phi),
            ("coord-surface-slope-as-angle", {"pi/6": sqrt(3), "pi/4": S(1), "pi/3": 1 / sqrt(3)}[name]),
        ]
        working = [f"On the cone tan φ = √(x² + y²)/z, which is {sympy.tan(phi)} here, so φ = {disp(phi)}."]
    return text, value, cands, working


def _hard_to_cylindrical(rng):
    kind = rng.choice(["circle_x", "circle_y", "paraboloid", "sphere", "hyperboloid"])
    if kind in ("circle_x", "circle_y"):
        a = rng.randint(1, 5)
        axis = "x" if kind == "circle_x" else "y"
        fn = sympy.cos if kind == "circle_x" else sympy.sin
        other = sympy.sin if kind == "circle_x" else sympy.cos
        text = f"The cylinder «x² + y² = {2 * a}{axis}» is written in cylindrical coordinates as «r = f(θ)», for the values of «θ» where «r > 0». Find «f(θ)», as an expression in theta."
        value = 2 * a * fn(theta_)
        cands = [
            ("coord-surface-sine-for-cosine", 2 * a * other(theta_)),
            ("coord-surface-coefficient-dropped", a * fn(theta_)),
            ("coord-surface-constant-radius", S(2 * a)),
            ("coord-surface-power-slip", 4 * a * a * fn(theta_) ** 2),
        ]
        for th in (0.3, 0.9, 1.2):
            rr = float(value.subs(theta_, th))
            px, py = rr * math.cos(th), rr * math.sin(th)
            check(
                abs(px * px + py * py - 2 * a * (px if axis == "x" else py)) < 1e-9,
                "circle does not satisfy its equation",
            )
        working = [
            f"x² + y² = r² and {axis} = r {'cos' if axis == 'x' else 'sin'} θ, so r² = {2 * a}r {'cos' if axis == 'x' else 'sin'} θ."
        ]
    elif kind == "paraboloid":
        c = rng.randint(2, 6)
        text = f"The paraboloid «z = {c}(x² + y²)» is written in cylindrical coordinates as «z = g(r)». Find «g(r)», as an expression in r."
        value = c * r_**2
        cands = [
            ("coord-surface-r-not-squared", c * r_),
            ("coord-surface-coefficient-dropped", r_**2),
            ("coord-surface-power-slip", c * r_**3),
            ("coord-surface-power-slip", sqrt(c) * r_),
        ]
        working = ["x² + y² = r², so z = c r²."]
    elif kind == "sphere":
        R = rng.randint(2, 7)
        text = f"The sphere «x² + y² + z² = {R * R}» is written in cylindrical coordinates as «r = h(z)», for «r ≥ 0». Find «h(z)», as an expression in z."
        value = sqrt(R * R - z_**2)
        cands = [
            ("coord-surface-sign-slip", R - z_),
            ("coord-surface-sign-slip", sqrt(R * R + z_**2)),
            ("coord-surface-root-missing", R * R - z_**2),
            ("coord-surface-power-slip", sqrt(R - z_**2)),
        ]
        working = ["x² + y² = r², so r² + z² = R² and r = √(R² - z²)."]
    else:
        c = rng.randint(1, 4)
        text = f"The hyperboloid «x² + y² - z² = {c * c}» is written in cylindrical coordinates as «r = h(z)», for «r ≥ 0». Find «h(z)», as an expression in z."
        value = sqrt(c * c + z_**2)
        cands = [
            ("coord-surface-sign-slip", sqrt(c * c - z_**2)),
            ("coord-surface-sign-slip", c + z_),
            ("coord-surface-root-missing", c * c + z_**2),
            ("coord-surface-power-slip", sqrt(c + z_**2)),
        ]
        working = ["x² + y² = r², so r² = c² + z²."]
    return text, value, cands, working


def _hard_from_curvilinear(rng):
    kind = rng.choice(["sph_sphere", "cyl_circle", "sph_cylinder", "sph_cone"])
    if kind == "sph_sphere":
        a = rng.randint(1, 6)
        text = f"In spherical coordinates the surface «ρ = {2 * a}cos φ» is a sphere. Find the radius of this sphere."
        value = S(a)
        cands = [
            ("coord-surface-diameter-for-radius", S(2 * a)),
            ("coord-surface-radius-squared", S(a * a)),
            ("coord-surface-halved-again", sympy.Rational(a, 2)),
        ]
        working = [f"ρ² = {2 * a}ρ cos φ gives x² + y² + z² = {2 * a}z, so x² + y² + (z - {a})² = {a * a}."]
    elif kind == "cyl_circle":
        a = rng.randint(1, 6)
        text = f"In cylindrical coordinates the curve «r = {2 * a}cos θ» is a circle in the xy-plane. Find the x-coordinate of its centre."
        value = S(a)
        cands = [
            ("coord-surface-diameter-for-radius", S(2 * a)),
            ("coord-surface-radius-squared", S(a * a)),
            ("coord-surface-halved-again", sympy.Rational(a, 2)),
        ]
        working = [f"r² = {2 * a}r cos θ gives x² + y² = {2 * a}x, so (x - {a})² + y² = {a * a}."]
    elif kind == "sph_cylinder":
        c = rng.randint(2, 8)
        text = f"In spherical coordinates the surface «ρ sin φ = {c}» is a cylinder about the z-axis. Find its radius."
        value = S(c)
        cands = [
            ("coord-surface-radius-squared", S(c * c)),
            ("coord-surface-sine-for-cosine", S(c) / 2),
            ("coord-surface-diameter-for-radius", S(2 * c)),
            ("coord-surface-halved-again", sympy.Rational(c, 2)),
        ]
        working = [f"ρ sin φ = r, so r = {c}."]
    else:
        name = rng.choice(["pi/6", "pi/4", "pi/3"])
        phi = {"pi/6": pi / 6, "pi/4": pi / 4, "pi/3": pi / 3}[name]
        text = (
            f"In spherical coordinates the surface «φ = {disp(phi)}» is a cone. Its upper half is the graph "
            "«z = k√(x² + y²)». Find «k»."
        )
        value = sympy.nsimplify(1 / sympy.tan(phi))
        cands = [
            ("coord-surface-slope-inverted", sympy.nsimplify(sympy.tan(phi))),
            ("coord-surface-sine-for-cosine", sympy.nsimplify(sympy.cos(phi))),
            ("coord-surface-cosine-for-cotangent", sympy.nsimplify(sympy.sin(phi))),
            ("coord-surface-angle-as-slope", sympy.nsimplify(phi)),
        ]
        working = ["On the cone z = ρ cos φ and √(x² + y²) = ρ sin φ, so k = cot φ."]
    return text, value, cands, working


def _hard(rng):
    pick = rng.random()
    if pick < 0.4:
        return _hard_to_spherical(rng)
    if pick < 0.75:
        return _hard_to_cylindrical(rng)
    return _hard_from_curvilinear(rng)


@generator(
    "s2a1_coordconv",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a1_coordconv(rng, tier="easy"):
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(60, "a coordinate question with enough distractors"):
        try:
            text, value, cands, working = build(rng)
        except VerificationError:
            continue
        signed = (
            re.search(r"coordinate «(x|y|z|θ)»", text) is not None
        )  # a quantity that can be negative or taken the wrong way round
        wrong = wrong_answers(value, cands, 4, kind="expression", allow_negative=signed)
        if len(wrong) >= 3:
            break
    kind, machine, answer_text = expression_answer(value)
    return dict(
        **stem(text),
        answer=machine,
        answer_text=answer_text,
        answer_latex=latex_of(value),
        answer_kind=kind,
        distractors=wrong,
        working=working,
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    rho, phi, theta, x, y, z = _spatial(rng)
    r = sympy.simplify(rho * sympy.sin(phi))
    check(x != 0 and y != 0 and z != 0, "the point is on a coordinate plane, which hides the errors")
    th = theta  # in [0, 2π) by construction; the point is rebuilt from it below
    fx, fy, fz, fr, fth, frho, fphi = (float(t) for t in (x, y, z, r, th, rho, phi))
    check(abs(fr - math.hypot(fx, fy)) < 1e-9 and abs(frho - math.sqrt(fr**2 + fz**2)) < 1e-9, "r or rho disagrees")
    check(abs(fr * math.cos(fth) - fx) < 1e-9 and abs(fr * math.sin(fth) - fy) < 1e-9, "theta disagrees")
    check(abs(math.cos(fphi) - fz / frho) < 1e-9, "phi disagrees")
    raw = sympy.atan(y / x) if x != 0 else None
    a_c = [
        ("coord-r-sum-not-root", abs(x) + abs(y)),
        ("coord-r-includes-z", sqrt(x**2 + y**2 + z**2)),
        ("coord-r-root-missing", x**2 + y**2),
        ("coord-r-difference-of-squares", sqrt(abs(x**2 - y**2))),
    ]
    b_c = [
        ("coord-theta-quadrant-ignored", raw),
        ("coord-theta-arctan-inverted", sympy.atan(x / y) if y != 0 else None),
        ("coord-theta-half-turn", (th + pi) % (2 * pi)),
        ("coord-theta-reference-angle", abs(raw) if raw is not None else None),
        ("coord-theta-mirrored", (2 * pi - th) % (2 * pi)),
    ]
    c_c = [
        ("coord-rho-planar", r),
        ("coord-rho-root-missing", r**2 + z**2),
        ("coord-rho-sum-not-root", r + abs(z)),
        ("coord-rho-height-only", abs(z)),
        ("coord-rho-difference-of-squares", sqrt(abs(r**2 - z**2))),
    ]
    d_c = [
        ("coord-phi-complement", pi / 2 - phi),
        ("coord-phi-reflected", pi - phi),
        ("coord-phi-arctan-inverted", sympy.atan(sympy.nsimplify(z / r))),
        ("coord-phi-sine-for-cosine", sympy.asin(sympy.nsimplify(z / rho))),
        ("coord-phi-quadrant-ignored", sympy.atan(sympy.nsimplify(r / z)) if z != 0 else None),
        ("coord-phi-cosine-for-angle", sympy.nsimplify(z / rho)),
    ]
    parts = [
        exam_part(
            "Find the distance «r» of the drone from the vertical axis through the origin, in metres.",
            r,
            1,
            ["r = √(x² + y²)."],
            a_c,
            limit=3,
        ),
        exam_part(
            "Find the cylindrical angle «θ» of the drone, in radians, with «0 ≤ θ < 2π».",
            th,
            2,
            ["The point is in the quadrant given by the signs of x and y, so θ is not just arctan(y/x)."],
            b_c,
            limit=3,
            allow_negative=True,
        ),
        exam_part(
            "Find the distance «ρ» of the drone from the origin, in metres.",
            rho,
            2,
            ["ρ² = r² + z²."],
            c_c,
            follow_through=f"sqrt(part_a^2 + ({str(z).replace('**', '^')})^2)",
            limit=3,
        ),
        exam_part(
            "Find the spherical angle «φ» of the drone, measured down from the vertical, in radians, with «0 ≤ φ ≤ π».",
            phi,
            3,
            ["cos φ = z/ρ."],
            d_c,
            follow_through=f"acos(({str(z).replace('**', '^')})/part_c)",
            limit=3,
            allow_negative=True,
        ),
    ]
    check(all(parts), "a part has too few distractors")
    return dict(
        **stem(
            f"A drone hovers at the point «P({disp(x)}, {disp(y)}, {disp(z)})» relative to a ground station at the origin, "
            "with the z-axis pointing straight up and all lengths in metres."
        ),
        parts=parts,
    )


@generator("exd_x_s2a1_coordconv", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a1_coordconv(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam coordinate chain"):
        try:
            return _exam_once(rng)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
