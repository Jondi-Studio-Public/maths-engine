"""Generates `s2a1_cylindrical` questions (easy, medium, hard) and the exam chain `exd_x_s2a1_cylindrical`.

``uc2-cylindrical-triple-integral`` (``multivariable-and-vector-calculus``, answer ``expression``): triple
integrals in cylindrical coordinates with exact answers in pi. Every solid is described in Cartesian words, so
the learner has to choose the coordinates, find the limits and remember the factor r in dV = r dz dr dθ.

* easy: an integrand in x²+y² and z over a solid cylinder or half cylinder; the volume between a paraboloid and a
  plane.
* medium: the region between a cone and a paraboloid (the learner finds where they meet), the mass of a solid with a
  density that depends on the distance from the axis or on the height, and a half cylinder under a sloping plane,
  where the term in cos θ must be integrated over half a turn.
* hard: the part of a ball inside a cylinder; a cylinder through the origin (x²+y² ≤ 2ax), whose angle runs over half
  a turn and whose radius depends on the angle; the solid between two paraboloids.
* exam (``exd_x_s2a1_cylindrical``): a silo or a bowl with density ρ = k + m·z: (a) the volume, (b) the mass
  (follows through from (a)), (c) the height of the centre of mass (follows through from (b)), (d) the average
  density (follows through from (a) and (b)).

Each key is worked out twice. The exact route integrates in cylindrical coordinates with SymPy. The second route is in
tests/test_s2a1_cylindrical.py: a Cartesian Monte Carlo that reads only the question text, run over many seeds.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import sympy

from ..core import VerificationError, attempts, check, generator
from ._s2a1_common import (
    say_value,
    PI,
    close,
    etext,
    exam_part,
    expression_answer,
    gauss_nested,
    latex_of,
    stem,
    wrong_answers,
)

x, y, z = sympy.symbols("x y z", real=True)
r, t = sympy.symbols("r theta", positive=True)
Z = sympy.Symbol("z_", real=True)
RAD = sympy.Symbol("rad", positive=True)  # stands for sqrt(x² + y²) in a wrong route

S = sympy.Integer
R_ = sympy.Rational


@functools.lru_cache(maxsize=4096)
def _angular(i: int, j: int, lo, hi):
    return sympy.integrate(sympy.cos(t) ** i * sympy.sin(t) ** j, (t, lo, hi))


def _integrate(expr, var, lo, hi):
    """Exact integral of `expr` in `var` between the limits: term by term when it is a polynomial in `var`."""
    expr = sympy.expand(expr)
    if not expr.has(var):
        return sympy.expand(expr * (hi - lo))
    if not expr.is_polynomial(var):
        if var == r:
            return _radial_with_root(expr, lo, hi)
        return sympy.expand(sympy.integrate(expr, (var, lo, hi)))
    poly = sympy.Poly(expr, var)
    return sympy.expand(sum(c * (hi ** (n + 1) - lo ** (n + 1)) / (n + 1) for (n,), c in poly.terms()))


def _radial_with_root(expr, lo, hi):
    """∫ of a sum of c·r^p·(K - r²)^e dr (p odd, e a half-integer) by the substitution u = K - r²."""
    u = sympy.Symbol("u", positive=True)
    total = S(0)
    for term in sympy.Add.make_args(sympy.expand(expr)):
        coeff, p, e, base = S(1), S(0), S(0), None
        for factor in sympy.Mul.make_args(term):
            b, ex = factor.as_base_exp()
            if b == r:
                p += ex
            elif b.has(r):
                e += ex
                base = b
            else:
                coeff *= factor
        if base is None or p % 2 != 1:
            raise ValueError("radial integral not in the form r^odd (K - r^2)^e")
        k_const = sympy.expand(base + r**2)
        check(not k_const.has(r), "the root is not of the form K - r^2")
        q = int((p - 1) / 2)
        poly = sympy.Poly(sympy.expand((k_const - u) ** q), u)
        lo_u, hi_u = k_const - hi**2, k_const - lo**2
        anti = sum(c * (hi_u ** (n + e + 1) - lo_u ** (n + e + 1)) / (n + e + 1) for (n,), c in poly.terms())
        total += coeff * anti / 2
    return sympy.expand(total)


def _integrate_angle(expr, lo, hi):
    expr = sympy.expand(expr)
    if not expr.has(t):
        return sympy.expand(expr * (hi - lo))
    poly = sympy.Poly(expr, sympy.cos(t), sympy.sin(t))
    return sympy.expand(sum(c * _angular(i, j, lo, hi) for (i, j), c in poly.terms()))


@functools.lru_cache(maxsize=4096)
def cyl(f, th, rl, zl, jacobian: bool = True):
    """Exact iterated integral of the Cartesian integrand f(x, y, z) in cylindrical coordinates."""
    g = f.subs({x: r * sympy.cos(t), y: r * sympy.sin(t), z: Z, RAD: r}) * (r if jacobian else 1)
    g = _integrate(g, Z, zl[0], zl[1])
    g = _integrate(g, r, rl[0], rl[1])
    return _integrate_angle(g, th[0], th[1])


def safe_cyl(*args):
    """`cyl`, or None for a wrong route whose integral is not elementary (it is then not offered as an option)."""
    try:
        return cyl(*args)
    except ValueError:
        return None


def _poly_text(expr) -> str:
    return str(sympy.expand(expr)).replace("**", "^").replace("*", "").replace("^2", "²").replace("^3", "³")


def _integrand_text(f) -> str:
    text = _poly_text(f)
    return f"({text})" if len(sympy.Add.make_args(sympy.expand(f))) > 1 else text


# ---------------------------------------------------------------------------------------------------------------
# the solids. Each returns words (Cartesian description), the cylindrical limits, a Cartesian membership test, the
# bounding half-widths and the wrong routes' inputs.


def _cylinder(rng):
    a, h = rng.randint(1, 4), rng.randint(1, 5)
    half = rng.random() < 0.4
    th = (S(0), PI) if half else (S(0), 2 * PI)
    words = f"x² + y² ≤ {a * a}, {'y ≥ 0, ' if half else ''}0 ≤ z ≤ {h}"
    return dict(
        words=words,
        th=th,
        rl=(S(0), S(a)),
        zl=(S(0), S(h)),
        contains=lambda px, py, pz: px * px + py * py <= a * a and (py >= 0 or not half) and 0 <= pz <= h,
        box=(a, a, h),
        a=a,
        h=h,
        half=half,
    )


def _bowl(rng, c=None):
    a = rng.randint(1, 3)
    c = c or rng.choice([1, 1, 2, 3])
    h = c * a * a
    q = "" if c == 1 else str(c)
    return dict(
        words=f"{q}(x² + y²) ≤ z ≤ {h}" if c > 1 else f"x² + y² ≤ z ≤ {h}",
        th=(S(0), 2 * PI),
        rl=(S(0), S(a)),
        zl=(c * r**2, S(h)),
        contains=lambda px, py, pz: c * (px * px + py * py) <= pz <= h,
        box=(a, a, h),
        a=a,
        h=h,
        c=c,
    )


# ---------------------------------------------------------------------------------------------------------------
# easy


def _easy(rng):
    if rng.random() < 0.55:
        s = _cylinder(rng)
        al, be = rng.randint(1, 3), rng.choice([0, 0, 1, 2, 3])
        f = al * (x**2 + y**2) + be * z
        text = f"Evaluate «∭ {_integrand_text(f)} dV» over the solid E: «{s['words']}»."
        value = cyl(f, s["th"], s["rl"], s["zl"])
        spans = 2 * PI if not s["half"] else PI
        a, h = s["a"], s["h"]
        cands = [
            ("cylindrical-jacobian-omitted", cyl(f, s["th"], s["rl"], s["zl"], False)),
            ("cylindrical-radius-not-squared", cyl(al * RAD + be * z, s["th"], s["rl"], s["zl"])),
            ("cylindrical-radius-limit-squared", cyl(f, s["th"], (S(0), S(a * a)), s["zl"])),
            ("cylindrical-angle-range-halved", value / 2 if not s["half"] else value * 2),
        ]
        working = [
            f"With x = r cos θ and y = r sin θ, x² + y² = r² and dV = r dz dr dθ, with 0 ≤ r ≤ {a} and 0 ≤ z ≤ {h}."
        ]
        spec = s
    else:
        s = _bowl(rng)
        f = S(1)
        text = f"Find the volume of the solid «{s['words']}»."
        value = cyl(f, s["th"], s["rl"], s["zl"])
        a, h = s["a"], s["h"]
        cands = [
            ("cylindrical-jacobian-omitted", cyl(f, s["th"], s["rl"], s["zl"], False)),
            ("cylindrical-radius-from-height", cyl(f, s["th"], (S(0), S(h)), (s["c"] * r**2, S(h)))),
            ("cylindrical-bounding-cylinder", PI * a * a * h),
            ("cylindrical-angle-range-halved", value / 2),
        ]
        working = [
            f"The paraboloid meets the plane where r = {a}, so V = ∫ from 0 to 2π ∫ from 0 to {a} of (z limits) r dr dθ."
        ]
        spec = s
    return text, value, cands, working


# ---------------------------------------------------------------------------------------------------------------
# medium


def _medium_cone(rng):
    a, n = rng.randint(1, 3), rng.choice([1, 1, 2])
    m = n * a
    th = (S(0), 2 * PI)
    rl = (S(0), S(a))
    zl = (n * r**2, m * r)
    dens = rng.choice([("volume", S(1)), ("mass", x**2 + y**2), ("mass", z)])
    kind, f = dens
    k = rng.randint(1, 4)
    contains = lambda px, py, pz: n * (px * px + py * py) <= pz <= m * math.hypot(px, py)  # noqa: E731
    solid = dict(th=th, rl=rl, zl=zl, contains=contains, box=(a, a, m * a))
    npart = "x² + y²" if n == 1 else f"{n}(x² + y²)"
    mpart = "√(x² + y²)" if m == 1 else f"{m}√(x² + y²)"
    words = f"{npart} ≤ z ≤ {mpart}"
    if kind == "volume":
        text = f"Find the volume of the solid «{words}»."
        fk = f
    else:
        fk = k * f
        text = (
            f"A solid occupies the region «{words}». Its density at the point «(x, y, z)» is «ρ = {_integrand_text(fk)}» "
            "kilograms per cubic metre, with lengths in metres. Find the mass of the solid in kilograms."
        )
    value = cyl(fk, th, rl, zl)
    cands = [
        ("cylindrical-jacobian-omitted", cyl(fk, th, rl, zl, False)),
        ("cylindrical-surfaces-swapped", -value),
        ("cylindrical-intersection-misread", cyl(fk, th, (S(0), S(m)), (n * r**2, m * r))),
        ("cylindrical-bounding-cylinder", cyl(fk, th, rl, (S(0), m * r))),
    ]
    working = [f"The cone and the paraboloid meet where {m}r = {n}r², at r = {a}; integrate z from {n}r² to {m}r."]
    return text, value, cands, working


def _medium_slope(rng):
    a, p, q = rng.randint(1, 3), rng.randint(2, 5), rng.randint(1, 2)
    p = max(p, q * a)
    th = (-PI / 2, PI / 2)
    rl = (S(0), S(a))
    zl = (S(0), p + q * r * sympy.cos(t))
    f = S(1)
    contains = lambda px, py, pz: px * px + py * py <= a * a and px >= 0 and 0 <= pz <= p + q * px  # noqa: E731
    solid = dict(th=th, rl=rl, zl=zl, contains=contains, box=(a, a, p + q * a))
    plane = f"{p} + {q if q > 1 else ''}x"
    text = f"Find the volume of the solid «x² + y² ≤ {a * a}, x ≥ 0, 0 ≤ z ≤ {plane}»."
    value = cyl(f, th, rl, zl)
    cands = [
        ("cylindrical-jacobian-omitted", cyl(f, th, rl, zl, False)),
        ("cylindrical-x-as-r", cyl(f, th, rl, (S(0), p + q * r))),
        ("cylindrical-slope-term-dropped", cyl(f, th, rl, (S(0), S(p)))),
        ("cylindrical-angle-range-halved", cyl(f, (S(0), PI / 2), rl, zl)),
    ]
    working = [f"With x = r cos θ the height is {p} + {q}r cos θ and θ runs from -π/2 to π/2 for x ≥ 0."]
    return text, value, cands, working


def _medium(rng):
    return _medium_cone(rng) if rng.random() < 0.6 else _medium_slope(rng)


# ---------------------------------------------------------------------------------------------------------------
# hard


# (R, a, h) with a² + h² = R², every leg at least 3, R at most 41
_BALL_TRIPLES = [(R, a, h) for R in range(5, 42) for a in range(3, R) for h in range(3, R) if a * a + h * h == R * R]


def _hard_ball(rng):
    R, a, h = rng.choice(_BALL_TRIPLES)
    which = rng.choice(["volume", "volume", "x2y2", "z2"])
    f = {"volume": S(1), "x2y2": x**2 + y**2, "z2": z**2}[which]
    th = (S(0), 2 * PI)
    rl = (S(0), S(a))
    root = sympy.sqrt(R * R - r**2)
    zl = (-root, root)
    contains = lambda px, py, pz: px * px + py * py <= a * a and px * px + py * py + pz * pz <= R * R  # noqa: E731
    solid = dict(th=th, rl=rl, zl=zl, contains=contains, box=(a, a, R))
    ball = f"x² + y² + z² ≤ {R * R}"
    cyl_words = f"x² + y² ≤ {a * a}"
    what = {
        "volume": "Find the volume of the solid",
        "x2y2": "Evaluate «∭ (x² + y²) dV» over the solid",
        "z2": "Evaluate «∭ z² dV» over the solid",
    }[which]
    text = f"{what} that lies inside both the ball «{ball}» and the cylinder «{cyl_words}»."
    value = cyl(f, th, rl, zl)
    cands = [
        ("cylindrical-jacobian-omitted", safe_cyl(f, th, rl, zl, False)),
        ("cylindrical-sphere-height-constant", cyl(f, th, rl, (-S(R), S(R)))),
        ("cylindrical-ball-only", cyl(f, th, (S(0), S(R)), (-sympy.sqrt(R * R - r**2), sympy.sqrt(R * R - r**2)))),
        ("cylindrical-upper-half-only", cyl(f, th, rl, (S(0), root))),
    ]
    working = [
        f"The cylinder meets the sphere at z = ±{h}; for 0 ≤ r ≤ {a} the height runs from -√({R * R} - r²) to √({R * R} - r²)."
    ]
    return text, value, cands, working


def _hard_offset(rng):
    a = rng.randint(1, 4)
    axis = rng.choice(["x", "y"])
    kind = rng.choice(["paraboloid", "paraboloid", "plane"])
    if axis == "x":
        th = (-PI / 2, PI / 2)
        rim = 2 * a * sympy.cos(t)
        words = f"x² + y² ≤ {2 * a}x"
        contains = lambda px, py: px * px + py * py <= 2 * a * px  # noqa: E731
    else:
        th = (S(0), PI)
        rim = 2 * a * sympy.sin(t)
        words = f"x² + y² ≤ {2 * a}y"
        contains = lambda px, py: px * px + py * py <= 2 * a * py  # noqa: E731
    if kind == "paraboloid":
        c = rng.choice([1, 2, 3])
        top = c * r**2
        top_words = "x² + y²" if c == 1 else f"{c}(x² + y²)"
        topf = lambda px, py: c * (px * px + py * py)  # noqa: E731
        what = f"Find the volume of the solid below the surface «z = {top_words}», above the plane «z = 0» and inside the cylinder «{words}»."
    else:
        hgt = rng.randint(2, 9)
        top = S(hgt)
        topf = lambda px, py: hgt  # noqa: E731
        what = f"Find the volume of the solid below the plane «z = {hgt}», above the plane «z = 0» and inside the cylinder «{words}»."
    rl = (S(0), rim)
    zl = (S(0), top)
    f = S(1)
    solid = dict(
        th=th,
        rl=rl,
        zl=zl,
        contains=lambda px, py, pz: contains(px, py) and 0 <= pz <= topf(px, py),
        box=(2 * a, 2 * a, 100 if kind == "plane" else 16 * a * a),
    )
    # sample z only up to the top of the solid's highest point
    top_max = hgt if kind == "plane" else c * 4 * a * a
    solid["box"] = (2 * a, 2 * a, top_max)
    # membership test samples z in [-top_max, top_max]; the cylindrical limits only describe z >= 0 so both agree
    value = cyl(f, th, rl, zl)
    full_turn = (S(0), 2 * PI)
    cands = [
        ("cylindrical-jacobian-omitted", cyl(f, th, rl, zl, False)),
        ("cylindrical-offset-angle-range", cyl(f, full_turn, rl, zl)),
        ("cylindrical-offset-radius", cyl(f, th, (S(0), S(2 * a)), zl)),
        ("cylindrical-offset-radius", cyl(f, th, (S(0), S(a)), zl)),
    ]
    working = [
        f"The cylinder is r = {2 * a}{'cos' if axis == 'x' else 'sin'} θ, so the angle runs over half a turn and r over 0 to {2 * a}{'cos' if axis == 'x' else 'sin'} θ."
    ]
    return what, value, cands, working


def _hard_paraboloids(rng):
    a = rng.randint(1, 7)
    top_c = 2 * a * a
    f = rng.choice([S(1), z, x**2 + y**2, z**2, z * (x**2 + y**2)])
    th, rl = (S(0), 2 * PI), (S(0), S(a))
    zl = (r**2, top_c - r**2)
    contains = lambda px, py, pz: (px * px + py * py) <= pz <= top_c - (px * px + py * py)  # noqa: E731
    solid = dict(th=th, rl=rl, zl=zl, contains=contains, box=(a, a, top_c))
    words = f"x² + y² ≤ z ≤ {top_c} - (x² + y²)"
    if f == 1:
        text = f"Find the volume of the solid «{words}»."
    elif f == z:
        text = f"A solid occupies «{words}», with lengths in metres and density «ρ = z» kilograms per cubic metre. Find its mass in kilograms."
    elif f == z**2:
        text = f"Evaluate «∭ z² dV» over the solid «{words}»."
    elif f == z * (x**2 + y**2):
        text = f"Evaluate «∭ z (x² + y²) dV» over the solid «{words}»."
    else:
        text = f"Evaluate «∭ (x² + y²) dV» over the solid «{words}»."
    value = cyl(f, th, rl, zl)
    cands = [
        ("cylindrical-jacobian-omitted", cyl(f, th, rl, zl, False)),
        ("cylindrical-intersection-misread", cyl(f, th, (S(0), S(top_c)), zl)),
        ("cylindrical-surfaces-swapped", -value),
        ("cylindrical-lower-surface-dropped", cyl(f, th, rl, (S(0), top_c - r**2))),
    ]
    working = [f"The paraboloids meet where r² = {top_c} - r², at r = {a}."]
    return text, value, cands, working


def _hard(rng):
    pick = rng.random()
    if pick < 0.35:
        return _hard_ball(rng)
    if pick < 0.7:
        return _hard_offset(rng)
    return _hard_paraboloids(rng)


@generator(
    "s2a1_cylindrical",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a1_cylindrical(rng, tier="easy"):
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(30, "a cylindrical integral with enough distractors"):
        text, value, cands, working = build(rng)
        check(sympy.sympify(value).is_real and float(value) > 0, "the integral is not positive")
        wrong = wrong_answers(value, cands, 4, kind="expression")
        if len(wrong) >= 3:
            break
    kind, machine, answer_text = expression_answer(value)
    out = stem(text)
    return dict(
        **out,
        answer=machine,
        answer_text=answer_text,
        answer_latex=latex_of(value),
        answer_kind=kind,
        distractors=wrong,
        working=working,
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_solid(rng):
    if rng.random() < 0.5:
        a, p, q = rng.randint(1, 3), rng.randint(3, 8), rng.randint(1, 2)
        p = max(p, q * a * a + 1)
        words = f"the silo «x² + y² ≤ {a * a}, 0 ≤ z ≤ {p} - {'' if q == 1 else str(q)}(x² + y²)»"
        solid = dict(
            th=(S(0), 2 * PI),
            rl=(S(0), S(a)),
            zl=(S(0), p - q * r**2),
            contains=lambda px, py, pz: px * px + py * py <= a * a and 0 <= pz <= p - q * (px * px + py * py),
            box=(a, a, p),
            top=p,
        )
        return words, solid
    a = rng.randint(1, 3)
    c = rng.choice([1, 2])
    h = c * a * a
    words = f"the bowl «{'x² + y²' if c == 1 else str(c) + '(x² + y²)'} ≤ z ≤ {h}»"
    solid = dict(
        th=(S(0), 2 * PI),
        rl=(S(0), S(a)),
        zl=(c * r**2, S(h)),
        contains=lambda px, py, pz: c * (px * px + py * py) <= pz <= h,
        box=(a, a, h),
        top=h,
    )
    return words, solid


def _exam_once(rng):
    words, s = _exam_solid(rng)
    k, m = rng.randint(1, 4), rng.randint(1, 3)
    th, rl, zl = s["th"], s["rl"], s["zl"]
    one, rho = S(1), k + m * z
    V = cyl(one, th, rl, zl)
    Iz = cyl(z, th, rl, zl)
    M = cyl(rho, th, rl, zl)
    Mz = cyl(z * rho, th, rl, zl)
    check(sympy.simplify(M - k * V - m * Iz) == 0, "mass is not kV + m times the integral of z")
    zbar = sympy.simplify(Mz / M)
    avg = sympy.simplify(M / V)
    check(0 < float(zbar) < s["top"], "centre of mass outside the solid")
    top = S(s["top"])
    check(zbar.is_Rational and avg.is_Rational, "the centre of mass or average density is not rational")

    a_c = [
        ("cylindrical-jacobian-omitted", cyl(one, th, rl, zl, False)),
        ("cylindrical-angle-range-halved", V / 2),
        ("cylindrical-bounding-cylinder", PI * rl[1] ** 2 * top),
    ]
    b_c = [
        ("triple-integral-density-term-dropped", k * V),
        ("triple-integral-density-maximum", (k + m * top) * V),
        ("triple-integral-density-midpoint", (k + m * top / 2) * V),
        ("cylindrical-jacobian-omitted", cyl(rho, th, rl, zl, False)),
    ]
    c_c = [
        ("triple-integral-moment-not-divided", Mz),
        ("triple-integral-centroid-uniform", sympy.simplify(Iz / V)),
        ("triple-integral-moment-inverted", M / Mz),
        ("triple-integral-wrong-coordinate", sympy.simplify(cyl(x * rho, th, rl, zl) / M) + top / 2),
    ]
    d_c = [
        ("triple-integral-moment-inverted", V / M),
        ("triple-integral-mass-not-divided", M),
        ("triple-integral-density-midpoint", k + m * top / 2),
        ("triple-integral-density-maximum", k + m * top),
    ]
    Iz_text = etext(Iz)
    Mz_text = etext(Mz)
    parts = [
        exam_part(
            "Find the volume of the solid, in cubic metres.",
            V,
            3,
            [f"V = ∫ from 0 to 2π ∫ over r ∫ over z of r dz dr dθ = {say_value(V)}."],
            a_c,
            sensible_range=(0, None),
        ),
        exam_part(
            "Hence find the mass of the solid, in kilograms.",
            M,
            3,
            [f"M = ∭ ρ dV = {k}V + {m}∭ z dV = {say_value(M)}."],
            b_c,
            follow_through=f"{k}*part_a + {m}*({Iz_text})",
            sensible_range=(0, None),
        ),
        exam_part(
            "Find the height of the centre of mass above the base of the vessel, in metres.",
            zbar,
            3,
            [f"z-bar = (∭ zρ dV)/M = ({say_value(Mz)})/M."],
            c_c,
            follow_through=f"({Mz_text})/part_b",
            sensible_range=(0, s["top"]),
        ),
        exam_part(
            "Find the average density of the solid, in kilograms per cubic metre.",
            avg,
            1,
            ["Average density = mass / volume."],
            d_c,
            follow_through="part_b/part_a",
            sensible_range=(0, None),
        ),
    ]
    check(all(parts), "a part has too few distractors")
    return dict(
        **stem(
            f"A storage vessel occupies {words}, with lengths in metres. The density of its contents at the point "
            f"«(x, y, z)» is «ρ = {k} + {m if m > 1 else ''}z» kilograms per cubic metre."
        ),
        parts=parts,
    )


@generator("exd_x_s2a1_cylindrical", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a1_cylindrical(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam cylindrical chain"):
        try:
            return _exam_once(rng)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
