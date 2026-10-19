"""Generates `s2a1_spherical` questions (easy, medium, hard) and the exam chain `exd_x_s2a1_spherical`.

``uc2-spherical-triple-integral`` (``multivariable-and-vector-calculus``, answer ``expression``): triple integrals in
spherical coordinates, where x = ρ sin φ cos θ, y = ρ sin φ sin θ, z = ρ cos φ and dV = ρ² sin φ dρ dφ dθ (φ is the
angle from the positive z-axis). Every solid is described by Cartesian inequalities, so the learner has to choose
the coordinates, read the cone's half-angle from its equation and write the limits.

* easy: a ball, a half ball or a spherical shell, with an integrand that is a function of the distance from the
  origin, of z, or of x² + y².
* medium: a ball cut by a cone (an ice-cream cone) with a half-angle of π/6, π/4 or π/3, and a shell in the first
  octant with an integrand such as xyz.
* hard: a ball through the origin (x² + y² + z² ≤ 2az, whose radius depends on φ); a spherical cap cut off by a
  plane (whose radius runs from h sec φ); the part of a shell above a cone, with a density.
* exam (``exd_x_s2a1_spherical``): a hemisphere or a cone-shaped sector of a ball, with density δ = k + m·(distance
  from the origin): (a) the volume, (b) the mass (follows through from (a)), (c) the height of the centre of mass
  (follows through from (b)), (d) the average density (follows through from (a) and (b)).

Each key is worked out twice. The exact route integrates in spherical coordinates with SymPy. The second route is in
tests/test_s2a1_spherical.py: a Cartesian Monte Carlo that reads only the question text, run over many seeds.
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
p, ph, th = sympy.symbols("rho phi theta", positive=True)
S = sympy.Integer
SIN, COS = sympy.sin, sympy.cos
JAC = p**2 * SIN(ph)
DIST = sympy.sqrt(x**2 + y**2 + z**2)

#: name -> (Cartesian integrand, spherical integrand without the Jacobian, text)
INTEGRANDS = {
    "one": (S(1), S(1), "1"),
    "r2": (x**2 + y**2 + z**2, p**2, "x² + y² + z²"),
    "r": (DIST, p, "√(x² + y² + z²)"),
    "z": (z, p * COS(ph), "z"),
    "z2": (z**2, p**2 * COS(ph) ** 2, "z²"),
    "x2y2": (x**2 + y**2, p**2 * SIN(ph) ** 2, "x² + y²"),
    "zr": (z * DIST, p**2 * COS(ph), "z√(x² + y² + z²)"),
    "x": (x, p * SIN(ph) * COS(th), "x"),
    "xyz": (x * y * z, p**3 * SIN(ph) ** 2 * COS(ph) * SIN(th) * COS(th), "xyz"),
}


# ---------------------------------------------------------------------------------------------------------------
# exact integration


@functools.lru_cache(maxsize=1024)
def _trig_antiderivative(var, i: int, j: int):
    return sympy.integrate(COS(var) ** i * SIN(var) ** j, var)


def _trig_monomial(var, i: int, j: int, lo, hi, add_lower: bool = False):
    anti = _trig_antiderivative(var, i, j)
    at_hi, at_lo = anti.subs(var, hi), anti.subs(var, lo)
    return at_hi + at_lo if add_lower else at_hi - at_lo


def _integrate_poly(expr, var, lo, hi):
    expr = sympy.expand(expr)
    if not expr.has(var):
        return sympy.expand(expr * (hi - lo))
    poly = sympy.Poly(expr, var)
    return sympy.expand(sum(c * (hi ** (n + 1) - lo ** (n + 1)) / (n + 1) for (n,), c in poly.terms()))


def _integrate_trig(expr, var, lo, hi, add_lower: bool = False):
    """Term by term: each term is (something free of `var`) times cos(var)^i sin(var)^j, with i, j integers."""
    expr = sympy.expand(expr)
    total = S(0)
    for term in sympy.Add.make_args(expr):
        powers = term.as_powers_dict()
        i, j = powers.get(COS(var), 0), powers.get(SIN(var), 0)
        rest = sympy.Mul(*[b**e for b, e in powers.items() if b not in (COS(var), SIN(var))])
        check(not rest.has(var), f"a term of the integrand has an unexpected dependence on {var}")
        total += rest * _trig_monomial(var, int(i), int(j), lo, hi, add_lower)
    return sympy.expand(total)


@functools.lru_cache(maxsize=4096)
def sph(f, thl, phl, rl, jacobian=JAC, add_lower_in_phi: bool = False):
    """Exact iterated integral of the spherical form `f` (times `jacobian`): ρ first, then φ, then θ."""
    g = _integrate_poly(f * jacobian, p, rl[0], rl[1])
    g = _integrate_trig(g, ph, phl[0], phl[1], add_lower_in_phi)
    return _integrate_trig(g, th, thl[0], thl[1])


def abs_value(v):
    return -v if float(v) < 0 else v


def safe_sph(*args):
    try:
        return sph(*args)
    except (VerificationError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------------------------------------------


def _text_of(key: str) -> str:
    return INTEGRANDS[key][2]


_SUMS = {"r2", "x2y2"}  # integrands that are sums, so need brackets when multiplied or integrated


def _wrapped(key: str) -> str:
    text = _text_of(key)
    return f"({text})" if key in _SUMS else text


def _coef(k: int, key: str) -> str:
    """'k times the integrand', as the density expression written in the text."""
    if k == 1:
        return _text_of(key)
    return f"{k}({_text_of(key)})" if key in _SUMS else f"{k}{_text_of(key)}"


# ---------------------------------------------------------------------------------------------------------------
# the solids


def _ball(rng, half: bool, shell: bool):
    a2 = rng.randint(2, 4)
    a1 = rng.randint(1, a2 - 1) if shell else 0
    words = f"{a1 * a1} ≤ x² + y² + z² ≤ {a2 * a2}" if shell else f"x² + y² + z² ≤ {a2 * a2}"
    if half:
        words += ", z ≥ 0"
    return dict(
        words=words,
        thl=(S(0), 2 * PI),
        phl=(S(0), PI / 2 if half else PI),
        rl=(S(a1), S(a2)),
        contains=lambda px, py, pz: a1 * a1 <= px * px + py * py + pz * pz <= a2 * a2 and (pz >= 0 or not half),
        box=(a2, a2, a2),
        a1=a1,
        a2=a2,
        half=half,
        shell=shell,
    )


_CONES = {  # half-angle -> (the condition in x, y, z; the cos of the half-angle)
    "pi/6": (PI / 6, "3(x² + y²) ≤ z²", sympy.sqrt(3) / 2),
    "pi/4": (PI / 4, "x² + y² ≤ z²", sympy.sqrt(2) / 2),
    "pi/3": (PI / 3, "x² + y² ≤ 3z²", S(1) / 2),
}


def _cone_contains(name):
    if name == "pi/6":
        return lambda px, py, pz: 3 * (px * px + py * py) <= pz * pz and pz >= 0
    if name == "pi/4":
        return lambda px, py, pz: px * px + py * py <= pz * pz and pz >= 0
    return lambda px, py, pz: px * px + py * py <= 3 * pz * pz and pz >= 0


def _cone_ball(rng, name=None, shell=False):
    name = name or rng.choice(list(_CONES))
    angle, cond, _ = _CONES[name]
    a2 = rng.randint(2, 4)
    a1 = rng.randint(1, a2 - 1) if shell else 0
    inner = f"{a1 * a1} ≤ x² + y² + z² ≤ {a2 * a2}" if shell else f"x² + y² + z² ≤ {a2 * a2}"
    cone = _cone_contains(name)
    return dict(
        words=f"{inner}, {cond}, z ≥ 0",
        thl=(S(0), 2 * PI),
        phl=(S(0), angle),
        rl=(S(a1), S(a2)),
        contains=lambda px, py, pz: a1 * a1 <= px * px + py * py + pz * pz <= a2 * a2 and cone(px, py, pz),
        box=(a2, a2, a2),
        a1=a1,
        a2=a2,
        cone=name,
        angle=angle,
    )


def _octant(rng):
    a2 = rng.randint(2, 4)
    a1 = rng.randint(0, a2 - 1)
    inner = f"{a1 * a1} ≤ x² + y² + z² ≤ {a2 * a2}" if a1 else f"x² + y² + z² ≤ {a2 * a2}"
    return dict(
        words=f"{inner}, x ≥ 0, y ≥ 0, z ≥ 0",
        thl=(S(0), PI / 2),
        phl=(S(0), PI / 2),
        rl=(S(a1), S(a2)),
        contains=lambda px, py, pz: (
            a1 * a1 <= px * px + py * py + pz * pz <= a2 * a2 and px >= 0 and py >= 0 and pz >= 0
        ),
        box=(a2, a2, a2),
        a1=a1,
        a2=a2,
    )


def _ask(rng, key: str, words: str, k: int = 1, mass_ok: bool = True):
    """The question text for the integrand `key` over the region `words`: an integral, or a mass."""
    f_text = _coef(k, key)
    if key == "one":
        return f"Find the volume of the solid «{words}»."
    if mass_ok and rng.random() < 0.45:
        return (
            f"A solid occupies the region «{words}», with lengths in metres. Its density at the point «(x, y, z)» is "
            f"«δ = {f_text}» kilograms per cubic metre. Find the mass of the solid in kilograms."
        )
    if k == 1:
        return f"Evaluate «∭ {_wrapped(key)} dV» over the solid «{words}»."
    return f"Evaluate «∭ {f_text} dV» over the solid «{words}»."


def _finish(rng, solid, key, k, wrong_inputs):
    cart, sphf, _ = INTEGRANDS[key]
    f = k * sphf
    value = sph(f, solid["thl"], solid["phl"], solid["rl"])
    return value, f


def _swap_angle_convention(f):
    return f.subs({SIN(ph): COS(ph), COS(ph): SIN(ph)}, simultaneous=True)


# ---------------------------------------------------------------------------------------------------------------
# easy


def _easy(rng):
    half = rng.random() < 0.45
    shell = rng.random() < 0.4
    solid = _ball(rng, half, shell)
    keys = ["r2", "r", "z2", "x2y2"] + (["z", "zr"] if half else [])
    key = rng.choice(keys)
    k = rng.choice([1, 1, 2, 3])
    text = _ask(rng, key, solid["words"], k)
    value, f = _finish(rng, solid, key, k, None)
    thl, phl, rl = solid["thl"], solid["phl"], solid["rl"]
    a1, a2 = solid["a1"], solid["a2"]
    cands = [
        ("spherical-jacobian-omitted", sph(f, thl, phl, rl, S(1))),
        ("spherical-sin-phi-omitted", sph(f, thl, phl, rl, p**2)),
        ("spherical-rho-squared-omitted", sph(f, thl, phl, rl, SIN(ph))),
        ("spherical-colatitude-confused", sph(_swap_angle_convention(f), thl, phl, rl)),
        ("spherical-radius-limit-squared", sph(f, thl, phl, (S(a1 * a1), S(a2 * a2)))),
        ("integral-constant-multiplier-dropped", value / k if k > 1 else None),
        ("spherical-angle-range-halved", sph(f, thl, (S(0), PI / 2), rl) if not solid["half"] else value * 2),
    ]
    working = [
        f"With dV = ρ² sin φ dρ dφ dθ, ρ runs from {a1} to {a2}, φ from 0 to {'π/2' if solid['half'] else 'π'} and θ from 0 to 2π."
    ]
    return text, value, cands, working


# ---------------------------------------------------------------------------------------------------------------
# medium


def _medium_cone(rng):
    solid = _cone_ball(rng, shell=rng.random() < 0.3)
    key = rng.choice(["one", "one", "z", "r2", "r", "zr"])
    k = rng.choice([1, 2, 3])
    text = _ask(rng, key, solid["words"], k if key != "one" else 1)
    value, f = _finish(rng, solid, key, k if key != "one" else 1, None)
    thl, phl, rl = solid["thl"], solid["phl"], solid["rl"]
    a1, a2 = solid["a1"], solid["a2"]
    cands = [
        ("spherical-jacobian-omitted", sph(f, thl, phl, rl, S(1))),
        ("spherical-sin-phi-omitted", sph(f, thl, phl, rl, p**2)),
        ("spherical-cone-angle-complement", sph(f, thl, (S(0), PI / 2 - solid["angle"]), rl)),
        ("spherical-lower-limit-added", abs_value(sph(f, thl, phl, rl, JAC, True))),
        ("spherical-polar-angle-range", sph(f, thl, (S(0), PI), rl)),
        ("spherical-hemisphere-as-cone", sph(f, thl, (S(0), PI / 2), rl)),
        ("spherical-colatitude-confused", sph(_swap_angle_convention(f), thl, phl, rl)),
        ("spherical-radius-limit-squared", sph(f, thl, phl, (S(a1 * a1), S(a2 * a2)))),
        ("integral-constant-multiplier-dropped", value / k if k > 1 else None),
    ]
    working = [
        f"The cone makes an angle of {solid['cone'].replace('pi', 'π')} with the z-axis, so φ runs from 0 to {solid['cone'].replace('pi', 'π')}."
    ]
    return text, value, cands, working


def _medium_octant(rng):
    solid = _octant(rng)
    key = rng.choice(["xyz", "xyz", "z", "x", "r", "r2"])
    k = rng.choice([1, 2, 3, 4])
    text = _ask(rng, key, solid["words"], k, mass_ok=key not in ("x",) or True)
    value, f = _finish(rng, solid, key, k, None)
    thl, phl, rl = solid["thl"], solid["phl"], solid["rl"]
    a1, a2 = solid["a1"], solid["a2"]
    cands = [
        ("spherical-jacobian-omitted", sph(f, thl, phl, rl, S(1))),
        ("spherical-sin-phi-omitted", sph(f, thl, phl, rl, p**2)),
        ("spherical-octant-ignored", sph(f, (S(0), 2 * PI), (S(0), PI), rl)),
        ("spherical-octant-angle-halved", sph(f, thl, (S(0), PI), rl)),
        ("spherical-colatitude-confused", sph(_swap_angle_convention(f), thl, phl, rl)),
        ("spherical-radius-limit-squared", sph(f, thl, phl, (S(a1 * a1), S(a2 * a2)))),
        ("integral-constant-multiplier-dropped", value / k if k > 1 else None),
    ]
    working = ["In the first octant θ runs from 0 to π/2 and φ from 0 to π/2."]
    return text, value, cands, working


def _medium(rng):
    return _medium_cone(rng) if rng.random() < 0.6 else _medium_octant(rng)


# ---------------------------------------------------------------------------------------------------------------
# hard


def _hard_origin_ball(rng):
    a = rng.randint(1, 3)
    key = rng.choice(["one", "z", "r2", "x2y2"])
    k = rng.choice([1, 2, 3]) if key != "one" else 1
    rl = (S(0), 2 * a * COS(ph))
    solid = dict(
        words=f"x² + y² + z² ≤ {2 * a}z",
        thl=(S(0), 2 * PI),
        phl=(S(0), PI / 2),
        rl=rl,
        contains=lambda px, py, pz: px * px + py * py + pz * pz <= 2 * a * pz,
        box=(a, a, 2 * a),
    )
    # z ranges over 0..2a only; sample z from -2a..2a
    solid["box"] = (a, a, 2 * a)
    text = _ask(rng, key, solid["words"], k)
    value, f = _finish(rng, solid, key, k, None)
    thl, phl = solid["thl"], solid["phl"]
    cands = [
        ("spherical-jacobian-omitted", sph(f, thl, phl, rl, S(1))),
        ("spherical-sin-phi-omitted", sph(f, thl, phl, rl, p**2)),
        ("spherical-offset-radius", sph(f, thl, phl, (S(0), S(2 * a)))),
        ("spherical-offset-angle-range", sph(f, thl, (S(0), PI), rl)),
        ("spherical-offset-radius", sph(f, thl, phl, (S(0), S(a)))),
    ]
    working = [f"The sphere is ρ = {2 * a} cos φ, so ρ runs from 0 to {2 * a} cos φ and φ from 0 to π/2."]
    return text, value, cands, working


def _hard_cap(rng):
    a, h = rng.choice([(5, 3), (10, 6), (13, 5), (17, 8), (15, 9)])
    key = rng.choice(["one", "one", "z"])
    k = rng.choice([1, 2]) if key != "one" else 1
    phi0 = sympy.acos(S(h) / a)
    rl = (h / COS(ph), S(a))
    solid = dict(
        words=f"x² + y² + z² ≤ {a * a}, z ≥ {h}",
        thl=(S(0), 2 * PI),
        phl=(S(0), phi0),
        rl=rl,
        contains=lambda px, py, pz: px * px + py * py + pz * pz <= a * a and pz >= h,
        box=(a, a, a),
    )
    text = _ask(rng, key, solid["words"], k)
    value, f = _finish(rng, solid, key, k, None)
    thl, phl = solid["thl"], solid["phl"]
    cands = [
        ("spherical-jacobian-omitted", safe_sph(f, thl, phl, rl, S(1))),
        ("spherical-cap-lower-limit-zero", sph(f, thl, phl, (S(0), S(a)))),
        ("spherical-plane-limit-cos", sph(f, thl, phl, (h * COS(ph), S(a)))),
        ("spherical-plane-ignored", sph(f, thl, (S(0), PI / 2), (S(0), S(a)))),
        ("spherical-cap-cone-instead", sph(f, thl, phl, (S(0), h / COS(ph)))),
        ("spherical-cap-complement", abs_value(sph(f, thl, (S(0), PI), (S(0), S(a))) - value)),
        ("integral-constant-multiplier-dropped", value / k if k > 1 else None),
    ]
    working = [
        f"The plane z = {h} is ρ = {h} sec φ; it meets the sphere where cos φ = {h}/{a}, so ρ runs from {h} sec φ to {a}."
    ]
    return text, value, cands, working


def _hard_cone_shell(rng):
    solid = _cone_ball(rng, shell=True)
    key = rng.choice(["z", "r", "zr", "z2"])
    k = rng.choice([1, 2, 3])
    text = _ask(rng, key, solid["words"], k)
    value, f = _finish(rng, solid, key, k, None)
    thl, phl, rl = solid["thl"], solid["phl"], solid["rl"]
    a1, a2 = solid["a1"], solid["a2"]
    cands = [
        ("spherical-jacobian-omitted", sph(f, thl, phl, rl, S(1))),
        ("spherical-sin-phi-omitted", sph(f, thl, phl, rl, p**2)),
        ("spherical-inner-sphere-ignored", sph(f, thl, phl, (S(0), S(a2)))),
        ("spherical-lower-limit-added", abs_value(sph(f, thl, phl, rl, JAC, True))),
        ("spherical-cone-angle-complement", sph(f, thl, (S(0), PI / 2 - solid["angle"]), rl)),
        ("spherical-radius-limit-squared", sph(f, thl, phl, (S(a1 * a1), S(a2 * a2)))),
        ("spherical-colatitude-confused", sph(_swap_angle_convention(f), thl, phl, rl)),
        ("integral-constant-multiplier-dropped", value / k if k > 1 else None),
    ]
    working = [f"ρ runs from {a1} to {a2} and φ from 0 to {solid['cone'].replace('pi', 'π')}."]
    return text, value, cands, working


def _hard(rng):
    pick = rng.random()
    if pick < 0.35:
        return _hard_origin_ball(rng)
    if pick < 0.7:
        return _hard_cap(rng)
    return _hard_cone_shell(rng)


@generator(
    "s2a1_spherical",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a1_spherical(rng, tier="easy"):
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(30, "a spherical integral with enough distractors"):
        text, value, cands, working = build(rng)
        check(sympy.sympify(value).is_real and float(value) > 0, "the integral is not positive")
        wrong = wrong_answers(value, cands, 4, kind="expression")
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


def _exam_solid(rng):
    a = rng.randint(1, 4)
    if rng.random() < 0.5:
        words = f"the hemispherical dome «x² + y² + z² ≤ {a * a}, z ≥ 0»"
        return words, dict(
            thl=(S(0), 2 * PI),
            phl=(S(0), PI / 2),
            rl=(S(0), S(a)),
            contains=lambda px, py, pz: px * px + py * py + pz * pz <= a * a and pz >= 0,
            box=(a, a, a),
            a=a,
            cosphi=S(0),
        )
    words = f"the conical sector «x² + y² + z² ≤ {a * a}, x² + y² ≤ 3z², z ≥ 0»"
    return words, dict(
        thl=(S(0), 2 * PI),
        phl=(S(0), PI / 3),
        rl=(S(0), S(a)),
        contains=lambda px, py, pz: (
            px * px + py * py + pz * pz <= a * a and px * px + py * py <= 3 * pz * pz and pz >= 0
        ),
        box=(a, a, a),
        a=a,
        cosphi=S(1) / 2,
    )


def _exam_once(rng):
    words, s = _exam_solid(rng)
    k, m = rng.randint(1, 5), rng.randint(1, 4)
    thl, phl, rl, a = s["thl"], s["phl"], s["rl"], s["a"]
    one = S(1)
    dens = k + m * p  # spherical form of delta = k + m * distance
    dens_cart = k + m * DIST
    V = sph(one, thl, phl, rl)
    Ir = sph(p, thl, phl, rl)
    M = sph(dens, thl, phl, rl)
    Mz = sph(p * COS(ph) * dens, thl, phl, rl)
    check(sympy.simplify(M - k * V - m * Ir) == 0, "mass is not kV + m times the integral of the distance")
    zbar = sympy.simplify(Mz / M)
    avg = sympy.simplify(M / V)
    check(zbar.is_Rational and avg.is_Rational, "the centre of mass or average density is not rational")
    check(0 < float(zbar) < a, "centre of mass outside the solid")
    Iz = sph(p * COS(ph), thl, phl, rl)
    a_c = [
        ("spherical-jacobian-omitted", sph(one, thl, phl, rl, S(1))),
        ("spherical-sin-phi-omitted", sph(one, thl, phl, rl, p**2)),
        ("spherical-lower-limit-added", abs_value(sph(one, thl, phl, rl, JAC, True))),
        ("spherical-polar-angle-range", sph(one, thl, (S(0), PI), rl)),
        ("spherical-hemisphere-as-cone", sph(one, thl, (S(0), PI / 2), rl) if s["cosphi"] != 0 else V * 2),
    ]
    b_c = [
        ("triple-integral-density-term-dropped", k * V),
        ("triple-integral-density-maximum", (k + m * a) * V),
        ("triple-integral-density-midpoint", (k + m * S(a) / 2) * V),
        ("spherical-sin-phi-omitted", sph(dens, thl, phl, rl, p**2)),
    ]
    c_c = [
        ("triple-integral-moment-not-divided", Mz),
        ("triple-integral-centroid-uniform", sympy.simplify(Iz_unit(thl, phl, rl, V))),
        ("triple-integral-moment-inverted", M / Mz),
        ("spherical-colatitude-confused", sympy.simplify(sph(p * SIN(ph) * dens, thl, phl, rl) / M)),
    ]
    d_c = [
        ("triple-integral-moment-inverted", V / M),
        ("triple-integral-mass-not-divided", M),
        ("triple-integral-density-midpoint", k + m * S(a) / 2),
        ("triple-integral-density-maximum", S(k + m * a)),
    ]
    Ir_text, Mz_text = etext(Ir), etext(Mz)
    parts = [
        exam_part(
            "Find the volume of the solid, in cubic metres.",
            V,
            3,
            [f"V = ∫ from 0 to 2π ∫ over φ ∫ from 0 to {a} of ρ² sin φ dρ dφ dθ = {say_value(V)}."],
            a_c,
            sensible_range=(0, None),
        ),
        exam_part(
            "Hence find the mass of the solid, in kilograms.",
            M,
            3,
            [f"M = ∭ δ dV = {k}V + {m}∭ ρ dV = {say_value(M)}."],
            b_c,
            follow_through=f"{k}*part_a + {m}*({Ir_text})",
            sensible_range=(0, None),
        ),
        exam_part(
            "Find the height of the centre of mass above the base of the solid, in metres.",
            zbar,
            3,
            [f"z-bar = (∭ zδ dV)/M = ({say_value(Mz)})/M."],
            c_c,
            follow_through=f"({Mz_text})/part_b",
            sensible_range=(0, a),
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
            f"A solid occupies {words}, with lengths in metres. At a point whose distance from the origin is "
            f"«d» metres the density is «δ = {k} + {m if m > 1 else ''}d» kilograms per cubic metre."
        ),
        parts=parts,
    )


def Iz_unit(thl, phl, rl, V):
    """Height of the centroid (uniform density): the moment of the volume about z over the volume."""
    return sph(p * COS(ph), thl, phl, rl) / V


@generator("exd_x_s2a1_spherical", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a1_spherical(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam spherical chain"):
        try:
            return _exam_once(rng)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
