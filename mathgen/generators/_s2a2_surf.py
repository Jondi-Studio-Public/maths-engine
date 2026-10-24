"""Parametrised surfaces for the second-year builds `s2a2_surface_*`: the families, their normals and their speeds.

A surface is r(u, v) = ⟨X, Y, Z⟩ on a rectangle (or a triangle) of the (u, v) plane. Every family carries

* ``r``: the position vector;
* ``N``: the normal r_u × r_v, written in closed form (and confirmed against a numerical cross product of central
  differences of ``r`` by `verify_surface`);
* ``speed``: the length |N| of that normal, positive on the region;
* ``u_range`` and ``v_range``: the region, as pairs of SymPy numbers.

The revolution family r = ⟨g(u) cos v, g(u) sin v, h(u)⟩ covers the cylinder, cone, paraboloid, sphere and torus with
the one formula N = ⟨-h' g cos v, -h' g sin v, g g'⟩ and |N| = g √(g'² + h'²).
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import sympy

from ..core import attempts, check
from . import _s2a2_vec as V
from ._s2a2_vec import u, v

PI = sympy.pi
UV = (u, v)


def _cyclic(comps, shift):
    """The components moved round by `shift` places. A cyclic move keeps the orientation, so the normal moves with it."""
    return tuple(comps[(i + shift) % 3] for i in range(3))


def surface(r, N, speed, u_range, v_range, label, shift=0, **extra):
    return SimpleNamespace(
        r=_cyclic(r, shift),
        N=_cyclic(N, shift),
        speed=speed,
        u_range=u_range,
        v_range=v_range,
        label=label,
        shift=shift,
        **extra,
    )


# ---------------------------------------------------------------------------------------------------------------
# polynomial surfaces


# A polynomial in u and v as {(i, j): coefficient} for u^i v^j. The searches below draw many candidate surfaces, so the
# arithmetic is done on these (whole numbers, no SymPy) and only the accepted surface is turned into an expression.


def _dpoly(rng, terms, max_degree, lo=-3, hi=4):
    monos = [(i, j) for i in range(max_degree + 1) for j in range(max_degree + 1 - i) if i + j >= 1]
    out = {}
    for mono in rng.sample(monos, min(terms, len(monos))):
        out[mono] = rng.choice([n for n in range(lo, hi + 1) if n])
    return out


def _dd(p, axis):
    out = {}
    for (i, j), c in p.items():
        k = (i, j)[axis]
        if k:
            mono = (i - 1, j) if axis == 0 else (i, j - 1)
            out[mono] = out.get(mono, 0) + c * k
    return out


def _dmul(p, q):
    out = {}
    for (a, b), c in p.items():
        for (e, f), d in q.items():
            out[(a + e, b + f)] = out.get((a + e, b + f), 0) + c * d
    return out


def _dsub(p, q):
    out = dict(p)
    for mono, c in q.items():
        out[mono] = out.get(mono, 0) - c
    return {mono: c for mono, c in out.items() if c}


def _dexpr(p):
    return sympy.Add(*[sympy.Integer(c) * u**i * v**j for (i, j), c in sorted(p.items())])


def _dval(p, uv):
    return sum(c * uv[0] ** i * uv[1] ** j for (i, j), c in p.items())


def _dcross(ru, rv):
    return (
        _dsub(_dmul(ru[1], rv[2]), _dmul(ru[2], rv[1])),
        _dsub(_dmul(ru[2], rv[0]), _dmul(ru[0], rv[2])),
        _dsub(_dmul(ru[0], rv[1]), _dmul(ru[1], rv[0])),
    )


def mixed_draw(rng, max_degree=2):
    """(r, N) as dictionaries: three polynomial components in u and v, and their cross product r_u x r_v."""
    for _ in attempts(200, "a mixed polynomial surface"):
        r = tuple(_dpoly(rng, rng.choice([1, 2]), max_degree) for _ in range(3))
        if not (any(j == 0 and i > 0 for p in r for (i, j) in p) and any(j > 0 for p in r for (i, j) in p)):
            continue
        ru = tuple(_dd(p, 0) for p in r)
        rv = tuple(_dd(p, 1) for p in r)
        N = _dcross(ru, rv)
        if all(N):
            return r, N
    raise AssertionError("unreachable")


def from_draw(r, N, label="mixed"):
    return surface(tuple(_dexpr(p) for p in r), tuple(_dexpr(p) for p in N), None, (-2, 2), (-2, 2), label)


def graph(rng, max_degree=2):
    """⟨u, v, f(u, v)⟩, with N = ⟨-f_u, -f_v, 1⟩."""
    for _ in attempts(100, "a graph surface"):
        f = _dpoly(rng, rng.choice([2, 3]), max_degree)
        if any(i for (i, j) in f) and any(j for (i, j) in f):
            fx = _dexpr(f)
            fu, fv = _dexpr(_dd(f, 0)), _dexpr(_dd(f, 1))
            return surface((u, v, fx), (-fu, -fv, sympy.Integer(1)), None, (-2, 2), (-2, 2), "graph", graph_f=fx)
    raise AssertionError("unreachable")


def mixed(rng, max_degree=2):
    return from_draw(*mixed_draw(rng, max_degree))


def linear(rng, whole=False):
    """⟨a1 u + b1 v + c1, ...⟩: a plane, with a constant normal (of whole-number length when `whole`)."""
    for _ in attempts(5000, "a plane"):
        A = tuple(rng.randint(-3, 3) for _ in range(3))
        B = tuple(rng.randint(-3, 3) for _ in range(3))
        C = tuple(rng.randint(-3, 3) for _ in range(3))
        n_ = V.cross(A, B)
        if whole and math.isqrt(sum(c * c for c in n_)) ** 2 != sum(c * c for c in n_):
            continue
        N = tuple(sympy.Integer(c) for c in n_)
        if all(c != 0 for c in N) and sum(1 for c in A if c) >= 2 and sum(1 for c in B if c) >= 2:
            r = tuple(a * u + b * v + c for a, b, c in zip(A, B, C))
            speed = sympy.sqrt(sum(c * c for c in N))
            return surface(r, N, speed, (-2, 2), (-2, 2), "plane", A=A, B=B, C=C)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# surfaces of revolution


def revolution(gfun, hfun, u_range, v_range, label, shift=0, speed=None, **extra):
    g, h = gfun, hfun
    gp, hp = sympy.diff(g, u), sympy.diff(h, u)
    N = (-hp * g * sympy.cos(v), -hp * g * sympy.sin(v), g * gp)
    N = tuple(sympy.expand(c) for c in N)
    root = sympy.sqrt(sympy.expand(gp**2 + hp**2))
    speed = speed if speed is not None else sympy.expand(g) * root
    r = (g * sympy.cos(v), g * sympy.sin(v), h)
    return surface(r, N, speed, u_range, v_range, label, shift, g=g, h=h, **extra)


def cylinder(rng, shift=0, full=None):
    a = rng.randint(1, 5)
    c = rng.choice([1, 1, 2, 3])
    v_end = full or rng.choice([2 * PI, PI, PI / 2])
    return revolution(sympy.Integer(a), c * u, (0, rng.randint(1, 4)), (0, v_end), "cylinder", shift, a=a, c=c)


def cone(rng, shift=0, full=None, pythag=None):
    pairs = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (3, 4), (4, 3), (1, 4), (5, 12), (2, 5)]
    if pythag:
        pairs = [(3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6)]
    a, h = rng.choice(pairs)
    v_end = full or rng.choice([2 * PI, PI, PI / 2])
    return revolution(a * u, h * u, (0, rng.randint(1, 4)), (0, v_end), "cone", shift, a=a, slope=h)


def paraboloid(rng, shift=0):
    c = rng.choice([1, 1, 2, 3])
    return revolution(u, c * u**2, (0, rng.randint(1, 3)), (0, rng.choice([2 * PI, PI])), "paraboloid", shift, c=c)


def sphere(rng, shift=0, alpha=None):
    a = rng.randint(1, 5)
    alpha = alpha or rng.choice([PI / 2, PI / 3, PI])
    return revolution(
        a * sympy.sin(u),
        a * sympy.cos(u),
        (0, alpha),
        (0, 2 * PI),
        "sphere",
        shift,
        speed=a * a * sympy.sin(u),
        a=a,
        alpha=alpha,
    )


def torus(rng, shift=0, v_end=None):
    r_ = rng.randint(1, 3)
    R = r_ + rng.randint(1, 4)
    return revolution(
        R + r_ * sympy.cos(u),
        r_ * sympy.sin(u),
        (0, rng.choice([PI, 2 * PI])),
        (0, v_end or 2 * PI),
        "torus",
        shift,
        speed=r_ * (R + r_ * sympy.cos(u)),
        R=R,
        minor=r_,
    )


def helicoid(rng, shift=0):
    c = rng.randint(1, 4)
    r = (u * sympy.cos(v), u * sympy.sin(v), c * v)
    N = (c * sympy.sin(v), -c * sympy.cos(v), u)
    return surface(
        r, N, sympy.sqrt(c * c + u * u), (0, rng.randint(1, 3)), (0, rng.choice([PI, 2 * PI])), "helicoid", shift, c=c
    )


# ---------------------------------------------------------------------------------------------------------------
# confirmation


def _num_cross(r, p):
    ru = [V.fd_partial(c, UV, p, 0) for c in r]
    rv = [V.fd_partial(c, UV, p, 1) for c in r]
    return V.cross(ru, rv)


def verify_surface(S, points=((0.7, 0.4), (1.3, 1.1))):
    """The closed-form normal and its length against a numerical cross product of central differences of r."""
    for p in points:
        num = _num_cross(S.r, p)
        exact = [V.float_of(V.at(c, p, UV)) for c in S.N]
        check(V.close_vec(num, exact, 1e-5), f"the normal of the {S.label} disagrees with the numerical cross product")
        if S.speed is not None:
            check(
                V.close(sum(c * c for c in num) ** 0.5, V.float_of(V.at(S.speed, p, UV)), 1e-5),
                f"the length of the normal of the {S.label} disagrees with the numerical length",
            )


def region_phrase(S):
    """'0 ≤ u ≤ 2 and 0 ≤ v ≤ π' as a formula piece."""
    (u0, u1), (v0, v1) = S.u_range, S.v_range
    le = V.raw(" ≤ ", r" \le ")
    return V.cat(
        V.m(sympy.sympify(u0)),
        le,
        V.raw("u"),
        le,
        V.m(sympy.sympify(u1)),
        V.raw(" and ", r" \text{ and } "),
        V.m(sympy.sympify(v0)),
        le,
        V.raw("v"),
        le,
        V.m(sympy.sympify(v1)),
    )


def rvec(S):
    return V.cat(V.R_OF_UV, V.mv(S.r))


def numeric_area(surf, u_range=None, v_range=None, n=24, integrand=None):
    """The area of the surface by Gauss-Legendre quadrature of |r_u x r_v|, with r differentiated by SymPy and every
    function lambdified: independent of the closed-form normal, its length and the exact integral. With an `integrand` g(x, y, z)
    the result is the surface integral of g, g being evaluated at r(u, v)."""
    gn = sympy.lambdify(V.XYZ, integrand, "math") if integrand is not None else None
    rn = sympy.lambdify(UV, list(surf.r), "math")
    ru = sympy.lambdify(UV, [sympy.diff(c, u) for c in surf.r], "math")
    rv = sympy.lambdify(UV, [sympy.diff(c, v) for c in surf.r], "math")

    def speed(a, b):
        p, q = ru(a, b), rv(a, b)
        cx = (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0])
        length = float(sum(c * c for c in cx) ** 0.5)
        return length if gn is None else length * float(gn(*rn(a, b)))

    ur = tuple(float(c) for c in (u_range or surf.u_range))
    vr = v_range or tuple(float(c) for c in surf.v_range)
    return V.gl2(speed, ur, vr, n)
