"""Generates `w5a_mobius` and `exd_x_w5a_mobius` questions: Möbius transformations.

One second-year skill, ``uc2-mobius-transformations`` (domain ``complex-analysis``). A Möbius transformation is held as a
matrix of Gaussian rationals ``(a, b, c, d)`` acting as ``T(z) = (az + b)/(cz + d)``, so every image, preimage, fixed
point and composition is exact. Questions are built backwards: the fixed points or the three prescribed points are chosen
first and the matrix is made from them.

Tiers:

- ``easy``: the image of a Gaussian-integer point under ``(az + b)/(cz + d)`` with integer coefficients (a complex
  division), or the image of ``infinity``/the preimage of a given point.
- ``medium``: a fixed point of a real transformation (``cz^2 + (d - a)z - b = 0``, the one with positive imaginary part or
  the larger real root, since the two are not distinguished by the question otherwise), or the point mapped to a given
  ``w``.
- ``hard``: the transformation sending three given points (possibly one of them ``infinity``) to ``0, 1, infinity``, typed
  as a single fraction; the image of a fourth point under the transformation that sends three given points to three
  given images; the second iterate ``T(T(z))`` as a single fraction; and the single fixed point of a parabolic
  transformation, or a fixed point of a loxodromic one with complex coefficients.
- ``exam`` (topic ``exd_x_w5a_mobius``): a four-part chain on a real transformation: the image of a point, the image of
  that image, the sum of the two fixed points, and the image of that sum.
"""

from __future__ import annotations

import random
from fractions import Fraction as Fr

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator
from ._w5a_support import (
    join_signed,
    pn,
    C,
    Link,
    X,
    astr,
    atext,
    cx_x,
    exam_dict,
    frac_x,
    msay,
    poly_x,
    qsym,
    single,
    zminus_x,
)

TOPIC = "w5a_mobius"
EXAM_TOPIC = "exd_x_w5a_mobius"

_Z = sympy.Symbol("z")
INF = None  # the point at infinity

# a Möbius matrix is a tuple (a, b, c, d) of C


def mob_apply(m, z):
    a, b, c, d = m
    if z is INF:
        return INF if c.is_zero() else a / c
    den = c * z + d
    if den.is_zero():
        return INF
    return (a * z + b) / den


def mob_norm(m):
    """Scale so that c = 1 (or d = 1 when c = 0): the normal form, unique for a transformation."""
    a, b, c, d = m
    k = c if not c.is_zero() else d
    return (a / k, b / k, c / k, d / k)


def mob_mul(m, n):
    a, b, c, d = m
    e, f, g, h = n
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def mob_det(m):
    a, b, c, d = m
    return a * d - b * c


def _same_m(m, n) -> bool:
    return mob_norm(m) == mob_norm(n)


def mob_x(m, var: str = "z") -> X:
    """(az + b)/(cz + d) as a single fraction (a linear polynomial when c = 0)."""
    a, b, c, d = m
    num = poly_x([b, a], var)
    if c.is_zero():
        den = d
        if den == C(1):
            return num
        return frac_x(num, poly_x([d], var))
    return frac_x(num, poly_x([d, c], var))


def mob_sym(m):
    a, b, c, d = m
    return (a.sym() * _Z + b.sym()) / (c.sym() * _Z + d.sym())


def _tfun(m) -> X:
    f = mob_x(m)
    return X("T(z) = " + f.p, "T(z) = " + f.t)


def _pt(z) -> X:
    return X("∞", r"\infty") if z is INF else cx_x(z)


def _point_eq(z) -> X:
    p = _pt(z)
    return X(f"z = {p.p}", f"z = {p.t}")


def _tidy(c: C, bound: int = 40) -> bool:
    return c.re.denominator <= bound and c.im.denominator <= bound


_REAL_ENTRIES = [-3, -2, -1, 1, 2, 3, 4]
_POINTS = [C(a, b) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)]


def _real_matrix(rng, lo=-3, hi=3):
    for _ in attempts(500, "a real Möbius matrix"):
        a, b, c, d = (C(rng.randint(lo, hi)) for _ in range(4))
        if c.is_zero() or mob_det((a, b, c, d)).is_zero():
            continue
        return a, b, c, d
    raise AssertionError


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy_image(rng):
    for _ in attempts(800, "an easy image"):
        m = _real_matrix(rng)
        if any(e.is_zero() for e in m):
            continue
        z0 = rng.choice([p for p in _POINTS if not p.is_real()] + [C(0), C(1), C(-1)])
        if (m[2] * z0 + m[3]).is_zero():
            continue
        w0 = mob_apply(m, z0)
        if w0 is INF or w0.is_zero() or not _tidy(w0) or w0 == z0:
            continue
        break
    value = w0.sym()
    num = m[0] * z0 + m[1]
    den = m[2] * z0 + m[3]
    wrongs = [
        ((C(1) / w0).sym(), "cx2-mobius-inverse-quotient"),
        (w0.conj().sym(), "cx2-mobius-conjugate-quotient"),
        (
            ((m[0] * z0 + m[1]) / (m[2] * z0 - m[3])).sym() if not (m[2] * z0 - m[3]).is_zero() else (-w0).sym(),
            "cx2-mobius-denominator-sign-slip",
        ),
        (((m[0] * z0 - m[1]) / (m[2] * z0 + m[3])).sym(), "cx2-mobius-numerator-sign-slip"),
        ((num / m[3]).sym() if not m[3].is_zero() else (num / m[2]).sym(), "cx2-mobius-denominator-dropped"),
        ((num / den.conj()).sym(), "cx2-mobius-conjugate-quotient"),
    ]
    tx = _tfun(m)
    pt = cx_x(z0)
    working = [
        f"T({pt.p}) = ({cx_x(num).p})/({cx_x(den).p}).",
        f"Multiplying top and bottom by the conjugate of the denominator gives {atext(value)}.",
    ]
    lead = [
        lambda: msay("Let ", tx, ". Find the image of the point ", _point_eq(z0), " under T."),
        lambda: msay("The Möbius transformation ", tx, " maps ", _point_eq(z0), " to a point w. Find w."),
        lambda: msay("Evaluate ", X(f"T({pt.p})", f"T({pt.t})"), " for ", tx, "."),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-mobius-numerator-sign-slip")


def _easy_special(rng):
    """The image of infinity, or the point that T sends to a stated value."""
    for _ in attempts(800, "an easy preimage"):
        m = _real_matrix(rng, -3, 4)
        mode = rng.choice(["infinity", "preimage"])
        if mode == "infinity":
            w = mob_apply(m, INF)
            if w.is_zero():
                continue
            value = w.sym()
            lead = lambda tx: msay("Let ", tx, ". Find the point that the point at infinity is mapped to.")  # noqa: E731
            wrongs = [
                ((m[1] / m[3]).sym() if not m[3].is_zero() else (m[0] / m[2] + 1).sym(), "cx2-mobius-image-of-zero"),
                ((m[2] / m[0]).sym() if not m[0].is_zero() else (m[1] / m[2]).sym(), "cx2-mobius-image-inverted"),
                ((-m[3] / m[2]).sym(), "cx2-mobius-pole-for-image-of-infinity"),
                ((-m[0] / m[2]).sym(), "calc-sign-slip"),
                ((m[0] * m[3] / m[2]).sym(), "cx2-mobius-image-inverted"),
            ]
            working = ["As z → ∞, (az + b)/(cz + d) → a/c.", f"Here a/c = {atext(value)}."]
            break
        w0 = rng.choice([C(1), C(2), C(0, 1), C(-1), C(1, 1), C(0), C(3)])
        # z = (d w - b)/(-c w + a)
        den = m[0] - m[2] * w0
        if den.is_zero():
            continue
        z0 = (m[3] * w0 - m[1]) / den
        if not _tidy(z0) or mob_apply(m, z0) != w0:
            continue
        value = z0.sym()
        wsym = cx_x(w0)
        lead = lambda tx, w0=w0, wsym=wsym: msay(
            "Let ", tx, ". Find the point z that T maps to ", X(f"w = {wsym.p}", f"w = {wsym.t}"), "."
        )  # noqa: E731
        wrongs = [
            (
                ((m[0] * w0 + m[1]) / (m[2] * w0 + m[3])).sym() if not (m[2] * w0 + m[3]).is_zero() else (-z0).sym(),
                "cx2-mobius-inverse-applied-as-forward",
            ),
            (
                ((m[3] * w0 + m[1]) / (m[2] * w0 + m[0])).sym() if not (m[2] * w0 + m[0]).is_zero() else (-z0).sym(),
                "cx2-mobius-inverse-sign-slip",
            ),
            (
                ((m[3] * w0 - m[1]) / (m[2] * w0 - m[0])).sym() if not (m[2] * w0 - m[0]).is_zero() else (-z0).sym(),
                "cx2-mobius-inverse-sign-slip",
            ),
            ((-z0).sym(), "calc-sign-slip"),
            ((C(1) / z0).sym() if not z0.is_zero() else C(1).sym(), "cx2-mobius-image-inverted"),
        ]
        working = [
            f"Solve (az + b)/(cz + d) = {wsym.p}: z = (d·w - b)/(a - c·w).",
            f"With the given coefficients z = {atext(value)}.",
        ]
        break
    tx = _tfun(m)
    return single(lead(tx), value, working, wrongs, fallback_tag="cx2-mobius-inverse-sign-slip")


# -- medium ----------------------------------------------------------------------------------------------------------


def _fixed_point_matrix(rng):
    """A real transformation with chosen fixed points; (matrix, kind, p) where p is the asked-for fixed point."""
    for _ in attempts(800, "a fixed-point matrix"):
        conj_pair = rng.random() < 0.6
        c = C(rng.choice([1, 1, 2]))
        a = C(rng.randint(-3, 3))
        if conj_pair:
            u, v = rng.randint(-2, 2), rng.randint(1, 3)
            d = a - c * C(2 * u)
            b = -c * C(u * u + v * v)
            p = C(u, v)
            ask = "positive imaginary part"
        else:
            p1, p2 = rng.sample([-3, -2, -1, 0, 1, 2, 3], 2)
            hi, lo = max(p1, p2), min(p1, p2)
            if hi == 0:
                continue
            d = a - c * C(hi + lo)
            b = -c * C(hi * lo)
            p = C(hi)
            ask = "larger"
            lo_pt = C(lo)
        m = (a, b, c, d)
        if mob_det(m).is_zero() or abs(b.re) > 20 or abs(d.re) > 20:
            continue
        check(mob_apply(m, p) == p, "p is a fixed point")
        return m, ask, p, conj_pair
    raise AssertionError


def _medium_fixed(rng):
    m, ask, p, conj_pair = _fixed_point_matrix(rng)
    a, b, c, d = m
    value = p.sym()
    other = p.conj()
    wrongs = [
        (other.sym() if conj_pair else ((a - d) / c - p).sym(), "cx2-mobius-wrong-fixed-point"),
        ((-p.conj()).sym() if conj_pair else (-p).sym(), "cx2-mobius-fixed-point-sign-slip"),
        (C(p.re * 2, p.im).sym() if conj_pair else (p + C(1)).sym(), "cx2-mobius-fixed-point-equation-slip"),
        (C(-p.re, p.im).sym() if conj_pair else (-p + C(1)).sym(), "cx2-mobius-fixed-point-sign-slip"),
        ((p / 2).sym() if not p.is_zero() else C(1).sym(), "cx2-mobius-fixed-point-equation-slip"),
        ((-b / c).sym(), "cx2-mobius-fixed-point-equation-slip"),
        ((a / c).sym(), "cx2-mobius-fixed-point-equation-slip"),
    ]
    tx = _tfun(m)
    wording = "with positive imaginary part" if conj_pair else "with the larger real value"
    lead = [
        lambda: msay("The transformation ", tx, " has two fixed points. Find the fixed point ", wording, "."),
        lambda: msay("Let ", tx, ". Solve T(z) = z, and give the solution ", wording, "."),
        lambda: msay("Find, in exact form, the fixed point ", wording, " of ", tx, "."),
    ][rng.randrange(3)]
    working = [
        "A fixed point satisfies (az + b)/(cz + d) = z, that is c z² + (d - a) z - b = 0.",
        f"Here {_quad_text(c, d - a, -b)} = 0, with roots {atext(p.sym())} and {atext((p.conj() if conj_pair else (a - d) / c - p).sym())}.",
    ]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-mobius-fixed-point-equation-slip")


def _quad_text(c: C, lin: C, const: C) -> str:
    return poly_x([const, lin, c]).p


def _gauss_matrix(rng):
    for _ in attempts(500, "a Gaussian Möbius matrix"):
        m = tuple(C(rng.randint(-2, 2), rng.randint(-1, 1)) for _ in range(4))
        if m[2].is_zero() or mob_det(m).is_zero() or sum(1 for e in m if e.is_zero()) > 1:
            continue
        return m
    raise AssertionError


def _medium_from_image(rng):
    """The point that a transformation with complex coefficients sends to a given w."""
    for _ in attempts(1500, "a Gaussian preimage"):
        m = _gauss_matrix(rng)
        z0 = rng.choice(_POINTS)
        w0 = mob_apply(m, z0)
        if w0 is INF or w0 == z0 or not _tidy(w0, 6) or abs(w0.re) > 8 or abs(w0.im) > 8:
            continue
        if (m[0] - m[2] * w0).is_zero():
            continue
        break
    value = z0.sym()
    a, b, c, d = m
    inv = (d, -b, -c, a)
    check(mob_apply(inv, w0) == z0, "inverse")
    wrongs = [
        (
            mob_apply((a, b, c, d), w0).sym() if mob_apply((a, b, c, d), w0) is not INF else (-z0).sym(),
            "cx2-mobius-inverse-applied-as-forward",
        ),
        (
            mob_apply((d, b, c, a), w0).sym() if mob_apply((d, b, c, a), w0) is not INF else (-z0).sym(),
            "cx2-mobius-inverse-sign-slip",
        ),
        (
            mob_apply((d, -b, c, -a), w0).sym() if mob_apply((d, -b, c, -a), w0) is not INF else (-z0).sym(),
            "cx2-mobius-inverse-sign-slip",
        ),
        ((-z0).sym(), "calc-sign-slip"),
        (z0.conj().sym(), "cx2-mobius-conjugate-quotient"),
        ((C(1) / z0).sym() if not z0.is_zero() else (z0 + C(1)).sym(), "cx2-mobius-image-inverted"),
    ]
    tx = _tfun(m)
    wsym = cx_x(w0)
    wx = X(f"w = {wsym.p}", f"w = {wsym.t}")
    lead = [
        lambda: msay("Let ", tx, ". Find the point z that T maps to ", wx, "."),
        lambda: msay("The Möbius transformation ", tx, " sends a point z to ", wx, ". Find z."),
        lambda: msay("Solve T(z) = ", X(wsym.p, wsym.t), " for ", tx, "."),
    ][rng.randrange(3)]
    working = [
        "Solving w = (az + b)/(cz + d) for z gives z = (d·w - b)/(a - c·w).",
        f"Substituting w = {wsym.p} gives z = {atext(value)}.",
    ]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-mobius-inverse-sign-slip")


# -- hard ------------------------------------------------------------------------------------------------------------


def _cross_matrix(z1, z2, z3):
    """The matrix of the transformation with z1 -> 0, z2 -> 1, z3 -> infinity (any one of them may be INF)."""
    if z1 is INF:
        return (C(0), z2 - z3, C(1), -z3)
    if z2 is INF:
        return (C(1), -z1, C(1), -z3)
    if z3 is INF:
        return (C(1), -z1, C(0), z2 - z1)
    return (z2 - z3, -z1 * (z2 - z3), z2 - z1, -z3 * (z2 - z1))


def _hard_cross(rng):
    for _ in attempts(800, "a three-point map"):
        pts = rng.sample(_POINTS, 3)
        use_inf = rng.random() < 0.3
        if use_inf:
            pts[rng.randrange(3)] = INF
        z1, z2, z3 = pts
        m = _cross_matrix(z1, z2, z3)
        if mob_det(m).is_zero():
            continue
        n = mob_norm(m)
        if not all(_tidy(e) for e in n):
            continue
        break
    check(
        mob_apply(m, z1) is not INF and mob_apply(m, z1).is_zero() if z1 is not INF else mob_apply(m, INF) == C(0),
        "T(z1) = 0",
    )
    if z2 is not INF:
        check(mob_apply(m, z2) == C(1), "T(z2) = 1")
    if z3 is not INF:
        check(mob_apply(m, z3) is INF, "T(z3) = ∞")
    # distractors: permuted roles of the three points
    wrongs_m = [
        _cross_matrix(z2, z1, z3),
        _cross_matrix(z1, z3, z2),
        _cross_matrix(z3, z2, z1),
        _swapped_numerator(m),
        _sign_slip(m),
    ]
    dist = _mob_dists(
        n,
        wrongs_m,
        [
            "cx2-mobius-point-roles-swapped",
            "cx2-mobius-point-roles-swapped",
            "cx2-mobius-point-roles-swapped",
            "cx2-mobius-inverse-quotient",
            "cx2-mobius-numerator-sign-slip",
        ],
    )
    shown = mob_x(n)
    pts_x = [_point_eq(z) for z in (z1, z2, z3)]
    values = ("0", "1", "∞")
    tex_values = ("0", "1", r"\infty")
    cond = X(
        ", ".join(f"T({_pt(z).p}) = {v}" for z, v in zip(pts, values)),
        ",\\ ".join(f"T({_pt(z).t}) = {v}" for z, v in zip(pts, tex_values)),
    )
    del pts_x
    working = [
        "The map with z1 → 0, z2 → 1, z3 → ∞ is T(z) = (z - z1)(z2 - z3) / ((z - z3)(z2 - z1)), with the factors that contain ∞ removed.",
        f"Simplifying gives T(z) = {shown.p}.",
    ]
    lead = [
        lambda: msay(
            "Find the Möbius transformation ", X("T", "T"), " with ", cond, ". Give T(z) as a single fraction."
        ),
        lambda: msay("There is exactly one Möbius transformation with ", cond, ". Write T(z) as a single fraction."),
        lambda: msay(
            "Determine ", X("T(z)", "T(z)"), ", the Möbius transformation satisfying ", cond, ", as a single fraction."
        ),
    ][rng.randrange(3)]
    return single(lead(), mob_sym(n), working, [], dist=dist, answer_text=shown.p, answer_latex=shown.t)


def _swapped_numerator(m):
    a, b, c, d = m
    return (c, d, a, b)


def _sign_slip(m):
    a, b, c, d = m
    return (a, -b, c, d)


def _mob_dists(answer_n, wrong_ms, tags) -> list:
    out: list[Distractor] = []
    seen = [answer_n]
    for wm, tag in zip(wrong_ms, tags):
        if mob_det(wm).is_zero():
            continue
        nm = mob_norm(wm)
        if any(nm == s for s in seen):
            continue
        seen.append(nm)
        out.append(Distractor(mob_x(nm).p, tag, astr(mob_sym(nm))))
    if len(out) < 3:
        a, b, c, d = answer_n
        for wm in (
            (a, b, c, -d),
            (a, -b, -c, d),
            (d, b, c, a),
            (a, c, b, d),
            (d, -b, -c, a),
            (a, b, c, d + C(1)),
            (a, b + C(1), c, d),
        ):
            if mob_det(wm).is_zero():
                continue
            nm = mob_norm(wm)
            if any(nm == s for s in seen):
                continue
            seen.append(nm)
            out.append(Distractor(mob_x(nm).p, "cx2-mobius-numerator-sign-slip", astr(mob_sym(nm))))
            if len(out) >= 3:
                break
    return out


def _hard_fourth_point(rng):
    for _ in attempts(1500, "a three-to-three map"):
        m = _real_matrix(rng, -2, 3)
        if rng.random() < 0.5:
            m = (
                C(rng.randint(-2, 2), rng.randint(-1, 1)),
                C(rng.randint(-2, 2)),
                C(rng.randint(-1, 1), rng.randint(-1, 1)),
                C(rng.randint(-2, 2), rng.randint(-1, 1)),
            )
        if mob_det(m).is_zero():
            continue
        zs = rng.sample(_POINTS, 4)
        ws = [mob_apply(m, z) for z in zs]
        if any(w is INF for w in ws) or len({w for w in ws}) < 4:
            continue
        if not all(_tidy(w, 12) for w in ws):
            continue
        if any(abs(w.re) > 12 or abs(w.im) > 12 for w in ws):
            continue
        break
    z1, z2, z3, z4 = zs
    w1, w2, w3, w4 = ws
    value = w4.sym()
    # the cross-ratio route as a second check
    cz = (z4 - z1) * (z2 - z3) / ((z4 - z3) * (z2 - z1))
    cw = (w4 - w1) * (w2 - w3) / ((w4 - w3) * (w2 - w1))
    check(cz == cw, "cross-ratio preserved")
    # wrong: use the map for z -> w with the roles of w2 and w3 swapped, or reuse the cross-ratio of the z's as w
    cz_swapped = (z4 - z1) * (z3 - z2) / ((z4 - z2) * (z3 - z1))
    ws2 = (w1, w3, w2)
    swapped = _solve_w(ws2, cz_swapped)
    wrongs = [
        (swapped.sym() if swapped is not None else (-w4).sym(), "cx2-mobius-point-roles-swapped"),
        (cz.sym(), "cx2-mobius-cross-ratio-not-solved-for-w"),
        (((w1 + w2 + w3) / 3).sym() if ((w1 + w2 + w3) / 3) != w4 else (w4 + C(1)).sym(), "cx2-mobius-points-averaged"),
        ((w4.conj()).sym() if w4.conj() != w4 else (-w4).sym(), "cx2-mobius-conjugate-quotient"),
        ((-w4).sym(), "calc-sign-slip"),
    ]
    conds = ", ".join(f"T({cx_x(z).p}) = {cx_x(w).p}" for z, w in zip((z1, z2, z3), (w1, w2, w3)))
    conds_t = ",\\ ".join(f"T({cx_x(z).t}) = {cx_x(w).t}" for z, w in zip((z1, z2, z3), (w1, w2, w3)))
    cond = X(conds, conds_t)
    working = [
        "A Möbius transformation preserves the cross-ratio, so (w - w1)(w2 - w3)/((w - w3)(w2 - w1)) = (z - z1)(z2 - z3)/((z - z3)(z2 - z1)).",
        f"At z = {cx_x(z4).p} the right side is {atext(cz.sym())}. Solving for w gives {atext(value)}.",
    ]
    lead = [
        lambda: msay(
            "The Möbius transformation T satisfies ", cond, ". Find ", X(f"T({cx_x(z4).p})", f"T({cx_x(z4).t})"), "."
        ),
        lambda: msay("A Möbius transformation T has ", cond, ". What is the image of ", _point_eq(z4), "?"),
        lambda: msay(
            "Given that T is a Möbius transformation with ",
            cond,
            ", determine ",
            X(f"T({cx_x(z4).p})", f"T({cx_x(z4).t})"),
            ".",
        ),
    ][rng.randrange(3)]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-mobius-cross-ratio-not-solved-for-w")


def _solve_w(ws, ratio):
    """The w with (w - w1)(w2 - w3)/((w - w3)(w2 - w1)) = ratio, or None."""
    w1, w2, w3 = ws
    # (w - w1) A = ratio (w - w3) B  with A = w2 - w3, B = w2 - w1
    a_, b_ = w2 - w3, w2 - w1
    den = a_ - ratio * b_
    if den.is_zero():
        return None
    return (w1 * a_ - ratio * w3 * b_) / den


def _hard_iterate(rng):
    for _ in attempts(500, "a second iterate"):
        m = _real_matrix(rng, -3, 3)
        sq = mob_mul(m, m)
        if sq[2].is_zero() or mob_det(sq).is_zero():
            continue
        nsq = mob_norm(sq)
        if not all(_tidy(e, 30) for e in nsq):
            continue
        if _same_m(sq, m):
            continue
        break
    a, b, c, d = m
    entry = (a * a, b * b, c * c, d * d)
    wrongs_m = [
        entry,
        (a * a + b * c, b * (a + d), c * (a + d), b * c + d * d)
        if False
        else (a * a + b * c, a * b + b * d, c * a, c * b + d * d),
        mob_mul(m, (a, c, b, d)),
        (a * a - b * c, b * (a - d), c * (a - d), d * d - b * c),
    ]
    dist = _mob_dists(
        nsq,
        wrongs_m,
        [
            "cx2-mobius-entrywise-square",
            "cx2-mobius-composition-slip",
            "cx2-mobius-composition-slip",
            "cx2-mobius-composition-slip",
        ],
    )
    shown = mob_x(nsq)
    tx = _tfun(m)
    working = [
        "T(T(z)) has matrix M² for the matrix M = [[a, b], [c, d]] of T.",
        f"M² = [[a² + bc, b(a + d)], [c(a + d), bc + d²]], so T(T(z)) = {shown.p}.",
    ]
    lead = [
        lambda: msay("Let ", tx, ". Find ", X("T(T(z))", "T(T(z))"), " as a single fraction in z."),
        lambda: msay(
            "The Möbius transformation ",
            tx,
            " is applied twice. Write the composite ",
            X("(T∘T)(z)", r"(T\circ T)(z)"),
            " as a single fraction.",
        ),
        lambda: msay(
            "For ", tx, ", determine the second iterate ", X("T(T(z))", "T(T(z))"), ", simplified to a single fraction."
        ),
    ][rng.randrange(3)]
    return single(lead(), mob_sym(nsq), working, [], dist=dist, answer_text=shown.p, answer_latex=shown.t)


def _loxodromic_matrix(p: C, q: C, k: C):
    """The transformation with fixed points p and q and multiplier k, scaled to Gaussian integers."""
    mi = (-q, p, C(-1), C(1))  # proportional to the inverse of z -> (z - p)/(z - q)
    pq = (C(1), -p, C(1), -q)
    diag = (k, C(0), C(0), C(1))
    t = mob_mul(mob_mul(mi, diag), pq)
    dens = [e.re.denominator for e in t] + [e.im.denominator for e in t]
    lcm = 1
    for dn in dens:
        lcm = lcm * dn // __import__("math").gcd(lcm, dn)
    return tuple(e * C(lcm) for e in t)


def _hard_fixed(rng):
    for _ in attempts(1500, "a fixed-point question"):
        if rng.random() < 0.45:
            # parabolic: one fixed point p, 1/(T - p) = 1/(z - p) + k
            p = rng.choice([C(1, 1), C(0, 1), C(1), C(-1, 1), C(2, -1), C(0, 2), C(-1)])
            k = rng.choice([C(1), C(2), C(-1), C(0, 1), C(1, 1), C(3)])
            m = (C(1) + k * p, -k * p * p, k, C(1) - k * p)
            fixed = [p]
        else:
            p, q = rng.sample([C(a, b) for a in (-2, -1, 0, 1, 2) for b in (-1, 0, 1, 2)], 2)
            if p.re == q.re:
                continue
            k = rng.choice([C(2), C(3), C(-1), C(0, 1), C(1, 1), C(2, 1), C(0, 2)])
            m = _loxodromic_matrix(p, q, k)
            fixed = [p, q]
        if mob_det(m).is_zero() or m[2].is_zero():
            continue
        if any(abs(e.re) > 30 or abs(e.im) > 30 for e in m):
            continue
        break
    n = mob_norm(m)
    a, b, c, d = n
    for f in fixed:
        check(mob_apply(m, f) == f, "fixed point")
    if len(fixed) == 2:
        p, q = fixed
        target = p if p.re > q.re else q
        other = q if target == p else p
        wording = "the fixed point with the larger real part"
    else:
        target = other = fixed[0]
        wording = "the only fixed point"
    value = target.sym()
    check(((d - a) ** 2 + C(4) * b * c).is_zero() == (len(fixed) == 1), "discriminant")
    wrongs = [
        (other.sym() if other != target else (-target).sym(), "cx2-mobius-wrong-fixed-point"),
        (
            ((a - d) / (c * 2)).sym()
            if len(fixed) == 1
            else ((a - d) / c - target).sym()
            if False
            else ((a - d) / (c * 2)).sym(),
            "cx2-mobius-fixed-point-equation-slip",
        ),
        (((d - a) / (c * 2)).sym(), "cx2-mobius-fixed-point-sign-slip"),
        ((-target).sym(), "calc-sign-slip"),
        (target.conj().sym() if target.conj() != target else (target + C(1)).sym(), "cx2-mobius-conjugate-quotient"),
        ((b / c).sym(), "cx2-mobius-fixed-point-equation-slip"),
    ]
    tx = _tfun(m)
    if len(fixed) == 1:
        lead = [
            lambda: msay("The Möbius transformation ", tx, " has exactly one fixed point. Find it."),
            lambda: msay("Find the fixed point of ", tx, ", which has only one."),
            lambda: msay("Let ", tx, ". Show that T(z) = z has a repeated root and give the fixed point."),
        ][rng.randrange(3)]
    else:
        lead = [
            lambda: msay("The Möbius transformation ", tx, " has two fixed points. Find ", X(wording, wording), "."),
            lambda: msay("Let ", tx, ". Solve T(z) = z and give ", X(wording, wording), "."),
            lambda: msay("Find, in exact form, ", X(wording, wording), " of ", tx, "."),
        ][rng.randrange(3)]
    working = [
        "Fixed points solve c z² + (d - a) z - b = 0.",
        f"Here the equation is {_quad_text(c, d - a, -b)} = 0 and the discriminant is {atext(((d - a) ** 2 + C(4) * b * c).sym())}.",
        f"The required fixed point is {atext(value)}.",
    ]
    return single(lead(), value, working, wrongs, fallback_tag="cx2-mobius-fixed-point-equation-slip")


_TIERS = {
    "easy": [_easy_image, _easy_image, _easy_special],
    "medium": [_medium_fixed, _medium_fixed, _medium_from_image],
    "hard": [_hard_cross, _hard_fourth_point, _hard_iterate, _hard_fixed],
}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5a_mobius(rng: random.Random, tier: str = "easy") -> dict:
    """An image, a preimage, a fixed point or a three-point map for a Möbius transformation, exact."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return rng.choice(_TIERS[tier])(rng)


# -- exam ------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5a_mobius(rng: random.Random, tier: str = "exam") -> dict:
    """A real transformation: T(z1), T(T(z1)), the sum of the fixed points, T of that sum."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a Möbius chain"):
        a, b, c, d = (rng.randint(-4, 4) for _ in range(4))
        if c == 0 or a * d - b * c == 0 or (d - a) ** 2 + 4 * b * c == 0:
            continue  # c = 0 or a repeated fixed point would not give two fixed points
        z1 = rng.choice([-3, -2, -1, 1, 2, 3])
        den1 = c * z1 + d
        if den1 == 0:
            continue
        w1 = Fr(a * z1 + b, den1)
        den2 = c * w1 + d
        if den2 == 0:
            continue
        w2 = (a * w1 + b) / den2
        s = Fr(a - d, c)
        den3 = c * s + d
        if den3 == 0:
            continue
        w3 = (a * s + b) / den3
        if len({w1, w2, w3, Fr(z1)}) < 4 or s == 0 or w1 == 0 or w2 == 0 or w3 == 0:
            continue
        if max(abs(w1.denominator), abs(w2.denominator), abs(w3.denominator)) > 60:
            continue
        break
    m = (C(a), C(b), C(c), C(d))
    tx = _tfun(m)
    stem = msay("Let ", tx, ", a Möbius transformation with real coefficients.")
    w_a = Link(
        msay("Find ", X(f"T({z1})", f"T({z1})"), "."),
        qsym(w1),
        "fraction",
        1,
        [
            f"T({z1}) = ({join_signed([f'{a}·{pn(z1)}', str(b)])})/({join_signed([f'{c}·{pn(z1)}', str(d)])}) = {atext(qsym(w1))}."
        ],
        [
            (qsym(Fr(1) / w1), "cx2-mobius-inverse-quotient"),
            (qsym(Fr(a * z1 - b, den1)), "cx2-mobius-numerator-sign-slip"),
            (qsym(Fr(a * z1 + b, c * z1 - d)) if c * z1 - d != 0 else qsym(-w1), "cx2-mobius-denominator-sign-slip"),
            (qsym(Fr(a * z1 + b, d)) if d != 0 else qsym(w1 + 1), "cx2-mobius-denominator-dropped"),
            (qsym(-w1), "calc-sign-slip"),
            (qsym(Fr(a, c)), "cx2-mobius-image-inverted"),
        ],
    )
    w_b = Link(
        msay("Hence find ", X(f"T(T({z1}))", f"T(T({z1}))"), "."),
        qsym(w2),
        "fraction",
        2,
        [f"T(T({z1})) = T({atext(qsym(w1))}) = {atext(qsym(w2))}."],
        [
            (qsym(Fr(1) / w2), "cx2-mobius-inverse-quotient"),
            (qsym(w1 * w1), "cx2-mobius-composition-slip"),
            (qsym(Fr(a * z1 + b, den1) * 0 + (a * w1 - b) / den2), "cx2-mobius-numerator-sign-slip"),
            (
                qsym(Fr(a * (a * z1 + b) + b, c * (c * z1 + d) + d)) if c * (c * z1 + d) + d != 0 else qsym(w2 + 1),
                "cx2-mobius-composition-slip",
            ),
        ],
        ft=f"({a}*part_a + {b})/({c}*part_a + {d})",
    )
    w_c = Link(
        msay("T has two fixed points (real or complex). Find their sum."),
        qsym(s),
        "fraction",
        2,
        [f"Fixed points solve {c}z² + ({d - a})z - ({b}) = 0, so their sum is -({d - a})/{c} = {atext(qsym(s))}."],
        [
            (-qsym(s), "cx2-mobius-fixed-point-sign-slip"),
            (qsym(Fr(-b, c)), "cx2-mobius-fixed-point-equation-slip"),
            (qsym(Fr(d - a, c)), "cx2-mobius-fixed-point-sign-slip"),
            (qsym(Fr(a + d, c)), "cx2-mobius-fixed-point-equation-slip"),
            (qsym(Fr(a, c)), "cx2-mobius-fixed-point-equation-slip"),
            (qsym(Fr(d, c)), "cx2-mobius-fixed-point-equation-slip"),
        ],
    )
    w_d = Link(
        msay("Find the image under T of the number you found in part (c)."),
        qsym(w3),
        "fraction",
        2,
        [f"T({atext(qsym(s))}) = {atext(qsym(w3))}."],
        [
            (qsym(Fr(1) / w3), "cx2-mobius-inverse-quotient"),
            (qsym(s), "cx2-mobius-fixed-point-equation-slip"),
            (qsym(Fr(a * s - b) / (c * s + d)), "cx2-mobius-numerator-sign-slip"),
            (qsym(Fr(a * s + b) / (c * s - d)) if c * s - d != 0 else qsym(w3 + 1), "cx2-mobius-denominator-sign-slip"),
        ],
        ft=f"({a}*part_c + {b})/({c}*part_c + {d})",
    )
    return exam_dict(stem, [w_a, w_b, w_c, w_d])
